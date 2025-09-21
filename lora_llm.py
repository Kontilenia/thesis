import json
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from transformers import (AutoTokenizer,
                          AutoModelForCausalLM,
                          BitsAndBytesConfig,
                          TrainingArguments,
                          Trainer,
                          pipeline,
                          DataCollatorForLanguageModeling,
                          PreTrainedTokenizer)
from peft import LoraConfig, get_peft_model
import huggingface_hub
import os
import logging
from tqdm import tqdm
from sklearn.metrics import (accuracy_score,
                             classification_report,
                             confusion_matrix)
from typing import List
import wandb

cache_dir = "/home/ec2-user/SageMaker"
os.environ['HF_HOME'] = cache_dir
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["WANDB_DISABLED"] = "false"

class CustomTextDataset(Dataset):
    """
    Custom dataset class for QEvasion Dataset.

    Attributes:
        texts (List[str]): A list of text samples to be tokenized.
        tokenizer (PreTrainedTokenizer): A tokenizer from the Hugging Face Transformers library.
    """

    def __init__(self,
                 texts: List[str],
                 tokenizer: PreTrainedTokenizer) -> None:
        self.texts = texts
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])

        encoding = self.tokenizer(
            text,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
        }


def create_prompted_text(dataset: pd.DataFrame,
                         label_name: str) -> List[str]:
    """
    Creates prompted text for classification from the dataset.

    Args:
        dataset (pd.DataFrame): The dataset containing interview questions and 
        answers.
        label_name (str): The name of the label column for classification.
        
    Returns:
        List[str]: A list of formatted prompt texts for each interview response.
    """
    texts = []
    global class_names
    class_names = list(dataset[label_name].unique())
    class_names_text = ', '.join(class_names)

    for _, row in dataset.iterrows():
        texts.append(
            f"You will be given a part of an interview. "
            f"Classify the response to the selected question "
            f"into one of the following categories: {class_names_text}"
            # f". \n Respond with only the category name.\n"
            f". \n\n ### Part of the interview ### \nIntervier:"
            f" {row['interview_question']} \nResponse:"
            f" {row['interview_answer']} \n\n### Selected Question ###\n"
            f"{row['question']} \n\nLabel: {row[label_name]}"
        )
    return texts


def load_qevasion_dataset(tokenizer: PreTrainedTokenizer,
                          label_name: str = "clarity_label") -> tuple:
    """
    Loads the QEvasion dataset, splits it into training and validation sets,
    and creates prompted texts for both sets.

    Args:
        tokenizer: The tokenizer to be used for text encoding.
        label_name (str): The name of the label column for classification.
        
    Returns:
        tuple: A tuple containing the training and validation datasets.
    """

    # Get train set data
    df = pd.read_csv('preprocessed_data/train_set.csv')[['question',
                                                         'interview_question',
                                                         'interview_answer',
                                                         label_name]]

    # Split train set to train and validation data
    np.random.seed(2024)
    msk = np.random.rand(len(df)) < 0.9
    train = df[msk]
    validation = df[~msk]

    train.reset_index(drop=True, inplace=True)
    validation.reset_index(drop=True, inplace=True)

    train_texts = create_prompted_text(train, label_name)
    validation_texts = create_prompted_text(validation,
                                            label_name)

    # print("Example of train test:" + train_texts[1])
    # print("Example of validation test:" + validation_texts[1])

    train_texts = train_texts #[:20]
    validation_texts = validation_texts #[:2]
    return (CustomTextDataset(train_texts, tokenizer),
            CustomTextDataset(validation_texts, tokenizer))


class CastOutputToFloat(nn.Sequential):
    def forward(self, x): return super().forward(x).to(torch.float32)


def print_trainable_parameters(model: nn.Module) -> None:
    """
    Prints the number of trainable parameters in the model.

    Args:
        model (nn.Module): The model for which to count trainable parameters.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"""trainable params: {trainable_params} || all params: {all_param}
        || trainable%: {100 * trainable_params / all_param}"""
    )


def finetuning(model_name: str,
               output_model_dir: str,
               label_taxonomy: str,
               lr: float,
               epochs: int) -> tuple:
    """
    Fine-tunes a pre-trained language model with LoRA.

    Args:
        model_name (str): The name of the pre-trained model.
        output_model_dir (str): Directory to save the fine-tuned model.
        label_taxonomy (str): The label taxonomy for the dataset.
        lr (float): Learning rate for training.
        epochs (int): Number of training epochs.

    Returns:
        tuple: The fine-tuned model and tokenizer.
    """

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16
        ),
        device_map='auto',
        dtype=torch.float16,
        cache_dir=cache_dir
    )

    if "llama" not in model_name:
        # Mistral/Ministral or other models        
        tokenizer = AutoTokenizer.from_pretrained(model_name,
                                                 cache_dir=cache_dir,
                                                 legacy=False) 
    else:
        # LLaMA models
        tokenizer = AutoTokenizer.from_pretrained(model_name,
                                              cache_dir=cache_dir)
    tokenizer.pad_token = tokenizer.eos_token

    for param in model.parameters():
        param.requires_grad = False  # Freeze the model - train adapters later
        if param.ndim == 1:
            # Cast the small parameters to fp32 for stability
            param.data = param.data.to(torch.float32)

    # Reduce number of stored activation
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model.lm_head = CastOutputToFloat(model.lm_head)

    config = LoraConfig(
        r=16,  # Attention heads
        lora_alpha=32,  # Alpha scaling
        # target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )

    model = get_peft_model(model, config)
    print_trainable_parameters(model)

    # Load data
    train_data, validation_data = load_qevasion_dataset(tokenizer,
                                                        label_taxonomy)

    print(f"""Found {len(train_data)} instances for training and
    {len(validation_data) } instances for validation.""")

    grad_accum_steps = 8

    # Train model
    print("Training...")

    trainer = Trainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=validation_data,
        args=TrainingArguments(
            per_device_train_batch_size=1,
            per_device_eval_batch_size=1,
            gradient_accumulation_steps=grad_accum_steps,
            eval_accumulation_steps=1,
            warmup_steps=100,
            max_steps=int((len(train_data)*epochs)/grad_accum_steps),
            learning_rate=lr,
            fp16=True,
            logging_steps=1,
            # eval_steps * int((len(train_data)*epochs)/grad_accum_steps)
            # if eval_steps < 1
            eval_steps=0.33 / epochs,
            eval_strategy="steps",
            do_eval=True,
            report_to="wandb",
            # save_steps= 2,
            # num_train_epochs=epochs,
            # output_dir=f'outputs_{out_dir}'
        ),
        data_collator=DataCollatorForLanguageModeling(tokenizer,
                                                      mlm=False)
    )

    # Silence the warnings
    model.config.use_cache = False
    trainer.train()

    # Save the model
    model.save_pretrained(output_model_dir)

    return model, tokenizer


def create_test_prompted_text(dataset: pd.DataFrame,
                              label_name: str) -> List[str]:
    """
    Creates prompted text for classification from the test dataset.

    Args:
        dataset (pd.DataFrame): The dataset containing interview questions
        and answers.
        label_name (str): The name of the label column for classification.

    Returns:
        List[str]: A list of formatted prompt texts for each interview response.
    """

    texts = []
    classes_names = ', '.join(list(dataset[label_name].unique()))

    for _, row in dataset.iterrows():
        texts.append(
            f"You will be given a part of an interview."
            f"Classify the response to the selected question"
            f"into one of the following categories: {classes_names}"
            f". \n\n ### Part of the interview ### \nIntervier:"
            f" {row['interview_question']} \nResponse:"
            f" {row['interview_answer']} \n\n### Selected Question ###\n"
            f"{row['question']} \n\nLabel:"
        )
    return texts


def create_test_prompted_text_name_summaries(dataset: pd.DataFrame,
                              label_name: str) -> List[str]:
    """
    Creates prompted text for classification from the test dataset.

    Args:
        dataset (pd.DataFrame): The dataset containing interview questions
        and answers.
        label_name (str): The name of the label column for classification.

    Returns:
        List[str]: A list of formatted prompt texts for each interview response.
    """

    texts = []
    classes_names = ', '.join(list(dataset[label_name].unique()))

    for _, row in dataset.iterrows():
        texts.append(
            f"You will be given a part of an interview."
            f"Classify the response to the selected question"
            f"into one of the following categories: {classes_names}"
            f". \n\n ### Part of the interview ### \nIntervier:"
            f" {row['interview_question']} \nResponse:"
            f" {row['interview_answer']} \n\n### Selected Question: ###\n"
            f" {row['names_information']} \n\n### Information about mentioned people: ###\n"
            f"{row['question']} \n\nLabel:"
        )
    return texts


def create_inference_prompted_text(dataset: pd.DataFrame,
                                   label_name: str) -> List[str]:
    """
    Creates prompted text for classification from the test dataset.

    Args:
        dataset (pd.DataFrame): The dataset containing interview questions
        and answers.
        label_name (str): The name of the label column for classification.

    Returns:
        List[str]: A list of formatted prompt texts for each interview response.
    """

    texts = []
    classes_names = ', '.join(list(dataset[label_name].unique()))

    for _, row in dataset.iterrows():
        texts.append(
            f"You will be given a part of an interview."
            f"Classify the response to the selected question"
            f"into one of the following categories: {classes_names}"
            f". \n\n ### Part of the interview ### \nIntervier:"
            f" {row['interview_question']} \nResponse:"
            f" {row['interview_answer']} \n\n### Selected Question ###\n"
            f"{row['question']} \n\nLabel:"
        )
    return texts


def predict(test: pd.DataFrame,
            categories: list,
            model: nn.Module,
            tokenizer: PreTrainedTokenizer,
            after_training: bool = False) -> list:
    """
    Generates predictions for the test dataset using the provided model
    and tokenizer.

    Args:
        test (pd.DataFrame): The test dataset containing prompts.
        categories (list): The list of possible categories for classification.
        model (nn.Module): The trained model for making predictions.
        tokenizer (PreTrainedTokenizer): The tokenizer associated with the model.

    Returns:
        list: A list of predicted labels for the test dataset.
    """
    batch_size = 8
    category_set = set(category.lower() for category in categories)
    model.eval()  # Set the model to evaluation mode
    model.config.use_cache = False

    with torch.no_grad(), torch.autocast("cuda"):
        y_pred = []

        if after_training:
            pipe = pipeline(task="text-generation",
                            model=model,
                            tokenizer=tokenizer,
                            max_new_tokens=5,
                            temperature=0.1)
        else:
            pipe = pipeline(task="text-generation",
                            model=model,
                            tokenizer=tokenizer,
                            temperature=0.1)

        terminators = [
            pipe.tokenizer.eos_token_id,
            pipe.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]

        for i in tqdm(range(0, len(test), batch_size)):
            prompts = test.iloc[i:i + batch_size]["text"].tolist()
            results = pipe(prompts,
                           eos_token_id=terminators)

            for result in results:
                answer = result[0]['generated_text'].split("Label:")[-1].strip()
                matched = False

                for category in category_set:
                    if category in answer.lower():
                        print(f"Right label: {answer.lower()}")
                        y_pred.append(category)
                        matched = True
                        break

                if not matched:
                    print(f"Wrong label: {answer.lower()}")
                    y_pred.append("none")

    return y_pred


def evaluation_report(y_true: pd.Series,
                      y_pred: pd.Series,
                      labels: list,
                      run=None) -> None:
    """
    Generates and prints an evaluation report including accuracy and 
    classification metrics.
    
    Args:
        y_true (np.ndarray): The true labels for the test dataset.
        y_pred (np.ndarray): The predicted labels for the test dataset.
        labels (list): The list of label names.
        run: Optional; a wandb run object for logging metrics.
    """
    mapping = {label: idx for idx, label in enumerate(labels)}

    def map_func(x):
        return mapping.get(x, -1)  # Map to -1 if not found

    y_true_mapped = np.vectorize(map_func)(y_true)
    y_pred_mapped = np.vectorize(map_func)(y_pred)

    if run:
        wandb_log_dict = {}

    # Calculate accuracy
    accuracy = accuracy_score(y_true=y_true_mapped, y_pred=y_pred_mapped)
    print(f'Accuracy: {accuracy:.2f}')
    if run:
        wandb_log_dict["Accuracy"] = accuracy

    # Generate accuracy report
    unique_labels = set(y_true_mapped)  # Get unique labels

    for label in unique_labels:
        label_indices = [i for i in range(len(y_true_mapped))
                         if y_true_mapped[i] == label]
        label_y_true = [y_true_mapped[i] for i in label_indices]
        label_y_pred = [y_pred_mapped[i] for i in label_indices]
        label_accuracy = accuracy_score(label_y_true, label_y_pred)
        print(f'Accuracy for label {labels[label]}: {label_accuracy:.2f}')
        if run:
            wandb_log_dict[f"Accuracy for label {labels[label]}"] = label_accuracy

    unsplit_labels = [label.replace(" ", "_") for label in labels]

    # Generate classification report
    class_report = classification_report(y_true=y_true_mapped,
                                         y_pred=y_pred_mapped,
                                         target_names=unsplit_labels,
                                         labels=list(range(len(labels))))
    print('\nClassification Report:')
    print(class_report)

    report_columns = ["Class", "Precision", "Recall", "F1-score", "Support"]
    report_table = []
    class_report = class_report.splitlines()
    for line in class_report[2:(len(labels)+2)]:
        report_table.append(line.split())

    if run:
        wandb_log_dict["Classification Report"] = wandb.Table(
            data=report_table,
            columns=report_columns)

    # For not predicted classes
    mask = y_pred_mapped != -1
    y_true_mapped2 = y_true_mapped[mask]
    y_pred_mapped2 = y_pred_mapped[mask]

    # Generate confusion matrix
    conf_matrix = confusion_matrix(y_true=y_true_mapped,
                                   y_pred=y_pred_mapped,
                                   labels=list(range(len(labels))))
    print('\nConfusion Matrix:')
    print(conf_matrix)

    if run:
        wandb_log_dict["Confusion Matix"] = wandb.plot.confusion_matrix(
            y_true=y_true_mapped2,
            preds=y_pred_mapped2,
            class_names=labels
        )
        run.log(wandb_log_dict)

def create_labels_train_set(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create labels for the dataset

    Arguments:
        df: Dataframe

    Returns:
        df: Dataframe with labels
    """

    clarity_mapping ={
        'explicit': 'direct reply',
        'implicit': 'indirect',
        'dodging': "indirect",
        'deflection': "indirect",
        'partial/half-answer': "indirect",
        'general': "indirect",
        'contradictory': "indirect",
        'declining_to_answer': "direct non-reply",
        'claims_ignorance': "direct non-reply",
        'clarification': "direct non-reply",
        'diffusion': "indirect",
    }
    
    df["clarity_label"] = df["label"].map(clarity_mapping)
    df.rename(columns={"label": "evasion_label"}, inplace=True)
    return df


def evaluate(base_model_name: str,
             fine_tuned_model_path: str,
             train_label_name: str,
             test_label_name: str,
             test_set_path: str = 'preprocessed_data/test_set.csv',
             added_name_summary: bool = False,
             model: nn.Module = None,
             tokenizer: PreTrainedTokenizer = None,
             run=None) -> None:
    """
    Evaluates the fine-tuned model on the test dataset and
    generates an evaluation report.

    Args:
        base_model_name (str): The name of the base model to load.
        fine_tuned_model_path (str): The path to the fine-tuned model.
        train_label_name (str): The name of the label column for classification
        in train set.
        test_label_name (str): The name of the label column for classification
        in test set.
        model (nn.Module): Optional; a pre-trained model to use for evaluation.
        tokenizer (PreTrainedTokenizer): Optional; a tokenizer to use for 
        evaluation.
        run: Optional; a wandb run object for logging metrics.
    """

    if not model:
        model = AutoModelForCausalLM.from_pretrained(
            fine_tuned_model_path,
            return_dict=True,
            low_cpu_mem_usage=True,
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16
            ),
            # dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
            offload_folder="offload/",
            cache_dir=cache_dir
        )
        
    if not tokenizer:
        if "llama" not in base_model_name:
            # Mistral/Ministral or other models        
            tokenizer = AutoTokenizer.from_pretrained(base_model_name,
                                                     cache_dir=cache_dir,
                                                     legacy=False) 
        else:
            # LLaMA models
            tokenizer = AutoTokenizer.from_pretrained(base_model_name,
                                                  cache_dir=cache_dir)   
        tokenizer.pad_token = tokenizer.eos_token

    list_of_columns = [
        'question',
        'interview_question',
        'interview_answer',
        test_label_name,
        train_label_name
    ]

    if added_name_summary:
        list_of_columns.extend(['names_information'])

    # Get test set data
    test_df = pd.read_csv(test_set_path)[list_of_columns]
    
    # creating bool series False for NaN values
    test_df = test_df[test_df["evasion_label"].notnull()] #.iloc[:10]

    if added_name_summary:
        test_texts = create_test_prompted_text_name_summaries(test_df, train_label_name)     
    else: 
        test_texts = create_test_prompted_text(test_df, train_label_name)

    dataset = pd.DataFrame(test_texts, columns=['text'])

    # NEW
    labels = [label.lower() for label in list(test_df[train_label_name].unique())]
    test_labels = [label.lower() for label in list(test_df[test_label_name].unique())]

    y_pred = predict(dataset, labels, model, tokenizer, True)
    y_pred = pd.DataFrame({"label": y_pred})
    y_pred_evasion_based = create_labels_train_set(y_pred)[test_label_name]
    y_true = test_df[test_label_name].str.lower()
    evaluation_report(y_true, y_pred_evasion_based, test_labels, run)

def inference(base_model_name: str,
              fine_tuned_model_path: str,
              label_name: str,
              model: nn.Module = None,
              tokenizer: PreTrainedTokenizer = None,
              ) -> None:
    """
    Inference a model on the test dataset and
    generates an evaluation report.

    Args:
        base_model_name (str): The name of the base model to load.
        fine_tuned_model_path (str): The path to the fine-tuned model.
        label_name (str): The name of the label column for classification.
        model (nn.Module): Optional; a pre-trained model to use for inference.
        tokenizer (PreTrainedTokenizer): Optional; a tokenizer to use for inference.
    """

    if not model:

        model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            return_dict=True,
            low_cpu_mem_usage=True,
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16
            ),
            # dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
            offload_folder="offload/",
            cache_dir=cache_dir
        )
    if not tokenizer:
        if "llama" not in base_model_name:
            # Mistral/Ministral or other models        
            tokenizer = AutoTokenizer.from_pretrained(base_model_name,
                                                     cache_dir=cache_dir,
                                                     legacy=False) 
        else:
            # LLaMA models
            tokenizer = AutoTokenizer.from_pretrained(base_model_name,
                                                  cache_dir=cache_dir)
        tokenizer.pad_token = tokenizer.eos_token

    # Get test set data
    test_df = pd.read_csv('preprocessed_data/test_set.csv')[[
        'question',
        'interview_question',
        'interview_answer',
        label_name
    ]] # [:20]

    test_texts = create_inference_prompted_text(test_df, label_name)
    dataset = pd.DataFrame(test_texts, columns=['text'])

    labels = [label.lower() for label in list(test_df[label_name].unique())]

    y_pred = predict(dataset, labels, model, tokenizer)
    y_pred = pd.Series(y_pred, name=label_name)
    y_true = test_df[label_name].str.lower()
    evaluation_report(y_true, y_pred, labels)