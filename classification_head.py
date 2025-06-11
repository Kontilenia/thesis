from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
import torch

# Load the model with quantization
model_name = "meta-llama/Meta-Llama-3-8B"
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type='nf4',
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16
)

model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    quantization_config=quantization_config,
    num_labels=4,
    device_map='auto'
)

# Configure LoRA
lora_config = LoraConfig(
    r=16,
    lora_alpha=8,
    target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj'],
    lora_dropout=0.05,
    bias='none',
    task_type='SEQ_CLS'
)

# Prepare the model for training with LoRA
model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, lora_config)

# Define training arguments
training_args = TrainingArguments(
    output_dir='sentiment_classification',
    learning_rate=1e-4,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=1,
    logging_steps=1,
    weight_decay=0.01,
    evaluation_strategy='epoch',
    save_strategy='epoch',
    load_best_model_at_end=True,
    report_to="none"
)

# Create a Trainer instance
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets['train'],
    eval_dataset=tokenized_datasets['val'],
    tokenizer=tokenizer,
    data_collator=collate_fn,
    compute_metrics=compute_metrics,
    class_weights=class_weights,
)

# Start training
train_result = trainer.train()
