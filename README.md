# Master's thesis: Prediction of clarity on political speech using LLMs

This repository is used for storing the implementation part of my master's thesis.

The goal of the thesis is to utilize the dataset from *"I Never Said That: A dataset, taxonomy and baselines on response clarity classification"* paper and LLMs to improve the metrics of the paper and explore different DL techniques.

Progress so far:

1) Created a cleaner version of train and test set: Train data have been taken from https://huggingface.co/datasets/ailsntua/QEvasion, while test data exist on data folder. The proccess to clean the data is on data_cleaning.py file. The results of that proccess are 4 files in preproccessed_data folder, two for training set and two for test set. For every set there is a full version incliding all the columns of initial datasets and a standard one including only the part needed for training.

2) Experiments on how to perform LoRA finetuning to LLMs: The experiments can be found on the lora_llm.ipynb notebook.

