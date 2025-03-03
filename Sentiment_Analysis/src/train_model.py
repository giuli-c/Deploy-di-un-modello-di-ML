# import torch
# import os
# import mlflow
# import mlflow.pyfunc
# from model import tokenizer, model
# from Sentiment_Analysis.test.test_model import dataset
# from transformers import TrainingArguments, Trainer
# from huggingface_hub import HfApi

# HF_USERNAME = "GiuliaC"
# HF_REPO = "{HF_USERNAME}/Sentiment_Aalysis_twitter_roberta"
# HF_TOKEN = os.getenv("HF_TOKEN")

# api = HfApi()

# try:
#     repo_list = [repo.id for repo in api.list_models(author=HF_USERNAME)]
    
#     if HF_REPO in repo_list:
#         print(f"Il repository {HF_REPO} esiste già. ")
#     else:
#         print(f"Creazione del repository {HF_REPO} su Hugging Face...")
#         api.create_repo(repo_id=HF_REPO, repo_type="model", private=False)
#         print("Repository creato con successo!")

# except Exception as e:
#     print(f"Errore nel verificare il repository: {e}")

# # Inizializzazione di MLflow per Hugging Face
# mlflow.set_tracking_uri("https://huggingface.co/" + HF_REPO)
# mlflow.set_experiment("Sentiment_Analysis")

# # Funzione di tokenizzazione
# def tokenize_function(examples):
#     return tokenizer(examples["text"], padding="max_length", truncation=True)

# # Applico la tokenizzazione
# tokenized_datasets = dataset.map(tokenize_function, batched=True)

# # Definisco i dataset di train e di test
# train_dataset = tokenized_datasets["train"]
# test_dataset = tokenized_datasets["test"]

# # Configurazione del training
# training_args = TrainingArguments(
#     output_dir="./results",
#     evaluation_strategy="epoch",
#     save_strategy="epoch",
#     per_device_train_batch_size=8,
#     per_device_eval_batch_size=8,
#     num_train_epochs=10,  
#     weight_decay=0.01,
#     push_to_hub=True,  
#     hub_model_id=HF_REPO,
#     hub_token=HF_TOKEN,  
# )

# # Trainer di Hugging Face
# trainer = Trainer(
#     model=model,
#     args=training_args,
#     train_dataset=train_dataset,
#     eval_dataset=test_dataset,
# )

# with mlflow.start_run():
#     print("Avvio del training...")
#     trainer.train()
    
#     print("Training completato!")
#     #Carico il modello su HuggingFace
#     trainer.push_to_hub()

# print(f"Modello caricato su Hugging Face: https://huggingface.co/{HF_REPO}")
