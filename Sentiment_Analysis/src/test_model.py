"""
CLASSE PER VERIFICARE SE IL MODELLO FA DELLE PREVISIONI CORRETTE.
"""
import pandas as pd
import random
from datasets import load_dataset
from model import test_model
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# Carico il dataset di test da Hugging Face
dataset = load_dataset("cardiffnlp/tweet_sentiment_multilingual", "italian")
df_test = dataset["test"].to_pandas()

map_dict = {0: "Negative", 1:"Neutral", 2:"Positive"}
df_test["label"] = df_test["label"].map(map_dict)

# Controllo valori nulli
if df_test.isnull().sum().sum() > 0:
    df_test.dropna(inplace=True)
    
# Controllo duplicati
if df_test.duplicated().sum() > 0:
    df_test.drop_duplicates(inplace=True)

# Selezione di 10 indici casuali
random_indices = random.sample(range(len(df_test)), 10)

for index in random_indices:
    text = df_test.loc[index, "text"]
    label = df_test.loc[index, "label"]
    prediction = test_model(text)

    print(f"Testo: {text}\n→ Previsione: {prediction} | Reale: {label}\n")

# 5. Test del modello su tutto il dataset di test**
print("\n**Calcolo le metriche di performance su tutto il dataset...**\n")

# Generiamo le predizioni su tutto il dataset di test
y_true = df_test["label"].values
y_pred = [test_model(text) for text in df_test["text"]]

# Metriche
accuracy = accuracy_score(y_true, y_pred)
report = classification_report(y_true, y_pred)
conf_matrix = confusion_matrix(y_true, y_pred)

print("\n✅ **Performance del modello:**")
print(f"🔹 Accuratezza: {accuracy:.2f}")
print("\n🔹 Report di classificazione:")
print(report)

print("\n🔹 Matrice di Confusione:")
print(conf_matrix)

if accuracy < 0.85:
    print("\n**Attenzione: Il modello ha un'accuratezza bassa!**")
