"""
CLASSE PER VERIFICARE SE IL MODELLO FA DELLE PREVISIONI CORRETTE.
"""
import random
from datasets import load_dataset
from model import test_model
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from train_model import train

# Carico il dataset di test da Hugging Face
dataset = load_dataset("cardiffnlp/tweet_sentiment_multilingual", "italian")
df_test = dataset["test"].to_pandas()

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
print("\n**Calcolo le metriche...**\n")

# Generiamo le predizioni su tutto il dataset di test
y_true = df_test["label"].values
y_pred = [test_model(text) for text in df_test["text"]]

# Metriche
accuracy = accuracy_score(y_true, y_pred)
report = classification_report(y_true, y_pred)

print("\n**Performance del modello:**")
print(f"Accuratezza: {accuracy:.2f}")
print("\nReport di classificazione:")
print(report)

THRESHOLD_ACCURACY = 0.85
training_required = "YES" if accuracy < THRESHOLD_ACCURACY else "NO"

# Uso un file di testo per salvare i risultati e sapere se devo fare il retrain del modello
with open("training_required.txt", "w") as f:
    f.write(training_required)