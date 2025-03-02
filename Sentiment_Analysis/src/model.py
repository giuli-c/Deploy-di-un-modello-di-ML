"""
Caricamento del modello e inferenza sui testi.
"""
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoConfig
import torch

MODEL_NAME = f"cardiffnlp/twitter-roberta-base-sentiment-latest"
LIST_SENTIMENT = ["Negative", "Neutral", "Positive"]

# 1.  SCARICO IL MODELLO 
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
config = AutoConfig.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

# Preprocess text (username and link placeholders)
def preprocess(text):
    new_text = []
    for t in text.split(" "):
        t = '@user' if t.startswith('@') and len(t) > 1 else t
        t = 'http' if t.startswith('http') else t
        new_text.append(t)
    return " ".join(new_text)

def test_model(text, sentiment_classes=LIST_SENTIMENT):
    text = preprocess(text)
    encoded_input = tokenizer(text, return_tensors='pt')
    output = model(**encoded_input)

    # Eseguo il modello su un testo di esempio:
    output = model(**encoded_input)

    # Interpretare il risultato

# output.logits è il risultato grezzo della rete neurale del modello Twitter-RoBERTa, 
# ovvero un tensore contenente 3 valori corrispondenti ai punteggi (logits): 
# Classe 0 → Sentiment negativo, Classe 1 → Sentiment neutro, Classe 2 → Sentiment positivo
# Softmax trasforma questi valori in probabilità comprese tra 0 e 1, assegnando un peso maggiore alla classe più probabile.
    probabilities = torch.nn.functional.softmax(output.logits, dim=-1)
# Identificazione della Classe Predetta: torch.argmax(probabilities) trova l'indice con la probabilità più alta
    sentiment = sentiment_classes[torch.argmax(probabilities)]

    return sentiment

if __name__ == "__main__":
    text = "Covid cases are increasing fast!"
    print(f"Testo: {text} → Sentiment: {test_model(text)}")