# 🧠 EEG Eye State Classification 🧠 

Progetto per classificare lo stato degli occhi (aperti o chiusi) a partire da segnali EEG a 14 canali, usando un modello di **Random Forest**.  
Il modello raggiunge circa **92% di accuratezza** sul test set.

---

## Struttura del progetto

EEG_AI_Project/
│
├── data/
│ └── EEG_Eye_State.arff # Dataset EEG
│
├── notebooks/
│ └── exploration.ipynb # Notebook per esplorare e visualizzare i dati
│
├── src/
│ ├── preprocessing.py # Normalizzazione e funzioni di preprocessing
│ ├── train_model.py # Script per allenare il modello
│ ├── predict.py # Script per predire nuovi segnali
│ └── utils.py # Funzioni comuni (caricamento dati, plot)
│
├── models/
│ ├── rf_model.pkl # Modello Random Forest salvato
│ └── scaler.pkl # Scaler per normalizzare i dati
│
├── .gitignore
├── requirements.txt
└── README.md

yaml
Copia codice

---

## Requisiti

- Python 3.10+
- Librerie installabili via pip:


pip install -r requirements.txt 
requirements.txt include:

pandas

numpy

scikit-learn

joblib

matplotlib

jupyter

liac-arff

Come usare il progetto
1️⃣ Allenare il modello
bash
Copia codice
python src/train_model.py
Carica il dataset EEG_Eye_State.arff

Divide i dati in train/test

Normalizza i segnali

Allena il modello Random Forest

Salva rf_model.pkl e scaler.pkl in models/

2️⃣ Testare nuove predizioni
bash
Copia codice
python src/predict.py
Inserisci 14 valori EEG separati da virgola

Il programma restituirà:

nginx
Copia codice
Occhi aperti
o

nginx
Copia codice
Occhi chiusi
3️⃣ Esplorazione dei dati
Apri il notebook:

bash
Copia codice
jupyter notebook notebooks/exploration.ipynb
Visualizza segnali EEG

Analizza distribuzione occhi aperti/chiusi

Plot dei canali e statistiche descrittive

Risultati
Accuracy del modello: ~92% sul test set

Matrice di confusione inclusa nello script train_model.py

Note
Il progetto è pronto per essere esteso: puoi aggiungere feature extraction (bande EEG), usare deep learning o creare una dashboard interattiva in Streamlit.

Tutti i file della venv sono ignorati grazie a .gitignore.

Autore: [Il tuo nome]
Dataset: EEG Eye State Dataset (UCI Repository)
