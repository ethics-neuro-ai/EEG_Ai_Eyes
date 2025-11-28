import joblib
import numpy as np

# Carica modello e scaler
clf = joblib.load("../models/rf_model.pkl")
scaler = joblib.load("../models/scaler.pkl")

# Input manuale di esempio
valori = input("Inserisci 14 valori EEG separati da virgola:\n")
valori = [float(x) for x in valori.split(",")]

valori_scaled = scaler.transform([valori])
stato = clf.predict(valori_scaled)[0]

print("Occhi aperti" if stato == 0 else "Occhi chiusi")
