# src/train_model.py
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix
import arff
import pandas as pd
import os

# --- Carica dataset ARFF ---
dataset_path = "/Volumes/backup/mail/EEG Eye State.arff"
with open(dataset_path, 'r') as f:
    dataset = arff.load(f)
df = pd.DataFrame(dataset['data'], columns=[attr[0] for attr in dataset['attributes']])

# --- Separa X e y ---
X = df.iloc[:, :-1]  # prime 14 colonne EEG
y = df.iloc[:, -1]   # ultima colonna eyeState

# --- Split train/test ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Normalizzazione ---
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --- Allena Random Forest ---
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train_scaled, y_train)

# --- Valutazione ---
y_pred = clf.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
print(f"Accuracy: {accuracy*100:.2f}%")
print("Confusion Matrix:")
print(cm)

# Percorso della cartella script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Percorso della cartella models nella root del progetto
MODELS_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), "models")
os.makedirs(MODELS_DIR, exist_ok=True)

# Salva modello e scaler
joblib.dump(clf, os.path.join(MODELS_DIR, "rf_model.pkl"))
joblib.dump(scaler, os.path.join(MODELS_DIR, "scaler.pkl"))
print(f"Modello salvato in {os.path.join(MODELS_DIR, 'rf_model.pkl')}")