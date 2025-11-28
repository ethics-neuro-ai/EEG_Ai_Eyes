import pandas as pd
import matplotlib.pyplot as plt
import arff

def load_csv(path):
    
    return pd.read_csv(path)

def load_arff(path):
   
    with open(path, 'r') as f:
        dataset = arff.load(f)
    df = pd.DataFrame(dataset['data'], columns=[attr[0] for attr in dataset['attributes']])
    return df

def plot_eeg_signal(df, channels=None, n_samples=500):
    """
    Plot dei segnali EEG.
    df: DataFrame con colonne EEG
    channels: lista di nomi delle colonne da plottare, default tutte
    n_samples: numero di campioni da visualizzare
    """
    if channels is None:
        channels = df.columns[:-1]  # tutte tranne la colonna target
    plt.figure(figsize=(12,6))
    for ch in channels:
        plt.plot(df[ch].iloc[:n_samples], label=ch)
    plt.xlabel("Campioni")
    plt.ylabel("Ampiezza EEG")
    plt.title("Segnali EEG")
    plt.legend()
    plt.show()
