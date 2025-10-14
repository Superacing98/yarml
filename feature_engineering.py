import pandas as pd
import math
import re
import os
from collections import Counter
from tqdm import tqdm

INPUT_CSV = 'dataset\\strings_dataset.csv'
OUTPUT_CSV = 'dataset\\features_dataset.csv'

def calculate_entropy(s):
    """Calculate the entropy of a string."""
    if not s:
        return 0
    p, lns = Counter(s), float(len(s))
    return - sum(count/lns * math.log(count/lns, 2) for count in p.values())

def create_features(df):
    tqdm.pandas(desc="Calcolo Features")
    print("Inizio calcolo features...Questa operazione potrebbe richiedere diversi minuti.")

    df['lunghezza'] = df['stringa'].str.len()
    df['entropia'] = df['stringa'].progress_apply(calculate_entropy)
    df['percentuale_numeri'] = df['stringa'].str.count(r'[0,9]') / df['lunghezza'].replace(0, 1)
    df['percentuale_lettere'] = df['stringa'].str.count(r'[a-zA-Z]') / df['lunghezza'].replace(0, 1)
    df['percentuale_simboli'] = df['stringa'].str.count(r'[^a-zA-Z0-9]') / df['lunghezza'].replace(0, 1)
    df['is_path_windows'] = df['stringa'].str.contains(r':\\[^\\/:"*?<>|]+|\\{2}[^\\/:"*?<>|]+', regex=True).astype(int)
    df['is_url'] = df['stringa'].str.contains(r'https?://', regex=True).astype(int)
    df['is_hex'] = df['stringa'].str.fullmatch(r'[0-9a-fA-F]+$').astype(int)

    df.fillna(0, inplace=True)
    print("Calcolo features completato.")
    return df

if not os.path.exists(INPUT_CSV):
    raise FileNotFoundError(f"Input file '{INPUT_CSV}' not found.")

print(f"Caricamento del dataset da '{INPUT_CSV}'...")
string_df = pd.read_csv(INPUT_CSV)
string_df.dropna(subset=['stringa'], inplace=True)
features_df = create_features(string_df)

try:
    features_df.to_csv(OUTPUT_CSV, index=False, encoding='utf-8')
    print(f"Dataset con features salvato in '{OUTPUT_CSV}'.")
except IOError as e:
    print(f"Errore nel salvataggio del file: {e}")
