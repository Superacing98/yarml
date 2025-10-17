import pandas as pd
import math
import os
from collections import Counter
from tqdm import tqdm

INPUT_CSV = 'dataset\\strings_dataset.csv'
OUTPUT_CSV = 'dataset\\features_dataset_v2.csv'

#List of suspicious APIs, commands, and registry keys
SUSPICIOUS_APIS = ['getprocaddress', 'loadlibrary', 'createremotethread', 'virtualalloc', 'isdebuggerpresent', 'writeprocessmemory']
CMD_COMMANDS = ['powershell', 'cmd.exe', 'net user', 'schtasks', 'rundll32', 'svchost']
REGISTRY_KEYS = ['hklm\\', 'hkcu\\', 'hkey_local_machine', 'hkey_current_user']

def get_max_consecutive_consonants(s):
    """Calculates the longest sequence of consecutive consonants."""
    max_len = 0
    current_len = 0
    vocals = "aeiouAEIOU"
    for char in s:
        if char.isalpha() and char not in vocals:
            current_len += 1
        else:
            max_len = max(max_len, current_len)
            current_len = 0
    max_len = max(max_len, current_len)
    return max_len


def calculate_entropy(s):
    """Calculate the entropy of a stringa."""
    if not s:
        return 0
    p, lns = Counter(s), float(len(s))
    return - sum(count/lns * math.log(count/lns, 2) for count in p.values())

def create_features(df):
    tqdm.pandas(desc="Calculate features")
    print("Starting feature calculation...")

    df['length'] = df['stringa'].str.len()
    df['entropy'] = df['stringa'].progress_apply(calculate_entropy)
    df['number_perc'] = df['stringa'].str.count(r'[0,9]') / df['length'].replace(0, 1)
    df['letter_perc'] = df['stringa'].str.count(r'[a-zA-Z]') / df['length'].replace(0, 1)
    df['symbol_perc'] = df['stringa'].str.count(r'[^a-zA-Z0-9]') / df['length'].replace(0, 1)
    df['is_path_windows'] = df['stringa'].str.contains(r':\\[^\\/:"*?<>|]+|\\{2}[^\\/:"*?<>|]+', regex=True).astype(int)
    df['is_url'] = df['stringa'].str.contains(r'https?://', regex=True).astype(int)
    df['is_hex'] = df['stringa'].str.fullmatch(r'[0-9a-fA-F]+$').astype(int)
    
    #Calculate advanced features
    string_lower = df['stringa'].str.lower()
    df['has_susp_api'] = string_lower.str.contains('|'.join(SUSPICIOUS_APIS), regex=True).astype(int)
    df['has_cmd_cmd'] = string_lower.str.contains('|'.join(CMD_COMMANDS), regex=True).astype(int)
    df['has_reg_key'] = string_lower.str.contains('|'.join(REGISTRY_KEYS), regex=True).astype(int)

    #Structural features
    df['dot_count'] = df['stringa'].str.count(r'\.')
    df['ratio_digit_alpha'] = df['number_perc'] / df['letter_perc'].replace(0, 1)
    df['max_seq_consonant'] = df['stringa'].progress_apply(get_max_consecutive_consonants)

    df.fillna(0, inplace=True)
    print("Calculation of basic features completed.")
    return df

if not os.path.exists(INPUT_CSV):
    raise FileNotFoundError(f"Input file '{INPUT_CSV}' not found.")

print(f"Loading dataset from '{INPUT_CSV}'...")
string_df = pd.read_csv(INPUT_CSV)
string_df.dropna(subset=['stringa'], inplace=True)
features_df = create_features(string_df)

try:
    features_df.to_csv(OUTPUT_CSV, index=False, encoding='utf-8')
    print(f"Dataset saved in '{OUTPUT_CSV}'.")
except IOError as e:
    print(f"Error while saving the file: {e}")
