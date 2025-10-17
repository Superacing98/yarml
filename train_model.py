import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import os
import joblib

INPUT_CSV = 'dataset\\features_dataset_v2.csv'
MODEL_OUTPUT_FILE = 'model\\yara_string_scorer_v2.joblib'

print("Addestramento del modello...")

if not os.path.exists(INPUT_CSV):
    print(f"Errore: il file di input '{INPUT_CSV}' non esiste. Assicurarsi di aver eseguito prima lo scritp 'feature_engineering.py'.")
else:
    print(f"Caricamento del dataset da '{INPUT_CSV}'...")
    df = pd.read_csv(INPUT_CSV)

    X = df.drop(columns=['stringa', 'label'], axis=1)
    y = df['label']

    print(f"Dataset caricato con {X.shape[0]} campioni e {X.shape[1]} features ciascuno.")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    print(f"Suddivisione del dataset: {X_train.shape[0]} campioni per l'addestramento, {X_test.shape[0]} per il test.")

    print("\nInizio addestramento del classificatore LightGBM...")
    lgbm_classifier = lgb.LGBMClassifier(objective='binary', random_state=42)
    lgbm_classifier.fit(X_train, y_train)
    print("Addestramento completato.")

    print("\nValutazione del modello sul set di test...")
    y_pred = lgbm_classifier.predict(X_test)

    print(f"Accuratezza: {accuracy_score(y_test, y_pred):.4f}")
    print("\nReport di classificazione:")
    print(classification_report(y_test, y_pred, target_names=['Benign(0)', 'Malicious(1)']))

    print("\nMatrice di confusione:")
    print(confusion_matrix(y_test, y_pred))
    print("Righe: Valori veri, Colonne: Predetti")

    print(f"\nSalvataggio del modello addestrato in '{MODEL_OUTPUT_FILE}'...")
    try:
        joblib.dump(lgbm_classifier, MODEL_OUTPUT_FILE)
        print(f"Modello salvato con successo in '{MODEL_OUTPUT_FILE}'.")
    except Exception as e:
        print(f"Errore nel salvataggio del modello: {e}")
