"""
Pipeline completa per Housing Price Prediction
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Carica i dati
def load_data(path):
    return pd.read_csv(path)

# Preprocessing
def preprocess(df):
    df = df.copy()
    for col in df.select_dtypes(include=['object']).columns:
        df[col].fillna(df[col].mode()[0], inplace=True)
        df[col] = LabelEncoder().fit_transform(df[col])
    for col in df.select_dtypes(include=['float', 'int']).columns:
        df[col].fillna(df[col].median(), inplace=True)
    scaler = StandardScaler()
    df[df.select_dtypes(include=['float', 'int']).columns] = scaler.fit_transform(df.select_dtypes(include=['float', 'int']))
    return df

# Train/test split
def split_data(df, target='SalePrice'):
    X = df.drop(target, axis=1)
    y = df[target]
    return train_test_split(X, y, test_size=0.2, random_state=42)

# Training
def train_model(X_train, y_train):
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

# Evaluation
def evaluate(model, X_test, y_test):
    preds = model.predict(X_test)
    mse = mean_squared_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    print(f"Mean Squared Error: {mse:.2f}")
    print(f"R^2 Score: {r2:.2f}")
    return preds

# Pipeline completa
def run_pipeline(train_path):
    df = load_data(train_path)
    df = preprocess(df)
    X_train, X_test, y_train, y_test = split_data(df)
    model = train_model(X_train, y_train)
    preds = evaluate(model, X_test, y_test)
    return model, preds

if __name__ == "__main__":
    # Modifica il path se necessario
    model, preds = run_pipeline("../data/train.csv")

    # Predizioni su test.csv e creazione submission
    test_df = load_data("../data/test.csv")
    test_df_proc = preprocess(test_df)
    # Assicurati che le colonne siano allineate con quelle usate nel training
    X_new = test_df_proc[model.feature_names_in_]
    predictions = model.predict(X_new)
    submission = pd.DataFrame({
        'Id': test_df['Id'] if 'Id' in test_df.columns else range(1, len(test_df)+1),
        'SalePrice': predictions
    })
    submission.to_csv("../data/submission.csv", index=False)
    print("Submission file creato: ../data/submission.csv")
