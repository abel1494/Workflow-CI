import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
import pickle
import ast
import os

PATH_DATASET = "stacksample_preprocessing"

def train_basic():
    df = pd.read_csv(f'{PATH_DATASET}/clean_data.csv')
    df['filtered_tags'] = df['filtered_tags'].apply(ast.literal_eval)
    
    with open(f'{PATH_DATASET}/tfidf_model.pkl', 'rb') as f: tfidf = pickle.load(f)
    with open(f'{PATH_DATASET}/mlb_model.pkl', 'rb') as f: mlb = pickle.load(f)

    X = tfidf.transform(df['text_clean'])
    y = mlb.transform(df['filtered_tags'])

    mlflow.sklearn.autolog()
    model = OneVsRestClassifier(LinearSVC())
    model.fit(X, y)
    
    print("Training Basic selesai dengan Autolog!")

if __name__ == "__main__":
    train_basic()
