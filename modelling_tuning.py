import pandas as pd
import mlflow
import mlflow.sklearn
import pickle
import ast
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import f1_score, classification_report

PATH_DATASET = "Membangun_model/stacksample_preprocessing"

mlflow.set_tracking_uri("https://dagshub.com/abel1494/Eksperimen_SML_Alsyabella_Saputra.mlflow")
mlflow.set_experiment("Advance_Experiment")

def train():
    df = pd.read_csv(f'{PATH_DATASET}/clean_data.csv')
    df['filtered_tags'] = df['filtered_tags'].apply(ast.literal_eval)
    with open(f'{PATH_DATASET}/tfidf_model.pkl', 'rb') as f: tfidf = pickle.load(f)
    with open(f'{PATH_DATASET}/mlb_model.pkl', 'rb') as f: mlb = pickle.load(f)

    X = tfidf.transform(df['text_clean'])
    y = mlb.transform(df['filtered_tags'])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    mlflow.sklearn.autolog(disable=True)

    with mlflow.start_run(run_name="Run_Poin_4_Advance"):
        grid = GridSearchCV(OneVsRestClassifier(LinearSVC(max_iter=3000)), {'estimator__C': [1]}, cv=2)
        grid.fit(X_train, y_train)

        f1 = f1_score(y_test, grid.predict(X_test), average='micro')
        mlflow.log_metric("f1_micro", f1)

        mlflow.sklearn.log_model(grid.best_estimator_, "model")

        with open("report.txt", "w") as f:
            f.write(classification_report(y_test, grid.predict(X_test)))
        mlflow.log_artifact("report.txt") 

        with open("best_model_svc.pkl", "wb") as f:
            pickle.dump(grid.best_estimator_, f)
        mlflow.log_artifact("best_model_svc.pkl")

if __name__ == "__main__":
    train()
