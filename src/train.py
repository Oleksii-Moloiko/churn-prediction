import pickle
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline

from preprocessing import build_preprocessor, clean_dataset, load_dataset, split_features_target


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "data.csv"
MODEL_PATH = BASE_DIR / "models" / "model.pkl"
RANDOM_STATE = 42


def build_models(preprocessor):
    return {
        "logistic_regression": Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)),
            ]
        ),
        "random_forest": Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE)),
            ]
        ),
        "logistic_regression_balanced": Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                (
                    "model",
                    LogisticRegression(
                        max_iter=1000,
                        random_state=RANDOM_STATE,
                        class_weight="balanced",
                    ),
                ),
            ]
        ),
    }


def evaluate_model(name, pipeline, x_train, x_test, y_train, y_test):
    pipeline.fit(x_train, y_train)
    predictions = pipeline.predict(x_test)

    print(f"\n{name}")
    print("-" * len(name))
    print(f"Accuracy: {accuracy_score(y_test, predictions):.4f}")
    print("\nClassification report:")
    print(classification_report(y_test, predictions))
    print("Confusion matrix:")
    print(confusion_matrix(y_test, predictions))

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    recall_scores = cross_val_score(pipeline, x_train, y_train, cv=cv, scoring="recall_macro")
    print(f"CV recall_macro: {recall_scores.mean():.4f} +/- {recall_scores.std():.4f}")

    return pipeline


def main():
    raw_df = load_dataset(DATA_PATH)
    cleaned_df = clean_dataset(raw_df)
    x, y = split_features_target(cleaned_df)

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    models = build_models(build_preprocessor())
    trained_models = {}

    for name, pipeline in models.items():
        trained_models[name] = evaluate_model(name, pipeline, x_train, x_test, y_train, y_test)

    final_model = trained_models["logistic_regression_balanced"]
    with open(MODEL_PATH, "wb") as model_file:
        pickle.dump(final_model, model_file)

    print(f"\nSaved final model to: {MODEL_PATH}")


if __name__ == "__main__":
    main()
