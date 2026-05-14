import os

import joblib
import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC
from sklearn.utils.class_weight import compute_class_weight


DATA_PATH = "data/cleaned_intent_dataset.csv"
REPORT_DIR = "reports"
MODEL_DIR = "models"
RANDOM_STATE = 42
TEST_SIZE = 0.20
CV_SPLITS = 5
MODEL_C = 0.2
RARE_CLASS_MAX_TRAIN_SAMPLES = 250
RARE_CLASS_WEIGHT_MULTIPLIER = 2.0


def build_vectorizer():
    word_vectorizer = TfidfVectorizer(
        max_features=30000,
        ngram_range=(1, 3),
        min_df=1,
        max_df=0.95,
        sublinear_tf=True,
    )

    char_vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        min_df=2,
        max_features=50000,
        sublinear_tf=True,
    )

    return FeatureUnion(
        [
            ("word_tfidf", word_vectorizer),
            ("char_tfidf", char_vectorizer),
        ]
    )


def build_class_weight(y_train):
    counts = y_train.value_counts()
    classes = counts.index.sort_values().to_numpy()
    weights = dict(
        zip(
            classes,
            compute_class_weight(
                class_weight="balanced",
                classes=classes,
                y=y_train,
            ),
        )
    )

    rare_classes = counts[counts < RARE_CLASS_MAX_TRAIN_SAMPLES].index.tolist()
    for label in rare_classes:
        weights[label] *= RARE_CLASS_WEIGHT_MULTIPLIER

    return weights, rare_classes


def build_base_model(class_weight):
    return LinearSVC(
        class_weight=class_weight,
        C=MODEL_C,
        random_state=RANDOM_STATE,
    )


def load_dataset():
    df = pd.read_csv(DATA_PATH)
    df = df[["text", "label"]].dropna().copy()
    df["text"] = df["text"].astype(str)
    df["label"] = df["label"].astype(str)
    return df.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)


def print_distribution(title, labels):
    print(f"\n{title}")
    print(labels.value_counts())


def run_cross_validation(X, y, class_weight):
    pipeline = Pipeline(
        [
            ("vectorizer", build_vectorizer()),
            ("model", build_base_model(class_weight)),
        ]
    )

    scoring = {
        "accuracy": "accuracy",
        "balanced_accuracy": "balanced_accuracy",
        "macro_f1": "f1_macro",
        "weighted_f1": "f1_weighted",
    }

    cv = StratifiedKFold(
        n_splits=CV_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    return cross_validate(
        pipeline,
        X,
        y,
        cv=cv,
        scoring=scoring,
        n_jobs=1,
    )


def summarize_cv_results(cv_results):
    summary = {}
    for metric in ["accuracy", "balanced_accuracy", "macro_f1", "weighted_f1"]:
        values = cv_results[f"test_{metric}"]
        summary[metric] = {
            "mean": values.mean(),
            "std": values.std(),
        }
    return summary


def save_confusion_matrices(y_test, y_pred, labels):
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    print("\nConfusion Matrix:")
    print(cm)

    fig, ax = plt.subplots(figsize=(11, 9))
    ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels).plot(
        ax=ax,
        cmap="Blues",
        values_format="d",
        colorbar=True,
        xticks_rotation=45,
    )
    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    fig.tight_layout()
    fig.savefig(os.path.join(REPORT_DIR, "confusion_matrix.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)

    cm_percent = confusion_matrix(y_test, y_pred, labels=labels, normalize="true") * 100
    fig, ax = plt.subplots(figsize=(11, 9))
    ConfusionMatrixDisplay(confusion_matrix=cm_percent, display_labels=labels).plot(
        ax=ax,
        cmap="Blues",
        values_format=".1f",
        colorbar=True,
        xticks_rotation=45,
    )
    ax.set_title("Normalized Confusion Matrix (%)")
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    fig.tight_layout()
    fig.savefig(
        os.path.join(REPORT_DIR, "confusion_matrix_normalized.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)


def save_reports(y_test, y_pred, holdout_metrics, cv_summary, train_size, test_size, rare_classes):
    os.makedirs(REPORT_DIR, exist_ok=True)

    report_df = pd.DataFrame(
        classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    ).transpose()
    report_df.to_csv(
        os.path.join(REPORT_DIR, "classification_report.csv"),
        encoding="utf-8-sig",
    )

    cv_df = pd.DataFrame(cv_summary).transpose()
    cv_df.to_csv(os.path.join(REPORT_DIR, "cross_validation_metrics.csv"), encoding="utf-8-sig")

    with open(os.path.join(REPORT_DIR, "model_metrics.txt"), "w", encoding="utf-8") as f:
        f.write("Model config\n")
        f.write("Model: LinearSVC\n")
        f.write(f"C: {MODEL_C}\n")
        f.write("Vectorizer: word TF-IDF + char_wb TF-IDF\n")
        f.write(f"Rare class max train samples: {RARE_CLASS_MAX_TRAIN_SAMPLES}\n")
        f.write(f"Rare class weight multiplier: {RARE_CLASS_WEIGHT_MULTIPLIER}\n")
        f.write(f"Rare classes: {', '.join(rare_classes)}\n")
        f.write("\n")
        f.write("Holdout metrics\n")
        f.write(f"Accuracy: {holdout_metrics['accuracy']:.4f}\n")
        f.write(f"Balanced accuracy: {holdout_metrics['balanced_accuracy']:.4f}\n")
        f.write(f"Macro F1: {holdout_metrics['macro_f1']:.4f}\n")
        f.write(f"Weighted F1: {holdout_metrics['weighted_f1']:.4f}\n")
        f.write(f"Train samples: {train_size}\n")
        f.write(f"Test samples: {test_size}\n")
        f.write("\nCross-validation metrics\n")
        for metric, values in cv_summary.items():
            f.write(f"{metric}: {values['mean']:.4f} +/- {values['std']:.4f}\n")


def save_model(model, vectorizer):
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(model, os.path.join(MODEL_DIR, "intent_model.pkl"))
    joblib.dump(vectorizer, os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl"))


def main():
    df = load_dataset()

    print("Dataset shape:", df.shape)
    print_distribution("Class distribution:", df["label"])

    X = df["text"]
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    print("\nTrain samples:", X_train.shape[0])
    print("Test samples:", X_test.shape[0])
    print_distribution("Train class distribution:", y_train)
    print_distribution("Test class distribution:", y_test)

    class_weight, rare_classes = build_class_weight(y_train)
    print("\nRare classes with extra weight:", ", ".join(rare_classes))

    print("\nRunning stratified cross-validation on training data...")
    cv_results = run_cross_validation(X_train, y_train, class_weight)
    cv_summary = summarize_cv_results(cv_results)
    for metric, values in cv_summary.items():
        print(f"{metric}: {values['mean']:.4f} +/- {values['std']:.4f}")

    vectorizer = build_vectorizer()
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    print("\nTF-IDF train matrix shape:", X_train_tfidf.shape)
    print("TF-IDF test matrix shape:", X_test_tfidf.shape)

    model = build_base_model(class_weight)
    model.fit(X_train_tfidf, y_train)

    print("\nModel training completed.")

    y_pred = model.predict(X_test_tfidf)
    holdout_metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_test, y_pred),
        "macro_f1": f1_score(y_test, y_pred, average="macro", zero_division=0),
        "weighted_f1": f1_score(y_test, y_pred, average="weighted", zero_division=0),
    }

    print("\nHoldout metrics:")
    for metric, value in holdout_metrics.items():
        print(f"{metric}: {value:.4f}")

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))

    labels = model.classes_
    save_confusion_matrices(y_test, y_pred, labels)
    save_reports(
        y_test,
        y_pred,
        holdout_metrics,
        cv_summary,
        X_train.shape[0],
        X_test.shape[0],
        rare_classes,
    )
    save_model(model, vectorizer)

    print("\nSaved report files:")
    print("reports/confusion_matrix.png")
    print("reports/confusion_matrix_normalized.png")
    print("reports/classification_report.csv")
    print("reports/cross_validation_metrics.csv")
    print("reports/model_metrics.txt")

    print("\nSaved model files:")
    print("models/intent_model.pkl")
    print("models/tfidf_vectorizer.pkl")

    sample_texts = [
        "Hastanin atesi yukseliyor ve nefes almakta zorlaniyor.",
        "Bu lupus olabilir mi?",
        "Hemen MR cekin.",
        "Tedaviye antibiyotikle baslayalim.",
        "Kan testinin sonucunu degerlendirmemiz gerekiyor.",
    ]

    sample_vectors = vectorizer.transform(sample_texts)
    sample_predictions = model.predict(sample_vectors)

    print("\nSample predictions:")
    for text, pred in zip(sample_texts, sample_predictions):
        print(f"Text: {text}")
        print(f"Predicted intent: {pred}")
        print("-" * 50)


if __name__ == "__main__":
    main()
