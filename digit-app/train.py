import json
from pathlib import Path

import joblib
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from digit_preprocessing import augment_training_digit, preprocess_training_digit


BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"


def main():
    STATIC_DIR.mkdir(parents=True, exist_ok=True)
    digits = load_digits()
    y = digits.target
    X = np.array([preprocess_training_digit(image) for image in digits.images])

    train_indices, test_indices = train_test_split(
        np.arange(len(y)), test_size=0.2, stratify=y, random_state=42
    )
    X_train, X_test = X[train_indices], X[test_indices]
    y_train, y_test = y[train_indices], y[test_indices]

    candidates = {
        "SVM": (
            SVC(probability=True, random_state=42),
            {
                "clf__C": [1, 10],
                "clf__gamma": ["scale", 0.001],
                "clf__kernel": ["rbf"],
            },
        ),
        "Random Forest": (
            RandomForestClassifier(random_state=42),
            {"clf__n_estimators": [100, 300], "clf__max_depth": [None, 15]},
        ),
        "Neural Network (MLP)": (
            MLPClassifier(max_iter=500, random_state=42),
            {
                "clf__hidden_layer_sizes": [(64,), (128, 64)],
                "clf__alpha": [1e-4, 1e-2],
            },
        ),
    }

    results = {}
    fitted_models = {}
    augmentation_rng = np.random.default_rng(2026)
    augmented_images = []
    augmented_labels = []
    stroke_modes = ("normal", "thick", "thin", "normal")
    for index, label in zip(train_indices, y_train, strict=True):
        augmented_images.append(X[index])
        augmented_labels.append(label)
        for stroke_mode in stroke_modes:
            augmented_images.append(
                augment_training_digit(
                    digits.images[index], augmentation_rng, stroke_mode
                )
            )
            augmented_labels.append(label)
    augmented_X = np.asarray(augmented_images)
    augmented_y = np.asarray(augmented_labels)

    stress_rng = np.random.default_rng(2027)
    stress_features = {
        mode: np.asarray(
            [
                augment_training_digit(digits.images[index], stress_rng, mode)
                for index in test_indices
            ]
        )
        for mode in ("thick", "thin", "normal")
    }

    best_name = None
    best_accuracy = -1.0
    best_model = None

    for name, (classifier, parameter_grid) in candidates.items():
        pipeline = Pipeline(
            [("scaler", StandardScaler()), ("clf", classifier)]
        )
        search = GridSearchCV(
            pipeline, parameter_grid, cv=5, scoring="accuracy", n_jobs=-1
        )
        search.fit(X_train, y_train)
        fitted_model = search.best_estimator_
        fitted_model.fit(augmented_X, augmented_y)
        test_accuracy = accuracy_score(y_test, fitted_model.predict(X_test))
        robustness_accuracy = float(
            np.mean(
                [
                    accuracy_score(y_test, fitted_model.predict(stress_X))
                    for stress_X in stress_features.values()
                ]
            )
        )
        results[name] = {
            "cv_accuracy": round(search.best_score_, 4),
            "test_accuracy": round(test_accuracy, 4),
            "robustness_accuracy": round(robustness_accuracy, 4),
            "best_params": {
                key.removeprefix("clf__"): str(value)
                for key, value in search.best_params_.items()
            },
        }
        fitted_models[name] = fitted_model
        print(name, results[name])

        if test_accuracy > best_accuracy:
            best_name = name
            best_accuracy = test_accuracy
            best_model = fitted_model

    if best_model is None:
        raise RuntimeError("Model training completed without selecting a model.")

    joblib.dump(fitted_models, BASE_DIR / "models.joblib")
    joblib.dump(best_model, BASE_DIR / "model.joblib")
    for model_name, metrics in results.items():
        print(
            model_name,
            "held-out variation accuracy:",
            metrics["robustness_accuracy"],
        )
    with (BASE_DIR / "metrics.json").open("w", encoding="utf-8") as metrics_file:
        json.dump(
            {
                "best_model": best_name,
                "default_model": "Random Forest",
                "results": results,
            },
            metrics_file,
            indent=2,
        )

    figure, axis = plt.subplots(figsize=(5, 5))
    ConfusionMatrixDisplay.from_estimator(
        best_model, X_test, y_test, ax=axis, colorbar=False
    )
    axis.set_title(f"Confusion matrix: {best_name}")
    figure.tight_layout()
    figure.savefig(STATIC_DIR / "confusion.png", dpi=90)
    plt.close(figure)

    scaled_data = StandardScaler().fit_transform(X)
    projected_data = PCA(n_components=2).fit_transform(scaled_data)
    figure, axis = plt.subplots(figsize=(5, 5))
    scatter = axis.scatter(
        projected_data[:, 0], projected_data[:, 1], c=y, cmap="tab10", s=8
    )
    axis.legend(
        *scatter.legend_elements(), title="Digit", fontsize=7, loc="best"
    )
    axis.set_title("PCA of digits (2 components)")
    figure.tight_layout()
    figure.savefig(STATIC_DIR / "pca.png", dpi=90)
    plt.close(figure)

    print("Saved best model:", best_name, best_accuracy)


if __name__ == "__main__":
    main()
