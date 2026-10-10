"""训练模型并保存为 model.joblib。"""

import joblib
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

from .data_pipeline import build_pipeline, load_dataset, model_path


def train() -> None:
    X, y = load_dataset()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_test)

    print(f"accuracy={accuracy_score(y_test, predictions):.3f}")
    print(classification_report(y_test, predictions, zero_division=0))
    joblib.dump(pipeline, model_path())
    print(f"saved model to {model_path()}")


if __name__ == "__main__":
    train()

