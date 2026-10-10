"""数据读取、清洗和可复用的 scikit-learn 训练 pipeline。"""

from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.datasets import load_iris
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

FEATURES = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
TARGET_NAMES = ["setosa", "versicolor", "virginica"]


def load_dataset() -> tuple[pd.DataFrame, pd.Series]:
    """读取数据，并模拟少量缺失值以演示清洗步骤。"""
    iris = load_iris(as_frame=True)
    X = iris.data.copy()
    X.columns = FEATURES
    y = pd.Series(iris.target, name="target")

    # 仅用于演示：真实项目中这里应替换为原始数据清洗逻辑。
    X.loc[0, "sepal_width"] = None
    X.loc[10, "petal_length"] = None
    return X, y


def build_pipeline() -> Pipeline:
    """返回完整的预处理 + 模型 pipeline，避免训练/预测处理不一致。"""
    numeric_steps = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[("numeric", numeric_steps, FEATURES)],
        remainder="drop",
    )
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", LogisticRegression(max_iter=500, random_state=42)),
        ]
    )


def model_path() -> Path:
    return Path(__file__).resolve().parent.parent / "model.joblib"

