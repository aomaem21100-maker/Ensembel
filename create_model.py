# create_model.py
import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import accuracy_score, roc_auc_score

# ---------- สร้างข้อมูลตัวอย่าง ----------
np.random.seed(42)
n = 500

data = pd.DataFrame({
    "study_hours": np.random.uniform(0, 12, n),
    "attendance_percent": np.random.uniform(30, 100, n),
    "assignment_score": np.random.uniform(20, 100, n),
    "previous_gpa": np.random.uniform(0, 4, n),
    "internet_access": np.random.choice(["Yes", "No"], n),
    "tutoring": np.random.choice(["Yes", "No"], n),
})

# สร้าง label จำลอง (ผ่าน/ไม่ผ่าน)
score = (
    data["study_hours"] * 3
    + data["attendance_percent"] * 0.5
    + data["assignment_score"] * 0.8
    + data["previous_gpa"] * 15
    + (data["internet_access"] == "Yes").astype(int) * 10
    + (data["tutoring"] == "Yes").astype(int) * 8
    + np.random.normal(0, 15, n)
)
data["pass_exam"] = (score > score.median()).astype(int)

X = data.drop(columns="pass_exam")
y = data["pass_exam"]

# ---------- สร้าง Pipeline ----------
numeric_features = [
    "study_hours", "attendance_percent",
    "assignment_score", "previous_gpa",
]
categorical_features = ["internet_access", "tutoring"]

numeric_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])

categorical_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore")),
])

preprocessor = ColumnTransformer([
    ("num", numeric_transformer, numeric_features),
    ("cat", categorical_transformer, categorical_features),
])

pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(n_estimators=100, random_state=42)),
])

# ---------- Train & Evaluate ----------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)
y_prob = pipeline.predict_proba(X_test)[:, 1]

accuracy = accuracy_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_prob)

print(f"Accuracy: {accuracy:.2%}")
print(f"ROC-AUC:  {roc_auc:.3f}")

# ---------- บันทึกโมเดล ----------
model_bundle = {
    "pipeline": pipeline,
    "metadata": {
        "metrics": {
            "accuracy": accuracy,
            "roc_auc": roc_auc,
        },
        "feature_columns": numeric_features + categorical_features,
        "class_names": ["ไม่ผ่าน", "ผ่าน"],
    },
}

with open("random_forest_model.pkl", "wb") as f:
    pickle.dump(model_bundle, f)

print("✅ บันทึก random_forest_model.pkl เรียบร้อย!")