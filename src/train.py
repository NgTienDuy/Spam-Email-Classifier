import os
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# ======================================
# 🔹 DANH SÁCH MÔ HÌNH CẦN HUẤN LUYỆN
# ======================================
MODELS = {
    "naive_bayes": MultinomialNB(),
    "logistic_regression": LogisticRegression(max_iter=1000),
    "svm": LinearSVC(),
    "random_forest": RandomForestClassifier(n_estimators=200, random_state=42),
    "knn": KNeighborsClassifier(n_neighbors=5)
}


# ======================================
# 🔹 HÀM DỌN SẠCH DỮ LIỆU TRƯỚC KHI TRAIN
# ======================================
def clean_training_data(df):
    # Ép text về string
    df["text"] = df["text"].astype(str)

    # Loại bỏ NaN, None, rỗng
    df = df[df["text"].notna()]
    df = df[df["text"].str.strip() != ""]

    # Làm sạch label
    df["label"] = df["label"].astype(str).str.strip()
    df = df[df["label"].notna()]
    df = df[df["label"] != ""]

    df = df.reset_index(drop=True)
    return df


# ======================================
# 🔹 HÀM HUẤN LUYỆN VỚI CROSS VALIDATION
# ======================================
def train_with_cross_validation(
    model_name: str, model, X, y, k_folds=5, model_dir="model/"
):
    print(f"\n🚀 Huấn luyện mô hình {model_name.upper()} với {k_folds}-Fold CV")

    skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)
    fold_scores = []

    # Duyệt qua từng fold
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        # DỌN SẠCH TRONG TỪNG FOLD
        mask_train = X_train.notna() & (X_train.str.strip() != "")
        mask_val = X_val.notna() & (X_val.str.strip() != "")

        X_train, y_train = X_train[mask_train], y_train[mask_train]
        X_val, y_val = X_val[mask_val], y_val[mask_val]

        # Pipeline: TF-IDF + Model
        pipe = Pipeline([
            ("tfidf", TfidfVectorizer(max_features=5000, stop_words="english")),
            ("clf", model)
        ])

        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_val)
        acc = accuracy_score(y_val, y_pred)
        fold_scores.append(acc)
        print(f"   📊 Fold {fold}: accuracy = {acc:.4f}")

    # Trung bình độ chính xác
    mean_acc = np.mean(fold_scores)
    print(f"✅ Trung bình {k_folds}-Fold accuracy: {mean_acc:.4f}")

    # Huấn luyện lại toàn bộ train để lưu mô hình cuối
    final_pipe = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=5000, stop_words="english")),
        ("clf", model)
    ])
    final_pipe.fit(X, y)

    Path(model_dir).mkdir(parents=True, exist_ok=True)
    model_path = os.path.join(model_dir, f"{model_name}.pkl")
    joblib.dump(final_pipe, model_path)
    print(f"💾 Đã lưu mô hình cuối tại: {model_path}\n")

    return mean_acc


# ======================================
# 🔹 MAIN ENTRY POINT
# ======================================
def main():
    # Đọc dữ liệu train
    df_train = pd.read_csv("data/train.csv")

    # DỌN SẠCH dữ liệu trước train
    df_train = clean_training_data(df_train)

    X, y = df_train["text"], df_train["label"]

    results = {}

    # Huấn luyện từng mô hình
    for name, model in MODELS.items():
        acc = train_with_cross_validation(name, model, X, y, k_folds=5)
        results[name] = acc

    # Tổng kết
    print("\n🎯 TỔNG KẾT KẾT QUẢ:")
    for name, acc in results.items():
        print(f"   {name:20s}: {acc:.4f}")

    # Lưu bảng kết quả ra file CSV
    pd.DataFrame(list(results.items()), columns=["model", "mean_accuracy"]).to_csv(
        "model/cv_results.csv", index=False
    )
    print("\n📊 Kết quả cross-validation đã lưu tại model/cv_results.csv")


if __name__ == "__main__":
    main()
