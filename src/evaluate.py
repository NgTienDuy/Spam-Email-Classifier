import os
import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

# ======================================
# 🔹 HÀM ĐÁNH GIÁ MỘT MÔ HÌNH
# ======================================
def evaluate_model(model_path, X_test, y_test):
    """Load mô hình .pkl và đánh giá trên tập test"""
    model = joblib.load(model_path)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="binary", zero_division=0)
    rec = recall_score(y_test, y_pred, average="binary", zero_division=0)
    f1 = f1_score(y_test, y_pred, average="binary", zero_division=0)

    return acc, prec, rec, f1


# ======================================
# 🔹 MAIN ENTRY POINT
# ======================================
def main(model_dir="model/", test_path="data/test.csv"):
    print("🚀 Bắt đầu đánh giá mô hình trên tập test...")

    # Đọc tập test
    if not os.path.exists(test_path):
        raise FileNotFoundError(f"❌ Không tìm thấy file test: {test_path}")

    df_test = pd.read_csv(test_path)
    X_test, y_test = df_test["text"], df_test["label"]

    # Lấy danh sách mô hình .pkl trong thư mục model/
    model_files = [f for f in os.listdir(model_dir) if f.endswith(".pkl")]

    if not model_files:
        raise FileNotFoundError("⚠️ Không tìm thấy mô hình nào trong thư mục model/")

    results = []

    for model_file in model_files:
        model_path = os.path.join(model_dir, model_file)
        print(f"\n🧠 Đang đánh giá mô hình: {model_file}")

        acc, prec, rec, f1 = evaluate_model(model_path, X_test, y_test)

        print(f"📊 Accuracy : {acc:.4f}")
        print(f"📊 Precision: {prec:.4f}")
        print(f"📊 Recall   : {rec:.4f}")
        print(f"📊 F1-score : {f1:.4f}")

        results.append({
            "model": model_file.replace(".pkl", ""),
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1_score": f1
        })

    # Lưu kết quả ra CSV
    result_df = pd.DataFrame(results)
    result_df.to_csv(os.path.join(model_dir, "evaluation_results.csv"), index=False)
    print("\n✅ Đã lưu kết quả tại:", os.path.join(model_dir, "evaluation_results.csv"))

    # In bảng tổng kết
    print("\n🎯 BẢNG TỔNG KẾT KẾT QUẢ:")
    print(result_df.sort_values(by="f1_score", ascending=False).to_string(index=False))


if __name__ == "__main__":
    main()
