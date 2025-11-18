import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

# ======================================
# 🔹 HÀM ĐÁNH GIÁ MỘT MÔ HÌNH
# ======================================
def evaluate_model(model_path, X_test, y_test):
    """Load mô hình .pkl và đánh giá trên tập test"""

    # 🔥 Chuẩn hoá dữ liệu test
    X_test = X_test.astype(str).fillna("")

    model = joblib.load(model_path)

    # Predict
    y_pred = model.predict(X_test)

    # 🔹 Chuyển y_pred về cùng kiểu với y_test
    if y_test.dtype.kind in 'if':  # int or float
        y_pred = pd.Series(y_pred).astype(float if y_test.dtype.kind=='f' else int)
    else:
        y_pred = pd.Series(y_pred).astype(str)

    # 🔥 Tìm pos_label tự động (chỉ cần khi dùng precision/recall/f1)
    unique_labels = sorted(set(y_test.unique()))
    pos_label = unique_labels[-1]

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, pos_label=pos_label, zero_division=0)
    rec = recall_score(y_test, y_pred, pos_label=pos_label, zero_division=0)
    f1 = f1_score(y_test, y_pred, pos_label=pos_label, zero_division=0)

    return acc, prec, rec, f1, y_pred

# ======================================
# 🔹 VẼ BIỂU ĐỒ
# ======================================
def plot_confusion_matrix(y_true, y_pred, model_name, save_path):
    cm = confusion_matrix(y_true, y_pred, labels=sorted(y_true.unique()))
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=sorted(y_true.unique()),
                yticklabels=sorted(y_true.unique()))
    plt.title(f"Confusion Matrix - {model_name}")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_model_scores(result_df, save_path):
    plt.figure(figsize=(10, 6))
    result_df_plot = result_df.set_index("model")[["accuracy", "precision", "recall", "f1_score"]]
    result_df_plot.plot(kind="bar", figsize=(12, 6))
    plt.title("So sánh các chỉ số đánh giá mô hình")
    plt.ylabel("Score (0 → 1)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

# ======================================
# 🔹 MAIN ENTRY POINT
# ======================================
def main(model_dir="model/", test_path="data/test.csv", result_dir="result/"):
    print("🚀 Bắt đầu đánh giá mô hình trên tập test...")

    # 🔹 Tạo thư mục result và plots nếu chưa tồn tại
    os.makedirs(result_dir, exist_ok=True)
    os.makedirs(os.path.join(result_dir, "plots"), exist_ok=True)

    # Đọc tập test
    if not os.path.exists(test_path):
        raise FileNotFoundError(f"❌ Không tìm thấy file test: {test_path}")

    df_test = pd.read_csv(test_path)
    df_test["text"] = df_test["text"].astype(str)
    df_test = df_test[df_test["text"].str.strip() != ""]
    df_test = df_test.dropna(subset=["text", "label"])
    df_test.reset_index(drop=True, inplace=True)

    X_test, y_test = df_test["text"], df_test["label"]

    model_files = [f for f in os.listdir(model_dir) if f.endswith(".pkl")]
    if not model_files:
        raise FileNotFoundError("⚠️ Không tìm thấy mô hình nào trong thư mục model/")

    results = []

    for model_file in model_files:
        model_path = os.path.join(model_dir, model_file)
        print(f"\n🧠 Đang đánh giá mô hình: {model_file}")

        acc, prec, rec, f1, y_pred = evaluate_model(model_path, X_test, y_test)

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

        # 🔹 Vẽ confusion matrix cho từng mô hình
        cm_path = os.path.join(result_dir, "plots", f"cm_{model_file.replace('.pkl','')}.png")
        plot_confusion_matrix(y_test, y_pred, model_file, cm_path)

    # 🔹 Lưu kết quả CSV
    result_df = pd.DataFrame(results)
    csv_path = os.path.join(result_dir, "evaluation_results.csv")
    result_df.to_csv(csv_path, index=False)
    print("\n✅ Đã lưu kết quả tại:", csv_path)

    # 🔹 Vẽ biểu đồ tổng hợp
    plot_score_path = os.path.join(result_dir, "plots", "model_scores.png")
    plot_model_scores(result_df, plot_score_path)
    print("📈 Đã tạo biểu đồ tổng hợp tại:", plot_score_path)

    # 🔹 In bảng tổng kết
    print("\n🎯 BẢNG TỔNG KẾT KẾT QUẢ:")
    print(result_df.sort_values(by="f1_score", ascending=False).to_string(index=False))


if __name__ == "__main__":
    main()
