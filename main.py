# main.py
from src.preprocessing import clean_text
import joblib
import os

# ==============================
# 🔹 Danh sách mô hình
# ==============================
MODEL_FILES = {
    "1": ("Naive Bayes", "model/naive_bayes.pkl"),
    "2": ("Logistic Regression", "model/logistic_regression.pkl"),
    "3": ("SVM", "model/svm.pkl"),
    "4": ("Random Forest", "model/random_forest.pkl"),
    "5": ("KNN", "model/knn.pkl"),
    "6": ("Tất cả", None)  # Placeholder cho chọn tất cả
}

# ==============================
# 🔹 Load tất cả mô hình
# ==============================
def load_models():
    loaded_models = {}
    for key, (name, path) in MODEL_FILES.items():
        if path is not None:
            if not os.path.exists(path):
                raise FileNotFoundError(f"Không tìm thấy file mô hình: {path}")
            vectorizer, model = joblib.load(path)
            loaded_models[name] = (vectorizer, model)
    return loaded_models

# ==============================
# 🔹 Chọn mô hình
# ==============================
def select_model():
    print("📬 SPAM EMAIL CLASSIFIER")
    print("=======================")
    print("Chọn mô hình muốn dùng:")
    for key, (name, _) in MODEL_FILES.items():
        print(f"{key}. {name}")
    choice = input("Nhập số tương ứng (1-6): ").strip()
    while choice not in MODEL_FILES:
        choice = input("Lựa chọn không hợp lệ. Nhập lại (1-6): ").strip()
    return choice

# ==============================
# 🔹 Hàm phân loại email cho 1 mô hình
# ==============================
def classify_email(vectorizer, model, text):
    clean = clean_text(text)
    vec = vectorizer.transform([clean])
    pred = model.predict(vec)[0]
    return "SPAM 🧨" if str(pred) == '1' else "NON-SPAM ✅"

# ==============================
# 🔹 Main
# ==============================
if __name__ == "__main__":
    loaded_models = load_models()
    while True:
        choice = select_model()
        if choice == "6":  # Tất cả
            text = input("\nNhập email/text (gõ 'exit' để thoát):\n> ")
            if text.lower() == 'exit':
                break
            print("\n📊 Kết quả dự đoán từ tất cả mô hình:")
            for name, (vectorizer, model) in loaded_models.items():
                result = classify_email(vectorizer, model, text)
                print(f"{name}: {result}")
            print()
        else:
            name, path = MODEL_FILES[choice]
            vectorizer, model = loaded_models[name]
            text = input(f"\nNhập email/text để phân loại bằng {name} (gõ 'exit' để thoát):\n> ")
            if text.lower() == 'exit':
                break
            result = classify_email(vectorizer, model, text)
            print(f"👉 Dự đoán bằng {name}: {result}\n")