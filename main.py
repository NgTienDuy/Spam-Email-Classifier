import tkinter as tk
from tkinter import messagebox, ttk
import joblib
import os

# ======= Load tất cả mô hình .pkl =======
model_folder = "model/"
models = {}
for file in os.listdir(model_folder):
    if file.endswith(".pkl"):
        # Tên model lấy theo tên file, có thể tùy chỉnh nếu cần
        name = file.replace(".pkl", "").replace("_", " ").title()
        models[name] = joblib.load(os.path.join(model_folder, file))

if not models:
    raise FileNotFoundError("Không tìm thấy file mô hình nào (.pkl). Hãy chạy train_models.py trước.")

# ======= Giao diện =======
root = tk.Tk()
root.title("Spam Email Classifier - Multi Model")
root.geometry("600x600")
root.config(bg="#f3f3f3")

tk.Label(root, text="Nhập nội dung email:", font=("Arial", 14), bg="#f3f3f3").pack(pady=10)
text_input = tk.Text(root, height=10, width=60, font=("Arial", 12))
text_input.pack(pady=5)

# ======= Dropdown chọn mô hình =======
tk.Label(root, text="Chọn mô hình phân loại:", font=("Arial", 13), bg="#f3f3f3").pack(pady=10)
model_names = list(models.keys())
model_names.insert(0, "Tất cả")  # Thêm lựa chọn tất cả
model_choice = ttk.Combobox(root, values=model_names, font=("Arial", 12), state="readonly")
model_choice.set("Tất cả")
model_choice.pack(pady=5)

# ======= Hàm dự đoán =======
def predict():
    email = text_input.get("1.0", tk.END).strip()
    if not email:
        messagebox.showwarning("Cảnh báo", "Vui lòng nhập nội dung email!")
        return

    # Map nhãn số sang nhãn chữ
    label_map = {
        0: "non-spam",
        1: "spam"
    }

    selected = model_choice.get()
    result_texts = []

    if selected == "Tất cả":
        # Dự đoán với tất cả mô hình
        for name, model in models.items():
            pred = model.predict([email])[0]
            pred_label = label_map.get(int(pred), str(pred))
            color = "red" if pred_label == "spam" else "green"
            result_texts.append(f"{name}: {pred_label.upper()}")
        result_label.config(text="\n".join(result_texts), fg="black")
    else:
        model = models[selected]
        pred = model.predict([email])[0]
        pred_label = label_map.get(int(pred), str(pred))
        color = "red" if pred_label == "spam" else "green"
        result_label.config(text=f"{selected}: {pred_label.upper()}", fg=color)

# ======= Nút và nhãn kết quả =======
tk.Button(root, text="Phân loại", font=("Arial", 13, "bold"), bg="#0078D7", fg="white", command=predict).pack(pady=10)
result_label = tk.Label(root, text="", font=("Arial", 14, "bold"), bg="#f3f3f3", justify="left")
result_label.pack(pady=10)

root.mainloop()
