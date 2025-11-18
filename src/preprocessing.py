import os
import re
import string
import pandas as pd
from pathlib import Path
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split

STOP_WORDS = set(stopwords.words("english"))
LEMMATIZER = WordNetLemmatizer()

# =============================
# 🔹 HÀM LÀM SẠCH NỘI DUNG
# =============================
def clean_text(text: str) -> str:
    """Làm sạch văn bản: bỏ ký tự đặc biệt, lowercase, stopwords, lemmatization"""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"http\S+", " ", text)           # bỏ link
    text = re.sub(f"[{string.punctuation}]", " ", text)  # bỏ dấu câu
    text = re.sub(r"\d+", " ", text)               # bỏ số
    text = re.sub(r"\s+", " ", text).strip()       # bỏ khoảng trắng thừa

    words = [LEMMATIZER.lemmatize(w) for w in text.split() if w not in STOP_WORDS]
    return " ".join(words)

# =============================
# 🔹 TIỀN XỬ LÝ DỮ LIỆU CHÍNH
# =============================
def preprocess_data(input_path="data/spam.csv", output_path="data/preprocessed.csv"):
    """
    Đọc dữ liệu từ file CSV (spam/ham và text),
    làm sạch text, mã hóa nhãn, và lưu lại.
    """
    print("🧹 Đang tiền xử lý dữ liệu...")

    # Đọc dữ liệu gốc
    df = pd.read_csv(input_path)
    df.columns = df.columns.str.lower()  # chuẩn hóa tên cột

    # Nếu file không có header
    if "label" not in df.columns or "text" not in df.columns:
        df = pd.read_csv(input_path, names=["label", "text"], header=None)

    # Chuẩn hóa nhãn
    df["label"] = df["label"].map(
        {"spam": 1, "ham": 0, "non-spam": 0, "not spam": 0, "legit": 0}
    ).fillna(df["label"])

    # Làm sạch text
    df["text"] = df["text"].astype(str).apply(clean_text)

    # Bỏ giá trị rỗng và trùng lặp
    df = df.dropna().drop_duplicates()

    # Tạo thư mục nếu chưa có
    Path(os.path.dirname(output_path)).mkdir(parents=True, exist_ok=True)

    # Lưu dữ liệu đã xử lý
    df.to_csv(output_path, index=False)
    print(f"✅ Dữ liệu đã xử lý và lưu tại: {output_path}")
    print(f"📊 Tổng số mẫu sau xử lý: {len(df)}")

    return df

# =============================
# 🔹 HÀM CHIA DỮ LIỆU
# =============================
def split_train_test(df, train_size=0.8):
    from sklearn.model_selection import train_test_split
    df_train, df_test = train_test_split(
        df, test_size=1 - train_size, stratify=df["label"], random_state=42
    )

    df_train.to_csv("data/train.csv", index=False)
    df_test.to_csv("data/test.csv", index=False)
    print(f"📚 Train: {len(df_train)} | Test: {len(df_test)}")
    return df_train, df_test



# =============================
# 🔹 MAIN ENTRY POINT
# =============================
if __name__ == "__main__":
    # 1️⃣ Tiền xử lý dữ liệu
    df_clean = preprocess_data()

    # 2️⃣ Chia train/val/test
    split_train_test(df_clean)
