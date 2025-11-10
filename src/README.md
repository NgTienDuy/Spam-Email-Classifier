# 📧 Spam Email Classification - Module Documentation

Tài liệu này mô tả 3 script chính trong dự án phân loại email spam: tiền xử lý dữ liệu, huấn luyện mô hình, và đánh giá mô hình.

---

## 1️⃣ Preprocess Script (`preprocess.py`)

### Mục đích
Tiền xử lý dữ liệu email trước khi huấn luyện mô hình:
- Làm sạch văn bản: lowercase, loại bỏ URL, số, dấu câu, ký tự đặc biệt, stopwords, lemmatization.
- Chuẩn hóa nhãn: `spam` → 1, `ham/non-spam/legit` → 0.
- Xử lý dữ liệu trống hoặc trùng lặp.
- Chia dữ liệu thành train và test (hoặc train/val/test).

### Cách dùng
python preprocess.py

Kết quả
data/preprocessed.csv (hoặc train.csv & test.csv)
In ra số lượng mẫu đã xử lý.

## 2️⃣ Train Model Script (train_model.py)
### Mục đích
Huấn luyện các mô hình Machine Learning cổ điển:
- Naive Bayes
- Logistic Regression
- SVM
- Random Forest
- KNN

### Quy trình
- Đọc dữ liệu train từ train.csv.
- Chuyển văn bản sang vector bằng TF-IDF.
- Huấn luyện với K-Fold Cross Validation (mặc định 5-fold).
- Huấn luyện lại toàn bộ tập train.
- Lưu từng mô hình .pkl vào thư mục model/.
- Lưu kết quả cross-validation vào model/cv_results.csv.

Cách dùng
python train_model.py

Kết quả
model/naive_bayes.pkl
model/logistic_regression.pkl
model/svm.pkl
model/random_forest.pkl
model/knn.pkl
model/cv_results.csv

## 3️⃣ Evaluate Script (evaluate.py)
### Mục đích
Đánh giá hiệu suất các mô hình đã huấn luyện trên tập test (test.csv).

### Quy trình
- Đọc tập test.
- Load tất cả các file .pkl trong model/.
- Dự đoán nhãn test và tính các chỉ số:
Accuracy
Precision
Recall
F1-score
- Lưu kết quả vào model/evaluation_results.csv.

## 🔗 Quy trình tổng thể
- Chạy tiền xử lý:
python preprocess.py
- Huấn luyện mô hình:
python train_model.py
- Đánh giá mô hình:
python evaluate.py

## 📝 Ghi chú
- TF-IDF được sử dụng để biểu diễn văn bản dưới dạng vector số.
- K-Fold Cross Validation giúp đánh giá mô hình ổn định.
- Các file .pkl có thể dùng trực tiếp để dự đoán email mới.