import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score


def evaluate_model(y_true, y_pred, model_name=""):
    print(f"\n{'='*20} KẾT QUẢ ĐÁNH GIÁ: {model_name} {'='*20}")
    print(f"Accuracy        : {accuracy_score(y_true, y_pred):.4f}")
    print(f"Macro Avg F1    : {f1_score(y_true, y_pred, average='macro'):.4f}")
    print(f"F1 - Clean (0)  : {f1_score(y_true, y_pred, pos_label=0):.4f}")
    print(f"F1 - Toxic (1)  : {f1_score(y_true, y_pred, pos_label=1):.4f}")
    print("=" * 60)


# 1. Tải dữ liệu (Đồng bộ cấu trúc cho cả 3 file)
train_df = pd.read_csv("final_train.csv").dropna(subset=["text", "label"])
test_df = pd.read_csv("final_test.csv").dropna(subset=["text", "label"])

X_train, y_train = train_df["text"].astype(str), train_df["label"].astype(int)
X_test, y_test = test_df["text"].astype(str), test_df["label"].astype(int)

# 2. Rút trích đặc trưng TF-IDF
print("Đang trích xuất đặc trưng TF-IDF...")
vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=15000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# 3. Huấn luyện Logistic Regression
print("Đang huấn luyện Logistic Regression...")
model = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)
model.fit(X_train_vec, y_train)

# 4. Dự đoán và Đánh giá
y_pred = model.predict(X_test_vec)
evaluate_model(y_test, y_pred, "Logistic Regression + TF-IDF")
