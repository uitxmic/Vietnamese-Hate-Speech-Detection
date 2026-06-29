import time
import pandas as pd
import numpy as np
import torch

from sentence_transformers import SentenceTransformer

from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import (
    accuracy_score,
    f1_score,
)


def evaluate_model(y_true, y_pred, model_name=""):
    print(f"\n{'='*20} KẾT QUẢ ĐÁNH GIÁ: {model_name} {'='*20}")
    print(f"Accuracy        : {accuracy_score(y_true, y_pred):.4f}")
    print(f"Macro Avg F1    : {f1_score(y_true, y_pred, average='macro'):.4f}")
    print(f"F1 - Clean (0)  : {f1_score(y_true, y_pred, pos_label=0):.4f}")
    print(f"F1 - Toxic (1)  : {f1_score(y_true, y_pred, pos_label=1):.4f}")
    print("=" * 60)


# =====================================================
# LOAD DATA
# =====================================================

print("Đang tải dữ liệu...")

train_df = pd.read_csv("final_train.csv").dropna(subset=["text", "label"])

test_df = pd.read_csv("final_test.csv").dropna(subset=["text", "label"])

X_train = train_df["text"].astype(str).tolist()
y_train = train_df["label"].astype(int).tolist()

X_test = test_df["text"].astype(str).tolist()
y_test = test_df["label"].astype(int).tolist()

print(f"Train Samples: {len(X_train):,}")
print(f"Test Samples : {len(X_test):,}")


# =====================================================
# LOAD LABSE
# =====================================================

device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"\nĐang tải LaBSE trên thiết bị: {device}")

embedder = SentenceTransformer("sentence-transformers/LaBSE", device=device)


# =====================================================
# ENCODE TRAIN
# =====================================================

print("\nĐang mã hóa Train...")

start = time.time()

X_train_vec = embedder.encode(
    X_train, batch_size=128, show_progress_bar=True, convert_to_numpy=True
)

train_encode_time = time.time() - start

print(f"Train Encoding Time: " f"{train_encode_time:.2f}s")


# =====================================================
# ENCODE TEST
# =====================================================

print("\nĐang mã hóa Test...")

start = time.time()

X_test_vec = embedder.encode(
    X_test, batch_size=128, show_progress_bar=True, convert_to_numpy=True
)

test_encode_time = time.time() - start

print(f"Test Encoding Time: " f"{test_encode_time:.2f}s")


# =====================================================
# FEATURE SCALING
# =====================================================

print("\nĐang chuẩn hóa vector...")

scaler = StandardScaler()

X_train_vec = scaler.fit_transform(X_train_vec)

X_test_vec = scaler.transform(X_test_vec)


# =====================================================
# TRAIN LINEAR SVM
# =====================================================

print("\nĐang huấn luyện LinearSVC...")

model = LinearSVC(class_weight="balanced", random_state=42, max_iter=10000)

start = time.time()

model.fit(X_train_vec, y_train)

train_time = time.time() - start

print(f"Training Time: " f"{train_time:.2f}s")


# =====================================================
# PREDICT
# =====================================================

print("\nĐang dự đoán...")

start = time.time()

y_pred = model.predict(X_test_vec)

predict_time = time.time() - start

print(f"Prediction Time: " f"{predict_time:.2f}s")


# =====================================================
# EVALUATION
# =====================================================

evaluate_model(y_test, y_pred, "LinearSVC + LaBSE")

print("\n================ THỐNG KÊ THỜI GIAN ================")

print(f"Train Encoding : {train_encode_time:.2f}s")

print(f"Test Encoding  : {test_encode_time:.2f}s")

print(f"Training       : {train_time:.2f}s")

print(f"Prediction     : {predict_time:.2f}s")

print("====================================================")
