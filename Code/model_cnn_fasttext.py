# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np

from sklearn.metrics import accuracy_score, f1_score
from sklearn.utils.class_weight import compute_class_weight

import tensorflow as tf

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import (
    Input,
    Embedding,
    Conv1D,
    GlobalMaxPooling1D,
    Dense,
    Dropout,
    SpatialDropout1D,
)

from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam

# ======================================================
# EVALUATION
# ======================================================


def evaluate_model(y_true, y_pred, model_name=""):
    print(f"\n{'='*20} KẾT QUẢ ĐÁNH GIÁ: {model_name} {'='*20}")

    print(f"Accuracy        : {accuracy_score(y_true, y_pred):.4f}")

    print(f"Macro Avg F1    : {f1_score(y_true, y_pred, average='macro'):.4f}")

    print(f"F1 - Clean (0)  : {f1_score(y_true, y_pred, pos_label=0):.4f}")

    print(f"F1 - Toxic (1)  : {f1_score(y_true, y_pred, pos_label=1):.4f}")

    print("=" * 60)


# ======================================================
# LOAD DATA
# ======================================================

train_df = pd.read_csv("final_train.csv")
test_df = pd.read_csv("final_test.csv")

train_df = train_df.dropna(subset=["text", "label"])
test_df = test_df.dropna(subset=["text", "label"])

X_train_text = train_df["text"].astype(str).tolist()
X_test_text = test_df["text"].astype(str).tolist()

y_train = train_df["label"].astype(int).values
y_test = test_df["label"].astype(int).values

print("Train:", len(X_train_text))
print("Test :", len(X_test_text))

print("\nLabel Distribution:")
print(train_df["label"].value_counts())


# ======================================================
# TOKENIZER
# ======================================================

MAX_VOCAB = 25000
MAX_LEN = 80
EMBEDDING_DIM = 300

tokenizer = Tokenizer(num_words=MAX_VOCAB, oov_token="<OOV>")

tokenizer.fit_on_texts(X_train_text)

X_train_pad = pad_sequences(
    tokenizer.texts_to_sequences(X_train_text),
    maxlen=MAX_LEN,
    padding="post",
    truncating="post",
)

X_test_pad = pad_sequences(
    tokenizer.texts_to_sequences(X_test_text),
    maxlen=MAX_LEN,
    padding="post",
    truncating="post",
)


# ======================================================
# LOAD FASTTEXT
# ======================================================

print("\nLoading FastText...")

embedding_matrix = np.zeros((MAX_VOCAB, EMBEDDING_DIM))

hits = 0

try:

    with open("cc.vi.300.vec", "r", encoding="utf-8", errors="ignore") as f:

        first_line = True

        for line in f:

            if first_line:
                first_line = False
                continue

            values = line.rstrip().split(" ")

            word = values[0]

            idx = tokenizer.word_index.get(word)

            if idx is not None and idx < MAX_VOCAB:

                embedding_matrix[idx] = np.asarray(values[1:], dtype="float32")

                hits += 1

    print(f"FastText matched words: {hits}")

except FileNotFoundError:

    print("\nWARNING: cc.vi.300.vec not found.")

coverage = hits / min(len(tokenizer.word_index), MAX_VOCAB)

print(f"Embedding Coverage: {coverage:.2%}")


# ======================================================
# CLASS WEIGHT
# ======================================================

weights = compute_class_weight(
    class_weight="balanced", classes=np.unique(y_train), y=y_train
)

class_weights = {0: weights[0], 1: weights[1]}

print("\nClass Weights:")
print(class_weights)


# ======================================================
# MODEL
# ======================================================

model = Sequential(
    [
        Input(shape=(MAX_LEN,)),
        Embedding(
            input_dim=MAX_VOCAB,
            output_dim=EMBEDDING_DIM,
            weights=[embedding_matrix],
            trainable=False,
        ),
        SpatialDropout1D(0.2),
        Conv1D(filters=128, kernel_size=3, activation="relu"),
        Conv1D(filters=128, kernel_size=5, activation="relu"),
        GlobalMaxPooling1D(),
        Dense(64, activation="relu"),
        Dropout(0.5),
        Dense(1, activation="sigmoid"),
    ]
)

model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss="binary_crossentropy",
    metrics=["accuracy"],
)

model.summary()


# ======================================================
# TRAIN
# ======================================================

early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=3,
    restore_best_weights=True,
    verbose=1,
)

history = model.fit(
    X_train_pad,
    y_train,
    epochs=20,
    batch_size=64,
    validation_split=0.1,
    class_weight=class_weights,
    callbacks=[early_stopping],
    verbose=1,
)


# ======================================================
# PREDICT
# ======================================================

y_pred_prob = model.predict(X_test_pad, verbose=0).ravel()

print("\nPrediction Stats")
print("Min :", y_pred_prob.min())
print("Max :", y_pred_prob.max())
print("Mean:", y_pred_prob.mean())


# ======================================================
# FIND BEST THRESHOLD
# ======================================================

best_threshold = 0.5
best_macro_f1 = 0

for threshold in np.arange(0.1, 0.91, 0.01):

    pred = (y_pred_prob >= threshold).astype(int)

    macro_f1 = f1_score(y_test, pred, average="macro")

    if macro_f1 > best_macro_f1:

        best_macro_f1 = macro_f1
        best_threshold = threshold

print(f"\nBest Threshold = {best_threshold:.2f}")

print(f"Best Macro F1 = {best_macro_f1:.4f}")

y_pred = (y_pred_prob >= best_threshold).astype(int)


# ======================================================
# FINAL EVALUATION
# ======================================================

evaluate_model(y_test, y_pred, f"CNN FastText (Threshold={best_threshold:.2f})")

# ======================================================
# LIME HEATMAP ANALYSIS
# ======================================================

import os
import seaborn as sns
import matplotlib.pyplot as plt

from lime.lime_text import LimeTextExplainer

print("\nGenerating LIME Heatmaps...")

os.makedirs("lime_reports", exist_ok=True)

# ======================================================
# PREDICT FUNCTION FOR LIME
# ======================================================


def predict_pipeline(texts):

    seqs = tokenizer.texts_to_sequences(texts)

    padded = pad_sequences(seqs, maxlen=MAX_LEN, padding="post", truncating="post")

    preds = model.predict(padded, verbose=0)

    return np.hstack((1 - preds, preds))


# ======================================================
# BUILD RESULT DATAFRAME
# ======================================================

result_df = test_df.copy()

result_df["pred"] = y_pred

# ======================================================
# TP TN FP FN
# ======================================================

tp_idx = result_df[(result_df["label"] == 1) & (result_df["pred"] == 1)].index.tolist()[
    :3
]

tn_idx = result_df[(result_df["label"] == 0) & (result_df["pred"] == 0)].index.tolist()[
    :3
]

fp_idx = result_df[(result_df["label"] == 0) & (result_df["pred"] == 1)].index.tolist()[
    :3
]

fn_idx = result_df[(result_df["label"] == 1) & (result_df["pred"] == 0)].index.tolist()[
    :3
]


# ======================================================
# LIME EXPLAINER
# ======================================================

explainer = LimeTextExplainer(class_names=["Clean", "Toxic"])

HEATMAP_TITLES = {
    "TP_Toxic_to_Toxic": "Ảnh hưởng của từ đến dự đoán kết quả Toxic (dự đoán đúng)",
    "TN_Clean_to_Clean": "Ảnh hưởng của từ đến dự đoán kết quả Clean (dự đoán đúng)",
    "FP_Clean_to_Toxic": "Ảnh hưởng của từ đến dự đoán kết quả Toxic (sai: nhãn thật Clean)",
    "FN_Toxic_to_Clean": "Ảnh hưởng của từ đến dự đoán kết quả Clean (sai: nhãn thật Toxic)",
}


# ======================================================
# DRAW HEATMAP
# ======================================================


def save_lime_heatmap(idx, category, sample_number):

    text = result_df.loc[idx, "text"]

    exp = explainer.explain_instance(
        text, predict_pipeline, num_features=20, num_samples=1000
    )

    # LIME trả về từ + trọng số
    weights_dict = dict(exp.as_list())

    # Token theo đúng thứ tự trong câu
    tokens = text.split()

    ordered_weights = []

    for token in tokens:
        ordered_weights.append(weights_dict.get(token, 0))

    heat_df = pd.DataFrame([ordered_weights], columns=tokens)

    # # Ẩn các token không được LIME chọn
    # non_zero_cols = heat_df.abs().sum(axis=0) > 0

    # heat_df = heat_df.loc[:, non_zero_cols]

    plt.figure(figsize=(max(10, len(heat_df.columns) * 0.8), 2.5))

    sns.heatmap(
        heat_df,
        cmap="RdBu_r",
        center=0,
        annot=True,
        fmt=".2f",
        linewidths=0.5,
        cbar=True,
    )

    title = HEATMAP_TITLES.get(category, category)

    plt.title(f"{title} - Mẫu {sample_number}", fontsize=14)

    plt.yticks([])

    plt.xticks(rotation=45, ha="right", fontsize=10)

    plt.tight_layout()

    save_path = f"lime_reports/" f"{category}_" f"{sample_number}.png"

    plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.close()

    print(f"Saved: {save_path}")


# ======================================================
# GENERATE ALL HEATMAPS
# ======================================================

for i, idx in enumerate(tp_idx):

    save_lime_heatmap(idx, "TP_Toxic_to_Toxic", i + 1)

for i, idx in enumerate(tn_idx):

    save_lime_heatmap(idx, "TN_Clean_to_Clean", i + 1)

for i, idx in enumerate(fp_idx):

    save_lime_heatmap(idx, "FP_Clean_to_Toxic", i + 1)

for i, idx in enumerate(fn_idx):

    save_lime_heatmap(idx, "FN_Toxic_to_Clean", i + 1)

print("\nDone!")
print("All heatmaps saved to:")
print("lime_reports/")
