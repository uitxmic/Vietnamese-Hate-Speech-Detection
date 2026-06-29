import pandas as pd
import re
import unicodedata
import emoji
from underthesea import word_tokenize
from sklearn.model_selection import train_test_split


# ==========================================
# HÀM TIỀN XỬ LÝ DỮ LIỆU CHUNG
# ==========================================
# Đọc file teencode.txt thành dictionary
def load_teencode_dict(filepath="teencode.txt"):
    teencode_dict = {}
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) >= 2:
                    teencode_dict[parts[0].lower()] = parts[1].lower()
    except FileNotFoundError:
        print(f"⚠️ Không tìm thấy file {filepath}. Vui lòng kiểm tra lại!")
    return teencode_dict


teencode_map = load_teencode_dict()


def text_preprocess(text):
    if not isinstance(text, str):
        return ""

    # 1. Chuẩn hóa Unicode (NFC)
    text = unicodedata.normalize("NFC", text)

    # 2. Chuẩn hóa chữ thường
    text = text.lower()

    # 3. Loại bỏ URL và Hashtag
    text = re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE)
    text = re.sub(r"#\w+", "", text)

    # 4. Chuẩn hóa biểu cảm (Emoji)
    # Chuyển icon ảnh thành dạng text (VD: 🤣 -> :rolling_on_the_floor_laughing:)
    text = emoji.demojize(text, delimiters=(" ", " "))
    # Thay thế các icon text phổ biến
    text = re.sub(r"\(\(\(\(\=", " cười ", text)
    text = re.sub(r":\)|:\-\)|=\)", " cười ", text)
    text = re.sub(r":\(|:\-\(|=\(", " buồn ", text)

    # 5. Loại bỏ ký tự đặc biệt (Chỉ giữ lại chữ cái, số, khoảng trắng và dấu '_' cho từ ghép)
    text = re.sub(r"[^\w\s_]", " ", text)

    # 6. Loại bỏ ký tự lặp (VD: hayyyyy -> hay)
    text = re.sub(
        r"([a-zàáảãạăằắẳẵặâầấẩẫậèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵđ])\1+",
        r"\1",
        text,
    )

    # 7. Thay thế từ viết tắt (Teencode)
    words = text.split()
    words = [teencode_map.get(w, w) for w in words]
    text = " ".join(words)

    # 8. Loại bỏ khoảng trống thừa
    text = re.sub(r"\s+", " ", text).strip()

    # 9. Tách từ (Word Segmentation với Underthesea)
    if text:
        text = word_tokenize(text, format="text")

    return text


# ==========================================
# PHẦN 1: ĐỌC VÀ LÀM SẠCH DỮ LIỆU
# ==========================================
print("1. Đang đọc dữ liệu Mạng xã hội gốc (ViHSD & ViCTSD)...")
vh_train = pd.read_csv("vihsd_train.csv")
vh_dev = pd.read_csv("vihsd_dev.csv")
vh_test = pd.read_csv("vihsd_test.csv")

vc_train = pd.read_csv("ViCTSD_train.csv")
vc_valid = pd.read_csv("ViCTSD_valid.csv")
vc_test = pd.read_csv("ViCTSD_test.csv")

print("2. Đang đọc 3 file dữ liệu Review sản phẩm mới...")
rev_train = pd.read_csv("train.csv")
rev_dev = pd.read_csv("dev.csv")
rev_test = pd.read_csv("test.csv")

# Xử lý tập Mạng xã hội
vh_all = pd.concat([vh_train, vh_dev, vh_test], ignore_index=True).dropna(
    subset=["free_text"]
)
vh_all["text"] = vh_all["free_text"]
vh_all["label"] = vh_all["label_id"].map({0: 0, 1: 1, 2: 1})
vh_df = vh_all[["text", "label"]]

vc_all = pd.concat([vc_train, vc_valid, vc_test], ignore_index=True).dropna(
    subset=["Comment"]
)
vc_all["text"] = vc_all["Comment"]
vc_all["label"] = vc_all["Toxicity"].map({0: 0, 1: 1})
vc_df = vc_all[["text", "label"]]

social_df = pd.concat([vh_df, vc_df], ignore_index=True)

# Xử lý tập Review
print("3. Đang đồng bộ hóa nhãn dữ liệu mới...")
rev_all = pd.concat([rev_train, rev_dev, rev_test], ignore_index=True).dropna(
    subset=["texts", "labels"]
)
rev_all["text"] = rev_all["texts"]
rev_all["label"] = rev_all["labels"].map({"POS": 0, "NEU": 0, "NEG": 1})
rev_df = rev_all[["text", "label"]].dropna(subset=["label"])
rev_df["label"] = rev_df["label"].astype(int)

# Gộp toàn bộ dữ liệu để tiền xử lý 1 lần cho đồng nhất
all_data = pd.concat([social_df, rev_df], ignore_index=True)

# ==========================================
# PHẦN 2: THỰC THI TIỀN XỬ LÝ (PREPROCESSING)
# ==========================================
print(
    "4. Đang chạy chu trình tiền xử lý dữ liệu (Sẽ mất một chút thời gian do Word Tokenize)..."
)
all_data["text"] = all_data["text"].apply(text_preprocess)

# Loại bỏ các câu bị trống sau khi tiền xử lý (VD: câu chỉ chứa mỗi URL)
all_data = all_data[all_data["text"].str.strip() != ""]
all_data = all_data.dropna(subset=["text"])

# Loại bỏ trùng lặp trên toàn bộ tập dữ liệu đã làm sạch
all_data = all_data.drop_duplicates(subset=["text"], keep="first")

# ==========================================
# PHẦN 3: PHÂN CHIA DỮ LIỆU TỐI ƯU
# ==========================================
print("5. Đang phân chia dữ liệu...")
# Tách lại tập Social và Review dựa trên tỷ lệ ban đầu
# (Lưu ý: Độ dài có thể thay đổi do loại bỏ trùng lặp và câu trống)
social_clean = all_data.iloc[: len(social_df.drop_duplicates(subset=["text"]).dropna())]
rev_clean = all_data.iloc[len(social_clean) :]

# Phân chia tập Mạng xã hội thành Train/Valid/Test (Tỷ lệ 80/10/10)
train_social, temp_df = train_test_split(
    social_clean, test_size=0.20, random_state=42, stratify=social_clean["label"]
)
valid_df, test_df = train_test_split(
    temp_df, test_size=0.50, random_state=42, stratify=temp_df["label"]
)

# Bổ sung Data chống Bias vào tập Train
final_train = pd.concat([train_social, rev_clean], ignore_index=True)

# Lọc trùng lặp lần cuối trên tập Train (Phòng trường hợp trùng giữa 2 nguồn)
final_train = final_train.drop_duplicates(subset=["text"], keep="first")

# ==========================================
# PHẦN 4: XUẤT DỮ LIỆU
# ==========================================
final_train.to_csv("final_train.csv", index=False)
valid_df.to_csv("final_valid.csv", index=False)
test_df.to_csv("final_test.csv", index=False)

print("\n🎉 HOÀN TẤT QUÁ TRÌNH TẠO VÀ LÀM SẠCH DỮ LIỆU!")
print(
    f" -> Kích thước tập Train mới (Đã kết hợp và chuẩn hóa): {len(final_train)} dòng"
)
print(f" -> Kích thước tập Valid: {len(valid_df)} dòng")
print(f" -> Kích thước tập Test: {len(test_df)} dòng")
