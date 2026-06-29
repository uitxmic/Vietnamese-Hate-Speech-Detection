from transformers import MarianMTModel, MarianTokenizer
import pandas as pd
import torch

# 1. Cấu hình thiết bị (Dùng GPU nếu có để nhanh hơn gấp 10-50 lần)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Đang chạy trên: {device}")

# 2. Tải model pre-trained (Trung -> Anh)
model_name = "Helsinki-NLP/opus-mt-zh-en"
tokenizer = MarianTokenizer.from_pretrained(model_name)
model = MarianMTModel.from_pretrained(model_name).to(device)

# 3. Hàm dịch theo Batch (Xử lý nhiều dòng cùng lúc để tăng tốc độ)
def translate_batch(texts, batch_size=32):
    results = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        # Tokenize
        inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
        # Generate translation
        translated = model.generate(**inputs)
        # Decode về text
        decoded = tokenizer.batch_decode(translated, skip_special_tokens=True)
        results.extend(decoded)
        
        if i % 100 == 0:
            print(f"Đã xử lý {i}/{len(texts)} dòng")
    return results

# 4. Thực thi
df = pd.read_csv('test.csv', encoding='gbk')

# Chuyển cột content thành list string và đảm bảo không có giá trị null
texts_to_translate = df['content'].fillna("").astype(str).tolist()

# Dịch
df['content_english'] = translate_batch(texts_to_translate, batch_size=16)

# 5. Lưu file
df.to_csv('toxic_comments_translated_hf.csv', index=False)