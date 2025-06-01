import pandas as pd

# Đọc dữ liệu
df = pd.read_parquet("hf://datasets/truongpdd/vietnamese_poetry/data/train-00000-of-00001.parquet")

# Lọc bài thơ bắt đầu bằng "thơ lục bát:"
df_luc_bat = df[df['text'].str.startswith('thơ lục bát:')]

print(f"Tổng số bài thơ: {len(df)}")
print(f"Số bài thơ lục bát: {len(df_luc_bat)}")

# Lưu thơ lục bát vào file
file_path = "data_poem.txt"
with open(file_path, "w", encoding="utf-8") as f:
    for item in df_luc_bat['text']:
        f.write("%s\n" % item)

print(f"\nĐã lưu {len(df_luc_bat)} bài thơ lục bát vào {file_path}")

# In ra một số ví dụ
print("\nVí dụ một số bài thơ lục bát:")
for i, poem in enumerate(df_luc_bat['text'].head(3)):
    print(f"\nBài {i+1}:")
    print(poem)