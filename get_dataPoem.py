from torch.utils.data import Dataset
from typing import List, Dict, Any
import re
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
class PoemDataset(Dataset):
    def __init__(self, poems: List[str], tokenizer, max_length: int = 256):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.examples = []
        
        # Tiền xử lý dữ liệu đặc biệt cho thơ lục bát
        processed_poems = self._preprocess_poems(poems)
        
        print(f"Số bài thơ sau khi tiền xử lý: {len(processed_poems)}")
        
        # Tokenize dữ liệu với format đặc biệt cho thơ
        for poem in processed_poems:
            try:
                # Sử dụng format đặc biệt cho thơ lục bát
                formatted_poem = f"<|thơ|>{poem}<|kết|>"
                
                # Tokenize
                encoding = tokenizer(
                    formatted_poem,
                    max_length=max_length,
                    truncation=True,
                    padding=False,
                    return_tensors="pt"
                )
                
                # Chỉ giữ lại những sequences có độ dài hợp lý cho thơ lục bát
                if len(encoding['input_ids'][0]) >= 20:  # Tối thiểu 20 tokens cho một bài thơ
                    self.examples.append(encoding['input_ids'][0])
                    
            except Exception as e:
                print(f"Lỗi khi tokenize: {str(e)}")
                continue
        
        print(f"Số examples sau khi tokenize: {len(self.examples)}")
        
        if len(self.examples) == 0:
            raise ValueError("Không có dữ liệu hợp lệ sau khi xử lý!")
    
    def _preprocess_poems(self, poems: List[str]) -> List[str]:
        """Tiền xử lý dữ liệu thơ lục bát đặc biệt"""
        processed = []
        
        for poem_text in poems:
            if not poem_text or len(poem_text.strip()) == 0:
                continue
                
            # clear prefix "thơ lục bát:"
            clean_poem = poem_text.replace("thơ lục bát:", "").strip()
            
            clean_poem = re.sub(r'\n{3,}', '\n\n', clean_poem)  # clear newline
            clean_poem = re.sub(r'[ \t]+', ' ', clean_poem)     # clear space
            clean_poem = clean_poem.strip()
            
        
            if 50 <= len(clean_poem) <= 1000:
                # Vietnamese word checked
                vietnamese_chars = re.findall(r'[àáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ]', clean_poem.lower())
                if len(vietnamese_chars) >= 10: 
                    processed.append(clean_poem)
        
        return processed
    
    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.examples[idx]
    
    def __len__(self) -> int:
        return len(self.examples)

def prepare_data():
    """Chuẩn bị dữ liệu training với focus vào chất lượng"""
    try:
        df = pd.read_parquet("hf://datasets/truongpdd/vietnamese_poetry/data/train-00000-of-00001.parquet")
        
        def is_quality_poem(text):
            clean_text = text.replace("thơ lục bát:", "").strip()
            # check length
            if not (50 <= len(clean_text) <= 1000):
                return False
            # check vietnamese characters
            vietnamese_chars = re.findall(r'[àáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ]', clean_text.lower())
            if len(vietnamese_chars) < 10:
                return False
            # check line break
            if '\n' not in clean_text:
                return False
            return True
        
        df_luc_bat = df[df['text'].str.contains('thơ lục bát:', na=False, case=False)]
        df_luc_bat = df_luc_bat[df_luc_bat['text'].notna()]
        
        # split data
        poems = df_luc_bat['text'].tolist()
        train_poems, val_poems = train_test_split(
            poems, 
            test_size=0.15,
            random_state=42,
            shuffle=True
        )
        
        print(f"Tổng số bài thơ: {len(df)}")
        print(f"Số bài thơ lục bát gốc: {len(df[df['text'].str.contains('thơ lục bát:', na=False, case=False)])}")
        print(f"Số bài thơ lục bát chất lượng: {len(df_luc_bat)}")
        print(f"Số bài thơ train: {len(train_poems)}")
        print(f"Số bài thơ validation: {len(val_poems)}")
        
        return train_poems, val_poems
        
    except Exception as e:
        print(f"Lỗi khi đọc dữ liệu: {e}")
        raise
