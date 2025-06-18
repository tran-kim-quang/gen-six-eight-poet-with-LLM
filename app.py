import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Cấu hình trang
st.set_page_config(
    page_title="Thơ Lục Bát AI",
    layout="centered"
)

# Tiêu đề và mô tả
st.title("Thơ Lục Bát AI")

# Tải mô hình và tokenizer
@st.cache_resource
def load_model():
    model_name = "meomeo163/luc-bat-poet-model"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    return model, tokenizer

# Tải mô hình
with st.spinner("Đang tải mô hình..."):
    model, tokenizer = load_model()

# Tạo form nhập liệu
with st.form("poem_form"):
    user_input = st.text_area(
        "Nhập câu thơ hoặc từ khóa để bắt đầu:",
        height=100
    )
    
    max_length = st.slider(
        "Độ dài tối đa của bài thơ:",
        min_value=50,
        max_value=500,
        value=200,
        step=50
    )
    
    temperature = st.slider(
        "Độ sáng tạo:",
        min_value=0.1,
        max_value=1.0,
        value=0.7,
        step=0.1,
        help="Giá trị càng cao, thơ càng sáng tạo nhưng có thể ít mạch lạc hơn"
    )
    
    submitted = st.form_submit_button("Sáng tác thơ")

if submitted and user_input:
    with st.spinner("Đang sáng tác thơ..."):
        # Chuẩn bị input
        input_text = f"<|thơ|>{user_input}"
        inputs = tokenizer(input_text, return_tensors="pt")
        
        # Setup tham số tạo thơ
        outputs = model.generate(
            inputs["input_ids"],
            max_length=max_length,
            temperature=temperature,
            num_return_sequences=1,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=True,
            top_k=30,      # get highest proba of next 30 token
            top_p=0.85,    # highest proba of token
            no_repeat_ngram_size=3,
            repetition_penalty=1.15,
        )
        
        # Giải mã kết quả
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Hiển thị kết quả
        st.markdown("### Bài thơ được tạo:")
        st.markdown(f"```\n{generated_text}\n```")
        
        # Thêm nút sao chép
        st.button("Sao chép bài thơ", on_click=lambda: st.write(generated_text))
