from transformers import pipeline
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = GPT2LMHeadModel.from_pretrained('meomeo163/luc-bat-poet-model')
tokenizer = GPT2Tokenizer.from_pretrained('meomeo163/luc-bat-poet-model')

pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, device=device)


prompt_text = "Ngẫm hay trăm sự tại trời"
print(f"\nPrompt: '{prompt_text}'")

try:
    generated_output = pipe(
        prompt_text,
        max_length=100, # max length of poet
        num_return_sequences=1, # max poet generated
        do_sample=True,
        temperature=0.7, # creative param
        top_k=30,      # get highest proba of next 30 token
        top_p=0.85,    # highest proba of token
        no_repeat_ngram_size=3,
        repetition_penalty=1.15,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    print(f"\nBài thơ với từ khoá {prompt_text}:")
    for i, seq in enumerate(generated_output):
        poem_text = seq['generated_text']
        print(f"Bài thơ {i+1}:")
        print(poem_text)
        print("-" * 30)

except Exception as e:
    print(f"Lỗi khi sử dụng pipeline: {e}")

print("-------------------------------------")
