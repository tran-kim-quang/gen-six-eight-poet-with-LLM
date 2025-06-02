from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments, DataCollatorForLanguageModeling, TrainerCallback

def setup_model_and_tokenizer(model_name: str = 'NlpHUST/gpt2-vietnamese'):
    """Thiết lập model và tokenizer với special tokens cho thơ lục bát"""
    try:
        # Load tokenizer và model
        tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        model = GPT2LMHeadModel.from_pretrained(model_name)
        
        # Thêm special tokens đặc biệt cho thơ lục bát
        special_tokens = {
            "pad_token": "<|pad|>",
            "eos_token": "<|kết|>",          # Kết thúc bài thơ
            "bos_token": "<|thơ|>",          # Bắt đầu bài thơ
            "additional_special_tokens": [
                "<|câu6|>",                   # Đánh dấu câu 6 tiếng
                "<|câu8|>",                   # Đánh dấu câu 8 tiếng
                "<|khổ|>",                    # Đánh dấu khổ thơ mới
            ]
        }
        
        # add token into tokenizer
        num_added_tokens = tokenizer.add_special_tokens(special_tokens)
        
        # Resize model embeddings
        if num_added_tokens > 0:
            model.resize_token_embeddings(len(tokenizer))
            print(f"Đã thêm {num_added_tokens} special tokens cho thơ lục bát")
        
        print(f"Vocab size: {len(tokenizer)}")
        print(f"Model vocab size: {model.config.vocab_size}")
        print(f"Special tokens: {tokenizer.special_tokens_map}")
        
        return model, tokenizer
        
    except Exception as e:
        print(f"Lỗi khi setup model: {e}")
        raise

def train_model(model, tokenizer, train_dataset, val_dataset):
    
    # Data collator cho language modeling
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # Causal LM cho generation
        pad_to_multiple_of=8,
    )
    
    # Cấu hình training arguments đặc biệt cho thơ
    training_args = TrainingArguments(
        output_dir="./luc-bat-poet-model",
        overwrite_output_dir=True,
        
        # Training parameters 
        num_train_epochs=8, 
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=32,  
        
        # Optimizer
        learning_rate=0.001, 
        weight_decay=0.01,
        adam_beta1=0.9,
        adam_beta2=0.999,
        adam_epsilon=1e-8,
        max_grad_norm=1.0,
        
        # Learning rate schedule
        lr_scheduler_type="cosine",
        warmup_steps=200,
        
        # Evaluation và logging
        eval_strategy="epoch",
        logging_steps=25,
        logging_dir="./logs",
        logging_first_step=True,
        
        # Saving
        save_strategy="epoch",
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        
        # Performance
        fp16=True,
        dataloader_num_workers=1,  # Giảm để tránh conflict
        dataloader_pin_memory=True,
        
        # Misc
        remove_unused_columns=False,
        disable_tqdm=False,
        report_to=None,
        seed=42,
        
        # Thêm các tham số để hiển thị thông tin
        log_level="info",
        logging_nan_inf_filter=False,
    )
    
    # Khởi tạo trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )
    
    print(f"Số bài thơ train: {len(train_dataset)}")
    print(f"Số bài thơ validation: {len(val_dataset)}")
    print(f"Batch size: {training_args.per_device_train_batch_size}")
    print(f"Gradient accumulation: {training_args.gradient_accumulation_steps}")
    print(f"Effective batch size: {training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps}")
    print(f"Total epochs: {training_args.num_train_epochs}")
    print(f"Learning rate: {training_args.learning_rate}")
    print(f"Max length: {train_dataset.max_length}")

    
    # Bắt đầu training
    trainer.train()
    
    # Lưu model cuối cùng
    print("\nĐang lưu model...")
    trainer.save_model("./luc-bat-poet")
    tokenizer.save_pretrained("./luc-bat-poet")
    
    return trainer