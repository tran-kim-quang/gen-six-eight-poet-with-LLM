from train import setup_model_and_tokenizer, train_model
from get_dataPoem import prepare_data, PoemDataset
from gen_poet import generate_poem
import torch

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tokenizer = setup_model_and_tokenizer()
    model=model.to(device)

    train_poems, val_poems = prepare_data()
    train_poems = train_poems[:10000]
    val_poems = val_poems[:1000]
    max_length = 261
    train_dataset = PoemDataset(train_poems, tokenizer, max_length=max_length)
    val_dataset = PoemDataset(val_poems, tokenizer, max_length=max_length)

    train_model(model, tokenizer, train_dataset, val_dataset)

    # make predict
    test = str(input("Nhập dòng thơ đầu tiên (hoặc không): "))

    generated_poems = generate_poem(model, tokenizer, test, max_length=max_length)

    print(generated_poems)

if __name__ == '__main__':
    main()