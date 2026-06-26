from src.data.loader import load_data
from src.models.lora_model import load_model
from src.training.train import train

def main():

    dataset = load_data()

    model, tokenizer = load_model("Qwen/Qwen2.5-0.5B")

    train(model, tokenizer, dataset)

if __name__ == "__main__":
    main()