from datasets import load_dataset, Dataset, concatenate_datasets
from src.data.synthetic import synthetic_finance_data

def load_data():

    fiqa = load_dataset("pauri32/fiqa-2018")["train"]

    def format_fn(example):
        return {
            "text": f"""
Instruction: You are a financial analyst.
Input: {example.get('question', '')}
Response: {example.get('answer', '')}
"""
        }

    fiqa = fiqa.map(format_fn)

    # SAFE LIMIT (no crash)
    fiqa = fiqa.select(range(min(2000, len(fiqa))))

    synthetic = Dataset.from_list(synthetic_finance_data())

    dataset = concatenate_datasets([fiqa, synthetic])

    return dataset