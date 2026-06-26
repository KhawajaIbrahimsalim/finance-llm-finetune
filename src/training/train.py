import torch
from transformers import TrainingArguments, Trainer
from transformers import DataCollatorForLanguageModeling


def train(model, tokenizer, dataset):
    has_cuda = torch.cuda.is_available()

    # VERY IMPORTANT (prevents padding bugs)
    tokenizer.pad_token = tokenizer.eos_token

    def tokenize(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=256,
        )

    tokenized_dataset = dataset.map(
        tokenize,
        batched=True,
        remove_columns=dataset.column_names,
    )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
        pad_to_multiple_of=8 if has_cuda else None,
    )

    args = TrainingArguments(
        output_dir="experiments/run_1",
        num_train_epochs=1,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        fp16=has_cuda and not torch.cuda.is_bf16_supported(),
        bf16=has_cuda and torch.cuda.is_bf16_supported(),
        logging_steps=10,
        save_steps=50,
        save_total_limit=2,
        dataloader_pin_memory=has_cuda,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
    )

    trainer.train()

    # SAVE ADAPTER (CRITICAL)
    model.save_pretrained("experiments/run_1")
    tokenizer.save_pretrained("experiments/run_1")
