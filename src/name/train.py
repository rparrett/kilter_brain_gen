import pprint
import re
import emoji

from datasets import Features, Value, load_dataset
from transformers import (
    DataCollatorForLanguageModeling,
    GPT2LMHeadModel,
    GPT2TokenizerFast,
    Trainer,
    TrainingArguments,
)

out_dir = "models/name"

features = Features(
    {
        "name": Value("string"),
        "quality_average": Value("float"),
    }
)


def is_valid(example):
    text = str(example["name"])
    return (
        not any(char in emoji.EMOJI_DATA for char in text)
        and not re.search(r"[\(\)\[\]]", text)
        and not re.search(r"\d", text)
    )


dataset = load_dataset(
    "csv", data_files="climbs.csv", delimiter=",", features=features, split="train"
)
dataset = dataset.filter(lambda example: example["name"] is not None)
dataset = dataset.filter(lambda example: example["quality_average"] >= 2.5)
dataset = dataset.filter(is_valid)

model_name = "gpt2"
tokenizer = GPT2TokenizerFast.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

model = GPT2LMHeadModel.from_pretrained(model_name)


def tokenize(batch):
    return tokenizer(
        batch["name"], truncation=True, padding="max_length", max_length=64
    )


tokenized = dataset.map(tokenize, batched=True)

print(f"Names: {len(dataset)}")
pprint.pprint(dataset[0])
pprint.pprint(tokenized[0])

tokenized = tokenized.remove_columns(["name", "quality_average"])

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

training_args = TrainingArguments(
    output_dir=out_dir,
    report_to="tensorboard",
    per_device_train_batch_size=8,
    num_train_epochs=5,
    save_steps=500,
    logging_steps=50,
    save_total_limit=2,
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=tokenized,
)

trainer.train()

model.save_pretrained(out_dir)
