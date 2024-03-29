import pprint
from datasets import Features, Value, load_dataset
from transformers import (
    TrainingArguments,
    GPT2Config,
    GPT2LMHeadModel,
    GPT2TokenizerFast,
    DataCollatorForLanguageModeling,
    Trainer,
)


out_dir = "name-model"

features = Features(
    {
        "name": Value("string"),
    }
)

dataset = load_dataset(
    "csv", data_files="climbs.csv", delimiter=",", features=features, split="train"
)
dataset = dataset.filter(lambda example: example["name"] != None)


datasets = dataset.train_test_split()

config = GPT2Config.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2", config=config)
tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
tokenizer.pad_token=tokenizer.eos_token
tokenizer.pad_token_id=tokenizer.eos_token_id

tokenized_train = datasets["train"].map(
    lambda examples: tokenizer(examples["name"]), batched=True
)
tokenized_test = datasets["test"].map(
    lambda examples: tokenizer(examples["name"]), batched=True
)

tokenized_train = tokenized_train.remove_columns(["name"])
tokenized_test = tokenized_test.remove_columns(["name"])

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=False, mlm_probability=0.15
)

training_args = TrainingArguments(
    output_dir=out_dir,  # output directory to where save model checkpoint
    evaluation_strategy="steps",  # evaluate each `logging_steps` steps
    overwrite_output_dir=True,
    num_train_epochs=2,  # number of training epochs, feel free to tweak
    per_device_train_batch_size=4,  # the training batch size, put it as high as your GPU memory fits
    gradient_accumulation_steps=1,  # accumulating the gradients before updating the weights
    per_device_eval_batch_size=8,  # evaluation batch size
    logging_steps=1000,  # evaluate, log and save model checkpoints every 1000 step
    save_steps=1000,
    report_to="tensorboard",
    remove_unused_columns=False,
    load_best_model_at_end=True,  # whether to load the best model (in terms of loss) at the end of training
    save_total_limit=3,  # whether you don't have much space so you let only 3 model weights saved in the disk
    weight_decay=0.01, # ??? people seem to do this stuff when fine-tuning?
    learning_rate=1e-5 # ??? people seem to do this stuff when fine-tuning?
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_test,
)

trainer.train()

model.save_pretrained(out_dir)
