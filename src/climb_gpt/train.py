import argparse
import pprint
from datetime import datetime

from data import load_training_datasets, preprocess_datasets
from train_tokenizer import train_tokenizer
from penalty_stats_callback import PenaltyStatsCallback
from transformers import (
    DataCollatorForLanguageModeling,
    GPT2Config,
    GPT2LMHeadModel,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
)

parser = argparse.ArgumentParser(description="Train CLM model")
parser.add_argument("--run-name", type=str, help="Name for this training run")
args = parser.parse_args()

# Create run name if not provided
if args.run_name is None:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    args.run_name = f"run_{timestamp}"

OUT_DIR = f"models/climb_gpt/{args.run_name}"

datasets = load_training_datasets()

print(f"Train: {len(datasets['train'])} Test: {len(datasets['test'])}")
pprint.pprint(datasets["train"][0])

# Train tokenizer for this run
tokenizer = train_tokenizer(datasets, OUT_DIR)

print(f"Train samples: {len(datasets['train'])} Test samples: {len(datasets['test'])}")
pprint.pprint(datasets["train"][0])
print("Processed.")
preprocess_datasets(datasets, tokenizer)
pprint.pprint(datasets["train"][0])

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

model_config = GPT2Config(
    **{
        "bos_token_id": tokenizer.bos_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "vocab_size": tokenizer.vocab_size,
        "n_embd": 192,  # 768
        "n_head": 3,  # 12
        "n_layer": 3,  # 12
        "n_positions": 128,  # 1024
        "n_ctx": 128,
    }
)
model = GPT2LMHeadModel(config=model_config)

print(
    f"Model parameter count: {model.num_parameters()}, vocab size: {model.config.vocab_size}"
)

training_args = TrainingArguments(
    output_dir=OUT_DIR,  # output directory to where save model checkpoint
    eval_strategy="steps",  # evaluate each `logging_steps` steps
    save_strategy="best",
    save_total_limit=3,  # save disk space
    logging_steps=500,  # evaluate and log model checkpoints every n steps
    overwrite_output_dir=True,
    num_train_epochs=30,
    per_device_train_batch_size=16,  # put it as high as your GPU memory fits; "if gradient_accumulation_steps > 1, this is the micro-batch size" --NanoGPT
    gradient_accumulation_steps=1,  # "used to simulate larger batch sizes"
    per_device_eval_batch_size=16,
    learning_rate=1e-5,
    adam_beta1=0.9,
    adam_beta2=0.999,
    report_to="tensorboard",
    remove_unused_columns=False,
    greater_is_better=False,  # whether the best model is the one with the highest or lowest evaluation metric, e.g. loss vs accuracy
    metric_for_best_model="eval_loss",  # use eval_loss to compare models
    load_best_model_at_end=True,  # whether to load the best model (in terms of loss) at the end of training
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=datasets["train"],
    eval_dataset=datasets["test"],
    callbacks=[
        EarlyStoppingCallback(
            early_stopping_patience=5, early_stopping_threshold=0.0001
        ),
        PenaltyStatsCallback(tokenizer),
    ],
)

trainer.train()

model.save_pretrained(OUT_DIR)
