import argparse
import pprint
from datetime import datetime
from pathlib import Path

from data import load_training_datasets, preprocess_datasets
from train_tokenizer import train_tokenizer
from transformers import (
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    GPT2Config,
    GPT2LMHeadModel,
    Trainer,
    TrainingArguments,
)
from transformers.trainer_utils import SchedulerType

parser = argparse.ArgumentParser(description="Train CLM model")
parser.add_argument("--run-name", type=str, help="Name for this training run")
args = parser.parse_args()

# Create run name if not provided
if args.run_name is None:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    f"run_{timestamp}"

OUT_DIR = f"models/climb_clm/{args.run_name}"

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

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=False, mlm_probability=0.15
)

model_config = GPT2Config(
    **{
        "bos_token_id": tokenizer.bos_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "vocab_size": tokenizer.vocab_size,
        "n_embd": 256,  # 768
        "n_head": 4,  # 12
        "n_layer": 4,  # 12
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
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=16,  # put it as high as your GPU memory fits; "if gradient_accumulation_steps > 1, this is the micro-batch size" --NanoGPT
    gradient_accumulation_steps=1,  # "used to simulate larger batch sizes"
    per_device_eval_batch_size=16,
    logging_steps=100,  # evaluate and log model checkpoints every n steps
    save_steps=400,  # save model checkpoints every n steps
    learning_rate=1e-5,
    adam_beta1=0.9,
    adam_beta2=0.999,
    report_to="tensorboard",
    remove_unused_columns=False,
    greater_is_better=False,  # whether the best model is the one with the highest or lowest evaluation metric, e.g. loss vs accuracy
    metric_for_best_model="eval_loss",  # use eval_loss to compare models
    load_best_model_at_end=True,  # whether to load the best model (in terms of loss) at the end of training
    save_total_limit=3,  # save disk space
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=datasets["train"],
    eval_dataset=datasets["test"],
)

trainer.train()

model.save_pretrained(OUT_DIR)
