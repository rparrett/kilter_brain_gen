import sys
import pprint
from pathlib import Path

from cfg import TRAINING_DIR, get_next_subdir_name
from data import load_training_datasets, preprocess_datasets
from transformers import (
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    EarlyStoppingCallback,
    GPT2Config,
    GPT2LMHeadModel,
    Trainer,
    TrainingArguments,
)

run_name = sys.argv[1] if len(sys.argv) > 1 else get_next_subdir_name()

CHECKPOINT_DIR = f"{TRAINING_DIR}/{run_name}"
tokenizer_dir = (
    sys.argv[2] if len(sys.argv) > 2 else f"{TRAINING_DIR}/climb_clm_tokenizer"
)  # TODO: just re-train the tokenizer each time, it's fast

if not (Path(tokenizer_dir) / "tokenizer_config.json").exists():
    print(
        f"Expecting a trained tokenizer in {CHECKPOINT_DIR}. Try `just train_tokenizer {run_name}`."
    )
    exit(1)

# Make sure to re-train tokenizer if you change the dataset
tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)

print(f"Checkpoint dir: {CHECKPOINT_DIR}")
print(f"Tokenizer: {tokenizer_dir}")

datasets = load_training_datasets()

print(f"Train samples: {len(datasets['train'])} Test samples: {len(datasets['test'])}")
pprint.pprint(datasets["train"][0])

preprocess_datasets(datasets, tokenizer)

pprint.pprint(datasets["train"][0])

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

model_config = GPT2Config(
    **{
        "bos_token_id": tokenizer.bos_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "n_embd": 256,  # 768
        "n_head": 4,  # 12
        "n_layer": 4,  # 12
        "n_positions": 128,  # 1024
        "n_ctx": 128,
        "vocab_size": tokenizer.vocab_size,
    }
)
model = GPT2LMHeadModel(config=model_config)

print(
    f"Model parameter count: {model.num_parameters()}, vocab size: {model.config.vocab_size}"
)

training_args = TrainingArguments(
    output_dir=CHECKPOINT_DIR,
    evaluation_strategy="steps",  # evaluate each `logging_steps` steps
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
    callbacks=[
        EarlyStoppingCallback(
            early_stopping_patience=5, early_stopping_threshold=0.0001
        ),
    ],
)

trainer.train()

model.save_pretrained(CHECKPOINT_DIR)
