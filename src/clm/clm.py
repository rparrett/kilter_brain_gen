from pathlib import Path

from rich.console import Console
from rich.pretty import pprint
from transformers import (AutoTokenizer, DataCollatorForLanguageModeling,
                          GPT2Config, GPT2LMHeadModel, Trainer,
                          TrainingArguments)
from transformers.trainer_utils import SchedulerType

from clm_data import load_training_datasets, preprocess_datasets

console = Console()

OUTPUT_DIR = "clm-model"
datasets = load_training_datasets()
console.print(f"Train: {len(datasets['train'])} Test: {len(datasets['test'])}")
pprint(datasets["train"][0])

if not (Path(OUTPUT_DIR) / "tokenizer_config.json").exists():
    console.print(f"Expecting a trained tokenizer @ {OUTPUT_DIR}. Try clm_tok.py.")
    exit(1)

tokenizer = AutoTokenizer.from_pretrained(OUTPUT_DIR)


def nearest_multiple_of_64(n):
    return 64 * ((n + 63) // 64)


model_config = GPT2Config(
    **{
        #   "activation_function": "gelu_new",
        #   "attn_pdrop": 0.1,
        #   "bos_token_id": 50256,
        "bos_token_id": tokenizer.bos_token_id,
        #   "embd_pdrop": 0.1,
        #   "eos_token_id": 50256,
        "eos_token_id": tokenizer.eos_token_id,
        #   "initializer_range": 0.02,
        #   "layer_norm_epsilon": 1e-05,
        #   "model_type": "gpt2",
        #   "n_embd": 768,
        "n_embd": 384,
        #   "n_head": 12,
        "n_head": 6,
        #   "n_inner": null,
        #   "n_layer": 12,
        "n_layer": 6,
        #   "n_positions": 1024,
        "n_positions": 128,
        #   "reorder_and_upcast_attn": false,
        #   "resid_pdrop": 0.1,
        #   "scale_attn_by_inverse_layer_idx": false,
        #   "scale_attn_weights": true,
        #   "summary_activation": null,
        #   "summary_first_dropout": 0.1,
        #   "summary_proj_to_labels": true,
        #   "summary_type": "cls_index",
        #   "summary_use_proj": true,
        #   "transformers_version": "4.39.1",
        #   "use_cache": true,
        #   "vocab_size": 50257
        "vocab_size": nearest_multiple_of_64(tokenizer.vocab_size),
    }
)
pprint(model_config)

model = GPT2LMHeadModel(config=model_config)
# Notice: resize_token_embeddings expect to receive the full size of the new vocabulary,
# i.e., the length of the tokenizer.
model.resize_token_embeddings(len(tokenizer))
set_vocab_size = max(nearest_multiple_of_64(tokenizer.vocab_size), len(tokenizer))
console.print(f"Vocab size: {set_vocab_size}")
assert tokenizer.cls_token == "<cls>"

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,  # output directory to where save model checkpoint
    evaluation_strategy="steps",  # evaluate each `logging_steps` steps
    overwrite_output_dir=True,
    num_train_epochs=5,  # number of training epochs, feel free to tweak
    per_device_train_batch_size=16,  # the training batch size, put it as high as your GPU memory fits; "if gradient_accumulation_steps > 1, this is the micro-batch size" --NanoGPT
    gradient_accumulation_steps=8,  # "used to simulate larger batch sizes" -- NanoGPT
    # gradient_accumulation_steps=1,  # accumulating the gradients before updating the weights
    per_device_eval_batch_size=16,  # evaluation batch size
    logging_steps=200,  # evaluate, log and save model checkpoints every 200 step
    save_steps=400,
    # learning_rate (`float`, *optional*, defaults to 5e-5):
    learning_rate=6e-4,  # same as NanoGPT
    lr_scheduler_type=SchedulerType.LINEAR.value,  # NanoGPT uses Cosine with warmup, which might not be possible here
    # adam_beta1 (`float`, *optional*, defaults to 0.9):
    adam_beta1=0.9,
    # adam_beta2 (`float`, *optional*, defaults to 0.999):
    adam_beta2=0.95,  # same as NanoGPT
    # weight_decay (`float`, *optional*, defaults to 0):
    weight_decay=0.01,  # same as NanoGPT
    report_to="tensorboard",
    remove_unused_columns=False,
    load_best_model_at_end=True,  # whether to load the best model (in terms of loss) at the end of training
    save_total_limit=3,  # whether you don't have much space so you let only 3 model weights saved in the disk
)

datasets = preprocess_datasets(datasets, tokenizer)

pprint(datasets["train"][0])
pprint(datasets["test"][0])

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=False, mlm_probability=0.15
    ),
    train_dataset=datasets["train"],
    eval_dataset=datasets["test"],
)

resume_from_checkpoint = True  # None
trainer.train(resume_from_checkpoint=resume_from_checkpoint)

# error sometimes when training is complete.
# 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████| 
# 898/898 [05:00<00:00,  4.55it/s]There were missing keys in the checkpoint model loaded: ['lm_head.weight'].

