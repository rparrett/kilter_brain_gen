import pprint
from pathlib import Path

from clm_data import load_training_datasets, preprocess_datasets
from clm_gen import Penalizer, alt_generate, get_named_route
from clm_trainer import CustomTrainer
from transformers import (
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    GPT2Config,
    GPT2LMHeadModel,
    TrainingArguments,
    EarlyStoppingCallback,
    TrainerCallback,
)

from transformers.trainer_utils import SchedulerType

OUT_DIR = "clm-model"


class SampleGenerationCallback(TrainerCallback):
    def __init__(self, tokenizer, model, num_samples=1):
        self.tokenizer = tokenizer
        self.model = model
        self.num_samples = num_samples
        self.penalizer = Penalizer(tokenizer)

    def on_evaluate(self, args, state, control, **kwargs):
        if state.global_step % 400 == 0:
            print("Generating samples...")
            self.generate_samples(state)

    def generate_samples(self, state):
        for _ in range(self.num_samples):
            penalties, generated_texts = alt_generate(
                "a20d15",
                self.model,
                self.tokenizer,
                self.penalizer,
                do_sample=True,
                num_beams=1,
                num_return_sequences=3,
            )
            for i, generated_text in enumerate(generated_texts):
                penalty = penalties[i]
                name, route = get_named_route(
                    generated_text, "train_epoch_" + str(state.epoch), 0, i
                )
                print(f"{penalty:.2f}\t{name}:\t{route}")


datasets = load_training_datasets()

print(f"Train: {len(datasets['train'])} Test: {len(datasets['test'])}")
pprint.pprint(datasets["train"][0])

if not (Path(OUT_DIR) / "tokenizer_config.json").exists():
    print(f"Expecting a trained tokenizer @ {OUT_DIR}. Try clm_tok.py.")
    exit(1)

tokenizer = AutoTokenizer.from_pretrained(OUT_DIR)

preprocess_datasets(datasets, tokenizer)

pprint.pprint(datasets["train"][0])

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

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
        "n_embd": 128,
        #   "n_head": 12,
        "n_head": 2,
        #   "n_inner": null,
        #   "n_layer": 12,
        "n_layer": 2,
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
model = GPT2LMHeadModel(config=model_config)

print(f"Model parameters: {model.num_parameters()}")

# When scaling the batch size by κ, scale the learning rate by sqrt(κ)
# Also, change the other hyperparameters, setting:
# β1=1−κ(1−β1)
# β2=1−κ(1−β2)
# ϵ=ϵ/sqrt(κ)

training_args = TrainingArguments(
    output_dir=OUT_DIR,  # output directory to where save model checkpoint
    evaluation_strategy="steps",  # evaluate each `logging_steps` steps
    overwrite_output_dir=True,
    num_train_epochs=7,  # number of training epochs, feel free to tweak
    per_device_train_batch_size=64,  # the training batch size, put it as high as your GPU memory fits; "if gradient_accumulation_steps > 1, this is the micro-batch size" --NanoGPT
    per_device_eval_batch_size=64,  # evaluation batch size
    gradient_accumulation_steps=1,  # default
    logging_steps=100,  # evaluate, log and save model checkpoints every n steps
    save_steps=200,
    learning_rate=5e-6,  # 1e-5,  # default: 5e-5 (note: larger batch size -> smaller learning rate)
    lr_scheduler_type=SchedulerType.LINEAR,  # default
    adam_beta1=0.9,  # default
    adam_beta2=0.995,  # 0.999,  # default
    weight_decay=0.00,  # default
    report_to="tensorboard",
    remove_unused_columns=False,
    load_best_model_at_end=True,  # whether to load the best model at the end of training
    save_total_limit=3,  # let only 3 model weights saved in the disk
    metric_for_best_model="eval_loss",  # use eval_loss to compare models
    greater_is_better=False,  # lower eval_loss is better
)

trainer = CustomTrainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=datasets["train"],
    eval_dataset=datasets["test"],
    tokenizer=tokenizer,
    callbacks=[
        EarlyStoppingCallback(
            early_stopping_patience=3, early_stopping_threshold=0.00001
        ),
        SampleGenerationCallback(tokenizer, model, num_samples=1),
    ],
)

trainer.train()

model.save_pretrained(OUT_DIR)


def nearest_multiple_of_64(n):
    return 64 * ((n + 63) // 64)
