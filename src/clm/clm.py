import pprint

from datasets import Features, Value, load_dataset
from tokenizers import Regex, Tokenizer, models, pre_tokenizers
from tokenizers.processors import TemplateProcessing
from tokenizers.trainers import WordLevelTrainer
from transformers import (
    DataCollatorForLanguageModeling,
    GPT2Config,
    GPT2LMHeadModel,
    PreTrainedTokenizerFast,
    Trainer,
    TrainingArguments,
)

out_dir = "clm-model"

features = Features(
    {
        "frames": Value("string"),
        "display_difficulty": Value("float"),
        "quality_average": Value("float"),
        "angle": Value("string"),
    }
)

dataset = load_dataset(
    "csv", data_files="climbs.csv", delimiter=",", features=features, split="train"
)


def add_prefix(example):
    a = "unk" if example["angle"] is None else example["angle"]
    d = (
        "unk"
        if example["display_difficulty"] is None
        else str(round(example["display_difficulty"]))
    )
    example["frames"] = "a" + a + "d" + d + example["frames"]
    return example


dataset = dataset.map(add_prefix)

datasets = dataset.train_test_split()

pprint.pprint(datasets)
pprint.pprint(datasets["train"][0])

# Train Tokenizer

max_length = 48

special_tokens = {"bos": "<s>", "eos": "</s>", "unk": "<unk>", "pad": "<pad>"}

trainer = WordLevelTrainer(special_tokens=list(special_tokens.values()))

tokenizer = Tokenizer(models.WordLevel(unk_token=special_tokens["unk"]))
tokenizer.enable_padding(length=max_length, pad_token=special_tokens["pad"])
tokenizer.enable_truncation(max_length=max_length)
tokenizer.add_special_tokens(list(special_tokens.values()))
tokenizer.pre_tokenizer = pre_tokenizers.Split(
    Regex(r"([ad]\d+|p\d+r\d+)"), behavior="isolated"
)

bos_token_id = tokenizer.token_to_id(special_tokens["bos"])
eos_token_id = tokenizer.token_to_id(special_tokens["eos"])

tokenizer.post_processor = TemplateProcessing(
    single=special_tokens["bos"] + " $A " + special_tokens["eos"],
    special_tokens=[
        (special_tokens["eos"], eos_token_id),
        (special_tokens["bos"], bos_token_id),
    ],
)

batch_size = 1000


def batch_iterator():
    for i in range(0, len(datasets["train"]), batch_size):
        yield datasets["train"][i : i + batch_size]["frames"]


tokenizer.train_from_iterator(batch_iterator(), trainer=trainer)

pprint.pprint(tokenizer.get_vocab())
pprint.pprint(tokenizer.encode("p1596r15p1597r14").tokens)
pprint.pprint(tokenizer.encode("p1595r15p1596r12").tokens)
pprint.pprint(tokenizer.encode("a40d15p1595r15p1596r12").tokens)

# Train Model

tokenizer_pretrained = PreTrainedTokenizerFast(
    tokenizer_object=tokenizer,
    model_max_length=max_length,
    padding_side="right",
    truncation_side="right",
    unk_token=special_tokens["unk"],
    pad_token=special_tokens["pad"],
)

tokenizer_pretrained.save_pretrained(out_dir)

# Tokenize datasets

tokenized_train = datasets["train"].map(
    lambda examples: tokenizer_pretrained(examples["frames"]), batched=True
)
tokenized_test = datasets["test"].map(
    lambda examples: tokenizer_pretrained(examples["frames"]), batched=True
)

tokenized_train = tokenized_train.remove_columns(
    ["frames", "display_difficulty", "quality_average", "angle"]
)
tokenized_test = tokenized_test.remove_columns(
    ["frames", "display_difficulty", "quality_average", "angle"]
)

pprint.pprint(tokenized_train[0])

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer_pretrained, mlm=False, mlm_probability=0.15
)

next_power_of_2 = lambda n: 2 ** (n - 1).bit_length()

model_config = GPT2Config(
    **{
        #   "activation_function": "gelu_new",
        #   "attn_pdrop": 0.1,
        #   "bos_token_id": 50256,
        "bos_token_id": bos_token_id,
        #   "embd_pdrop": 0.1,
        #   "eos_token_id": 50256,
        "eos_token_id": eos_token_id,
        #   "initializer_range": 0.02,
        #   "layer_norm_epsilon": 1e-05,
        #   "model_type": "gpt2",
        #   "n_embd": 768,
        "n_embd": 192,
        #   "n_head": 12,
        "n_head": 3,
        #   "n_inner": null,
        #   "n_layer": 12,
        "n_layer": 3,
        #   "n_positions": 1024,
        "n_positions": 256,
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
        "vocab_size": next_power_of_2(tokenizer.get_vocab_size()),
    }
)
model = GPT2LMHeadModel(config=model_config)

print(model.num_parameters())

training_args = TrainingArguments(
    output_dir=out_dir,  # output directory to where save model checkpoint
    evaluation_strategy="steps",  # evaluate each `logging_steps` steps
    overwrite_output_dir=True,
    num_train_epochs=2,  # number of training epochs, feel free to tweak
    per_device_train_batch_size=8,  # the training batch size, put it as high as your GPU memory fits
    gradient_accumulation_steps=1,  # accumulating the gradients before updating the weights
    per_device_eval_batch_size=64,  # evaluation batch size
    logging_steps=200,  # evaluate, log and save model checkpoints every 1000 step
    save_steps=1000,
    report_to="tensorboard",
    remove_unused_columns=False,
    load_best_model_at_end=True,  # whether to load the best model (in terms of loss) at the end of training
    save_total_limit=3,  # whether you don't have much space so you let only 3 model weights saved in the disk
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
