import pprint

from tokenizers import Tokenizer, Regex, models, pre_tokenizers
from tokenizers.trainers import WordLevelTrainer
from tokenizers.processors import TemplateProcessing
from datasets import Features, Value, load_dataset
from transformers import (
    TrainingArguments,
    PreTrainedTokenizerFast,
    BertConfig,
    BertForMaskedLM,
    DataCollatorForLanguageModeling,
    Trainer,
)

out_dir = "mlm-model"

features = Features(
    {
        "frames": Value("string"),
        "display_difficulty": Value("float"),
        "quality_average": Value("float"),
    }
)

dataset = load_dataset(
    "csv", data_files="climbs.csv", delimiter=",", features=features, split="train"
)

datasets = dataset.train_test_split()

pprint.pprint(datasets)
pprint.pprint(datasets["train"][0])

# Train Tokenizer

max_length = 48

special_tokens = ["[UNK]", "[PAD]", "[CLS]", "[SEP]", "[MASK]"]
trainer = WordLevelTrainer(special_tokens=special_tokens)

tokenizer = Tokenizer(models.WordLevel(unk_token="[UNK]"))
tokenizer.enable_padding(length=max_length)
tokenizer.enable_truncation(max_length=max_length)
tokenizer.add_special_tokens(special_tokens)
tokenizer.pre_tokenizer = pre_tokenizers.Split(Regex(r"p\d+r\d+"), behavior="isolated")

batch_size = 1000


def batch_iterator():
    for i in range(0, len(datasets["train"]), batch_size):
        yield datasets["train"][i : i + batch_size]["frames"]


tokenizer.train_from_iterator(batch_iterator(), trainer=trainer)

cls_token_id = tokenizer.token_to_id("[CLS]")
sep_token_id = tokenizer.token_to_id("[SEP]")
tokenizer.post_processor = TemplateProcessing(
    single="[CLS]:0 $A:0 [SEP]:0",
    pair="[CLS]:0 $A:0 [SEP]:0 $B:1 [SEP]:1",
    special_tokens=[
        ("[CLS]", cls_token_id),
        ("[SEP]", sep_token_id),
    ],
)

pprint.pprint(tokenizer.get_vocab())
pprint.pprint(tokenizer.encode("p1596r15p1597r14").tokens)
pprint.pprint(tokenizer.encode("p1595r15p1596r12").tokens)

# Train Model

tokenizer_pretrained = PreTrainedTokenizerFast(
    tokenizer_object=tokenizer,
    model_max_length=max_length,
    padding_side="right",
    truncation_side="right",
    unk_token="[UNK]",
    sep_token="[SEP]",
    pad_token="[PAD]",
    cls_token="[CLS]",
    mask_token="[MASK]",
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
    ["frames", "display_difficulty", "quality_average"]
)
tokenized_test = tokenized_test.remove_columns(
    ["frames", "display_difficulty", "quality_average"]
)

pprint.pprint(tokenized_train[0])

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer_pretrained, mlm=True, mlm_probability=0.15
)

next_power_of_2 = lambda n: 2 ** (n - 1).bit_length()

model_config = BertConfig(
    hidden_size=384,  # 786 default
    intermediate_size=1536,  # 3072 default
    max_position_embeddings=256,  # 512 default
    num_attention_heads=4,  # 12 default
    num_hidden_layers=4,  # default 6
    vocab_size=next_power_of_2(tokenizer_pretrained.vocab_size),
)
model = BertForMaskedLM(config=model_config)

print(model.num_parameters())

training_args = TrainingArguments(
    output_dir=out_dir,  # output directory to where save model checkpoint
    evaluation_strategy="steps",  # evaluate each `logging_steps` steps
    overwrite_output_dir=True,
    num_train_epochs=2,  # number of training epochs, feel free to tweak
    per_device_train_batch_size=8,  # the training batch size, put it as high as your GPU memory fits
    gradient_accumulation_steps=8,  # accumulating the gradients before updating the weights
    per_device_eval_batch_size=64,  # evaluation batch size
    logging_steps=50,  # evaluate, log and save model checkpoints every 1000 step
    save_steps=100,
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
