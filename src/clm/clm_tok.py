import pprint

from clm_data import batch_iterator, load_training_datasets
from tokenizers import Regex, Tokenizer, models, pre_tokenizers
from tokenizers.processors import TemplateProcessing
from tokenizers.trainers import WordLevelTrainer
from transformers import PreTrainedTokenizerFast

OUT_DIR = "clm-model"

datasets = load_training_datasets()

max_length = 48

special_tokens = {
    "bos_token": "<s>",
    "eos_token": "</s>",
    "unk_token": "<unk>",
    "pad_token": "<pad>",
}

tokenizer = Tokenizer(models.WordLevel(unk_token=special_tokens["unk_token"]))
tokenizer.enable_padding(length=max_length, pad_token=special_tokens["pad_token"])
tokenizer.enable_truncation(max_length=max_length)
added = tokenizer.add_special_tokens(list(special_tokens.values()))

print(f"Added {added} tokens to Tokenizer")

tokenizer.pre_tokenizer = pre_tokenizers.Split(
    Regex(r"([ad]\d+|p\d+r\d+)"), behavior="isolated"
)

bos_token_id = tokenizer.token_to_id(special_tokens["bos_token"])
eos_token_id = tokenizer.token_to_id(special_tokens["eos_token"])

tokenizer.post_processor = TemplateProcessing(
    single=special_tokens["bos_token"] + " $A " + special_tokens["eos_token"],
    special_tokens=[
        (special_tokens["eos_token"], eos_token_id),
        (special_tokens["bos_token"], bos_token_id),
    ],
)

print("Training tokenizer")

batch_size = 1000
trainer = WordLevelTrainer(special_tokens=list(special_tokens.values()))
tokenizer.train_from_iterator(batch_iterator(datasets, batch_size), trainer=trainer)

pprint.pprint(tokenizer.get_vocab())
pprint.pprint(tokenizer.encode("p1596r15p1597r14").tokens)
pprint.pprint(tokenizer.encode("p1595r15p1596r12").tokens)
pprint.pprint(tokenizer.encode("a40d15p1595r15p1596r12").tokens)

tokenizer_pretrained = PreTrainedTokenizerFast(
    tokenizer_object=tokenizer,
    model_max_length=max_length,
    padding_side="right",
    truncation_side="right",
)
pprint.pprint(special_tokens)
added = tokenizer_pretrained.add_special_tokens(special_tokens)
print(f"Added {added} special tokens to PreTrainedTokenizer")

print(f"Saving to {OUT_DIR}")

tokenizer_pretrained.save_pretrained(OUT_DIR)
