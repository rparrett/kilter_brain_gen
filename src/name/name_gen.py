import pprint
import re

from transformers import GPT2TokenizerFast, GPT2LMHeadModel, GPT2Config, pipeline

checkpoint = "4000"

token_dir = "name-model"
model_dir = token_dir if checkpoint is None else token_dir + "/checkpoint-" + checkpoint

tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
config = GPT2Config.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained(model_dir)

tokenizer.pad_token=tokenizer.eos_token
tokenizer.pad_token_id=tokenizer.eos_token_id
tokenizer.padding_side = "left"

generator = pipeline(
    "text-generation", model=model, tokenizer=tokenizer, max_new_tokens=20
)

for _ in range(10):
    out = generator("", do_sample=True, num_beams=1)[0]
    print(out)