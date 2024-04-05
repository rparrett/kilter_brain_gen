import pprint
import re
import time

from transformers import GPT2TokenizerFast, GPT2LMHeadModel, GPT2Config, pipeline

checkpoint = "5000"

token_dir = "name-model"
model_dir = token_dir if checkpoint is None else token_dir + "/checkpoint-" + checkpoint

tokenizer = GPT2TokenizerFast.from_pretrained('gpt2', bos_token='<|startoftext|>', eos_token='<|endoftext|>', pad_token='<|pad|>')
config = GPT2Config.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained(model_dir)

tokenizer.pad_token=tokenizer.eos_token
tokenizer.pad_token_id=tokenizer.eos_token_id
tokenizer.padding_side = "left"

generator = pipeline(
    "text-generation", model=model, tokenizer=tokenizer, max_new_tokens=20
)

params = [
    {
        "do_sample": True,
    },
    {
        "do_sample": True,
        "top_k": 25,
        "temperature": 0.2
    },
    {
        "do_sample": True,
        "top_k": 50,
        "temperature": 0.8
    },
    {
        "do_sample": True,
        "top_k": 50,
        "temperature": 0.2
    },
    {
        "do_sample": True,
        "top_k": 50,
        "temperature": 0.8
    },
    {
        "do_sample": True,
        "num_beams": 1
    },
    {
        "do_sample": True,
        "num_beams": 4
    },
    {
        "do_sample": False,
        "num_beams": 1
    },
]

for p in params:
    pprint.pprint(p);
    print()

    start = time.time()
    for _ in range(5):
        out = generator("<|startoftext|>", **p)[0]['generated_text']
        print("  " + out)
    end = time.time()
    print("%.2fms" % ((end - start) * 1000))
    print()