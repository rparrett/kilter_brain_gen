import time

from rich.console import Console
from rich.pretty import pprint
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

console = Console()

checkpoint = "19500"

token_dir = "models/name"
model_dir = token_dir if checkpoint is None else token_dir + "/checkpoint-" + checkpoint

tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained(model_dir)

params = [
    {"do_sample": True, "max_new_tokens": 10},
    {"do_sample": True, "top_k": 25, "temperature": 0.8, "max_new_tokens": 10},
]

for p in params:
    pprint(p)
    print()

    start = time.time()
    for _ in range(5):
        inputs = tokenizer(tokenizer.bos_token, return_tensors="pt")

        output_ids = model.generate(
            inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            pad_token_id=tokenizer.eos_token_id,
            **p,
        )

        output = tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()

        print(output)

    end = time.time()
    print()
    console.print("%.2fms" % ((end - start) * 1000), style="cyan")
    print()
