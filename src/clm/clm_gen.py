import pprint
import re

from transformers import AutoTokenizer, GPT2LMHeadModel, GPT2Config, pipeline

checkpoint = "17900"

token_dir = "clm-model"
model_dir = token_dir if checkpoint is None else token_dir + "/checkpoint-" + checkpoint

tokenizer = AutoTokenizer.from_pretrained(token_dir)
config = GPT2Config.from_pretrained(model_dir)
model = GPT2LMHeadModel.from_pretrained(model_dir)

generator = pipeline(
    "text-generation", model=model, tokenizer=tokenizer, max_new_tokens=20
)

prompts = [
    "a10 p1201r12p1202r12",
    "a40 p1201r12p1202r12",
    "a60 p1201r12p1202r12",
]

for (pn, prompt) in enumerate(prompts):
    for n in range(5):
        out = generator(prompt, do_sample=True, num_beams=1)[0]
        out = out["generated_text"].replace(" ", "")
        out = re.sub(r"^a(\d+)", "", out)
        out = model_dir + "." + str(pn) + "." + str(n) + "," + out

        print(out)
