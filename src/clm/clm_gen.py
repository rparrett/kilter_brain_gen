import pprint
import re

from transformers import AutoTokenizer, GPT2Config, GPT2LMHeadModel, pipeline

checkpoint = None  # "17900"

token_dir = "clm-model"
model_dir = token_dir if checkpoint is None else token_dir + "/checkpoint-" + checkpoint

tokenizer = AutoTokenizer.from_pretrained(token_dir)
config = GPT2Config.from_pretrained(model_dir)
model = GPT2LMHeadModel.from_pretrained(model_dir)

generator = pipeline(
    "text-generation", model=model, tokenizer=tokenizer, max_new_tokens=20
)

prompts = [
    "a20 d20",  # v5 at 20 degrees
    "a40 d15",  # v2 at 40 degrees
]


def remove_non_pr(match):
    non_pr.append(match.group(1))
    return ""


for pn, prompt in enumerate(prompts):
    for n in range(5):
        out = generator(prompt, do_sample=True, num_beams=1)[0]

        non_pr = []
        out = re.sub(r"([^pr\d]\d+)", remove_non_pr, out["generated_text"])

        out = out.replace(" ", "")

        out = (
            model_dir
            + "."
            + str(pn)
            + "."
            + str(n)
            + "."
            + ".".join(non_pr)
            + ","
            + out
        )

        print(out)
