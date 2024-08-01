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
    "a20d20",  # v5 at 20 degrees
    "a40d15",  # v2 at 40 degrees
]

def remove_and_get_non_pr(output):
    non_pr = []

    def remove_non_pr(match):
        non_pr.append(match.group(1))
        return ""

    output = re.sub(r"([^pr\d]\d+|aunk|dunk)", remove_non_pr, output)

    return (output, non_pr)

for pn, prompt in enumerate(prompts):
    for n in range(5):
        result = generator(prompt, do_sample=True, num_beams=1)[0]

        (out, non_pr) = remove_and_get_non_pr(result["generated_text"])
        out = out.replace(" ", "")

        a = None
        d = None
        for thing in non_pr:
            if d == None and thing[0] == "d":
                d = thing
            if a == None and thing[0] == "a":
                a = thing

        name = ".".join([model_dir, str(pn), str(n), a, d])

        print(name + "," + out)
