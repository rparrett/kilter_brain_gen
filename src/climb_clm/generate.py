import re
import sys
from pprint import pprint

from transformers import AutoTokenizer, GPT2Config, GPT2LMHeadModel, pipeline

from cfg import get_latest_checkpoint_path
from data import Penalizer

checkpoint_path = (
    sys.argv[1] if len(sys.argv) > 1 else get_latest_checkpoint_path().as_posix()
)

TRAINING_DIR = "models"
tokenizer_dir = f"{TRAINING_DIR}/climb_clm_tokenizer"

print(f"Checkpoint dir: {checkpoint_path}")
print(f"Tokenizer: {tokenizer_dir}")
tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)
config = GPT2Config.from_pretrained(checkpoint_path)
model = GPT2LMHeadModel.from_pretrained(checkpoint_path)

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


penalizer = Penalizer(tokenizer)


def get_ad(non_pr):
    a = None
    d = None
    for thing in non_pr:
        if d is None and thing[0] == "d":
            d = thing
        if a is None and thing[0] == "a":
            a = thing
    return a, d


for pn, prompt in enumerate(prompts):
    for n in range(5):
        result = generator(prompt, do_sample=True, num_beams=1)[0]
        print(result["generated_text"])
        (out, non_pr) = remove_and_get_non_pr(result["generated_text"])
        out = out.replace(" ", "")

        toks = tokenizer(out)["input_ids"]
        print()
        print("tokens:")
        print(toks)
        print()
        penalties = penalizer.compute_penalties(toks)
        penalty_score = penalizer.compute_penalty_score(penalties)
        print("penalty score:", penalty_score)
        pprint(
            list(f"{k}: {v}" for k, v in sorted(penalties.items(), key=lambda x: x[1]))
        )

        a, d = get_ad(non_pr)

        name = ".".join([checkpoint_path, str(pn), str(n), a, d])
        print()
        print("out:")
        print(name + "," + out)
