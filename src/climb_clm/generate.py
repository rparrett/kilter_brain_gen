import re
import sys
from pprint import pprint
from rich.console import Console

console = Console()
from transformers import AutoTokenizer, GPT2Config, GPT2LMHeadModel, pipeline

from cfg import get_latest_checkpoint_path
from data import Penalizer
from model import AltGenerator, get_pipeline

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

pipe = get_pipeline(model, tokenizer)
# generator = pipe
alt_gen = AltGenerator(model, tokenizer)
generator = alt_gen.generate

prompts = [
    "a20d20",  # v5 at 20 degrees
    "a40d15",  # v2 at 40 degrees
    "a10d10",  # v? at 10 degrees
]


def remove_and_get_non_pr(output):
    non_pr = []

    def remove_non_pr(match):
        non_pr.append(match.group(1))
        return ""

    output = re.sub(r"([^pr\d]\d+|aunk|dunk)", remove_non_pr, output)

    return (output, non_pr)


penalizer = Penalizer(tokenizer)
# console.print(penalizer.get_token_sets()) # debugging


def get_ad(non_pr):
    a = None
    d = None
    for thing in non_pr:
        if d is None and thing[0] == "d":
            d = thing
        if a is None and thing[0] == "a":
            a = thing
    return a, d


# Note: Repeated padding tokens are bad for penalty
# also:
#     'angle_tokens': [
#        ('a65', 1056),
#        ('<pad>', 3),

stats = []
n = 5
verbose = False
for pn, prompt in enumerate(prompts):
    gen_result = generator(prompt, num_return_sequences=n, do_sample=True, num_beams=1)
    if verbose:
        print(gen_result)

    pipe_result = pipe(
        prompt,
        num_return_sequences=n,
        do_sample=True,
        num_beams=1,
    )
    if verbose:
        print(pipe_result)

    for n, result in enumerate(gen_result):
        text = result["generated_text"]

        (out, non_pr) = remove_and_get_non_pr(text)
        out = out.replace(" ", "")

        a, d = get_ad(non_pr)

        name = ".".join([checkpoint_path, str(pn), str(n), a, d])
        print()
        console.print("out len:", len(out))
        full_name = name + "," + out
        print(full_name)

        if "tokens" in result:
            toks = [
                x
                for x in result["tokens"].flatten().tolist()
                if x != tokenizer.pad_token_id
            ]
            penalties = penalizer.compute_penalties(toks)
            penalty_score = penalizer.compute_penalty_score(penalties)

            if verbose:
                print()
                print("tokens:")
                print(toks)
                print("penalty score:", sum(penalties.values()), penalty_score)
                pprint(
                    list(
                        f"{k}: {v}"
                        for k, v in sorted(penalties.items(), key=lambda x: x[1])
                    )
                )
            stats.append(
                (
                    prompt,
                    len(toks),
                    penalty_score,
                    full_name,
                    ", ".join(k for k in penalties.keys()),
                )
            )

print("Stats:")
# console.print(stats)
print()
for stat in sorted(stats, key=lambda x: x[2], reverse=False):
    console.print(
        stat[:3], "[cyan]" + stat[3] + "[/cyan]", "[yellow]" + stat[4] + "[/yellow]"
    )
