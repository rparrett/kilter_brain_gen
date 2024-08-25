import re
import sys
from pprint import pprint
from rich.console import Console

from util import print_colored_role_names

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
    "<s>",  # empty prompt
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

import torch

token_sets = penalizer.get_token_sets()
start_tokens = [g[1] for g in token_sets["start_hold_tokens"]]
end_tokens = [g[1] for g in token_sets["end_hold_tokens"]]
angle_tokens = [g[1] for g in token_sets["angle_tokens"]]
diff_tokens = [g[1] for g in token_sets["difficulty_tokens"]]


def prefix_allowed_tokens_fn(batch_id, input_ids):
    has_a_start_hold = any(g in start_tokens for g in input_ids)
    has_an_end_hold = any(g in end_tokens for g in input_ids)

    has_angle_token = any(g in angle_tokens for g in input_ids)
    has_diff_token = any(g in diff_tokens for g in input_ids)

    filter_set = set()
    if not has_a_start_hold:
        return start_tokens  # aggressive
    if not has_an_end_hold:
        return end_tokens
    if not has_angle_token:
        return angle_tokens
    if not has_diff_token:
        return diff_tokens

    if has_a_start_hold:
        filter_set.update(start_tokens)
    if has_an_end_hold:
        filter_set.update(end_tokens)
    if has_angle_token:
        filter_set.update(angle_tokens)
    if has_diff_token:
        filter_set.update(diff_tokens)

    filter_set.update(
        set([t for t in input_ids if t not in penalizer.expected_repeats])
    )

    return [tid for tid in range(tokenizer.vocab_size) if tid not in filter_set]


def get_ad(non_pr):
    a = "unk"
    d = "unk"
    for thing in non_pr:
        if d == "unk" and thing[0] == "d":
            d = thing
        if a == "unk" and thing[0] == "a":
            a = thing
    return a, d


# Note: Repeated padding tokens are bad for penalty
# also:
#     'angle_tokens': [
#        ('a65', 1056),
#        ('<pad>', 3),
# repeated:
# </s>
"""
(e.g. one for creative text generation with sampling, and one for summarization with beam search)
"""
stats = []
n = 3  # ValueError: Greedy methods without beam search do not support `num_return_sequences` different than 1 (got 4).
verbose = False
samplers_all = {
    "multinomial": dict(
        do_sample=True,
        num_beams=1,
    ),
    "multinomial_with_prefix_allow": dict(
        do_sample=True,
        num_beams=1,
        top_k=50,
        temperature=0.7,
        no_repeat_ngram_size=2,
        max_new_tokens=20,
        min_new_tokens=10,
        num_return_sequences=n,
        gen_direct_args=dict(prefix_allowed_tokens_fn=prefix_allowed_tokens_fn),
    ),
    "contrastive": dict(
        penalty_alpha=0.6, top_k=4, max_new_tokens=20, num_return_sequences=n
    ),
    "top_k_50": dict(do_sample=True, top_k=50, num_return_sequences=n),
    "top_p_092": dict(
        do_sample=True,
        top_p=0.92,
        top_k=0,
        num_return_sequences=n,
    ),
}
samplers = {
    "multinomial_with_prefix_allow": samplers_all["multinomial_with_prefix_allow"]
}

for pn, prompt in enumerate(prompts):
    for sampler_name, kwargs in samplers.items():
        gen_result = generator(
            prompt,
            **kwargs,
        )
        if verbose:
            print(gen_result)

        # pipe_result = pipe(
        #    prompt,
        #    num_return_sequences=n,
        #    do_sample=True,
        #    num_beams=1,
        # )

        # if verbose:
        #    print(pipe_result)

        for n, result in enumerate(gen_result):
            text = result["generated_text"]

            (out, non_pr) = remove_and_get_non_pr(text)
            out = out.replace(" ", "")
            a, d = get_ad(non_pr)
            name = ".".join([checkpoint_path, str(pn), str(n), a, d])

            full_name = name + "," + out
            # print()
            # console.print("out len:", len(out))
            # print(full_name)

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
                        sampler_name,
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
        stat[:4], "[cyan]" + stat[4] + "[/cyan]", "\n[yellow]" + stat[5] + "[/yellow]\n"
    )
print_colored_role_names()
