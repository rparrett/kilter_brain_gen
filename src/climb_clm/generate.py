import re
import sys
from pathlib import Path

from transformers import AutoTokenizer, GPT2Config, GPT2LMHeadModel, pipeline


def find_latest_checkpoint(base_dir):
    """Find the most recent checkpoint directory in the given base directory."""
    base_path = Path(base_dir)

    # Find all checkpoint directories recursively
    checkpoint_dirs = []
    for checkpoint_dir in base_path.rglob("checkpoint-*"):
        if checkpoint_dir.is_dir():
            checkpoint_dirs.append(checkpoint_dir)

    if not checkpoint_dirs:
        return None

    # Sort by modification time and return the most recent
    latest = max(checkpoint_dirs, key=lambda p: p.stat().st_mtime)
    return latest


base_dir = "models/climb_clm"

checkpoint_path = None
if len(sys.argv) > 1:
    checkpoint_path = sys.argv[1]

if checkpoint_path is None:
    # Find the most recent checkpoint automatically
    latest_checkpoint_dir = find_latest_checkpoint(base_dir)
    if latest_checkpoint_dir:
        model_dir = str(latest_checkpoint_dir)
        token_dir = str(latest_checkpoint_dir.parent)
        print(f"Using latest checkpoint: {model_dir}")
        print(f"Tokenizer: {token_dir}")
    else:
        # Fallback to base directory
        token_dir = base_dir
        model_dir = base_dir
        print("No checkpoints found, using base model directory")
else:
    # Use specified checkpoint path (relative to models/climb_clm/)
    base_path = Path(base_dir)
    model_dir = str(base_path / checkpoint_path)
    token_dir = str((base_path / checkpoint_path).parent)
    print(f"Using specified checkpoint: {model_dir}")

tokenizer = AutoTokenizer.from_pretrained(token_dir)
config = GPT2Config.from_pretrained(model_dir)
model = GPT2LMHeadModel.from_pretrained(model_dir)

generator = pipeline(
    "text-generation", model=model, tokenizer=tokenizer, max_new_tokens=20
)

prompts = [
    "a20d20",  # v5 at 20 degrees
    "a40d15",  # v2 at 40 degrees
    "<s>",     # empty prompt
    "a20d20p1143r12p1162r12p1394r14",  # partial climb
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
