import sys
from pathlib import Path

from generator import generate_tokens, tokens_to_climb
from data import DIFFICULTY
from penalizer import Penalizer
from transformers import AutoTokenizer, GPT2Config, GPT2LMHeadModel

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils import find_latest_checkpoint  # noqa: E402

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

penalizer = Penalizer(tokenizer)

device = next(model.parameters()).device

prompts = [
    ("a20d20", "v5 at 20 degrees"),
    ("a40d15", "v2 at 40 degrees"),
    ("a20d20p1143r12p1162r12p1394r14", "partial climb (v5 at 20 degrees)"),
    ("", "empty"),
]

for prompt_i, (prompt, prompt_label) in enumerate(prompts):
    all_penalties = {}
    any_penalty_count = 0

    print(f"Prompt: {prompt} ({prompt_label})")

    NUM = 20

    for i in range(NUM):
        tokens = generate_tokens(tokenizer, model, prompt)

        climb_penalties = penalizer.compute_penalties(tokens)

        climb = tokens_to_climb(tokenizer, tokens)

        difficulty = DIFFICULTY.get(climb["difficulty"], "V?")

        name = " ".join(
            [
                model_dir,
                f"p{str(prompt_i)}",
                f"#{str(i + 1)}",
                f"{climb['angle']}°",
                difficulty,
            ]
        )

        print(name + "," + climb["frames"])

        if climb_penalties:
            any_penalty_count += 1

        # Count climbs affected by each penalty type
        for penalty_name in climb_penalties.keys():
            all_penalties[penalty_name] = all_penalties.get(penalty_name, 0) + 1

    print()
    penalty_free_pct = (any_penalty_count / NUM) * 100
    print(f"any_penalty: {any_penalty_count}/{NUM} ({penalty_free_pct:.1f}%)")
    for penalty_name, climb_count in all_penalties.items():
        print(f"{penalty_name}: {climb_count}/{NUM}")
    print()
