import re
import sys
from pathlib import Path

from generator import generate_climb
from data import DIFFICULTY
from transformers import AutoTokenizer, GPT2Config, GPT2LMHeadModel, pipeline

# Import from the same directory
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))
from data import find_latest_checkpoint

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

device = next(model.parameters()).device

prompts = [
    "a20d20",  # v5 at 20 degrees
    "a40d15",  # v2 at 40 degrees
    "a20d20p1143r12p1162r12p1394r14",  # partial climb
    "",  # empty
]

for prompt_i, prompt in enumerate(prompts):
    for i in range(5):
        climb = generate_climb(tokenizer, model, prompt)

        difficulty = DIFFICULTY.get(climb["difficulty"], "V?")

        name = " ".join(
            [model_dir, f"p{str(prompt_i)}", f"#{str(i+1)}", f"{climb["angle"]}°", difficulty]
        )

        print(name + "," + climb["frames"])
