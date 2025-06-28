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

device = next(model.parameters()).device

prompts = [
    "a20d20",  # v5 at 20 degrees
    "a40d15",  # v2 at 40 degrees
    "a20d20p1143r12p1162r12p1394r14",  # partial climb
    "",  # empty
]


def process_output(output):
    """Process generated text and extract frames, angle, and difficulty."""
    # Extract frames
    frames = re.findall(r"p\d+r\d+", output)
    frames_str = "".join(frames)

    # Extract angle
    angle_match = re.search(r"a(\d+|unk)", output)
    angle = angle_match.group(1) if angle_match else "unk"

    # Extract difficulty
    difficulty_match = re.search(r"d(\d+|unk)", output)
    difficulty = difficulty_match.group(1) if difficulty_match else "unk"

    return {"frames": frames_str, "angle": angle, "difficulty": difficulty}


for prompt_i, prompt in enumerate(prompts):
    for i in range(5):
        model_inputs = tokenizer(
            # explicitly add the bos token since we are using add_special_tokens=False
            [tokenizer.bos_token + prompt],
            return_tensors="pt",
            # don't add eos token to the prompt, because the model will not generate anything after that token
            add_special_tokens=False,
        ).to(device)

        generated_ids = model.generate(
            **model_inputs,
            num_beams=1,
            do_sample=True,
            # transformers is emitting a warning saying that it is doing this, so we will just do it
            pad_token_id=tokenizer.eos_token_id,
        )
        result = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

        parsed = process_output(result)

        name = ".".join(
            [model_dir, str(prompt_i), str(i), parsed["angle"], parsed["difficulty"]]
        )

        print(name + "," + parsed["frames"])
