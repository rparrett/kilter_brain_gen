import time
import sys
from pathlib import Path

from rich.console import Console
from rich.pretty import pprint
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils import find_latest_checkpoint  # noqa: E402

console = Console()

base_dir = "models/name"

checkpoint_path = None
if len(sys.argv) > 1:
    checkpoint_path = sys.argv[1]

if checkpoint_path is None:
    # Find the most recent checkpoint automatically
    latest_checkpoint_dir = find_latest_checkpoint(base_dir)
    if latest_checkpoint_dir:
        checkpoint_path = str(latest_checkpoint_dir)
        print(f"Using latest checkpoint: {checkpoint_path}")
    else:
        checkpoint_path = base_dir
        print(f"No checkpoints found, using base model: {checkpoint_path}")
else:
    print(f"Using specified checkpoint: {checkpoint_path}")

model_dir = checkpoint_path

tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained(model_dir)

params = [
    {"do_sample": True, "max_new_tokens": 10},
    {"do_sample": True, "top_k": 25, "temperature": 0.8, "max_new_tokens": 10},
]

for p in params:
    pprint(p)
    print()

    start = time.time()
    for _ in range(5):
        inputs = tokenizer(tokenizer.bos_token, return_tensors="pt")

        output_ids = model.generate(
            inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            pad_token_id=tokenizer.eos_token_id,
            **p,
        )

        output = tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()

        print(output)

    end = time.time()
    print()
    console.print("%.2fms" % ((end - start) * 1000), style="cyan")
    print()
