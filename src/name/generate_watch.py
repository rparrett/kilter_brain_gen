import pprint
import re
import time

from rich.console import Console
from rich.pretty import pprint
from transformers import GPT2Config, GPT2LMHeadModel, GPT2TokenizerFast, pipeline
from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

console = Console()


class Handler(FileSystemEventHandler):
    @staticmethod
    def on_any_event(event):
        if event.is_directory and event.event_type == "created":
            console.print(event.src_path, style="black on white")
            print()

            time.sleep(1)

            model = GPT2LMHeadModel.from_pretrained(event.src_path)
            tokenizer = GPT2TokenizerFast.from_pretrained(
                "gpt2", bos_token="<|startoftext|>", eos_token="<|endoftext|>", pad_token="<|pad|>"
            )
            model.resize_token_embeddings(len(tokenizer))

            generator = pipeline(
                "text-generation", model=model, tokenizer=tokenizer, max_new_tokens=20
            )

            params = [
                {
                    "do_sample": True,
                },
                {"do_sample": True, "top_k": 25, "temperature": 0.8},
                {"do_sample": True, "top_k": 50, "temperature": 0.8},
                {"do_sample": True, "num_beams": 1},
                {"do_sample": True, "num_beams": 4},
            ]

            for p in params:
                pprint(p)
                print()

                p["prefix"] = "<|startoftext|>"

                start = time.time()
                for _ in range(5):
                    out = generator("", **p)[0]["generated_text"]
                    print(out)
                end = time.time()
                print()
                console.print("%.2fms" % ((end - start) * 1000), style="cyan")
                print()


model_dir = "models/name"

tokenizer = GPT2TokenizerFast.from_pretrained(
    "gpt2", bos_token="<|startoftext|>", eos_token="<|endoftext|>", pad_token="<|pad|>"
)
config = GPT2Config.from_pretrained("gpt2")

observer = Observer()
handler = Handler()
observer.schedule(handler, model_dir, recursive=True)
observer.start()

try:
    while True:
        time.sleep(5)
except:
    observer.stop()
    print("Observer Stopped")

observer.join()
