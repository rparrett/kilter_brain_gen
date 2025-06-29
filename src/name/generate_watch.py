import time

from rich.console import Console
from rich.pretty import pprint
from transformers import GPT2Config, GPT2LMHeadModel, GPT2TokenizerFast
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
            tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
            tokenizer.pad_token = tokenizer.eos_token

            params = [
                {
                    "do_sample": True,
                    "max_new_tokens": 10,
                },
                {
                    "do_sample": True,
                    "top_k": 25,
                    "temperature": 0.8,
                    "max_new_tokens": 10,
                },
                # {"do_sample": True, "top_k": 50, "temperature": 0.8, "max_new_tokens": 20},
                # {"do_sample": True, "num_beams": 1, "max_new_tokens": 20},
                # {"do_sample": True, "num_beams": 4, "max_new_tokens": 20},
            ]

            for p in params:
                pprint(p)
                print()

                start = time.time()
                for i in range(5):
                    inputs = tokenizer(tokenizer.bos_token, return_tensors="pt")

                    output_ids = model.generate(
                        inputs["input_ids"],
                        attention_mask=inputs["attention_mask"],
                        pad_token_id=tokenizer.eos_token_id,
                        **p,
                    )

                    result = tokenizer.decode(
                        output_ids[0], skip_special_tokens=True
                    ).strip()
                    print(f"{i + 1}: {result}")
                end = time.time()
                print()
                console.print("%.2fms" % ((end - start) * 1000), style="cyan")
                print()


model_dir = "models/name"

tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token
config = GPT2Config.from_pretrained("gpt2")

observer = Observer()
handler = Handler()
observer.schedule(handler, model_dir, recursive=True)
observer.start()

try:
    while True:
        time.sleep(5)
except KeyboardInterrupt:
    observer.stop()
    print("Observer Stopped")

observer.join()
