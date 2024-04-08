import pprint
import re
import time
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from transformers import GPT2TokenizerFast, GPT2LMHeadModel, GPT2Config, pipeline

class Handler(FileSystemEventHandler):
    @staticmethod
    def on_any_event(event):
        if event.is_directory and event.event_type == 'created':
            print(event.src_path)
            print()

            time.sleep(1)

            model = GPT2LMHeadModel.from_pretrained(event.src_path)

            generator = pipeline(
                "text-generation", model=model, tokenizer=tokenizer, max_new_tokens=20
            )

            params = [
                {
                    "do_sample": True,
                },
                {
                    "do_sample": True,
                    "top_k": 25,
                    "temperature": 0.2
                },
                {
                    "do_sample": True,
                    "top_k": 50,
                    "temperature": 0.8
                },
                {
                    "do_sample": True,
                    "top_k": 25,
                    "temperature": 0.2
                },
                {
                    "do_sample": True,
                    "top_k": 50,
                    "temperature": 0.8
                },
                {
                    "do_sample": True,
                    "num_beams": 1
                },
                {
                    "do_sample": True,
                    "num_beams": 4
                },
            ]

            for p in params:
                pprint.pprint(p);
                print()

                start = time.time()
                for _ in range(5):
                    out = generator("<|startoftext|>", **p)[0]['generated_text']
                    print("  " + out)
                end = time.time()
                print("%.2fms" % ((end - start) * 1000))
                print()


model_dir = "name-model"

tokenizer = GPT2TokenizerFast.from_pretrained('gpt2', bos_token='<|startoftext|>', eos_token='<|endoftext|>', pad_token='<|pad|>')
config = GPT2Config.from_pretrained("gpt2")

observer = Observer()
handler = Handler()
observer.schedule(handler, model_dir, recursive = True)
observer.start()

try:
    while True:
        time.sleep(5)
except:
    observer.stop()
    print("Observer Stopped")

observer.join()

