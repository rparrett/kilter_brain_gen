import pprint

from transformers import AutoTokenizer, GPT2LMHeadModel, GPT2Config, pipeline

model_dir = "clm-model"
checkpoint = "/checkpoint-6300"

tokenizer = AutoTokenizer.from_pretrained(model_dir)
config = GPT2Config.from_pretrained(model_dir + checkpoint)
model = GPT2LMHeadModel.from_pretrained(model_dir + checkpoint)

generator = pipeline(
    "text-generation", model=model, tokenizer=tokenizer, max_new_tokens=20
)

prompts = [
    "p1201r12p1202r12",
    # This one generates complete nonsense. Just plasters a bunch of footholds
    # on. It seems like it only works well for initial inputs that exist in the
    # database...
    "p1128r12p1462r15p1458r15",
    "p1383r14"
]

for prompt in prompts:
    print("> " + prompt)

    for _n in range(5):
        out = generator(prompt, do_sample=True, num_beams=1)[0]
        out = out["generated_text"].replace(" ", "")

        print(out)
