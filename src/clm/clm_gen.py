import pprint

from transformers import AutoTokenizer, GPT2LMHeadModel, GPT2Config, pipeline

model_dir = "clm-model"

tokenizer = AutoTokenizer.from_pretrained(model_dir)
config = GPT2Config.from_pretrained(model_dir)
model = GPT2LMHeadModel.from_pretrained(model_dir)

generator = pipeline("text-generation", model=model, tokenizer=tokenizer)

for n in range(10):
    # Using p1128r12p1462r15p1458r15 seems to generate garbage. It seems like
    # it only works well for initial inputs that exist in the database...

    out = generator("p1201r12p1202r12", do_sample=True, num_beams=1)[0]
    out = out['generated_text'].replace(" ", "")

    print(out)
    print()
