import random

from transformers import AutoTokenizer, BertForMaskedLM, pipeline

tokenizer = AutoTokenizer.from_pretrained("models/climb_mlm")
model = BertForMaskedLM.from_pretrained("models/climb_mlm")

for n in range(10):
    start = "p1201r12p1202r12"
    num = random.randint(10, 14)

    inputs = tokenizer(start + ("[MASK]" * num), return_tensors="pt")

    unmasker = pipeline("fill-mask", model=model, tokenizer=tokenizer)

    for i in range(num):
        r = random.choice(unmasker(start + "[MASK]"))["token_str"]
        start += r

    print(start)
    print()
