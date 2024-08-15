import re
import sys

from transformers import AutoTokenizer, GPT2Config, GPT2LMHeadModel, pipeline
from transformers.generation.configuration_utils import GenerationConfig
from clm_data import Penalizer


def remove_and_get_non_pr(output):
    non_pr = []

    def remove_non_pr(match):
        non_pr.append(match.group(1))
        return ""

    output = re.sub(r"([^pr\d]\d+|aunk|dunk)", remove_non_pr, output)

    return (output, non_pr)


def alt_generate(prompt: str, model, tokenizer, penalizer, **kwargs):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    default_gen_args = dict(
        max_new_tokens=20,
        num_return_sequences=5,
        do_sample=True,
        top_k=50,
        temperature=0.8,
        pad_token_id=tokenizer.pad_token_id,
    )
    default_gen_args.update(**kwargs)
    gen_config = GenerationConfig(**default_gen_args)
    outputs = model.generate(**inputs, generation_config=gen_config)
    penalties = [penalizer.compute_penalties(output) for output in outputs.tolist()]
    return penalties, tokenizer.batch_decode(
        outputs, skip_special_tokens=True, clean_up_tokenization_spaces=True
    )


def get_ad(non_pr):
    a = "unk"
    d = "unk"
    for thing in non_pr:
        d = thing if d == "unk" and thing[0] == "d" else d
        a = thing if a == "unk" and thing[0] == "a" else a
    return a, d


def get_named_route(generated_text, model_dir="", pn=0, n=0):
    (out, non_pr) = remove_and_get_non_pr(generated_text)
    out = out.replace(" ", "")
    a, d = get_ad(non_pr)
    name = ".".join([model_dir, str(pn), str(n), a, d])
    return name, out


if __name__ == "__main__":
    checkpoint = None
    if len(sys.argv) > 1:
        checkpoint = sys.argv[1]

    token_dir = "clm-model"
    model_dir = (
        token_dir if checkpoint is None else token_dir + "/checkpoint-" + checkpoint
    )

    tokenizer = AutoTokenizer.from_pretrained(token_dir)
    config = GPT2Config.from_pretrained(model_dir)
    model = GPT2LMHeadModel.from_pretrained(model_dir)

    generator = pipeline(
        "text-generation", model=model, tokenizer=tokenizer, max_new_tokens=20
    )

    prompts = [
        "a20d20",  # v5 at 20 degrees
        "a40d15",  # v2 at 40 degrees
        "a40",  # ???
        "a20",  # ???
        "<s>",  # ???
    ]

    penalizer = Penalizer(tokenizer)

    final_results = []
    for pn, prompt in enumerate(prompts):
        labels = ["num_beams=1", "temp=0.7", "temp=0.3", "k=50, p=0.95"]
        results = []
        results.append(
            alt_generate(
                prompt, model, tokenizer, penalizer, do_sample=True, num_beams=1
            )
        )
        results.append(
            alt_generate(
                prompt, model, tokenizer, penalizer, do_sample=True, temperature=0.7
            )
        )
        results.append(
            alt_generate(
                prompt, model, tokenizer, penalizer, do_sample=True, temperature=0.3
            )
        )
        results.append(
            alt_generate(
                prompt,
                model,
                tokenizer,
                penalizer,
                do_sample=True,
                top_k=50,
                top_p=0.95,
            )
        )

        for label, result in zip(labels, results):
            penalties, generated_texts = result
            for n in range(len(generated_texts)):
                penalty = penalties[n]
                generated_text = generated_texts[n]
                name, out = get_named_route(generated_text, model_dir, pn, n)
                final_results.append([prompt, label, penalty, name, out])

    for prompt, label, penalty, name, out in sorted(final_results, key=lambda x: x[2]):
        print(f"{penalty:.2f}\t{label}\tprompt='{prompt}'\t{name}" + "," + out)
