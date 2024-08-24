from transformers.generation.configuration_utils import GenerationConfig
from transformers import pipeline


class AltGenerator:
    def __init__(self, model, tokenizer, gen_args: dict | None = None):
        self.model = model
        self.tokenizer = tokenizer
        base_gen_args = dict(
            max_new_tokens=20,
            min_new_tokens=5,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        if gen_args:
            base_gen_args.update(**gen_args)
        self.gen_args = base_gen_args

    def generate(self, prompt: str, **kwargs):
        gen_args = {**self.gen_args, **kwargs}
        gen_config = GenerationConfig(**gen_args)
        # print(gen_config)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        outputs = self.model.generate(**inputs, generation_config=gen_config)
        decoded = self.tokenizer.batch_decode(
            outputs, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )
        return [
            {"generated_text": text, "tokens": output}
            for text, output in zip(decoded, outputs)
        ]


def get_pipeline(model, tokenizer):
    return pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=20,
        min_new_tokens=5,
    )
