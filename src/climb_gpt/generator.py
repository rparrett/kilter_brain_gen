import re


def tokens_to_climb(tokenizer, token_ids):
    """Process generated tokens and extract frames string, angle, and difficulty."""
    decoded = "".join(tokenizer.batch_decode(token_ids, skip_special_tokens=True))

    # Extract frames
    frames = re.findall(r"p\d+r\d+", decoded)
    frames_str = "".join(frames)

    # Extract angle
    angle_match = re.search(r"a(\d+|unk)", decoded)
    angle = angle_match.group(1) if angle_match else "unk"

    # Extract difficulty
    difficulty_match = re.search(r"d(\d+|unk)", decoded)
    difficulty = difficulty_match.group(1) if difficulty_match else "unk"

    # TODO remove duplicates

    return {"frames": frames_str, "angle": angle, "difficulty": difficulty}


def generate_tokens(tokenizer, model, prompt):
    device = next(model.parameters()).device

    model_inputs = tokenizer(
        # explicitly add the bos token since we are using add_special_tokens=False
        [tokenizer.bos_token + prompt],
        return_tensors="pt",
        # don't add eos token to the prompt, because the model will not generate anything after that token
        add_special_tokens=False,
    ).to(device)

    output = model.generate(
        **model_inputs,
        num_beams=1,
        do_sample=True,
        # transformers is emitting a warning saying that it is doing this, so we will just do it
        pad_token_id=tokenizer.eos_token_id,
    )

    tokens = [x for x in output.flatten().tolist() if x != tokenizer.pad_token_id]

    return tokens
