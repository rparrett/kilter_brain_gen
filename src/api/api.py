from uuid import uuid4
from flask import Flask
from flask_cors import CORS, cross_origin
import pprint
import re

from transformers import AutoTokenizer, GPT2LMHeadModel, GPT2Config, pipeline

checkpoint = None  # "17900"
token_dir = "clm-model"
model_dir = token_dir if checkpoint is None else token_dir + "/checkpoint-" + checkpoint

tokenizer = AutoTokenizer.from_pretrained(token_dir)
config = GPT2Config.from_pretrained(model_dir)
model = GPT2LMHeadModel.from_pretrained(model_dir)

generator = pipeline(
    "text-generation", model=model, tokenizer=tokenizer, max_new_tokens=20
)

difficulties = {
    "1": "1a/V0",
    "2": "1b/V0",
    "3": "1c/V0",
    "4": "2a/V0",
    "5": "2b/V0",
    "6": "2c/V0",
    "7": "3a/V0",
    "8": "3b/V0",
    "9": "3c/V0",
    "10": "4a/V0",
    "11": "4b/V0",
    "12": "4c/V0",
    "13": "5a/V1",
    "14": "5b/V1",
    "15": "5c/V2",
    "16": "6a/V3",
    "17": "6a+/V3",
    "18": "6b/V4",
    "19": "6b+/V4",
    "20": "6c/V5",
    "21": "6c+/V5",
    "22": "7a/V6",
    "23": "7a+/V7",
    "24": "7b/V8",
    "25": "7b+/V8",
    "26": "7c/V9",
    "27": "7c+/V10",
    "28": "8a/V11",
    "29": "8a+/V12",
    "30": "8b/V13",
    "31": "8b+/V14",
    "32": "8c/V15",
    "33": "8c+/V16",
    "34": "9a/V17",
    "35": "9a+/V18",
    "36": "9b/V19",
    "37": "9b+/V20",
    "38": "9c/V21",
    "39": "9c+/V22",
}


def remove_and_get_non_pr(output):
    non_pr = []

    def remove_non_pr(match):
        non_pr.append(match.group(1))
        return ""

    output = re.sub(r"([^pr\d]\d+|aunk|dunk)", remove_non_pr, output)

    return (output, non_pr)


app = Flask(__name__)
cors = CORS(app)
app.config["CORS_HEADERS"] = "Content-Type"


@app.route("/generate/<prompt>")
@cross_origin()
def generate(prompt):
    result = generator(prompt, do_sample=True, num_beams=1)[0]
    (out, _non_pr) = remove_and_get_non_pr(result["generated_text"])
    out = out.replace(" ", "")

    a = None
    d = None
    for thing in _non_pr:
        if d == None and thing[0] == "d":
            d = thing[1:]
            d = difficulties[d]
        if a == None and thing[0] == "a":
            a = thing[1:]
            a = None if a == "unk" else int(a)

    return {
        "uuid": uuid4().hex,
        "frames": out,
        "name": "TODO",  # use separate model for this
        "description": "beep boop",
        "angle": a,
        "difficulty": d,
    }
