import json
import os.path
import pprint
import randomname
import requests

from random import randint
from uuid import uuid4
from flask import Flask, request
from flask_cors import CORS, cross_origin
from transformers import (
    AutoTokenizer,
    GPT2Config,
    GPT2LMHeadModel,
    GPT2TokenizerFast,
    pipeline,
)

import sys
from pathlib import Path

# Add the src directory to the Python path so we can import from sibling packages
src_dir = Path(__file__).parent.parent
sys.path.insert(0, str(src_dir))

from climb_clm.data import find_latest_checkpoint
from climb_clm.generator import generate_climb

def get_frames_model():
    base_dir = "models/climb_clm"

    latest_checkpoint_dir = find_latest_checkpoint(base_dir)
    if latest_checkpoint_dir:
        model_dir = str(latest_checkpoint_dir)
        token_dir = str(latest_checkpoint_dir.parent)
    else:
        raise Exception("Couldn't find a checkpoint")

    return (AutoTokenizer.from_pretrained(token_dir), GPT2LMHeadModel.from_pretrained(model_dir))

def get_name_generator():
    checkpoint = "1100"

    token_dir = "models/name"
    model_dir = (
        token_dir if checkpoint is None else token_dir + "/checkpoint-" + checkpoint
    )

    tokenizer = GPT2TokenizerFast.from_pretrained(
        "gpt2",
        bos_token="<|startoftext|>",
        eos_token="<|endoftext|>",
        pad_token="<|pad|>",
    )
    config = GPT2Config.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained(model_dir)

    generator = pipeline(
        "text-generation", model=model, tokenizer=tokenizer, max_new_tokens=20
    )

    return generator


(tokenizer, model) = get_frames_model()
# name_generator = get_name_generator()

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

app = Flask(__name__)
cors = CORS(app)
app.config["CORS_HEADERS"] = "Content-Type"


@app.route("/generate", methods=["POST"])
@cross_origin()
def generate():
    data = request.json

    num = min(data.get("num", 1), 10)

    # TODO error if no prompt

    climbs = []

    for _ in range(num):
        climb = generate_climb(tokenizer, model, data["prompt"])

        name = randomname.generate() + "-" + str(randint(100, 999))

        climbs.append(
            {
                "uuid": uuid4().hex,
                "frames": climb["frames"],
                "name": name,
                "description": "beep boop",
                "angle": climb["angle"],
                "difficulty": climb["difficulty"],
            }
        )

    return climbs


@app.route("/publish", methods=["POST"])
@cross_origin()
def publish():
    data = request.json

    is_draft = data.get("is_draft", False)

    if not os.path.isfile("token.json"):
        return {"error": "No stored token"}

    f = open("token.json")
    token = json.load(f)

    url = "https://api.kilterboardapp.com/v2/climbs/%s" % data["uuid"]

    to_kilter = {
        "uuid": data["uuid"],
        "layout_id": 1,
        "setter_id": token["session"]["user_id"],
        "name": data["name"],
        "description": data["description"],
        "is_draft": is_draft,
        "frames_count": 1,
        "frames_pace": 0,
        "frames": data["frames"],
        "angle": data["angle"],
    }

    res = requests.put(
        url,
        json=to_kilter,
        headers={
            "Accept-Encoding": "gzip",
            "Authorization": "Bearer %s" % token["session"]["token"],
            "User-Agent": "Dalvik/2.1.0 (Linux; U; Android 11; Pixel 2 XL Build/RP1A.201005.004.A1)",
        },
    )

    pprint.pprint(res)
    pprint.pprint(res.content)

    return {}
