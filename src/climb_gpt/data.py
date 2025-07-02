from datasets import Features, Value, load_dataset


def add_prefix(example):
    a = "unk" if example["angle"] is None else example["angle"]
    d = (
        "unk"
        if example["display_difficulty"] is None
        else str(round(example["display_difficulty"]))
    )
    example["frames"] = "a" + a + "d" + d + example["frames"]
    return example


def load_training_datasets():
    dataset = load_dataset(
        "csv",
        data_files="data/climbs.csv",
        delimiter=",",
        features=Features(
            {
                "frames": Value("string"),
                "display_difficulty": Value("float"),
                "quality_average": Value("float"),
                "angle": Value("string"),
            }
        ),
        split="train",
    )

    dataset = dataset.filter(lambda example: example["quality_average"] >= 2.5)

    datasets = dataset.map(add_prefix).train_test_split()
    return datasets


def tokenize_dataset(dataset, tokenizer):
    return dataset.map(lambda example: tokenizer(example["frames"]), batched=True)


def preprocess_datasets(datasets, tokenizer):
    for dataset_name in ("train", "test"):
        col_names = datasets[dataset_name].column_names
        datasets[dataset_name] = tokenize_dataset(
            datasets[dataset_name], tokenizer
        ).remove_columns(col_names)
    return datasets


def batch_iterator(datasets, batch_size):
    for i in range(0, len(datasets["train"]), batch_size):
        yield datasets["train"][i : i + batch_size]["frames"]


DIFFICULTY = {
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
