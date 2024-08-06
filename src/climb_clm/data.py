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
        data_files="climbs.csv",
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
