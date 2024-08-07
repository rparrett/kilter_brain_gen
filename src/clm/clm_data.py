from datasets import Features, Value, load_dataset


class Penalizer:
    start_hold_tokens = set()
    end_hold_tokens = set()
    any_hold_tokens = set()
    angle_tokens = set()
    difficulty_tokens = set()

    def __init__(self, tokenizer):
        for s, t in tokenizer.get_vocab().items():
            if "r12" in s:
                self.start_hold_tokens.add(t)
            elif "r14" in s:
                self.end_hold_tokens.add(t)
            elif "r13" in s:
                self.any_hold_tokens.add(t)
            elif "a" in s:
                self.angle_tokens.add(t)
            elif "d" in s:
                self.difficulty_tokens.add(t)

    def compute_penalties(self, y):
        penalty_factor = 0.1

        penalties = 0.0

        start_holds = 0
        end_holds = 0
        any_holds = 0
        angle_tokens = 0
        difficulty_tokens = 0

        for pred in [y]:
            unique_tokens = set()
            for token in pred:
                # No token should appear more than once
                if token in unique_tokens:
                    penalties += penalty_factor
                unique_tokens.add(token)

                if token in self.start_hold_tokens:
                    start_holds += 1
                elif token in self.end_hold_tokens:
                    end_holds += 1
                elif token in self.any_hold_tokens:
                    any_holds += 1
                elif token in self.angle_tokens:
                    angle_tokens += 1
                elif token in self.difficulty_tokens:
                    difficulty_tokens += 1

        if start_holds < 1:
            penalties += penalty_factor
        elif start_holds > 2:
            penalties += (start_holds - 2) * penalty_factor

        if end_holds < 1:
            penalties += penalty_factor
        elif end_holds > 2:
            penalties += (end_holds - 2) * penalty_factor

        if any_holds < 1:
            penalties += penalty_factor

        if difficulty_tokens < 1:
            penalties += penalty_factor
        elif difficulty_tokens > 1:
            penalties += (difficulty_tokens - 1) * penalty_factor

        if angle_tokens < 1:
            penalties += penalty_factor
        elif angle_tokens > 1:
            penalties += (angle_tokens - 1) * penalty_factor

        return penalties


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

    datasets = dataset.map(add_prefix).train_test_split(
        test_size=0.2, shuffle=True, seed=42
    )
    return datasets


def tokenize_dataset(dataset, tokenizer):
    return dataset.map(lambda example: tokenizer(example["frames"]), batched=True)


def filter_dataset(dataset, tokenizer, penalizer):
    def is_positive_example(example):
        penalty = penalizer.compute_penalties(example["input_ids"])
        # if penalty > 0:
        #    print(example)
        return penalty == 0

    n_before = dataset.num_rows
    dataset = dataset.filter(
        lambda example: is_positive_example(example), batched=False
    )
    n_after = dataset.num_rows
    print(f"Removed {n_before - n_after} of {n_before} examples")
    return dataset


def preprocess_datasets(datasets, tokenizer):
    penalizer = Penalizer(tokenizer)

    for dataset_name in ("train", "test"):
        col_names = datasets[dataset_name].column_names
        datasets[dataset_name] = tokenize_dataset(
            datasets[dataset_name], tokenizer
        ).remove_columns(col_names)
        filter_dataset(datasets[dataset_name], tokenizer, penalizer)
    return datasets


def batch_iterator(datasets, batch_size):
    for i in range(0, len(datasets["train"]), batch_size):
        yield datasets["train"][i : i + batch_size]["frames"]
