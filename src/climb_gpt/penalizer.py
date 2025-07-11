from collections import defaultdict


class Penalizer:
    start_hold_tokens = set()
    end_hold_tokens = set()
    any_hold_tokens = set()
    angle_tokens = set()
    difficulty_tokens = set()

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.expected_repeats = set(
            [
                self.tokenizer.pad_token_id,
                self.tokenizer.eos_token_id,
                self.tokenizer.bos_token_id,
            ]
        )
        for s, t in tokenizer.get_vocab().items():
            if "r12" in s:
                self.start_hold_tokens.add(t)
            elif "r14" in s:
                self.end_hold_tokens.add(t)
            elif "r13" in s:
                self.any_hold_tokens.add(t)
            elif "a" in s and s != "<pad>":
                self.angle_tokens.add(t)
            elif "d" in s and s != "<pad>":
                self.difficulty_tokens.add(t)

    def get_token_sets(self):
        vocab = self.tokenizer.get_vocab()
        lookup = {v: k for k, v in vocab.items()}
        return {
            "start_hold_tokens": [(lookup[t], t) for t in self.start_hold_tokens],
            "end_hold_tokens": [(lookup[t], t) for t in self.end_hold_tokens],
            "any_hold_tokens": [(lookup[t], t) for t in self.any_hold_tokens],
            "angle_tokens": [(lookup[t], t) for t in self.angle_tokens],
            "difficulty_tokens": [(lookup[t], t) for t in self.difficulty_tokens],
        }

    def compute_penalties(self, tokens):
        penalties = defaultdict(int)

        start_holds = 0
        end_holds = 0
        any_holds = 0
        angle_tokens = 0
        difficulty_tokens = 0
        # lookup = {v: k for k, v in self.tokenizer.get_vocab().items()}

        unique_tokens = set()
        for token in tokens:
            # No token should appear more than once
            if token in unique_tokens and token not in self.expected_repeats:
                penalties["repeated_tokens"] += 1
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
            penalties["start_holds_missing"] += 1
        elif start_holds > 2:
            penalties["start_holds_excessive"] += (start_holds - 2) * 1

        if end_holds < 1:
            penalties["end_holds_missing"] += 1
        elif end_holds > 2:
            penalties["end_holds_excessive"] += (end_holds - 2) * 1

        if any_holds < 1:
            penalties["holds_missing"] += 1

        if difficulty_tokens < 1:
            penalties["difficulty_token_missing"] += 1
        elif difficulty_tokens > 1:
            penalties["difficulty_token_excessive"] += (difficulty_tokens - 1) * 1

        if angle_tokens < 1:
            penalties["angle_token_missing"] += 1
        elif angle_tokens > 1:
            penalties["angle_token_excessive"] += (angle_tokens - 1) * 1

        return penalties

    def compute_penalty_score(self, penalties):
        penalty_factor = 0.1
        penalty_score = 0
        for v in penalties.values():
            penalty_score += v * penalty_factor
        return penalty_score
