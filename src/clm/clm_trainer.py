from torch import nn, argmax
from transformers import Trainer
import pprint

class CustomTrainer(Trainer):
    start_hold_tokens = set()
    end_hold_tokens = set()
    any_hold_tokens = set()

    def __init__(self, *args, tokenizer, **kwargs):
        super().__init__(*args, **kwargs)

        for s, t in tokenizer.get_vocab().items():
            if "r12" in s:
                self.start_hold_tokens.add(t)
            elif "r14" in s:
                self.end_hold_tokens.add(t)
            elif "r13" in s:
                self.any_hold_tokens.add(t)

    def compute_loss(self, model, inputs, return_outputs=False):
        (loss, outputs) = super().compute_loss(model, inputs, return_outputs=True)

        logits = outputs.logits

        # Compute custom penalties
        predictions = argmax(logits, dim=-1)
        penalties = self.compute_penalties(predictions)

        loss += penalties

        return (loss, outputs) if return_outputs else loss

    def compute_penalties(self, predictions):
        penalty_factor = 5.0

        penalties = 0.0

        start_holds = 0
        end_holds = 0
        any_holds = 0

        for pred in predictions:
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

        return penalties