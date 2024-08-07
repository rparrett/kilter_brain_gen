import torch
import torch.nn.functional as F
from transformers import Trainer


class CustomTrainer(Trainer):
    def __init__(self, *args, tokenizer, **kwargs):
        super().__init__(*args, **kwargs)

        vocab = tokenizer.get_vocab()
        self.start_hold_mask = torch.tensor(
            [1 if "r12" in s else 0 for s in vocab.keys()]
        )
        self.end_hold_mask = torch.tensor(
            [1 if "r14" in s else 0 for s in vocab.keys()]
        )
        self.any_hold_mask = torch.tensor(
            [1 if "r13" in s else 0 for s in vocab.keys()]
        )
        self.angle_mask = torch.tensor([1 if "a" in s else 0 for s in vocab.keys()])
        self.difficulty_mask = torch.tensor(
            [1 if "d" in s else 0 for s in vocab.keys()]
        )
        self._device_of_masks = "cpu"
        self.penalty_alpha = 5e-1

    def compute_loss(self, model, inputs, return_outputs=False):
        (loss, outputs) = super().compute_loss(model, inputs, return_outputs=True)

        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)
        attention_mask = inputs.get("attention_mask")  # Get attention mask from inputs
        penalties, seq_lengths = self.compute_penalties(predictions, attention_mask)

        total_loss = loss + penalties

        self.log(
            {
                "original_loss": loss.item(),
                "custom_penalty": penalties.item(),
                "total_loss": total_loss.item(),
                "avg_seq_length": seq_lengths.float().mean().item(),
                "min_seq_length": seq_lengths.min().item(),
                "max_seq_length": seq_lengths.max().item(),
            }
        )
        # Log sequence length distribution (in bins)
        # seq_length_bins = torch.histc(seq_lengths.float(), bins=10, min=0, max=seq_lengths.max())
        # for i, count in enumerate(seq_length_bins):
        #    self.log({f"seq_length_bin_{i}": count.item()})

        return (total_loss, outputs) if return_outputs else total_loss

    def compute_penalties(self, predictions, attention_mask=None):
        device = predictions.device

        # Move mask tensors to the same device as predictions
        if self._device_of_masks != device:
            self.start_hold_mask = self.start_hold_mask.to(device)
            self.end_hold_mask = self.end_hold_mask.to(device)
            self.any_hold_mask = self.any_hold_mask.to(device)
            self.angle_mask = self.angle_mask.to(device)
            self.difficulty_mask = self.difficulty_mask.to(device)
            self._device_of_masks = device  # Update the device tracker

        batch_size, seq_length = predictions.shape

        # Create non-padding token mask
        if attention_mask is None:
            # If no attention mask is provided, assume all tokens are non-padding
            non_padding_mask = torch.ones_like(
                predictions, dtype=torch.float, device=device
            )
        else:
            non_padding_mask = attention_mask.float()

        # Calculate actual sequence lengths
        actual_seq_lengths = non_padding_mask.sum(dim=1)

        # One-hot encode predictions
        one_hot = F.one_hot(predictions, num_classes=len(self.start_hold_mask)).float()

        # Apply non-padding mask to one-hot encodings
        one_hot = one_hot * non_padding_mask.unsqueeze(-1)
        # Count occurrences of each token type
        start_holds = torch.sum(one_hot * self.start_hold_mask, dim=(1, 2))
        end_holds = torch.sum(one_hot * self.end_hold_mask, dim=(1, 2))
        any_holds = torch.sum(one_hot * self.any_hold_mask, dim=(1, 2))
        angle_tokens = torch.sum(one_hot * self.angle_mask, dim=(1, 2))
        difficulty_tokens = torch.sum(one_hot * self.difficulty_mask, dim=(1, 2))

        penalties = torch.zeros(batch_size, device=device)

        # Penalties for repeated holds/tokens
        penalties += (torch.sum(one_hot, dim=1) > 1).sum().float()

        # Penalties for start and end holds
        # Values less than 1 or greater than 2 are penalized
        start_hold_penalties = torch.max(1 - start_holds, start_holds - 2).clamp(min=0)
        end_hold_penalties = torch.max(1 - end_holds, end_holds - 2).clamp(min=0)
        penalties += start_hold_penalties.sum() + end_hold_penalties.sum()

        # Penalties for any_holds, difficulty_tokens, and angle_tokens
        # Values less than 1 or greater than 1 are penalized
        any_hold_penalties = torch.abs(any_holds - 1)
        difficulty_penalties = torch.abs(difficulty_tokens - 1)
        angle_penalties = torch.abs(angle_tokens - 1)
        penalties += (any_hold_penalties + difficulty_penalties + angle_penalties).sum()

        # Normalize by actual sequence lengths
        normalized_penalties = penalties / actual_seq_lengths

        return torch.mean(normalized_penalties) * self.penalty_alpha, actual_seq_lengths

    def log(self, logs):
        """
        Add custom_penalty to the existing logs and pass to the original log method.
        """
        if self.state.epoch is not None:
            logs["epoch"] = round(self.state.epoch, 2)

        super().log(logs)
