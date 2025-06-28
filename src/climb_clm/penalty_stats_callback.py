from transformers.integrations import TensorBoardCallback
from generator import generate_tokens, tokens_to_climb
from penalizer import Penalizer


class PenaltyStatsCallback(TensorBoardCallback):
    """Callback to generate routes and log penalty statistics to TensorBoard during evaluation."""

    def __init__(self, tokenizer, num_routes=100, prompts=None):
        super().__init__()
        self.tokenizer = tokenizer
        self.num_routes = num_routes
        self.penalizer = Penalizer(tokenizer)

        # Default prompts if none provided
        if prompts is None:
            self.prompts = [
                ("a20d20", "angle_difficulty"),
                ("a20d20p1143r12p1162r12p1394r14", "partial_climb"),
                ("", "empty"),
            ]
        else:
            self.prompts = prompts

    def on_evaluate(self, args, state, control, model, logs=None, **kwargs):
        """Generate routes and log penalty statistics during evaluation."""

        # Get the current step for logging
        current_step = state.global_step

        print()
        # Generate routes for each prompt and collect statistics
        penalty_logs = {}
        for prompt, prompt_name in self.prompts:
            print(
                f"\nCollecting penalty stats for '{prompt_name}' at step {current_step}:"
            )

            penalty_stats = self._generate_and_analyze_routes(model, prompt)

            # Collect penalty stats for logging
            penalty_logs[f"generation_{prompt_name}/penalty_free_pct"] = penalty_stats[
                "penalty_free_pct"
            ]
            penalty_logs[f"generation_{prompt_name}/penalty_score"] = penalty_stats[
                "penalty_score"
            ]
            penalty_logs[f"generation_{prompt_name}/unique_pct"] = penalty_stats[
                "unique_pct"
            ]

            # Print summary to console
            print(
                f"  Penalty-free: {penalty_stats['penalty_free_count']}/{self.num_routes} ({penalty_stats['penalty_free_pct']:.1f}%)"
            )
            print(
                f"  Unique climbs: {penalty_stats['unique_count']}/{self.num_routes} ({penalty_stats['unique_pct']:.1f}%)"
            )
            print(f"  Avg penalty score: {penalty_stats['penalty_score']:.3f}")

        # Log directly to TensorBoard using self.tb_writer
        if self.tb_writer:
            for metric_name, value in penalty_logs.items():
                self.tb_writer.add_scalar(metric_name, value, current_step)
        else:
            print("Warning: TensorBoard writer not available")

    def _generate_and_analyze_routes(self, model, prompt):
        """Generate routes for a given prompt and return penalty statistics."""
        penalty_free_count = 0
        total_penalty_score = 0
        unique_climbs = set()

        for _ in range(self.num_routes):
            # Generate tokens for the route
            tokens = generate_tokens(self.tokenizer, model, prompt)

            # Convert tokens to climb to get frames for uniqueness check
            climb = tokens_to_climb(self.tokenizer, tokens)
            unique_climbs.add(climb["frames"])

            # Compute penalties and penalty score
            climb_penalties = self.penalizer.compute_penalties(tokens)
            penalty_score = self.penalizer.compute_penalty_score(climb_penalties)

            # Count penalty-free routes
            if not climb_penalties:
                penalty_free_count += 1

            # Accumulate penalty scores
            total_penalty_score += penalty_score

        # Calculate penalty-free percentage
        penalty_free_pct = (penalty_free_count / self.num_routes) * 100

        # Calculate average penalty score
        avg_penalty_score = total_penalty_score / self.num_routes

        # Calculate unique percentage
        unique_count = len(unique_climbs)
        unique_pct = (unique_count / self.num_routes) * 100

        return {
            "penalty_free_count": penalty_free_count,
            "penalty_free_pct": penalty_free_pct,
            "penalty_score": avg_penalty_score,
            "unique_count": unique_count,
            "unique_pct": unique_pct,
        }
