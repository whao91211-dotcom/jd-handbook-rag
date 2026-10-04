# Interrupted synthesis-deadline probe

The corrected transport sent the intended retry budgets, but the first complex case reached the 60-second synthesis deadline. The ordinary quality baseline needed 63.8 seconds end-to-end with 3 attempts; agent orchestration completed and gathered all reference chunks, then synthesis timed out. This is a budget limit, not evidence of answer correctness or a speed improvement.

The final default synthesis deadline was increased to 120 seconds, still shared across at most three attempts. Final paired results are in `../agent_experiment_2026-10-04/`. Retain this partial run as a timeout diagnostic; do not combine it with final metrics.
