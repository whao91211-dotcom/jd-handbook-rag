# Interrupted retry-transport diagnostic run

This partial run tried escalation by passing per-call max_tokens. The installed LlamaIndex model default overrode that value, so all synthesis requests still used 2048 tokens even though local usage metadata displayed intended higher budgets. The request-transport regression test reproduced this issue; final code now changes the model default per attempt.

This run was stopped. Treat its `max_tokens` fields as intended budgets, not actual transmitted budgets. Do not combine it with the final paired evaluation at `../agent_experiment_2026-10-04/` or claim a gain from incomplete answers.
