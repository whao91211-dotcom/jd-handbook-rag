# Collaboration and experiment rules

- Explain changes for an undergraduate preparing for AI application development internships: reason, affected metric, measured baseline/new result, dataset and change. Never invent gains.
- After each coherent project change and verification, make an explicit local Git commit. Preserve unrelated user changes. Push a larger verified batch to the configured remote; do not push credentials or local configuration.
- Keep baseline retrieval and generation available for ablation. Never encode evaluation question IDs, gold chunk IDs or expected answers into retrieval logic.
- These 36 simulated cases are a development/regression set, not a held-out real-user benchmark. Report model review separately from user adjudication.
