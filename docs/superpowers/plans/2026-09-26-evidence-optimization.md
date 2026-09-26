# Evidence optimization implementation plan

**Goal:** Implement separately selectable recall, evidence selection and generation improvements, with reproducible comparisons and local commits.

**Architecture:** Keep the existing retrieve/generate interfaces and baseline behavior. A pure evidence module provides bounded vocabulary expansion and selection across subqueries. New keyword flags choose experiments; no new external model or dependency. A sequentially controlled generation experiment stores exact contexts, prompts and per-request usage.

**Spec:** User-approved three-stage design in this conversation; EVIDENCE_LOSS_DIAGNOSIS.md supplies measured failure locations.

**Execution:** Native implementation in this session, authorized by the user. Local Git commit for each verified unit; push the completed verified batch.

## Constraints and review focus

- Existing baseline remains accessible. Model, temperature=0.2, final k=8 and 6000-character context limit remain fixed in comparisons.
- No question-specific IDs, gold sources or expected-answer text inside production retrieval rules.
- Empty/no-topic queries retain original behavior. Expansion is capped at original plus two queries and deduplicated.
- Dense-only and BM25-only ablations preserve their retrieval paths; rank scores across different queries must be comparable by rank fusion.
- Evidence selection deduplicates sources and preserves space for original-query results and supplemental facets.
- Missing information, institutional boundaries and unknown current-version validity are not guessed.
- Generated output is reviewed independently of regex citation existence; costs require actual per-request usage including retries.

## Tasks

- [ ] Commit existing evidence-reviewed baseline and diagnosis, leaving user .gitignore change unstaged.
- [ ] Recall: write tests for query cap, original-query retention, unfamiliar questions, asset-damage vocabulary and payroll facets. Run failing tests; add pure expansion and opt-in multi-query RRF integration; pass tests; commit.
- [ ] Selection: write tests for retaining distinct facet evidence, no duplicates and k bounds. Run failing tests; implement facet coverage strategy without gold IDs; pass tests; commit.
- [ ] Generation: preserve baseline prompt; add evidence-oriented prompt/context version and per-request usage sink. Tests cover exact source labels, context budget and usage isolation using a fake API boundary; verify and commit.
- [ ] Experiment: evaluate recall/selection on frozen36 questions and original35 positive retrieval cases. Generate same-period A baseline/B expanded/C coverage/D coverage+generation variants on36, preserving contexts and all usage. Review outputs, compare metrics and failure/regression cases; commit artifacts and report. Do not claim human-reviewed accuracy without actual review.
- [ ] Review diff for baseline compatibility, secrets and unnecessary changes. Verify tests and saved result integrity, then push this completed larger batch.

## Verification commands

Run `python -m unittest discover -s tests -v` using the InternVL environment. Run the experiment scripts against the existing index; ensure every raw result contains variant, question, evidence IDs, context and prompt hashes, answer, error and request usage. Compare complete-gold coverage only on positive cases, not negative related chunks. Do not overwrite frozen baseline_v1 snapshots.
