# Generation Fusion Evaluation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Test whether the optional Guard/BGE fusion improves generated answers relative to current Guard.

**Architecture:** Reuse audited frozen A/D selected evidence from the 123-chunk retrieval experiment. Rebuild the actual evidence context using the existing 6000-character budget; generate fresh answers with unchanged evidence prompt, low thinking, temperature and retry policy. Labels are used only in separate evaluation. Freeze source-grounded references before generating; evaluate faithfulness against input context alone, correctness against references, context metrics independently of generated answers, and official BERTScore as an auxiliary signal.

**Tech Stack:** Existing Python/OpenAI-compatible generator; stdlib experiment orchestration and auditable LLM metric formulas; official bert-score installed into an ignored evaluation dependency directory; locally pinned Chinese encoder and existing BGE encoder.

**Spec:** User-approved in-chat generation evaluation design: A/D same settings, frozen references, actual admitted context, separate refusal/clarification/citation checks. User explicitly authorized continuing tests; no additional approval required for routine implementation or configured API calls.

## Global Constraints

- Original46 and development36 reported separately; 82 cases, two variants, three generation repetitions =492 runs. Same development/regression data, no real-user accuracy claim.
- References annotated from frozen corpus, never from tested answers; source-grounded Codex annotation is distinct from human adjudication.
- Generator deepseek-v4-flash; judge deepseek-v4-pro as exposed by the configured provider. Both same family; record potential correlated bias. No new credentials or model-service destination.
- Faithfulness judge receives no reference answer/gold. Context metrics receive no generated answer. Correctness judge sees reference/rubric. Blind variant labels.
- Context Recall/Precision and numeric reference similarity scoped to positive cases; negative/ambiguous cases get expected-behavior grading; never infer rejection from number of hits.
- Empty, failed, truncated and invalid judge outputs explicitly recorded; never count as successful perfect scores. Report completion and all-attempt token use.
- Preserve raw prompts, responses, source mapping and hashes; freeze code/model/reference versions; no tuning after scoring.
- CPU scoring is recorded separately; frozen retrieval inputs mean this is a generation experiment, not a fresh end-to-end timing run.
- Do not change production retrieval/generation/default UI behavior. Preserve .gitignore user hash 639F6EF7D7D0B518EBCCBC46ACF66D31E9317224710CD99D35D975D989D325DB; weights, secrets and local config excluded.

## Tasks

- [ ] Create source-grounded original46/development36 reference answers, factual claims, quotes, forbidden inferences and clarification requirements; independently review and freeze.
- [ ] Implement experiment generation/judge/summary tooling with meaningful failure, isolation, context and pairing tests; review code; commit before full runs.
- [ ] Prepare isolated bert-score dependency and pinned local encoder; verify official artifacts; run actual generator/judge/BERTScore smoke without altering runtime defaults.
- [ ] Generate492 fresh A/D answers; score admitted contexts once per case/variant and evaluate all complete answers; preserve failures and actual token/latency data.
- [ ] Independently audit aggregates and reviewed cases; report measured changes/limitations/recommendation; commit, integrate and push verified batch under existing project authorization.

## Review Focus

- Stale reference IDs, unsupported reference claims or excessive required detail versus question.
- Reference leakage into faithfulness/context selection or generator.
- Failed/empty/truncated outputs, no-claim answers and partial paired completion.
- Prompt/response JSON parsing, invalid evidence/citation IDs and retry accounting.
- Chinese encoder truncation, semantic similarity versus fact correctness, judge model-family bias and cached retrieval timing attribution.
