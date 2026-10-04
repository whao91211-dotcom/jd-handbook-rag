# Complex-question RAG Agent Implementation Plan

> Execute inline with superpowers:executing-plans and test-driven-development. The user explicitly approved implementation of the proposed scope.

**Goal:** Add a bounded LlamaIndex-powered complex-question mode alongside existing RAG modes.

**Architecture:** Existing guarded retrieval becomes a FunctionTool. OpenAILike supplies the tool-calling decisions in a bounded loop; existing evidence generation remains the final synthesis stage.

**Tech Stack:** Python 3.10, LlamaIndex core/OpenAILike, Chroma/BGE/BM25, FastAPI, native browser UI.

**Spec:** ../specs/2026-10-04-complex-rag-agent.md

## Global constraints

Preserve baseline/quality/fast. No evaluation labels in runtime. Caps: 3 searches, 2 reads, 8 orchestration calls, 120-second orchestration, 60-second synthesis, 2048 orchestration output tokens, 6000-character evidence. Missing usage stays unknown. No secrets in browser or committed files.

## Review focus

- Model requests multiple tools in one message: enforce each cap before execution.
- Sources overflow context: final citations must use only admitted evidence.
- Repeated or malformed calls: never execute unchecked arguments or arbitrary reads.
- Timeout or API failure after retrieval: retain evidence, return explicit incomplete status.
- Missing token counts: never publish partial totals as complete costs.

## Task 1: Bounded agent and dependency compatibility

Files: requirements-agent.txt, ragcore/agent.py, tests/test_agent.py.
Interface: async `collect_evidence(question, *, llm=None, search=None, read=None, limits=None)` returns chunks, trace, usage, status, timings; sync `answer_question(question)` returns the web response shape.

- [ ] Create isolated dependency environment; run original unittest suite and import installed LlamaIndex APIs.
- [ ] Write offline scripted-LLM tests that expect a second search to add missing evidence, reject a fourth search/third read, reject unknown anchor IDs and preserve evidence on timeout/errors. Run `python -m unittest discover -s tests -p test_agent.py -v` and observe missing-feature failures.
- [ ] Implement FunctionTool adapters and bounded loop using `achat_with_tools`, `get_tool_calls_from_response`, ChatMessage and tool messages. Final synthesis uses the existing evidence prompt and prepare_context.
- [ ] Run targeted tests and full `python -m unittest discover -s tests -v`, then commit `feat: add bounded LlamaIndex evidence agent`.

## Task 2: API, CLI and browser integration

Files: webapp.py, ask.py, web/index.html, web/app.js, tests/test_webapp.py.
Interface: profile `agent` returns the same answer/sources/usage/metrics fields plus trace and limit status.

- [ ] Add API tests accepting agent mode, retaining sources on failure, mapping final-context sources, and ensuring retrieval-only does not call an agent. Observe failures before integration.
- [ ] Route agent requests through answer_question; add CLI switch and browser mode/trace details. Preserve existing branches and credential redaction.
- [ ] Run full unittest suite, CLI help, browser JavaScript syntax and local browser smoke; commit `feat: expose complex-question mode in API CLI and browser`.

## Task 3: Paired evaluation, evidence and delivery

Files: evaluation/run_agent_experiment.py, evaluation/complex_cases_v1.jsonl, tests/test_agent_evaluation.py, COMPLEX_RAG_AGENT.md, README.md.

- [ ] Test that evaluation passes only question text to runtime and separates gathered versus final-context coverage; run red test.
- [ ] Implement serial paired quality/agent runs with revision/corpus hashes, warm-up timing notes, raw attempts, tool traces, final context and manual review templates. No gold labels are sent into the pipeline.
- [ ] Run small live compatibility and paired development probe; include failures and measured latency/tokens without inventing correctness gains.
- [ ] Review whole diff, run full tests, update plan checkboxes and report limitations; commit documentation/evaluation batch and push codex/complex-rag-agent.

## Execution rulings

- Work in a dedicated feature branch in the existing clean checkout so the configured local index remains available. Do not create an extra worktree or request another approval.
- Implement inline; no multi-agent implementation dispatch is required.
- Live browser synthesis exhausted 2048 shared reasoning/output tokens. Reuse existing 2048/4096/8192 retries within one async synthesis deadline; count every attempt. Existing synchronous generate cannot enforce a shared cancellation deadline, so reuse its prompt/context/budgets rather than calling that function.
