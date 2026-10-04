# Complex-question RAG Agent

Approved in chat on 2026-10-04: a manually selected complex-question mode using LlamaIndex, keeping baseline/quality/fast unchanged.

## Behavior

A single agent calls the existing guarded retriever and reads adjacent chunks. It retains explicit eligibility conditions in queries and decides whether the evidence warrants another search. It gathers evidence, then final synthesis uses the existing evidence prompt, context builder and generation retry budgets through a cancellable async LlamaIndex request. The agent's intermediate prose is never treated as source evidence. Missing personal conditions produce clarifying questions in the final answer; this release does not add persistent multi-turn conversation state.

At most 3 searches and 2 adjacent reads per question, 8 orchestration model calls, 120 seconds for orchestration, 120 seconds for final generation, and 2048 output tokens per orchestration call. Final synthesis retries empty/truncated responses with the existing 2048/4096/8192 output budgets, at most 3 requests within one shared 120-second deadline. Search results and adjacent reads share a 6000-character evidence context. Explicit limit/failure statuses retain evidence and traces. No silent fallback to ordinary RAG. Every model attempt is counted; missing usage is unknown, not zero. API errors/credentials remain out of browser responses.

## Integration

Use LlamaIndex FunctionTool and the function-calling LLM interface with OpenAILike against the configured API. A bounded application-owned loop controls execution and cancellation, rather than relying on an agent's prompt to enforce budgets. Tools return raw chunks with page/path/id. Adjacent reads are restricted to chunk IDs already retrieved, and retrieve the anchor and immediate neighbors in source order. Final citations map only to chunks actually admitted to the final context.

Add `agent` profile in API, CLI, and browser. Retrieval-only requests do not invoke any model and use guarded retrieval. Trace displays tool names, queries, returned source IDs and stop reasons; never hidden reasoning. Keep dependency installation isolated from the original conda environment.

## Evaluation and acceptance

Meaningful offline tests cover adaptive retrieval, caps, duplicates, invalid tool calls, timeout, empty evidence, unknown citations, API failures, source mapping and usage. Existing suite must pass. Live smoke verifies configured model function calling and real indexed retrieval. Paired evaluation uses the same model, corpus and evidence prompt for quality and agent profiles; the existing 36 simulated cases are regression data only. Newly authored complex cases are development probes, not held-out real-user benchmarks. Report final-context coverage separately from collected evidence coverage; report model review separately from human adjudication. Do not claim answer quality gains without comparable review.

Commit coherent verified batches locally, then push the feature branch. Leave source data and credentials ignored.
