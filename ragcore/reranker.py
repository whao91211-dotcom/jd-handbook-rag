"""Opt-in, offline BGE scoring; no evaluation labels enter this module."""
from __future__ import annotations

import functools
import math

MODEL_ID = 'BAAI/bge-reranker-base'
MODEL_REVISION = '2cfc18c9415c912f9d8155881c133215df768a70'
BATCH_SIZE = 8
MAX_LENGTH = 512
CANDIDATE_LIMIT = 40
MODEL_FILES = ('config.json', 'model.safetensors', 'sentencepiece.bpe.model',
               'special_tokens_map.json', 'tokenizer.json', 'tokenizer_config.json')


def candidate_pool(fused: list[str], baseline: list[str], anchors: list[str]) -> list[str]:
    """Reserve protected IDs, fill by RRF, then restore the RRF tie order."""
    available = set(fused)
    protected = list(dict.fromkeys(baseline + anchors))
    if len(protected) > CANDIDATE_LIMIT or not set(protected) <= available:
        raise ValueError('Protected candidates exceed the pool or are not recalled')
    selected = set(protected)
    for iid in fused:
        if len(selected) >= CANDIDATE_LIMIT:
            break
        selected.add(iid)
    return list(dict.fromkeys(iid for iid in fused if iid in selected))


def score_order(ids: list[str], scores: dict[str, float]) -> list[str]:
    if any(iid not in scores or not math.isfinite(scores[iid]) for iid in ids):
        raise ValueError('Missing or non-finite reranker score')
    return sorted(ids, key=lambda iid: -scores[iid])


@functools.lru_cache(maxsize=1)
def get_model():
    # Local snapshots only: prepare weights before running any retrieval request.
    from huggingface_hub import snapshot_download
    from sentence_transformers import CrossEncoder
    import torch

    path = snapshot_download(MODEL_ID, revision=MODEL_REVISION, local_files_only=True,
                             allow_patterns=list(MODEL_FILES))
    return CrossEncoder(path, device='cpu', local_files_only=True,
                        max_length=MAX_LENGTH, activation_fn=torch.nn.Identity(),
                        model_kwargs={'dtype': torch.float32})


def score_candidates(query: str, candidates: list[dict]) -> tuple[dict, dict]:
    if not candidates:
        return {}, {}
    model = get_model()
    texts = [c['text'] for c in candidates]
    # Count the exact same pair format before longest-first truncation in predict.
    tokens = model.tokenizer([query] * len(texts), texts, padding=False, truncation=False)
    lengths = {c['id']: len(ids) for c, ids in zip(candidates, tokens['input_ids'])}
    values = model.predict([(query, text) for text in texts], batch_size=BATCH_SIZE,
                           show_progress_bar=False, convert_to_numpy=True)
    if len(values) != len(candidates):
        raise ValueError('Reranker returned an unexpected score count')
    scores = {c['id']: float(value) for c, value in zip(candidates, values)}
    score_order([c['id'] for c in candidates], scores)
    return scores, lengths
