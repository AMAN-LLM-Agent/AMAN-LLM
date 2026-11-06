# tests/conftest.py
"""
Shared fixtures for Layer 3 tests.

Real runs of claim_verifier.py / output_validator.py need
sentence-transformers to download 'paraphrase-multilingual-MiniLM-L12-v2'
from huggingface.co. That's correct for production, but it makes tests
slow, network-dependent, and non-deterministic in CI.

`fake_embeddings` replaces the embedding call sites with a deterministic,
offline bag-of-words cosine-similarity stand-in: same text -> sim 1.0,
disjoint vocabulary -> sim ~0. This is NOT a substitute for evaluating
real model accuracy (see src/layer3/evaluation/benchmark.py for that,
which is meant to run against the real model) — it exists purely so the
*plumbing* (fallback wiring, pass/fail branching, latency instrumentation)
can be tested quickly and deterministically without network access.

Tests that need the real model should be marked
`@pytest.mark.requires_model` and skipped in CI environments without
huggingface.co access (see pytest.ini / -m "not requires_model").
"""
from __future__ import annotations

import re

import pytest
import torch


def _tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


def _bow_vector(text: str, dim: int = 256) -> torch.Tensor:
    v = torch.zeros(dim)
    for tok in _tokenize(text):
        v[hash(tok) % dim] += 1.0
    norm = v.norm()
    return v / norm if norm > 0 else v


def _fake_embed(texts):
    return torch.stack([_bow_vector(t) for t in texts])


def _fake_embed_single(text: str):
    return _bow_vector(text)


@pytest.fixture
def fake_embeddings(monkeypatch):
    """
    Patch claim_verifier's embedding entry points with the deterministic
    bag-of-words stand-in described above. Yields nothing; apply by adding
    `fake_embeddings` as a test argument.
    """
    import src.layer3.claim_verifier as cv

    monkeypatch.setattr(cv, "_embed", _fake_embed)
    monkeypatch.setattr(cv, "_embed_single_cached", _fake_embed_single)
    yield


@pytest.fixture
def fake_output_validator_embedding_model(monkeypatch):
    """
    Patch output_validator's SentenceTransformer loader with a fake model
    exposing the same .encode(...) surface it relies on, so
    _semantic_consistency_score doesn't need network access either.
    """
    import src.layer3.output_validator as ov

    class _FakeModel:
        def encode(self, texts, convert_to_tensor=True, normalize_embeddings=True):
            return torch.stack([_bow_vector(t) for t in texts])

    monkeypatch.setattr(ov, "_get_embedding_model", lambda name: _FakeModel())
    yield
