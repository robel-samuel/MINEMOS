"""The embedding protocol the encoder comparison relies on is the engine's.

The V4 multilingual encoder comparison encodes every candidate with one fixed
protocol: attention-masked mean pooling, truncation at 128 tokens and L2
normalisation.  That protocol is only a valid description of what MINEMOS does
if it reproduces ``DenseSemanticEncoder`` exactly, so it is checked here against
an independent reimplementation rather than asserted in prose.

The test uses the installed English encoder, because that is the encoder
MINEMOS ships; it says nothing about any candidate's retrieval quality.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from minemos_engine.intelligence.dense import DenseSemanticEncoder  # noqa: E402

TEXTS = (
    "I live in Berlin and work at Acme.",
    "ስሜ ዳዊት ነው።",
    "Vivo en Madrid y trabajo en Ecopetrol.",
    "J'habite à Lyon et je travaille chez Renault.",
    "I live in ጎንደር.",
)


def protocol_encode(encoder, text: str) -> np.ndarray:
    """The documented comparison protocol, reimplemented independently."""
    inputs = encoder.tokenizer(text, padding=True, truncation=True,
                               max_length=128, return_tensors="pt")
    with torch.no_grad():
        outputs = encoder.model(**inputs)
    mask = (inputs["attention_mask"].unsqueeze(-1)
            .expand(outputs.last_hidden_state.size()).float())
    summed = torch.sum(outputs.last_hidden_state * mask, 1)
    counts = torch.clamp(mask.sum(1), min=1e-9)
    pooled = torch.nn.functional.normalize(summed / counts, p=2, dim=1)
    return pooled[0].cpu().numpy().astype(np.float32)


@pytest.fixture(scope="module")
def encoder() -> DenseSemanticEncoder:
    return DenseSemanticEncoder()


def test_protocol_reproduces_engine_vectors(encoder) -> None:
    for text in TEXTS:
        expected = encoder.encode(text)
        actual = protocol_encode(encoder, text)
        assert actual.shape == expected.shape == (384,)
        cosine = float(np.dot(expected, actual) /
                       (np.linalg.norm(expected) * np.linalg.norm(actual)))
        assert cosine >= 0.99999


def test_engine_vectors_are_unit_length_and_finite(encoder) -> None:
    for text in TEXTS:
        vector = encoder.encode(text)
        assert vector.dtype == np.float32
        assert np.all(np.isfinite(vector))
        assert float(np.linalg.norm(vector)) == pytest.approx(1.0, abs=1e-5)


def test_every_script_is_encodable(encoder) -> None:
    vectors = [encoder.encode(text) for text in TEXTS]
    for vector in vectors:
        assert vector.shape == (384,)
        assert np.all(np.isfinite(vector))
    # Different sentences must not collapse onto one another: a degenerate
    # encoder would make every comparison result meaningless.
    for left in range(len(vectors)):
        for right in range(left + 1, len(vectors)):
            cosine = float(np.dot(vectors[left], vectors[right]))
            assert cosine < 0.99
