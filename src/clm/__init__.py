"""CLM — contrastive language model inference engine and System One API.

Client (no torch needed):

    from clm import CLMClient, Choice, Noul, Score
    r = CLMClient().system_one(state, {"ok": Noul(instructions="Is this fine?")})

Engine (in-process, needs ``pip install -r requirements.txt`` from the repo and an embedder endpoint):

    from clm import Engine
    Engine().answer(state, {"ok": {"type": "noul", "instructions": "Is this fine?"}})

No embedder server? ``Engine(embedder=UnslothEmbedder())`` runs Qwen3-8B in-process
(``clm.unsloth_embedder``; Unsloth when installed, else transformers).

Server: ``clm-serve`` (``--local-encoder`` for the in-process encoder).  Checkpoint: ``clm-download``.
"""
from .client import (Answer, Choice, ChoiceAnswer, CLMClient, CLMError, Noul, NoulAnswer, Question, Score,
                     ScoreAnswer, SystemOneResponse, Usage)
from .schema import answer_from_logits, answer_from_probs, build_pairs, candidates, state_text

__version__ = "0.1.0"
__all__ = ["CLMClient", "CLMError", "Noul", "Choice", "Score", "Question", "Answer", "NoulAnswer", "ChoiceAnswer",
           "ScoreAnswer", "SystemOneResponse", "Usage", "Engine", "Embedder", "UnslothEmbedder", "HeadPair",
           "build_pairs", "candidates", "state_text", "answer_from_logits", "answer_from_probs", "__version__"]


def __getattr__(name):  # lazy: Engine / Embedder / HeadPair pull in numpy+torch only when used
    if name == "Engine":
        from .engine import Engine
        return Engine
    if name == "Embedder":
        from .embedder import Embedder
        return Embedder
    if name == "UnslothEmbedder":
        from .unsloth_embedder import UnslothEmbedder
        return UnslothEmbedder
    if name == "HeadPair":
        from .heads import HeadPair
        return HeadPair
    raise AttributeError(name)
