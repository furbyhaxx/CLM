"""In-process encoder: Qwen3-8B last-token embeddings without a vLLM server.

    from clm import Engine
    from clm.unsloth_embedder import UnslothEmbedder
    engine = Engine(embedder=UnslothEmbedder())        # loads Qwen3-8B into this process

It reproduces what ``vllm serve Qwen/Qwen3-8B --runner pooling`` returns: the
hidden state of the last token after the final RMSNorm, L2-normalised.  Text is
tokenized without special tokens and truncated to its last ``max_tokens``
tokens, like ``truncate_prompt_tokens``.

The backbone is loaded with Unsloth's ``FastLanguageModel`` when it is installed
(``backend="unsloth"``) and with plain transformers otherwise (``backend="hf"``).
``load_in_4bit=None`` picks 4-bit on GPUs with less than 20 GB (a Colab T4) and
16-bit otherwise.  The released heads were trained on 16-bit embeddings, so a
4-bit encoder is a close approximation rather than bit-exact.

Batches are right-padded and never masked: attention is causal, so no real
token ever attends to the padding after it, and the last real token's state is
the same as it would be unpadded.
"""
from __future__ import annotations

import threading
from typing import Any

import numpy as np

from .embedder import Embedder

DEFAULT_ENCODER = "Qwen/Qwen3-8B"
SMALL_GPU_BYTES = 20 * 2**30      # below this, the 8B encoder is loaded in 4-bit


def backbone(model):
    """The decoder stack (``Qwen3Model``) of a causal LM, a PEFT wrapper, or the stack itself."""
    m = model.get_base_model() if hasattr(model, "get_base_model") else model
    return m.model if hasattr(m, "lm_head") else m


def last_token_states(model, id_lists: list[list[int]], pad_id: int = 0):
    """-> [n, hidden] final-norm hidden states of each sequence's last token (model dtype, not
    normalised, differentiable when grad is enabled).  Sequences are right-padded."""
    import torch
    body = backbone(model)
    device = body.embed_tokens.weight.device
    n, width = len(id_lists), max(len(ids) for ids in id_lists)
    ids = torch.full((n, width), pad_id, dtype=torch.long)
    for i, seq in enumerate(id_lists):
        ids[i, :len(seq)] = torch.as_tensor(seq, dtype=torch.long)
    last = torch.tensor([len(seq) - 1 for seq in id_lists], device=device)
    out = body(input_ids=ids.to(device), use_cache=False)
    hidden = out.last_hidden_state if hasattr(out, "last_hidden_state") else out[0]
    return hidden[torch.arange(n, device=device), last]


def _cuda_total() -> int:
    try:
        import torch
        return torch.cuda.get_device_properties(0).total_memory if torch.cuda.is_available() else 0
    except Exception:  # noqa: BLE001
        return 0


def default_backend() -> str:
    try:
        import unsloth  # noqa: F401
        return "unsloth"
    except Exception:  # noqa: BLE001  (ImportError, or unsloth refusing to start without a GPU)
        return "hf"


def resolve_4bit(load_in_4bit: bool | None) -> bool:
    """``None`` -> 4-bit on a GPU under 20 GB (a Colab T4), 16-bit otherwise."""
    if load_in_4bit is None:
        total = _cuda_total()
        return 0 < total < SMALL_GPU_BYTES
    return bool(load_in_4bit)


def load_encoder(model: str = DEFAULT_ENCODER, max_seq_length: int = 2048, load_in_4bit: bool | None = None,
                 backend: str | None = None, token: str | None = None):
    """-> (model, tokenizer) ready for inference.

    ``model`` may also be a LoRA adapter directory saved by Unsloth / PEFT; Unsloth
    loads its base model and applies the adapter.
    """
    import torch
    backend = backend or default_backend()
    load_in_4bit = resolve_4bit(load_in_4bit)
    if backend == "unsloth":
        from unsloth import FastLanguageModel
        m, tok = FastLanguageModel.from_pretrained(model_name=model, max_seq_length=max_seq_length,
                                                   load_in_4bit=load_in_4bit, dtype=None, token=token)
        FastLanguageModel.for_inference(m)
    elif backend == "hf":
        from transformers import AutoModel, AutoTokenizer
        cuda = torch.cuda.is_available()
        dtype = (torch.bfloat16 if cuda and torch.cuda.is_bf16_supported() else torch.float16) if cuda else torch.float32
        kw: dict[str, Any] = {"torch_dtype": dtype, "token": token}
        if load_in_4bit:
            from transformers import BitsAndBytesConfig
            kw["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=dtype)
        if cuda:
            kw["device_map"] = {"": 0}
        m = AutoModel.from_pretrained(model, **kw).eval()
        tok = AutoTokenizer.from_pretrained(model, token=token)
    else:
        raise ValueError(f"unknown backend {backend!r}; expected 'unsloth' or 'hf'")
    return m, tok


class LocalEncoder:
    """A loaded backbone that turns token ids into L2-normalised last-token embeddings.

    ``batch_tokens`` caps the padded tokens per forward pass (default: 16k, 64k on
    GPUs with 40 GB or more); sequences are sorted by length so batches pad little.
    """

    def __init__(self, model: str = DEFAULT_ENCODER, max_tokens: int = 2048, load_in_4bit: bool | None = None,
                 backend: str | None = None, batch_tokens: int | None = None, max_batch: int = 64,
                 drop_lm_head: bool = True, token: str | None = None, loaded: tuple | None = None):
        import torch
        self.name, self.max_tokens, self.max_batch = model, max_tokens, max_batch
        self.load_in_4bit = None if loaded else resolve_4bit(load_in_4bit)   # None: caller's own model
        self.model, self.tok = loaded or load_encoder(model, max_tokens, self.load_in_4bit, backend, token)
        body = backbone(self.model)
        if loaded is not None:
            self.name = getattr(body.config, "_name_or_path", None) or model
        self.hidden = int(body.config.hidden_size)
        self.pad_id = self.tok.pad_token_id if self.tok.pad_token_id is not None else 0
        total = _cuda_total()
        self.batch_tokens = batch_tokens or (65536 if total >= 40 * 2**30 else 16384)
        lm = self.model.get_base_model() if hasattr(self.model, "get_base_model") else self.model
        head = getattr(lm, "lm_head", None)
        if drop_lm_head and loaded is None and isinstance(head, torch.nn.Linear) \
                and head.weight.data_ptr() != body.embed_tokens.weight.data_ptr():
            # 151k x 4096 logits projection: ~1.2 GB the encoder never uses
            lm.lm_head = torch.nn.Identity()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        self._lock = threading.Lock()

    @classmethod
    def from_model(cls, model, tokenizer, max_tokens: int = 2048, **kw) -> "LocalEncoder":
        """Wrap a model already in memory (e.g. one being LoRA-trained); it is used as is."""
        return cls(max_tokens=max_tokens, loaded=(model, tokenizer), **kw)

    def text_ids(self, text: str, keep: str = "tail", cap: int | None = None) -> list[int]:
        cap = cap or self.max_tokens
        ids = list(self.tok(text, add_special_tokens=False)["input_ids"])
        if not ids:
            ids = list(self.tok(" ", add_special_tokens=False)["input_ids"])
        if not cap:
            return ids
        return ids[-cap:] if keep == "tail" else ids[:cap]

    def embed_ids(self, id_lists: list[list[int]], progress: bool = False) -> np.ndarray:
        """-> [n, hidden] float32 L2-normalised embeddings, in input order."""
        import torch
        import torch.nn.functional as F
        n = len(id_lists)
        out = np.empty((n, self.hidden), dtype=np.float32)
        order = sorted(range(n), key=lambda i: -len(id_lists[i]))    # longest first: an OOM shows up at once
        done, i = 0, 0
        with self._lock, torch.inference_mode():
            while i < n:
                width = max(1, len(id_lists[order[i]]))
                idx = order[i:i + max(1, min(self.max_batch, self.batch_tokens // width))]
                h = last_token_states(self.model, [id_lists[j] for j in idx], self.pad_id)
                out[idx] = F.normalize(h.float(), dim=-1).cpu().numpy()
                i += len(idx)
                if progress and (i - done >= 2048 or i == n):
                    print(f"[embed] {i}/{n}", flush=True)
                    done = i
        return out

    def embed_texts(self, texts: list[str], keep: str = "tail", progress: bool = False) -> np.ndarray:
        return self.embed_ids([self.text_ids(t, keep) for t in texts], progress)


class UnslothEmbedder(Embedder):
    """Drop-in for :class:`clm.embedder.Embedder` that runs the encoder in this process.

    Same LRU cache and ``embed(texts) -> (vectors, tokens)`` contract, so ``Engine``
    and ``clm.server.create_app`` use it unchanged.  Extra keyword arguments go to
    :class:`LocalEncoder` (``load_in_4bit``, ``backend``, ``batch_tokens``, ...).
    """

    def __init__(self, model: str = DEFAULT_ENCODER, max_tokens: int = 2048, cache_size: int = 200_000,
                 encoder: LocalEncoder | None = None, **kw):
        super().__init__(url=f"local://{model}", model=model, max_tokens=max_tokens, cache_size=cache_size)
        self.encoder = encoder or LocalEncoder(model, max_tokens=max_tokens, **kw)
        self.batch = 1 << 30              # the encoder batches by tokens itself

    def _fetch(self, texts: list[str]) -> tuple[list[np.ndarray], int]:
        ids = [self.encoder.text_ids(t, "tail", self.max_tokens) for t in texts]
        return list(self.encoder.embed_ids(ids)), sum(len(x) for x in ids)

    def healthy(self) -> bool:
        return True
