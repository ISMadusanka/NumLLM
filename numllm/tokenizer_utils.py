"""Build & persist the extended tokenizer.

The numeric scheme in encode_decode.py introduces new tokens (derived from
HIERARCHY_NAMES). We add them as `additional_special_tokens` so each is a
single, unsplittable id, then every step loads the *same* saved tokenizer.
"""

from __future__ import annotations

import os
from typing import List

from encode_decode import magnitude_label


def build_special_tokens(cfg) -> List[str]:
    """Only the special tokens the *configured* encoding actually emits.

    (Adding every scheme's tokens would, for FLAT, needlessly add a bare ']'
    and remap every ']' in normal text.)
    """
    enc = cfg.encoding.type.lower()
    mgi = cfg.encoding.max_group_index
    mfg = cfg.encoding.max_frac_groups
    mag_labels = [magnitude_label(i) for i in range(mgi)]   # U, K, M, ..., G<n>
    toks: List[str] = []

    if enc in ("flat", "tree"):
        toks += ["<NUM>", "</NUM>", "<NEG>", "</NEG>", "<FRAC>", "</FRAC>"]
        for lab in mag_labels:
            toks += [f"<{lab}>", f"</{lab}>"]
        for i in range(1, mfg + 1):
            toks += [f"<F{i}>", f"</F{i}>"]
    elif enc == "hierarchy":
        toks += ["<NUM>", "</NUM>", "<NEG>", "</NEG>", "<FRAC>", "</FRAC>"]
        for i in range(1, max(mgi, mfg) + 1):
            toks += [f"<L{i}>", f"</L{i}>"]
    elif enc == "bracket":
        toks += ["<NEG>", "</NEG>", "[.]", "]"]
        for lab in mag_labels:
            toks += [f"[{lab}:"]
        for i in range(1, mfg + 1):
            toks += [f"[F{i}:"]
    else:
        raise ValueError(f"Unknown encoding type: {enc!r}")

    seen, unique = set(), []
    for t in toks:
        if t not in seen:
            seen.add(t)
            unique.append(t)
    return unique


def get_extended_tokenizer(cfg, save: bool = True):
    """Load the persisted extended tokenizer if present; otherwise build it
    from the base model, add the numeric special tokens, and save it.

    Returns (tokenizer, newly_added_count). newly_added_count is 0 when the
    tokenizer was loaded from disk (tokens already present).
    """
    from transformers import AutoTokenizer

    tok_dir = cfg.paths.tokenizer
    if os.path.isdir(tok_dir) and os.listdir(tok_dir):
        tok = AutoTokenizer.from_pretrained(tok_dir, trust_remote_code=cfg.model.trust_remote_code)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        return tok, 0

    tok = AutoTokenizer.from_pretrained(
        cfg.model.base_model, trust_remote_code=cfg.model.trust_remote_code
    )
    added = 0
    if cfg.encoding.enabled:
        specials = build_special_tokens(cfg)
        added = tok.add_special_tokens({"additional_special_tokens": specials})
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    if save:
        os.makedirs(tok_dir, exist_ok=True)
        tok.save_pretrained(tok_dir)
    return tok, added


def load_plain_tokenizer(cfg):
    """The base model's tokenizer with no numeric tokens (for base-model eval)."""
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(
        cfg.model.base_model, trust_remote_code=cfg.model.trust_remote_code
    )
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok
