"""Text-level numeric encoding.

`encode_decode.py` turns a single number into a token string. This module
finds numbers *inside free text* and swaps them for their encoded form
(leaving all non-numeric text untouched), and reverses that on model output
so answers can be scored/read as plain digits.
"""

from __future__ import annotations

import re
from decimal import Decimal
from typing import Tuple

from encode_decode import EncodingType, encode_number, decode_number


_ENC_MAP = {
    "flat": EncodingType.FLAT,
    "tree": EncodingType.TREE,
    "bracket": EncodingType.BRACKET,
    "hierarchy": EncodingType.HIERARCHY,
}


def encoding_from_str(name: str) -> EncodingType:
    try:
        return _ENC_MAP[name.lower()]
    except KeyError:
        raise ValueError(f"Unknown encoding {name!r}; choose from {list(_ENC_MAP)}")


# A "number" = optional sign, then either comma-grouped or plain digits, with an
# optional decimal part.
#   (?<![\w.])  - don't start glued to a word char or dot (skip ids, .5, mid-number)
#   (?!\.\d)    - don't stop right before ".<digit>"  => skips versions/IPs (3.2.1)
#   (?![\w])    - don't stop glued to a word char (skip 1990s, 3rd, v2)
# A trailing sentence period is fine: "It is 42.5." still matches "42.5".
_NUMBER_RE = re.compile(
    r"""
    (?<![\w.])
    (?P<sign>[+-])?
    (?P<num>
        \d{1,3}(?:,\d{3})+(?:\.\d+)?     # 1,234 or 1,234.56
      | \d+(?:\.\d+)?                     # 1234 or 1234.56
    )
    (?!\.\d)
    (?![\w])
    """,
    re.VERBOSE,
)


class NumberCodec:
    """Configurable numeric encoder/decoder for whole strings."""

    def __init__(self, encoding: str = "flat", enabled: bool = True,
                 max_int_digits: int = 30, max_frac_digits: int = 15):
        self.enabled = enabled
        self.enc_type = encoding_from_str(encoding)
        self.max_int_digits = max_int_digits
        self.max_frac_digits = max_frac_digits
        self._span_re = self._build_span_re(self.enc_type)

    @classmethod
    def from_config(cls, cfg) -> "NumberCodec":
        e = cfg.encoding
        return cls(encoding=e.type, enabled=e.enabled,
                   max_int_digits=e.max_int_digits, max_frac_digits=e.max_frac_digits)

    # ---------------------------------------------------------- encode side
    def _too_long(self, matched: str) -> bool:
        digits = matched.lstrip("+-").replace(",", "")
        int_part, _, frac_part = digits.partition(".")
        return len(int_part) > self.max_int_digits or len(frac_part) > self.max_frac_digits

    def encode_text(self, text: str) -> Tuple[str, int]:
        """Return (encoded_text, count_of_numbers_encoded)."""
        if not self.enabled or not text:
            return text, 0
        count = 0

        def _repl(m: "re.Match") -> str:
            nonlocal count
            whole = m.group(0)
            if self._too_long(whole):
                return whole
            try:
                enc = encode_number(whole, self.enc_type)
            except Exception:
                return whole
            count += 1
            return enc

        return _NUMBER_RE.sub(_repl, text), count

    def count_numbers(self, text: str) -> int:
        if not text:
            return 0
        return sum(1 for m in _NUMBER_RE.finditer(text) if not self._too_long(m.group(0)))

    # ---------------------------------------------------------- decode side
    @staticmethod
    def _build_span_re(enc: EncodingType) -> re.Pattern:
        if enc == EncodingType.BRACKET:
            group = r"(?:\[[A-Za-z0-9]+:\d*\]|\[\.\])+"
            return re.compile(rf"<NEG>{group}</NEG>|{group}")
        # flat / tree / hierarchy all wrap in <NUM>...</NUM>, optionally in <NEG>
        return re.compile(r"<NEG>\s*<NUM>.*?</NUM>\s*</NEG>|<NUM>.*?</NUM>", re.DOTALL)

    def decode_text(self, text: str) -> str:
        """Replace every encoded number span with its plain decimal string."""
        if not self.enabled or not text:
            return text

        def _repl(m: "re.Match") -> str:
            span = m.group(0)
            try:
                return format(decode_number(span, self.enc_type), "f")
            except Exception:
                return span

        return self._span_re.sub(_repl, text)
