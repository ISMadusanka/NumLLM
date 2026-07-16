from enum import Enum
from typing import List, Tuple, Union
from decimal import Decimal, InvalidOperation
import re


class EncodingType(Enum):
    FLAT = "flat"
    TREE = "tree"
    BRACKET = "bracket"
    HIERARCHY = "hierarchy"


# -------------------------------------------------------
# Hierarchy names (short scale) — extendable to arbitrary size
# -------------------------------------------------------

HIERARCHY_NAMES = [
    "U",    # 10^0   units
    "K",    # 10^3   thousand
    "M",    # 10^6   million
    "B",    # 10^9   billion
    "T",    # 10^12  trillion
    "Qa",   # 10^15  quadrillion
    "Qi",   # 10^18  quintillion
    "Sx",   # 10^21  sextillion
    "Sp",   # 10^24  septillion
    "Oc",   # 10^27  octillion
    "No",   # 10^30  nonillion
    "Dc",   # 10^33  decillion
    "Ud",   # 10^36  undecillion
    "Dd",   # 10^39  duodecillion
    "Td",   # 10^42  tredecillion
    "Qad",  # 10^45  quattuordecillion
    "Qid",  # 10^48  quindecillion
    "Sxd",  # 10^51  sexdecillion
    "Spd",  # 10^54  septendecillion
    "Ocd",  # 10^57  octodecillion
    "Nod",  # 10^60  novemdecillion
    "Vg",   # 10^63  vigintillion
]


def magnitude_label(idx: int) -> str:
    """
    Label for a group at magnitude index `idx`
    (0 = units, 1 = thousands, 2 = millions, ...).

    Falls back to a generic 'G<idx>' label once the named hierarchy
    is exhausted, so arbitrarily large numbers are still supported.
    """
    if idx < len(HIERARCHY_NAMES):
        return HIERARCHY_NAMES[idx]
    return f"G{idx}"


# -------------------------------------------------------
# Special tokens — generated dynamically since the label set
# is now open-ended (large numbers / fraction groups)
# -------------------------------------------------------

def generate_special_tokens(max_group_index: int = 25, max_frac_groups: int = 10) -> List[str]:
    """
    Build a tokenizer-vocab-style list of every special token this
    scheme can emit, up to `max_group_index` integer groups (i.e. up
    to 3*max_group_index digits) and `max_frac_groups` fraction groups.
    """

    tokens = [
        "<NUM>", "</NUM>",
        "<NEG>", "</NEG>",
        "<FRAC>", "</FRAC>",
        "[.]", "]",
    ]

    for idx in range(max_group_index):
        label = magnitude_label(idx)
        tokens += [f"<{label}>", f"</{label}>", f"[{label}:"]

    for i in range(1, max_frac_groups + 1):
        label = f"F{i}"
        tokens += [f"<{label}>", f"</{label}>", f"[{label}:"]

    for i in range(1, max_group_index + 1):
        tokens += [f"<L{i}>", f"</L{i}>"]

    seen = set()
    unique_tokens = []
    for t in tokens:
        if t not in seen:
            seen.add(t)
            unique_tokens.append(t)

    return unique_tokens


SPECIAL_TOKENS = generate_special_tokens()


# -------------------------------------------------------
# Parsing utilities — handle int / float / str / Decimal,
# negative numbers, and very large / precise numbers
# -------------------------------------------------------

def to_plain_string(number: Union[int, float, str, Decimal]) -> str:
    """
    Convert any supported numeric input into a plain (non-scientific)
    decimal string, e.g. Decimal('1.2E+5') -> '120000'.

    Uses Decimal (not float math) so large integers keep full precision.
    Note: if you pass a Python `float`, you inherit float's ~15-17
    significant-digit precision limit — pass int, Decimal, or str for
    exact very large / very precise values.
    """

    if isinstance(number, Decimal):
        d = number
    elif isinstance(number, int):
        d = Decimal(number)
    elif isinstance(number, float):
        d = Decimal(str(number))  # str() avoids float's full binary expansion
    elif isinstance(number, str):
        s = number.strip().replace(",", "")
        try:
            d = Decimal(s)
        except InvalidOperation:
            raise ValueError(f"Invalid number string: {number!r}")
    else:
        raise TypeError(f"Unsupported number type: {type(number)}")

    return format(d, "f")


def parse_number(number: Union[int, float, str, Decimal]) -> Tuple[str, str, str]:
    """
    Parse a number into (sign, integer_part, fractional_part) strings.

    Example:
        -1234.500 -> ("-", "1234", "5")
        42        -> ("",  "42",   "")
    """

    s = to_plain_string(number)

    sign = ""
    if s.startswith("-"):
        sign = "-"
        s = s[1:]
    elif s.startswith("+"):
        s = s[1:]

    if "." in s:
        int_part, frac_part = s.split(".", 1)
    else:
        int_part, frac_part = s, ""

    int_part = int_part.lstrip("0") or "0"
    frac_part = frac_part.rstrip("0")  # drop insignificant trailing zeros

    if int_part == "0" and frac_part == "":
        sign = ""  # canonicalize "-0" -> "0"

    return sign, int_part, frac_part


# -------------------------------------------------------
# Grouping utilities
# -------------------------------------------------------

def split_into_groups(digits: str) -> List[str]:
    """
    Split a digit string into 3-digit groups from the right (integer part).

    Example:
        '483721956' -> ['483', '721', '956']
        '9832'      -> ['9', '832']
    """

    s = digits
    groups = []

    while len(s) > 3:
        groups.insert(0, s[-3:])
        s = s[:-3]

    groups.insert(0, s)

    return groups


def split_fraction_into_groups(digits: str) -> List[str]:
    """
    Split a fractional digit string into 3-digit groups from the LEFT
    (significance decreases left-to-right after the decimal point).

    Example: '4837' -> ['483', '7']
    """

    groups = []
    s = digits

    while len(s) > 3:
        groups.append(s[:3])
        s = s[3:]

    if s:
        groups.append(s)

    return groups


def group_labels(groups: List[str]) -> List[Tuple[str, str]]:
    """
    Assign magnitude labels to integer-part groups, most-significant first.
    Supports arbitrarily large numbers via the 'G<n>' fallback.
    """

    n = len(groups)
    return [(magnitude_label(n - 1 - i), value) for i, value in enumerate(groups)]


def fraction_labels(groups: List[str]) -> List[Tuple[str, str]]:
    """Assign sequential F1, F2, ... labels to fractional groups, left to right."""
    return [(f"F{i}", value) for i, value in enumerate(groups, start=1)]


# -------------------------------------------------------
# Encoders
# -------------------------------------------------------

def encode_flat(number):

    sign, int_str, frac_str = parse_number(number)
    int_items = group_labels(split_into_groups(int_str))
    frac_items = fraction_labels(split_fraction_into_groups(frac_str)) if frac_str else []

    text = "<NUM>"
    for label, value in int_items:
        text += f"<{label}>{value}</{label}>"

    if frac_items:
        text += "<FRAC>"
        for label, value in frac_items:
            text += f"<{label}>{value}</{label}>"
        text += "</FRAC>"

    text += "</NUM>"

    return f"<NEG>{text}</NEG>" if sign == "-" else text


def encode_bracket(number):

    sign, int_str, frac_str = parse_number(number)
    int_items = group_labels(split_into_groups(int_str))
    frac_items = fraction_labels(split_fraction_into_groups(frac_str)) if frac_str else []

    text = "".join(f"[{label}:{value}]" for label, value in int_items)

    if frac_items:
        text += "[.]" + "".join(f"[{label}:{value}]" for label, value in frac_items)

    return f"<NEG>{text}</NEG>" if sign == "-" else text


def encode_hierarchy(number):

    sign, int_str, frac_str = parse_number(number)
    int_items = group_labels(split_into_groups(int_str))
    frac_items = fraction_labels(split_fraction_into_groups(frac_str)) if frac_str else []

    text = "<NUM>"
    for i, (_, value) in enumerate(int_items, start=1):
        text += f"<L{i}>{value}</L{i}>"

    if frac_items:
        text += "<FRAC>"
        for i, (_, value) in enumerate(frac_items, start=1):
            text += f"<L{i}>{value}</L{i}>"
        text += "</FRAC>"

    text += "</NUM>"

    return f"<NEG>{text}</NEG>" if sign == "-" else text


def encode_tree(number):

    sign, int_str, frac_str = parse_number(number)
    int_items = group_labels(split_into_groups(int_str))
    frac_items = fraction_labels(split_fraction_into_groups(frac_str)) if frac_str else []

    if not int_items:
        body = "<NUM></NUM>"
    else:
        result = "<NUM>\n"
        indent = ""

        for label, value in int_items:
            result += indent + f"<{label}>{value}\n"
            indent += "    "

        for label, _ in reversed(int_items):
            indent = indent[:-4]
            result += indent + f"</{label}>\n"

        if frac_items:
            result += "<FRAC>\n"
            indent = "    "

            for label, value in frac_items:
                result += indent + f"<{label}>{value}\n"
                indent += "    "

            for label, _ in reversed(frac_items):
                indent = indent[:-4]
                result += indent + f"</{label}>\n"

            result += "</FRAC>\n"

        result += "</NUM>"
        body = result

    return f"<NEG>\n{body}\n</NEG>" if sign == "-" else body


# -------------------------------------------------------
# Main encode API
# -------------------------------------------------------

def encode_number(number, encoding: EncodingType):

    if encoding == EncodingType.FLAT:
        return encode_flat(number)
    elif encoding == EncodingType.TREE:
        return encode_tree(number)
    elif encoding == EncodingType.BRACKET:
        return encode_bracket(number)
    elif encoding == EncodingType.HIERARCHY:
        return encode_hierarchy(number)

    raise ValueError("Unknown encoding")


# =========================================================
# DECODERS
# =========================================================

def _extract_neg(text: str) -> Tuple[bool, str]:
    """Strip an outer <NEG>...</NEG> wrapper, if present."""
    t = text.strip()
    if t.startswith("<NEG>") and t.endswith("</NEG>"):
        return True, t[len("<NEG>"):-len("</NEG>")].strip()
    return False, t


def _assemble_value(is_negative: bool, int_groups: List[str], frac_groups: List[str]) -> Decimal:
    """Reassemble digit groups (already in left-to-right order) into a Decimal."""

    int_str = "".join(int_groups) or "0"
    frac_str = "".join(frac_groups)

    s = int_str + ("." + frac_str if frac_str else "")
    value = Decimal(s)

    return -value if is_negative else value


def decode_flat(text: str) -> Decimal:

    is_neg, body = _extract_neg(text)

    m = re.fullmatch(r"<NUM>(.*)</NUM>", body, re.DOTALL)
    if not m:
        raise ValueError(f"Malformed flat encoding: {text!r}")

    inner = m.group(1)
    frac_m = re.search(r"<FRAC>(.*)</FRAC>", inner, re.DOTALL)
    int_part = inner[:frac_m.start()] if frac_m else inner
    frac_part = frac_m.group(1) if frac_m else ""

    int_groups = [v for _, v in re.findall(r"<([A-Za-z0-9]+)>(\d*)</\1>", int_part)]
    frac_groups = [v for _, v in re.findall(r"<(F\d+)>(\d*)</\1>", frac_part)]

    return _assemble_value(is_neg, int_groups, frac_groups)


def decode_bracket(text: str) -> Decimal:

    is_neg, body = _extract_neg(text)

    parts = body.split("[.]", 1)
    int_part = parts[0]
    frac_part = parts[1] if len(parts) > 1 else ""

    int_groups = [v for _, v in re.findall(r"\[([A-Za-z0-9]+):(\d*)\]", int_part)]
    frac_groups = [v for _, v in re.findall(r"\[(F\d+):(\d*)\]", frac_part)]

    return _assemble_value(is_neg, int_groups, frac_groups)


def decode_hierarchy(text: str) -> Decimal:

    is_neg, body = _extract_neg(text)

    m = re.fullmatch(r"<NUM>(.*)</NUM>", body, re.DOTALL)
    if not m:
        raise ValueError(f"Malformed hierarchy encoding: {text!r}")

    inner = m.group(1)
    frac_m = re.search(r"<FRAC>(.*)</FRAC>", inner, re.DOTALL)
    int_part = inner[:frac_m.start()] if frac_m else inner
    frac_part = frac_m.group(1) if frac_m else ""

    int_groups = [v for _, v in re.findall(r"<(L\d+)>(\d*)</\1>", int_part)]
    frac_groups = [v for _, v in re.findall(r"<(L\d+)>(\d*)</\1>", frac_part)]

    return _assemble_value(is_neg, int_groups, frac_groups)


def decode_tree(text: str) -> Decimal:

    t = text.strip()
    is_neg = False
    if t.startswith("<NEG>") and t.endswith("</NEG>"):
        is_neg = True
        t = t[len("<NEG>"):-len("</NEG>")].strip()

    frac_m = re.search(r"<FRAC>(.*)</FRAC>", t, re.DOTALL)
    int_part = t[:frac_m.start()] if frac_m else t
    frac_part = frac_m.group(1) if frac_m else ""

    # opening tags only, e.g. "<M>483" — closing tags start with "</" and won't match
    int_groups = [v for _, v in re.findall(r"<([A-Za-z0-9]+)>(\d+)", int_part)]
    frac_groups = [v for _, v in re.findall(r"<(F\d+)>(\d+)", frac_part)]

    return _assemble_value(is_neg, int_groups, frac_groups)


# -------------------------------------------------------
# Main decode API
# -------------------------------------------------------

def decode_number(text: str, encoding: EncodingType) -> Decimal:
    """
    Decode a previously-encoded string back into a Decimal (chosen over
    int/float so precision is preserved for very large/precise numbers).
    """

    if encoding == EncodingType.FLAT:
        return decode_flat(text)
    elif encoding == EncodingType.TREE:
        return decode_tree(text)
    elif encoding == EncodingType.BRACKET:
        return decode_bracket(text)
    elif encoding == EncodingType.HIERARCHY:
        return decode_hierarchy(text)

    raise ValueError("Unknown encoding")


# -------------------------------------------------------
# Example
# -------------------------------------------------------

if __name__ == "__main__":

    examples = [
        1000000,
        -1000000,
        3.14159,
        -42.5,
        0,
        123456789012345678901234567890,          # 30-digit int
        "-987654321.123456",
        Decimal("1000000000000000000000000000000.25"),  # beyond quintillion + fraction
    ]

    for n in examples:
        print("#" * 70)
        print(f"ORIGINAL: {n}")
        print()

        for enc in EncodingType:
            encoded = encode_number(n, enc)
            decoded = decode_number(encoded, enc)
            expected = Decimal(to_plain_string(n))

            print("=" * 70)
            print(enc.value.upper())
            print()
            print(encoded)
            print()
            print(f"DECODED : {decoded}")
            print(f"MATCH   : {decoded == expected}")
            print()