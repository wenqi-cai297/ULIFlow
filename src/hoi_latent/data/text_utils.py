from __future__ import annotations

from typing import Optional

__all__ = ["clean_interact_caption"]


def clean_interact_caption(raw_text: Optional[str]) -> Optional[str]:
    """Extract a single natural-language caption from an InterAct ``text.txt``.

    InterAct stores text in the HumanML3D annotation format. Each line is::

        <caption>#<pos-tagged tokens>#<start_sec>#<end_sec>

    and a file may hold several such lines (multiple annotations for one clip).
    The natural-language caption is only the part before the first ``#``; the
    rest is part-of-speech tokens and timing markers meant for tooling, not for a
    text encoder.

    Feeding the raw file to a text encoder is doubly wrong: the POS tokens and
    extra captions dilute the semantics (collapsing text-motion retrieval toward
    chance), and concatenated multi-line files can exceed the encoder's
    positional limit (DistilBERT caps at 512 tokens) and raise at runtime.

    Returns the first non-empty caption, or ``None`` if the input is empty/None.
    """

    if raw_text is None:
        return None
    for line in raw_text.splitlines():
        caption = line.split("#", 1)[0].strip()
        if caption:
            return caption
    return None
