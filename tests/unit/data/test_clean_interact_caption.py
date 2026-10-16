from __future__ import annotations

from hoi_latent.data.text_utils import clean_interact_caption


def test_strips_pos_tags_and_timing_from_single_line() -> None:
    raw = "A person picks up a box.#A/DET person/NOUN pick/VERB up/ADP box/NOUN#0.0#0.0"
    assert clean_interact_caption(raw) == "A person picks up a box."


def test_takes_first_caption_from_multiline_file() -> None:
    # InterAct text.txt holds several annotations; only one caption should be used.
    raw = (
        "A person lifts an airplane.#A/DET person/NOUN lift/VERB#0.0#0.0\n"
        "Someone raises a model plane with one hand.#...#0.0#0.0\n"
        "The subject picks up the toy.#...#0.0#0.0\n"
    )
    assert clean_interact_caption(raw) == "A person lifts an airplane."


def test_already_clean_caption_is_idempotent() -> None:
    assert clean_interact_caption("sit on the chair") == "sit on the chair"
    assert clean_interact_caption(clean_interact_caption("sit on the chair")) == "sit on the chair"


def test_skips_leading_blank_lines() -> None:
    assert clean_interact_caption("\n\n  \nthrow the ball#...#0#0") == "throw the ball"


def test_none_and_empty_return_none() -> None:
    assert clean_interact_caption(None) is None
    assert clean_interact_caption("") is None
    assert clean_interact_caption("\n  \n") is None


def test_reduces_overlong_multiline_text_below_encoder_limit() -> None:
    # A raw 3-line file can tokenize past DistilBERT's 512 cap; the caption is short.
    raw = "\n".join(f"caption number {i} describing the motion.#" + "tok/NOUN " * 200 + "#0.0#0.0" for i in range(3))
    cleaned = clean_interact_caption(raw)
    assert cleaned == "caption number 0 describing the motion."
    assert len(cleaned.split()) < 20
