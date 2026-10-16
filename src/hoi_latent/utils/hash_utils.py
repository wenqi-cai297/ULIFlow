from __future__ import annotations

import hashlib


def _digest(text: str, seed: int = 0) -> bytes:
    payload = f"{seed}:{text}".encode("utf-8")
    return hashlib.blake2b(payload, digest_size=8).digest()


def stable_hash_int(text: str, seed: int = 0) -> int:
    return int.from_bytes(_digest(text, seed=seed), byteorder="big", signed=False)


def stable_float01(text: str, seed: int = 0) -> float:
    return stable_hash_int(text, seed=seed) / float(2**64 - 1)
