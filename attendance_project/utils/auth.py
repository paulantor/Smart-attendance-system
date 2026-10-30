"""
utils/auth.py
==============
Password hashing/verification using Python's stdlib hashlib.pbkdf2_hmac
(no extra dependency like bcrypt/passlib needed). Stored format:

    "<salt_hex>:<hash_hex>"

Never store or compare plain-text passwords anywhere else in the codebase.
"""

from __future__ import annotations

import hashlib
import hmac
import secrets

_ITERATIONS = 200_000


def hash_password(plain_password: str) -> str:
    salt = secrets.token_hex(16)
    digest = hashlib.pbkdf2_hmac("sha256", plain_password.encode("utf-8"), bytes.fromhex(salt), _ITERATIONS)
    return f"{salt}:{digest.hex()}"


def verify_password(plain_password: str, stored_hash: str) -> bool:
    try:
        salt, digest_hex = stored_hash.split(":")
    except (ValueError, AttributeError):
        return False

    new_digest = hashlib.pbkdf2_hmac("sha256", plain_password.encode("utf-8"), bytes.fromhex(salt), _ITERATIONS)
    return hmac.compare_digest(new_digest.hex(), digest_hex)