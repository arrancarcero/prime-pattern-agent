"""
pss_demo.py
===========
RSA-PSS signatures, built by hand from RFC 8017 (EMSA-PSS, MGF1 with SHA-256,
32-byte salt), then checked three ways:

  1. Independent check: every signature made here is verified by the real
     `cryptography` library (OpenSSL). If our hand-built PSS were wrong, it would fail.
  2. Salt quality: the salts go through the randomness tuple. A reused salt and a
     predictable-but-uniform generator are tested too, to show what the tuple can
     and cannot catch.
  3. Key strength: two 1024-bit keys go to the factoring tool tuple. One is badly
     generated (its primes are close together); a tool factors it, and a signature
     is forged that the real library accepts. The other is properly generated and
     no tool cracks it within budget.

Run from the repository folder:  python examples/pss_demo.py
Needs: pip install cryptography
"""
import hashlib
import math
import os
import random
import secrets
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from ppa.truth.randomness_tuple import RANDOMNESS_TUPLE, sieve_of_randomness  # noqa: E402
from ppa.truth.tool_tuple import TOOL_TUPLE, fingerprint, verify_tools          # noqa: E402

H = hashlib.sha256
HLEN = 32
SLEN = 32
E = 65537


# ---------------------------------------------------------------- keys
def probable_prime(n, rounds=40):
    if n < 2:
        return False
    for p in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37):
        if n % p == 0:
            return n == p
    d, s = n - 1, 0
    while d % 2 == 0:
        d //= 2; s += 1
    for _ in range(rounds):
        a = secrets.randbelow(n - 3) + 2
        x = pow(a, d, n)
        if x in (1, n - 1):
            continue
        for _ in range(s - 1):
            x = x * x % n
            if x == n - 1:
                break
        else:
            return False
    return True


def random_prime(bits):
    while True:
        c = secrets.randbits(bits) | (1 << (bits - 1)) | 1
        if probable_prime(c) and math.gcd(E, c - 1) == 1:
            return c


def next_prime_after(n):
    n |= 1
    while not (probable_prime(n) and math.gcd(E, n - 1) == 1):
        n += 2
    return n


def make_key(p, q):
    n = p * q
    d = pow(E, -1, (p - 1) * (q - 1))
    return {"n": n, "e": E, "d": d, "p": p, "q": q}


def strong_key(bits=1024):
    while True:
        p, q = random_prime(bits // 2), random_prime(bits // 2)
        if p != q and (p * q).bit_length() == bits:
            return make_key(p, q)


def weak_key_close_primes(bits=1024):
    """A badly generated key: q is the next prime after p + 2^20.
    Real keys like this have been found in the wild (Fermat's method breaks them)."""
    while True:
        p = random_prime(bits // 2)
        q = next_prime_after(p + (1 << 20))
        if (p * q).bit_length() == bits:
            return make_key(p, q)


# ---------------------------------------------------------------- PSS (RFC 8017, section 9.1)
def mgf1(seed, length):
    out = b""
    counter = 0
    while len(out) < length:
        out += H(seed + counter.to_bytes(4, "big")).digest()
        counter += 1
    return out[:length]


def emsa_pss_encode(message, em_bits, salt):
    em_len = (em_bits + 7) // 8
    m_hash = H(message).digest()
    m_prime = b"\x00" * 8 + m_hash + salt
    h = H(m_prime).digest()
    ps = b"\x00" * (em_len - SLEN - HLEN - 2)
    db = ps + b"\x01" + salt
    mask = mgf1(h, em_len - HLEN - 1)
    masked_db = bytes(a ^ b for a, b in zip(db, mask))       # the XOR mirror
    zero_bits = 8 * em_len - em_bits
    masked_db = bytes([masked_db[0] & (0xFF >> zero_bits)]) + masked_db[1:]
    return masked_db + h + b"\xbc"


def emsa_pss_verify(message, em, em_bits):
    em_len = (em_bits + 7) // 8
    if len(em) != em_len or em[-1] != 0xBC:
        return False, None
    masked_db, h = em[: em_len - HLEN - 1], em[em_len - HLEN - 1: -1]
    zero_bits = 8 * em_len - em_bits
    if masked_db[0] & ~(0xFF >> zero_bits) & 0xFF:
        return False, None
    mask = mgf1(h, em_len - HLEN - 1)
    db = bytes(a ^ b for a, b in zip(masked_db, mask))        # same mirror undoes it
    db = bytes([db[0] & (0xFF >> zero_bits)]) + db[1:]
    ps_len = em_len - HLEN - SLEN - 2
    if any(db[:ps_len]) or db[ps_len] != 0x01:
        return False, None
    salt = db[-SLEN:]                                           # verification recovers the salt
    ok = H(b"\x00" * 8 + H(message).digest() + salt).digest() == h
    return ok, salt


def sign(key, message, salt=None):
    salt = os.urandom(SLEN) if salt is None else salt
    em_bits = key["n"].bit_length() - 1
    m = int.from_bytes(emsa_pss_encode(message, em_bits, salt), "big")
    s = pow(m, key["d"], key["n"])                              # the e-th root: needs p and q
    return s.to_bytes((key["n"].bit_length() + 7) // 8, "big"), salt


def verify(pub, message, signature):
    em_bits = pub["n"].bit_length() - 1
    m = pow(int.from_bytes(signature, "big"), pub["e"], pub["n"])  # raise to e: anyone can
    em_len = (em_bits + 7) // 8
    if m.bit_length() > 8 * em_len:
        return False, None
    return emsa_pss_verify(message, m.to_bytes(em_len, "big"), em_bits)


# ---------------------------------------------------------------- independent check
def library_accepts(pub, message, signature):
    from cryptography.exceptions import InvalidSignature
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.asymmetric import padding, rsa
    key = rsa.RSAPublicNumbers(pub["e"], pub["n"]).public_key()
    try:
        key.verify(signature, message,
                   padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=SLEN), hashes.SHA256())
        return True
    except InvalidSignature:
        return False


def salt_bits_to_signs(salts):
    bits = np.unpackbits(np.frombuffer(b"".join(salts), dtype=np.uint8))
    return bits.astype(np.int8) * 2 - 1


def tuple_verdict(signs):
    z = sieve_of_randomness(signs)
    bad = [r.name for r in RANDOMNESS_TUPLE if abs(z[r.name]) > 4]
    return ("indistinguishable from random" if not bad else "NOT random: " + ", ".join(bad)), z


# ---------------------------------------------------------------- the demo
def main():
    msg = b"Pay Arran 10 dollars"
    print("1) Hand-built RSA-PSS, checked by the real library")
    key = strong_key()
    pub = {"n": key["n"], "e": key["e"]}
    sigs = [sign(key, msg) for _ in range(3)]
    for i, (s, salt) in enumerate(sigs, 1):
        ok, rec = verify(pub, msg, s)
        print(f"   signature {i}: starts {s.hex()[:16]}...  ours verifies: {ok}  salt recovered: {rec == salt}"
              f"  OpenSSL accepts: {library_accepts(pub, msg, s)}")
    print(f"   same message, 3 different signatures: {len({s for s, _ in sigs}) == 3}")
    tampered = b"Pay Arran 1000 dollars"
    print(f"   tampered message accepted? ours: {verify(pub, tampered, sigs[0][0])[0]}  "
          f"OpenSSL: {library_accepts(pub, tampered, sigs[0][0])}")

    print("\n2) Salt quality through the randomness tuple (2,000 salts = 512,000 bits each)")
    good = [os.urandom(SLEN) for _ in range(2000)]
    reused = [b"\x5a" * SLEN] * 2000
    mt = random.Random(42)
    predictable = [mt.getrandbits(256).to_bytes(32, "big") for _ in range(2000)]
    for label, salts in (("os.urandom (correct)", good), ("one salt reused", reused),
                         ("Mersenne Twister, seed 42", predictable)):
        verdict, _ = tuple_verdict(salt_bits_to_signs(salts))
        print(f"   {label:28} {verdict}")
    s1, _ = sign(key, msg, reused[0]); s2, _ = sign(key, msg, reused[0])
    print(f"   with a reused salt, two signatures identical: {s1 == s2}  (PSS has become deterministic)")
    print("   Mersenne Twister passes every row, yet anyone who learns the seed predicts every salt:")
    print("   passing the tuple is necessary, not sufficient. Salts must come from os.urandom or secrets.")

    print(f"\n3) Key strength vs the factoring tool tuple (fingerprint {fingerprint()})")
    if not all(ok and nul for _, ok, nul in verify_tools()):
        print("   tools failed calibration; stopping"); return
    for label, k in (("badly generated (close primes)", weak_key_close_primes()), ("properly generated", strong_key())):
        N = k["n"]; cracked = None; t0 = time.time()
        for t in TOOL_TUPLE:
            f, steps = t.fn(N, t.budget)
            if f is not None and 1 < f < N and N % f == 0:
                cracked = (t.name, f, steps); break
        dt = time.time() - t0
        if cracked:
            name, f, steps = cracked
            p, q = f, N // f
            forged_key = make_key(p, q)
            forged, _ = sign(forged_key, b"Pay the attacker 1,000,000 dollars")
            pub_k = {"n": N, "e": E}
            print(f"   {label}: cracked by {name} in {steps:,} step(s), {dt:.2f}s")
            print(f"      forged signature accepted by OpenSSL: "
                  f"{library_accepts(pub_k, b'Pay the attacker 1,000,000 dollars', forged)}")
        else:
            print(f"   {label}: no tool cracked it within budget ({dt:.1f}s)")


if __name__ == "__main__":
    main()
