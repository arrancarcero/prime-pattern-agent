"""Factoring techniques, each built to catch a different structure in the hidden factors.

  trial      a small factor
  fermat     two factors close together          (the mirror N = a^2 - b^2)
  rho        any factor up to ~1e12, birthday     (Pollard rho: a walk mod p that loops)
  p_minus_1  p - 1 built from small primes        (Pollard p-1: a loop around a circle of size p-1)
  ecm        some elliptic curve has smooth size  (Lenstra: size p + 1 - a_p, inside the Hasse range)

Each takes (N, budget) and returns (factor or None, steps used).
"""
import math
import random

# ---------- primes ----------
def is_prime(n):
    if n < 2:
        return False
    for p in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37):
        if n % p == 0:
            return n == p
    d, s = n - 1, 0
    while d % 2 == 0:
        d //= 2; s += 1
    for a in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37):
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


def small_primes(n):
    s = bytearray([1]) * (n + 1); s[0] = s[1] = 0
    for i in range(2, int(n ** .5) + 1):
        if s[i]:
            s[i * i::i] = bytearray(len(s[i * i::i]))
    return [i for i in range(n + 1) if s[i]]

PRIMES = small_primes(200_000)

# ---------- techniques: each returns (factor or None, steps used) ----------
def trial(N, budget):
    for i, p in enumerate(PRIMES[:budget]):
        if N % p == 0 and p != N:
            return p, i + 1
    return None, budget


def fermat(N, budget):
    a = math.isqrt(N)
    if a * a < N:
        a += 1
    for i in range(budget):
        b2 = a * a - N
        b = math.isqrt(b2)
        if b * b == b2 and a - b > 1:
            return a - b, i + 1
        a += 1
    return None, budget


def moire(N, budget):
    """Moire beat. Grids with spacings p and q beat at frequency 1/p - 1/q = (q - p)/N,
    so the beat period is N / d with d = q - p. Knowing d and p*q = N:
        p = (sqrt(d^2 + 4N) - d) / 2
    Scan the beat gap d = 0, 2, 4, ... (p, q odd, so d is even) until d^2 + 4N is a
    perfect square. This is Fermat's method in another costume: N = a^2 - b^2 with b = d/2.
    Inverting the grids (dark <-> light) changes only the average brightness, not the
    beat, so the inverted moire gives the same d."""
    if N % 2 == 0:
        return 2, 1
    for i in range(budget):
        d = 2 * i
        s2 = d * d + 4 * N
        s = math.isqrt(s2)
        if s * s == s2:
            p = (s - d) // 2
            if 1 < p < N:
                return p, i + 1
    return None, budget


def zone_plate(N, budget):
    """Zone plate (Fresnel lens): brightness cos(pi x^2 / N), rings getting finer outward.
    It repeats with period N. Sample it only every s-th point: the samples follow
    cos(pi s^2 m^2 / N), which repeats after N / gcd(s^2, N) samples. If s shares a
    factor with N, the sampled plate collapses into a smaller plate (fewer rings per
    cycle): the factor shows up as aliasing. Trying strides s = 2, 3, 5, 7, ... is trial
    division seen through a lens, so it catches exactly what trial division catches."""
    for i, s in enumerate(PRIMES[:budget]):
        period = N // math.gcd(s * s, N)        # length of one cycle of the sampled plate
        if period < N and s != N:
            return math.gcd(s, N), i + 1
    return None, budget


def jacobi_all(N):
    """Jacobi symbol (a / N) for every a = 0..N-1, odd N, by quadratic reciprocity:
    flip and swap only, so it never needs the factors of N."""
    import numpy as np
    a = np.arange(N, dtype=np.int64)
    n = np.full_like(a, N)
    r = np.ones_like(a)
    while True:
        act = a != 0
        if not act.any():
            break
        while True:                              # take out factors of 2
            ev = act & (a % 2 == 0)
            if not ev.any():
                break
            a[ev] //= 2
            m8 = n[ev] % 8
            r[ev] *= np.where((m8 == 3) | (m8 == 5), -1, 1)
        flip = act & (a % 4 == 3) & (n % 4 == 3)  # reciprocity sign flip
        r[flip] *= -1
        a_new = np.where(act, n % np.where(a == 0, 1, a), a)
        n = np.where(act, a, n)
        a = a_new
    return np.where(n == 1, r, 0)


def sign_fft(N, budget):
    """Sign flips + FFT. The Jacobi sign pattern (a / N) is two hidden sign patterns
    (periods p and q) multiplied together. Its FFT is flat, every frequency exactly
    sqrt(N) bright, except dark at multiples of p and q (a Gauss-sum fact for primitive
    characters). The first dark frequency k gives gcd(k, N) = a factor.
    Deciphers any odd N with distinct prime factors, but the pattern is N long, so it
    costs ~N steps (trial division needs ~sqrt N). Shor's quantum algorithm runs the same
    'hide the factors in a period, then Fourier transform' move on about log N qubits."""
    import numpy as np
    if N % 2 == 0:
        return 2, 1
    r = math.isqrt(N)
    if r * r == N:                               # p^2: the pattern is not 'primitive'
        return (r, 1) if r > 1 else (None, 1)
    if N > budget:
        return None, 0                           # too long to build within budget
    mag = np.abs(np.fft.fft(jacobi_all(N).astype(np.float64)))
    dark = np.nonzero(mag[1:] < 1e-6 * math.sqrt(N))[0] + 1
    for k in dark:
        g = math.gcd(int(k), N)
        if 1 < g < N:
            return g, N
    return None, N


def rho(N, budget, seed=1):
    if N % 2 == 0:
        return 2, 1
    rng = random.Random(seed)
    x = y = rng.randrange(2, N); c = rng.randrange(1, N); g = 1; steps = 0
    f = lambda v: (v * v + c) % N
    while g == 1 and steps < budget:
        x = f(x); y = f(f(y)); g = math.gcd(abs(x - y), N); steps += 1
    return (g, steps) if 1 < g < N else (None, steps)


def p_minus_1(N, budget):
    a = 2; steps = 0
    for p in PRIMES:
        if steps >= budget:
            break
        pk = p
        while pk * p <= 10 ** 7:
            pk *= p
        a = pow(a, pk, N); steps += 1
        if steps % 50 == 0:
            g = math.gcd(a - 1, N)
            if 1 < g < N:
                return g, steps
            if g == N:
                return None, steps
    g = math.gcd(a - 1, N)
    return (g, steps) if 1 < g < N else (None, steps)


class FoundFactor(Exception):
    def __init__(self, g):
        self.g = g


def ecm(N, budget, curves=40, seed=7):
    """Lenstra ECM with affine points; a failed inverse mod N reveals a factor."""
    rng = random.Random(seed); steps = 0
    per_curve = max(1, budget // curves)

    def inv(v):
        g = math.gcd(v % N, N)
        if g != 1:
            raise FoundFactor(g)
        return pow(v, -1, N)

    def add(P, Q, a):
        if P is None: return Q
        if Q is None: return P
        x1, y1 = P; x2, y2 = Q
        if x1 == x2 and (y1 + y2) % N == 0:
            return None
        if P == Q:
            l = (3 * x1 * x1 + a) * inv(2 * y1) % N
        else:
            l = (y2 - y1) * inv(x2 - x1) % N
        x3 = (l * l - x1 - x2) % N
        return x3, (l * (x1 - x3) - y1) % N

    def mul(k, P, a):
        R = None
        while k:
            if k & 1: R = add(R, P, a)
            P = add(P, P, a); k >>= 1
        return R

    for _ in range(curves):
        x, y, a = rng.randrange(N), rng.randrange(N), rng.randrange(N)
        P = (x, y)                      # b is implied: y^2 = x^3 + a x + b
        try:
            for p in PRIMES[:per_curve]:
                pk = p
                while pk * p <= 2000:
                    pk *= p
                P = mul(pk, P, a); steps += 1
                if P is None:
                    break
        except FoundFactor as e:
            if 1 < e.g < N:
                return e.g, steps
    return None, steps




# ---------- helpers for building test numbers ----------
def next_prime(n):
    while not is_prime(n):
        n += 1
    return n


def smooth_minus_one_prime(rng, bits):
    """A prime p where p - 1 is built only from small primes (Pollard p-1 catches it)."""
    while True:
        m = 2
        while m.bit_length() < bits:
            m *= rng.choice(PRIMES[:60])
        if is_prime(m + 1):
            return m + 1


def hardened_prime(rng, bits):
    """A prime p where p - 1 has a large prime factor, so Pollard p-1 cannot catch it."""
    while True:
        big = next_prime(rng.getrandbits(bits - 6) | (1 << (bits - 7)))
        for k in range(2, 200, 2):
            p = k * big + 1
            if is_prime(p):
                return p
