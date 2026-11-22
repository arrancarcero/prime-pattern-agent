"""Natural measurements through the same sieve (needs: pip install statsmodels scipy pooch)."""
import numpy as np

from ppa.core.ledger import Finding
from ppa.skills.base import Skill
from ppa.truth.nature_truth import NATURE_TRUTH
from ppa.truth.randomness_tuple import RANDOMNESS_TUPLE, sieve_of_randomness


def load():
    import statsmodels.api as sm
    from scipy import datasets, signal
    d = {"sunspots": sm.datasets.sunspots.load_pandas().data["SUNACTIVITY"].to_numpy(),
         "nile": sm.datasets.nile.load_pandas().data["volume"].to_numpy(),
         "co2": sm.datasets.co2.load_pandas().data["co2"].interpolate().to_numpy(),
         "elnino": sm.datasets.elnino.load_pandas().data.drop(columns="YEAR").to_numpy().ravel()}
    ecg = datasets.electrocardiogram()
    peaks, _ = signal.find_peaks(ecg, distance=150, prominence=1.0)
    d["heartbeat_intervals"] = np.diff(peaks) / 360.0
    # Add your own series here, e.g. spinning-fluid speeds or tire runout:
    # d["fluid_speed"] = np.loadtxt("my_tracks.csv")
    return d


def dominant_period(x):
    t = np.arange(len(x))
    x = x - np.polyval(np.polyfit(t, x, 2), t)
    S = np.abs(np.fft.rfft(x)) ** 2
    f = np.fft.rfftfreq(len(x))
    return 1.0 / f[1 + int(np.argmax(S[1:]))]


def run(ctx, **_):
    try:
        data = load()
    except Exception as e:  # missing optional packages or no network
        return [Finding("nature", "natural measurements", "check",
                        f"skipped: {e.__class__.__name__}: {e}")]
    lim = ctx.cfg["thresholds"]["randomness_z"]
    out = []
    for t in NATURE_TRUTH:
        got = dominant_period(data[t.series])
        ok = abs(got - t.period) <= t.tolerance
        out.append(Finding("nature", f"known period: {t.series}", "established",
                           f"found {got:.2f} {t.unit}, expected {t.period} ({t.source})",
                           survived=ok))
    for k, x in data.items():
        s = np.sign(np.diff(x))
        s = s[s != 0]
        z = sieve_of_randomness(s)
        bad = [r.name for r in RANDOMNESS_TUPLE if abs(z[r.name]) > lim]
        kind = ("momentum (flips too rarely)" if z["sign_flips"] < -lim else
                "correction (flips too often)" if z["sign_flips"] > lim else "random-like flips")
        out.append(Finding("nature", f"up/down signs: {k}", "established",
                           f"{kind}; rows failed: {bad or 'none'} (n={len(s)})",
                           survived=not bad, details={"z": z}))
    return out


SKILL = Skill("nature", "randomness tuple + known natural periods",
              "Where do natural measurements sit between momentum, correction and random?",
              run, needs_primes=False)
