"""Known periods in natural measurements: the instrument must find these unaided."""
from collections import namedtuple

NaturalTruth = namedtuple("NaturalTruth", "series period unit tolerance source")

NATURE_TRUTH: tuple[NaturalTruth, ...] = (
    NaturalTruth("sunspots", 11.0, "years", 1.5, "Schwabe solar cycle"),
    NaturalTruth("co2", 52.18, "weeks", 1.0, "yearly plant growth cycle"),
    NaturalTruth("elnino", 12.0, "months", 0.5, "yearly sea-surface cycle"),
)
