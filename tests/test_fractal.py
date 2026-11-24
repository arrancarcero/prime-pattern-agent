import math

import numpy as np

from ppa.core.fractal import box_dimension, koch_curve


def test_calibration_shapes():
    assert abs(box_dimension(np.array([0, 1 + 0.3j])) - 1) < 0.05
    assert abs(box_dimension(koch_curve(8)) - math.log(4) / math.log(3)) < 0.05
