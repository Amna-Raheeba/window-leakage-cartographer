"""Window Leakage Cartographer.

Maps the sliding-window leakage that random train/test splits create
in time-series ML: overlapping windows from one trajectory land on
both sides of the split, and the model is scored partly on data it
has effectively already seen. The cartographer builds the
window-overlap graph (shared raw timesteps plus tolerance-banded
near-duplicate edges), finds the connected components that straddle
the split, reports leakage mass per test window, and - by removal
replay with a simple kNN predictor - attributes the score inflation
to each leakage component individually, not just as a total.
"""

__version__ = "0.1.0"
__author__ = "Amna Raheeba"
