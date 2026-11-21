import pytest

from window_leakage.cartographer import audit_split
from window_leakage.generate import generate_fleet
from window_leakage.windows import build_windows, random_split


@pytest.fixture(scope="module")
def fleet():
    return generate_fleet(5, 8)


@pytest.fixture(scope="module")
def windows(fleet):
    return build_windows(fleet, 10, 5)


@pytest.fixture(scope="module")
def split(windows):
    return random_split(windows, 0.25, seed=11)


@pytest.fixture(scope="module")
def audit(fleet, windows, split):
    train, test = split
    return audit_split(windows, train, test, fleet, k=3)
