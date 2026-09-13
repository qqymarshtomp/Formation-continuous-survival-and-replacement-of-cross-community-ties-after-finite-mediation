"""Checks of the conditional update identity and forecast independence."""
import itertools
import unittest
import numpy as np
from model import Config, NetworkModel
from network_forecast import OpinionExpectation, forecast


class ForecastTests(unittest.TestCase):
    def test_conditional_opinion_expectation(self):
        i, j = np.array([0, 0, 0]), np.array([1, 2, 3])
        x, s = np.array([.2, -.7, .8, .1]), np.array([.07, .23, .4])
        exact = np.zeros(4)
        for bits in itertools.product([0, 1], repeat=3):
            success = np.array(bits, dtype=bool)
            probability = np.prod(np.where(success, s, 1-s))
            if success.any():
                exact[0] += probability*(x[j[success]]-x[0]).mean()
            exact[j[success]] += probability*(x[0]-x[j[success]])
        got = OpinionExpectation(4, i, j).increment(x, s)
        np.testing.assert_allclose(got, exact, atol=1e-14)

    def test_forecast_does_not_mutate_or_read_future_stream(self):
        a = NetworkModel(Config(n=100), 55)
        for _ in range(15):
            a.step("context")
        b = a.clone()
        b.rng_nat.random(10000)
        first = forecast(a, duration=10, particles=8)
        second = forecast(b, duration=10, particles=8)
        np.testing.assert_array_equal(first, second)
        for key in ["w", "r", "x"]:
            np.testing.assert_array_equal(getattr(a, key), getattr(b, key))
        self.assertEqual(a.tick, 15)

    def test_constant_response_reduces_to_frozen_forecast(self):
        a = NetworkModel(Config(n=100, trust_learning=0, opinion_learning=0), 57)
        first = forecast(a, duration=30, particles=8, mode="coupled")
        second = forecast(a, duration=30, particles=8, mode="frozen")
        np.testing.assert_array_equal(first, second)


if __name__ == "__main__":
    unittest.main(verbosity=2)
