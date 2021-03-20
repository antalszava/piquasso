#
# Copyright (C) 2020 by TODO - All rights reserved.
#

import pytest
import numpy as np

import piquasso as pq
from piquasso.api import constants


@pytest.fixture
def state(program):
    return program.state


def test_xp_representation(state, assets):
    assert np.allclose(
        assets.load("expected_xp_mean"),
        state.xp_mean,
    )
    assert np.allclose(
        assets.load("expected_xp_corr"),
        state.xp_corr,
    )
    assert np.allclose(
        assets.load("expected_xp_cov"),
        state.xp_cov,
    )


def test_quad_representation(state, assets):
    assert np.allclose(
        assets.load("expected_mean"),
        state.mean,
    )
    assert np.allclose(
        assets.load("expected_corr"),
        state.corr,
    )
    assert np.allclose(
        assets.load("expected_cov"),
        state.cov,
    )


def test_representation_roundtrip(state):
    initial_mean = state.mean
    initial_cov = state.cov

    state.mean = initial_mean
    state.cov = initial_cov

    final_mean = state.mean
    final_cov = state.cov

    assert np.allclose(final_mean, initial_mean)
    assert np.allclose(final_cov, initial_cov)


def test_wigner_function(state, assets):
    quadrature_array = np.array(
        [
            [1, 2, 3, 4, 5, 6],
            [5, 6, 7, 8, 9, 0],
        ]
    )

    actual_result = state.wigner_function(quadrature_array)

    expected_result = assets.load("expected_wigner_function_result")

    assert np.allclose(expected_result, actual_result)


def test_reduced_rotated_mean_and_cov(state, assets):
    modes = (0, 2)
    phi = np.pi/2

    mean, cov = state.reduced_rotated_mean_and_cov(modes, phi)

    expected_mean = assets.load("expected_mean")
    expected_cov = assets.load("expected_cov")

    assert np.allclose(mean, expected_mean)
    assert np.allclose(cov, expected_cov)


class TestGaussianStateOperations:
    @pytest.fixture
    def mean(self):
        return np.array([  2.,  -4.,   6.,   8.,   4., -10.])

    @pytest.fixture
    def cov(self):
        return np.array(
            [
                [ 25.,  -3.,  -1.,   5.,  -8.,   8.],
                [ -3.,  12.,   0., -14.,   3.,  -9.],
                [ -1.,   0.,   2.,   0.,   0.,   3.],
                [  5., -14.,   0.,  38., -14.,  13.],
                [ -8.,   3.,   0., -14.,  18.,   9.],
                [  8.,  -9.,   3.,  13.,   9.,  35.],
            ]
        )

    @pytest.fixture(autouse=True)
    def setup(self, mean, cov):
        self.state = pq.GaussianState(d=(len(mean) // 2))
        self.state.mean = mean
        self.state.cov = cov

    def test_rotated(self):
        phi = np.pi / 2
        rotated_state = self.state.rotated(phi)

        expected_rotated_mean = np.array([ -4.,  -2.,   8.,  -6., -10.,  -4.])
        expected_rotated_cov = np.array(
            [
                [ 12.,   3., -14.,  -0.,  -9.,  -3.],
                [  3.,  25.,  -5.,  -1.,  -8.,  -8.],
                [-14.,  -5.,  38.,   0.,  13.,  14.],
                [ -0.,  -1.,   0.,   2.,  -3.,   0.],
                [ -9.,  -8.,  13.,  -3.,  35.,  -9.],
                [ -3.,  -8.,  14.,   0.,  -9.,  18.],
            ]
        )

        assert np.allclose(
            expected_rotated_mean,
            rotated_state.mean,
        )
        assert np.allclose(
            expected_rotated_cov,
            rotated_state.cov,
        )

    def test_reduced(self):
        modes = (0, 2)

        reduced_state = self.state.reduced(modes)

        expected_reduced_mean = np.array([ 2.,  -4.,   4., -10.])
        expected_reduced_cov = np.array(
            [
                [25., -3., -8.,  8.],
                [-3., 12.,  3., -9.],
                [-8.,  3., 18.,  9.],
                [ 8., -9.,  9., 35.],
            ]
        )

        assert np.allclose(
            expected_reduced_mean,
            reduced_state.mean,
        )
        assert np.allclose(
            expected_reduced_cov,
            reduced_state.cov,
        )


class TestGaussianStateVacuum:
    def test_vacuum_covariance_is_proportional_to_identity(self):
        d = 2

        state = pq.GaussianState(d=d)

        expected_mean = np.zeros(2 * d)
        expected_covariance = np.identity(2 * d) * constants.HBAR

        assert np.allclose(state.mean, expected_mean)
        assert np.allclose(state.cov, expected_covariance)


def test_mean_and_covariance(program, assets):
    program.execute()
    program.state.validate()

    expected_mean = assets.load("expected_mean") * np.sqrt(2 * constants.HBAR)

    expected_cov = assets.load("expected_cov") * constants.HBAR

    assert np.allclose(program.state.mean, expected_mean)
    assert np.allclose(program.state.cov, expected_cov)


def test_mean_and_covariance_with_different_HBAR(program, assets):
    program.execute()
    program.state.validate()

    constants.HBAR = 42

    expected_mean = assets.load("expected_mean") * np.sqrt(2 * constants.HBAR)
    expected_cov = assets.load("expected_cov") * constants.HBAR

    assert np.allclose(program.state.mean, expected_mean)
    assert np.allclose(program.state.cov, expected_cov)

    # TODO: We need to reset the value of HBAR. Create a better teardown for it!
    constants.HBAR = constants._HBAR_DEFAULT
