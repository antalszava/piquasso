#
# Copyright 2021 Budapest Quantum Computing Group
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pytest
import numpy as np
from scipy.stats import unitary_group

from piquasso import _math

from piquasso.decompositions.clements import T, Clements


@pytest.fixture
def dummy_unitary():
    def func(d):
        return np.array(unitary_group.rvs(d))

    return func


@pytest.fixture(scope="session")
def tolerance():
    return 1e-9


def test_T_beamsplitter_is_unitary():
    theta = np.pi / 3
    phi = np.pi / 4

    beamsplitter = T({"params": [theta, phi], "modes": [0, 1]}, d=2)

    assert _math.linalg.is_unitary(beamsplitter)


def test_eliminate_lower_offdiagonal_2_modes(dummy_unitary, tolerance):

    U = dummy_unitary(d=2)

    decomposition = Clements(U, decompose=False)

    operation = decomposition.eliminate_lower_offdiagonal(1, 0)

    beamsplitter = T(operation, 2)

    rotated_U = beamsplitter @ U

    assert np.abs(rotated_U[1, 0]) < tolerance


def test_eliminate_lower_offdiagonal_3_modes(dummy_unitary, tolerance):

    U = dummy_unitary(d=3)

    decomposition = Clements(U, decompose=False)

    operation = decomposition.eliminate_lower_offdiagonal(1, 0)

    beamsplitter = T(operation, 3)

    rotated_U = beamsplitter @ U

    assert np.abs(rotated_U[1, 0]) < tolerance


def test_eliminate_upper_offdiagonal_2_modes(dummy_unitary, tolerance):

    U = dummy_unitary(d=2)

    decomposition = Clements(U, decompose=False)

    operation = decomposition.eliminate_upper_offdiagonal(1, 0)

    beamsplitter = T.i(operation, 2)

    rotated_U = U @ beamsplitter

    assert np.abs(rotated_U[1, 0]) < tolerance


def test_eliminate_upper_offdiagonal_3_modes(dummy_unitary, tolerance):

    U = dummy_unitary(d=3)

    decomposition = Clements(U, decompose=False)

    operation = decomposition.eliminate_upper_offdiagonal(1, 0)

    beamsplitter = T.i(operation, 3)

    rotated_U = U @ beamsplitter

    assert np.abs(rotated_U[1, 0]) < tolerance


@pytest.mark.parametrize("n", [2, 3, 4, 5])
def test_clements_decomposition_on_n_modes(n, dummy_unitary, tolerance):

    U = dummy_unitary(d=n)

    decomposition = Clements(U)

    diagonalized = decomposition.U

    assert np.abs(diagonalized[0, 1]) < tolerance
    assert np.abs(diagonalized[1, 0]) < tolerance

    assert (
        sum(sum(np.abs(diagonalized))) - sum(np.abs(np.diag(diagonalized))) < tolerance
    ), "Absolute sum of matrix elements should be equal to the "
    "diagonal elements, if the matrix is properly diagonalized."


@pytest.mark.parametrize("n", [2, 3, 4, 5])
def test_clements_decomposition_and_composition_on_n_modes(n, dummy_unitary, tolerance):

    U = dummy_unitary(d=n)

    decomposition = Clements(U)

    diagonalized = decomposition.U

    assert (
        sum(sum(np.abs(diagonalized))) - sum(np.abs(np.diag(diagonalized))) < tolerance
    ), "Absolute sum of matrix elements should be equal to the "
    "diagonal elements, if the matrix is properly diagonalized."

    original = Clements.from_decomposition(decomposition)

    assert (U - original < tolerance).all()
