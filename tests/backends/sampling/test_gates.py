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

import numpy as np

import piquasso as pq


def test_program():
    U = np.array([[0.5, 0, 0], [0, 0.5j, 0], [0, 0, -1]], dtype=complex)

    with pq.Program() as program:
        pq.Q(all) | pq.StateVector([1, 1, 1, 0, 0])

        pq.Q(0, 1) | pq.Beamsplitter(0.5)
        pq.Q(1, 2, 3) | pq.Interferometer(U)
        pq.Q(3) | pq.Phaseshifter(0.5)
        pq.Q(4) | pq.Phaseshifter(0.5)
        pq.Q() | pq.Sampling()

    simulator = pq.SamplingSimulator(d=5)
    result = simulator.execute(program, shots=10)

    assert len(result.samples) == 10


def test_interferometer():
    U = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=complex)

    with pq.Program() as program:
        pq.Q(all) | pq.StateVector([1, 1, 1, 0, 0])

        pq.Q(4, 3, 1) | pq.Interferometer(U)

    simulator = pq.SamplingSimulator(d=5)
    state = simulator.execute(program).state

    expected_interferometer = np.array(
        [
            [1, 0, 0, 0, 0],
            [0, 9, 0, 8, 7],
            [0, 0, 1, 0, 0],
            [0, 6, 0, 5, 4],
            [0, 3, 0, 2, 1],
        ],
        dtype=complex,
    )

    assert np.allclose(state.interferometer, expected_interferometer)


def test_phaseshifter():
    phi = np.pi / 2

    with pq.Program() as program:
        pq.Q(all) | pq.StateVector([1, 1, 1, 0, 0])

        pq.Q(2) | pq.Phaseshifter(phi)

    simulator = pq.SamplingSimulator(d=5)
    state = simulator.execute(program).state

    x = np.exp(1j * phi)
    expected_interferometer = np.array(
        [
            [1, 0, 0, 0, 0],
            [0, 1, 0, 0, 0],
            [0, 0, x, 0, 0],
            [0, 0, 0, 1, 0],
            [0, 0, 0, 0, 1],
        ],
        dtype=complex,
    )

    assert np.allclose(state.interferometer, expected_interferometer)


def test_beamsplitter():
    theta = np.pi / 4
    phi = np.pi / 3

    with pq.Program() as program:
        pq.Q(all) | pq.StateVector([1, 1, 1, 0, 0])

        pq.Q(1, 3) | pq.Beamsplitter(theta, phi)

    simulator = pq.SamplingSimulator(d=5)
    state = simulator.execute(program).state

    t = np.cos(theta)
    r = np.exp(1j * phi) * np.sin(theta)
    rc = np.conj(r)
    expected_interferometer = np.array(
        [
            [1, 0, 0, 0, 0],
            [0, t, 0, -rc, 0],
            [0, 0, 1, 0, 0],
            [0, r, 0, t, 0],
            [0, 0, 0, 0, 1],
        ],
        dtype=complex,
    )

    assert np.allclose(state.interferometer, expected_interferometer)


def test_lossy_program():
    r"""
    This test checks the average number of particles in the lossy BS.
    We expect average number to be smaller than initial one.
    """
    losses = 0.5

    simulator = pq.SamplingSimulator(d=5)

    with pq.Program() as program:
        pq.Q(all) | pq.StateVector([1, 1, 1, 0, 0])

        pq.Q() | pq.Loss(losses)
        pq.Q(0) | pq.Loss(transmissivity=0.0)
        pq.Q() | pq.Sampling()

    result = simulator.execute(program, shots=1)
    sample = result.samples[0]
    assert sum(sample) < sum(result.state.initial_state)
