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


def test_PureFockState_reduced():
    with pq.Program() as program:
        pq.Q() | pq.StateVector([0, 1]) / 2

        pq.Q() | pq.StateVector([0, 2]) / 2
        pq.Q() | pq.StateVector([2, 0]) / np.sqrt(2)

    simulator = pq.PureFockSimulator(d=2)
    state = simulator.execute(program).state

    with pq.Program() as reduced_program:
        pq.Q() | pq.DensityMatrix(ket=(0,), bra=(0,)) / 2

        pq.Q() | pq.DensityMatrix(ket=(1,), bra=(1,)) / 4
        pq.Q() | pq.DensityMatrix(ket=(2,), bra=(2,)) / 4

        pq.Q() | pq.DensityMatrix(ket=(1,), bra=(2,)) / 4
        pq.Q() | pq.DensityMatrix(ket=(2,), bra=(1,)) / 4

    reduced_program_simulator = pq.FockSimulator(d=1)
    reduced_program_state = reduced_program_simulator.execute(reduced_program).state

    expected_reduced_state = reduced_program_state

    reduced_state = state.reduced(modes=(1,))

    assert expected_reduced_state == reduced_state


def test_PureFockState_fock_probabilities_map():
    with pq.Program() as program:
        pq.Q() | pq.StateVector([0, 1]) / 2

        pq.Q() | pq.StateVector([0, 2]) / 2
        pq.Q() | pq.StateVector([2, 0]) / np.sqrt(2)

    simulator = pq.PureFockSimulator(d=2)
    state = simulator.execute(program).state

    expected_fock_probabilities = {
        (0, 0): 0.0,
        (0, 1): 0.25,
        (1, 0): 0.0,
        (0, 2): 0.25,
        (1, 1): 0.0,
        (2, 0): 0.5,
        (0, 3): 0.0,
        (1, 2): 0.0,
        (2, 1): 0.0,
        (3, 0): 0.0,
    }

    actual_fock_probabilities = state.fock_probabilities_map

    for occupation_number, expected_probability in expected_fock_probabilities.items():
        assert np.isclose(
            actual_fock_probabilities[occupation_number],
            expected_probability,
        )


def test_PureFockState_quadratures_mean():
    with pq.Program() as program:
        pq.Q() | pq.Vacuum()
        pq.Q(0) | pq.Displacement(r=0.5, phi=0)
        pq.Q(1) | pq.Squeezing(r=1, phi=0)

    simulator = pq.PureFockSimulator(d=2, config=pq.Config(cutoff=10))
    result = simulator.execute(program)

    mean_1 = result.state.quadratures_mean(modes=(0,))
    assert round(mean_1) == 1.0
