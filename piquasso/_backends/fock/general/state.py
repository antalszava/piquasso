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

from typing import Tuple, Any, Generator, Dict, Mapping

import numpy as np

from piquasso.api.config import Config
from piquasso.api.instruction import Instruction
from piquasso.api.errors import InvalidState
from piquasso._math.linalg import is_selfadjoint
from piquasso._math.fock import cutoff_cardinality, FockOperatorBasis, FockBasis

from ..state import BaseFockState


class FockState(BaseFockState):
    """Object to represent a general bosonic state in the Fock basis.

    Note:
        If you only work with pure states, it is advised to use
        :class:`~piquasso._backends.fock.pure.state.PureFockState` instead.

    Args:
        density_matrix (numpy.ndarray, optional): The initial density matrix.
        d (int): The number of modes.
        cutoff (int): The Fock space cutoff.
    """

    _instruction_map = {
        "DensityMatrix": "_density_matrix_instruction",
        **BaseFockState._instruction_map
    }

    def __init__(self, *, d: int, config: Config = None) -> None:
        super().__init__(d=d, config=config)

        self._density_matrix = self._get_empty()

    def _get_empty(self) -> np.ndarray:
        return np.zeros(shape=(self._space.cardinality, ) * 2, dtype=complex)

    def _vacuum(self, *_args: Any, **_kwargs: Any) -> None:
        self._density_matrix = self._get_empty()
        self._density_matrix[0, 0] = 1.0

    @classmethod
    def from_fock_state(cls, state: BaseFockState) -> "FockState":
        """Instantiation using another :class:`BaseFockState` instance.

        Args:
            state (BaseFockState):
                The instance from which a :class:`FockState` instance is created.
        """

        new_instance = cls(d=state.d, config=state._config)

        new_instance._density_matrix = state.density_matrix

        return new_instance

    def _passive_linear(self, instruction: Instruction) -> None:
        operator: np.ndarray = instruction._all_params["passive_block"]

        index = self._get_operator_index(instruction.modes)

        embedded_operator = np.identity(self._space.d, dtype=complex)

        embedded_operator[index] = operator

        fock_operator = self._space.get_passive_fock_operator(embedded_operator)

        self._density_matrix = (
            fock_operator @ self._density_matrix @ fock_operator.conjugate().transpose()
        )

    def _get_probability_map(
        self, *, modes: Tuple[int, ...]
    ) -> Dict[FockBasis, float]:
        probability_map: Dict[FockBasis, float] = {}

        for index, basis in self._space.operator_basis_diagonal_on_modes(modes=modes):
            coefficient = float(self._density_matrix[index])

            subspace_basis = basis.ket.on_modes(modes=modes)

            if subspace_basis in probability_map:
                probability_map[subspace_basis] += coefficient
            else:
                probability_map[subspace_basis] = coefficient

        return probability_map

    @staticmethod
    def _get_normalization(
        probability_map: Mapping[FockBasis, float], sample: FockBasis
    ) -> float:
        return 1 / probability_map[sample]

    def _project_to_subspace(
        self, *, subspace_basis: FockBasis, modes: Tuple[int, ...], normalization: float
    ) -> None:
        projected_density_matrix = self._get_projected_density_matrix(
            subspace_basis=subspace_basis,
            modes=modes,
        )

        self._density_matrix = projected_density_matrix * normalization

    def _get_projected_density_matrix(
        self, *, subspace_basis: FockBasis, modes: Tuple[int, ...]
    ) -> np.ndarray:
        new_density_matrix = self._get_empty()

        index = self._space.get_projection_operator_indices(
            subspace_basis=subspace_basis,
            modes=modes,
        )

        new_density_matrix[index] = self._density_matrix[index]

        return new_density_matrix

    def _create(self, instruction: Instruction) -> None:
        operator = self._space.get_creation_operator(instruction.modes)

        self._density_matrix = operator @ self._density_matrix @ operator.transpose()

        self.normalize()

    def _annihilate(self, instruction: Instruction) -> None:
        operator = self._space.get_annihilation_operator(instruction.modes)

        self._density_matrix = operator @ self._density_matrix @ operator.transpose()

        self.normalize()

    def _add_occupation_number_basis(
        self, *, ket: Tuple[int, ...], bra: Tuple[int, ...], coefficient: complex
    ) -> None:
        index = self._space.index(ket)
        dual_index = self._space.index(bra)

        self._density_matrix[index, dual_index] = coefficient

    def _kerr(self, instruction: Instruction) -> None:
        mode = instruction.modes[0]
        xi = instruction._all_params["xi"]

        for index, (basis, dual_basis) in self._space.operator_basis:
            number = basis[mode]
            dual_number = dual_basis[mode]

            coefficient = np.exp(
                1j * xi * (
                   number * (2 * number + 1)
                   - dual_number * (2 * dual_number + 1)
                )
            )

            self._density_matrix[index] *= coefficient

    def _cross_kerr(self, instruction: Instruction) -> None:
        modes = instruction.modes
        xi = instruction._all_params["xi"]

        for index, (basis, dual_basis) in self._space.operator_basis:
            coefficient = np.exp(
                1j * xi * (
                    basis[modes[0]] * basis[modes[1]]
                    - dual_basis[modes[0]] * dual_basis[modes[1]]
                )
            )

            self._density_matrix[index] *= coefficient

    def _linear(self, instruction: Instruction) -> None:
        operator = self._space.get_linear_fock_operator(
            modes=instruction.modes,
            cache_size=self._config.cache_size,
            auxiliary_modes=self._get_auxiliary_modes(instruction.modes),
            passive_block=instruction._all_params["passive_block"],
            active_block=instruction._all_params["active_block"],
            displacement=instruction._all_params["displacement_vector"],
        )

        self._density_matrix = (
            operator
            @ self._density_matrix
            @ operator.conjugate().transpose()
        )

        self.normalize()

    def _density_matrix_instruction(self, instruction: Instruction) -> None:
        self._add_occupation_number_basis(**instruction.params)

    @property
    def nonzero_elements(
        self
    ) -> Generator[Tuple[complex, FockOperatorBasis], Any, None]:
        for index, basis in self._space.operator_basis:
            coefficient = self._density_matrix[index]
            if coefficient != 0:
                yield coefficient, basis

    def __repr__(self) -> str:
        return " + ".join(
            [
                str(coefficient) + str(basis)
                for coefficient, basis in self.nonzero_elements
            ]
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, FockState):
            return False
        return np.allclose(self._density_matrix, other._density_matrix)

    @property
    def density_matrix(self) -> np.ndarray:
        cardinality = cutoff_cardinality(d=self.d, cutoff=self._config.cutoff)

        return self._density_matrix[:cardinality, :cardinality]

    def get_particle_detection_probability(self, occupation_number: tuple) -> float:
        index = self._space.index(occupation_number)
        return np.diag(self._density_matrix)[index].real

    @property
    def fock_probabilities(self) -> np.ndarray:
        cardinality = cutoff_cardinality(d=self.d, cutoff=self._config.cutoff)

        return np.diag(self._density_matrix).real[:cardinality]

    def reduced(self, modes: Tuple[int, ...]) -> "FockState":
        modes_to_eliminate = self._get_auxiliary_modes(modes)

        reduced_state = FockState(d=len(modes))

        for index, (basis, dual_basis) in self._space.operator_basis_diagonal_on_modes(
            modes=modes_to_eliminate
        ):
            reduced_basis = basis.on_modes(modes=modes)
            reduced_dual_basis = dual_basis.on_modes(modes=modes)

            reduced_index = reduced_state._space.index(reduced_basis)
            reduced_dual_index = reduced_state._space.index(reduced_dual_basis)

            reduced_state._density_matrix[reduced_index, reduced_dual_index] += (
                self._density_matrix[index]
            )

        return reduced_state

    def normalize(self) -> None:
        """Normalizes the density matrix to have a trace of 1.

        Raises:
            RuntimeError: Raised if the current norm of the state is too close to 0.
        """
        if np.isclose(self.norm, 0):
            raise InvalidState("The norm of the state is 0.")

        self._density_matrix = self._density_matrix / self.norm

    def validate(self) -> None:
        """Validates the represented state.

        Raises:
            InvalidState:
                Raised, if the density matrix is not positive semidefinite, not
                self-adjoint or the trace of the density matrix is not 1.
        """
        if not is_selfadjoint(self._density_matrix):
            raise InvalidState(
                "The density matrix is not self-adjoint:\n"
                f"density_matrix={self._density_matrix}"
            )

        fock_probabilities = self.fock_probabilities

        if not np.all(fock_probabilities >= 0.0):
            raise InvalidState(
                "The density matrix is not positive semidefinite.\n"
                f"fock_probabilities={fock_probabilities}"
            )

        trace_of_density_matrix = sum(fock_probabilities)

        if not np.isclose(trace_of_density_matrix, 1.0):
            raise InvalidState(
                f"The trace of the density matrix is {trace_of_density_matrix}, "
                "instead of 1.0:\n"
                f"fock_probabilities={fock_probabilities}"
            )
