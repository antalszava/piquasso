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

from typing import Tuple

import functools

import numpy as np

from scipy.special import comb


def get_operator_index(modes: Tuple[int, ...]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Note:
        For indexing of numpy arrays, see
        https://numpy.org/doc/stable/reference/arrays.indexing.html#advanced-indexing
    """

    transformed_columns = np.array([modes] * len(modes))
    transformed_rows = transformed_columns.transpose()

    return transformed_rows, transformed_columns


def get_auxiliary_operator_index(
    modes: Tuple[int, ...], auxiliary_modes: Tuple[int, ...]
) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
    auxiliary_rows = tuple(np.array([modes] * len(auxiliary_modes)).transpose())

    return auxiliary_rows, auxiliary_modes


@functools.lru_cache()
def combinations_with_repetition(*, boxes: int, particles: int) -> int:
    return comb(boxes + particles - 1, particles, exact=True)


@functools.lru_cache()
def get_index_in_particle_subspace(element: Tuple[int, ...]) -> int:
    boxes = len(element)
    particles = sum(element)

    if boxes == 1:
        return 0

    return sum(
        [
            combinations_with_repetition(
                boxes=boxes - 1, particles=particles - particles_in_first_box
            )
            for particles_in_first_box in range(element[0])
        ]
    ) + get_index_in_particle_subspace(element[1:])


@functools.lru_cache()
def cumulated_combinations_with_repetition(*, boxes: int, cutoff: int) -> int:
    r"""
    Calculates the cumulated combinations with repetations up to `cutoff` with the
    equation

    ..math::
        \sum_{i=0}^{c - 1} {d + i - 1 \choose i} = {d + c - 1 \choose c - 1}.
    """

    return comb(boxes + cutoff - 1, cutoff - 1, exact=True)


@functools.lru_cache()
def get_index_in_fock_space(element: Tuple[int, ...]) -> int:
    boxes = len(element)
    particles = sum(element)

    return cumulated_combinations_with_repetition(
        boxes=boxes, cutoff=particles
    ) + get_index_in_particle_subspace(element)
