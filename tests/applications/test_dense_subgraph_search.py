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
import networkx as nx
import piquasso as pq
from scipy.linalg import polar, sinhm, coshm, expm


def test_dense_subgraph_search():
    adjacency_matrix = np.array(
        [
            [0, 1, 1, 0, 1],
            [1, 0, 1, 0, 1],
            [1, 1, 0, 1, 1],
            [0, 0, 1, 0, 1],
            [1, 1, 1, 1, 0],
        ]
    )
    shots = 2

    pq.constants.seed(40)

    with pq.Program() as program:
        pq.Q() | pq.Graph(adjacency_matrix)
        pq.Q() | pq.ParticleNumberMeasurement()

    state = pq.GaussianState(d=len(adjacency_matrix))
    result = state.apply(program)
    state.validate()

    # Convert sample results to subgraph nodes
    subgraphs = result.to_subgraph_nodes()

    # Create nx graph from the adjacency_matrix
    TA_graph = nx.Graph(adjacency_matrix)

    # Search for dense subgraphs with size between 2 and 3
    dense = result.dense_search([subgraphs[0]], TA_graph, 2, 3, max_count=3)

    assert (dense == {
        2: [(1.0, [1, 4])],
        3: [(1.0, [1, 2, 4])]
    } or dense == {
        2: [(1.0, [1, 4])],
        3: [(1.0, [0, 1, 4])]
    })
