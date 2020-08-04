#
# Copyright (C) 2020 by TODO - All rights reserved.
#

"""A simple quantum state implementation based on numpy."""

import numpy as np

from piquasso.operator import BaseOperator
from piquasso.backend import FockBackend


class FockState(BaseOperator):
    """
    Implements the density operator from quantum mechanics in Fock
    representation.
    """

    backend_class = FockBackend

    def __new__(cls, representation):
        return np.array(representation).view(cls)

    @classmethod
    def from_state_vector(cls, state_vector):
        """Creates a density operator from a state vector."""
        return np.outer(state_vector, state_vector).view(cls)

    @property
    def d(self):
        return self.shape[0]
