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

"""
.. note::

   When multiple shots are specified and the evolution of the choses simulated state is
   possible, the measurement outcome corresponding to the last shot is used to evolve
   the state.

"""

import numpy as np

from piquasso.api.instruction import Instruction
from piquasso.api.errors import InvalidParameter
from piquasso._math.linalg import is_positive_semidefinite, symplectic_form


class ParticleNumberMeasurement(Instruction):
    r"""Particle number measurement.

    A non-Gaussian projective measurement with the probability density given by

    .. math::
        p(n) = \operatorname{Tr} \left [ \rho | n \rangle \langle n | \right ]


    The generated samples are non-negative integer values corresponding to the detected
    photon number.

    .. note::

        When used with :class:`~piquasso._backends.gaussian.state.GaussianState`, the
        state is not evolved, since that would be non-Gaussian.

    Args:
        cutoff (int): The Fock space cutoff.
        shots (int): The number of samples to generate.
    """

    def __init__(self, cutoff=5, shots=1):
        super().__init__(params=dict(cutoff=cutoff, shots=shots))


class ThresholdMeasurement(Instruction):
    """Threshold measurement.

    Similar to :class:`ParticleNumberMeasurement`, but only measuring whether or not
    the measured mode contains any photon.

    The generated samples contain :math:`0` or :math:`1`, where :math:`0` corresponds to
    no photon being detected, and :math:`1` corresponds to detection of at least one
    photon.

    Args:
        shots (int): The number of samples to generate.
    """

    def __init__(self, shots=1):
        super().__init__(params=dict(shots=shots))


class GeneraldyneMeasurement(Instruction):
    r"""General-dyne measurement.

    The probability density is given by

    .. math::
        p(r_m) = \frac{
            \exp \left (
                (r_m - r)^T
                \frac{1}{\sigma + \sigma_m}
                (r_m - r)
            \right )
        }{
            \pi^d \sqrt{ \operatorname{det} (\sigma + \sigma_m) }
        },

    where :math:`r_m \in \mathbb{C}^d`, :math:`\sigma` is the covariance matrix of the
    current state, :math:`r \in \mathbb{C}^d` is the first moment of the current state,
    and :math:`\sigma_m` is the covariance corresponding to a non-displaced Gaussian
    state characterizing the general-dyne detection. Notably, the heterodyne detection
    would correspond to a non-displaced Gaussian state with covariance
    :math:`\sigma_m = I_{d \times d}`.

    Args:
        detection_covariance (numpy.ndarray):
            A 2-by-2 symplectic matrix corresponding to a purely quadratic Hamiltonian.
        shots (int): The number of samples to generate.
    """

    def __init__(self, detection_covariance, shots=1):
        if not is_positive_semidefinite(detection_covariance + 1j * symplectic_form(1)):
            raise InvalidParameter(
                "The parameter 'detection_covariance' is invalid, since it doesn't "
                "fulfill the Robertson-Schrödinger uncertainty relation."
            )

        super().__init__(
            params=dict(
                detection_covariance=detection_covariance,
                shots=shots,
            )
        )


class HomodyneMeasurement(Instruction):
    r"""Homodyne measurement.

    Corresponds to measurement of the quadrature operator

    .. math::
        \hat{x}_{\phi} = \cos \phi \hat{x} + \sin \phi \hat{p}

    with outcome probability density given by

    .. math::
        p(x_{\phi}) = \langle x_{\phi} | \rho | x_{\phi} \rangle,

    where :math:`x_{\phi}` correspond to the eigenvalues of :math:`\hat{x}_{\phi}`.

    In optical setups, the measurement is performed by mixing the state :math:`\rho`
    with a strong coherent state :math:`| \alpha \rangle`, where :math:`\alpha >> 1`,
    then subtracting the detected intensities of the two outputs.
    The mixing is performed with a 50:50 beamsplitter.

    Args:
        phi (float): Phase space rotation angle.
        z (float):
            Squeezing amplitude. In the limit of `z` going to infinity one would recover
            the pure homodyne measurement in the so-called strong oscillator limit.
            Conversely, setting `z = 1` would correspond to
            :class:`HeterodyneMeasurement`.
        shots (int): The number of samples to generate.
    """

    def __init__(self, phi=0.0, z=1e-4, shots=1):
        super().__init__(
            params=dict(
                phi=phi,
                z=z,
                shots=shots,
            ),
            extra_params=dict(
                detection_covariance=np.array(
                    [
                        [z ** 2, 0],
                        [0, (1 / z) ** 2],
                    ]
                ),
            ),
        )


class HeterodyneMeasurement(Instruction):
    r"""Heterodyne measurement.

    The probability density is given by

    .. math::
        p(x_{\phi}) = \frac{1}{\pi} \operatorname{Tr} (
            \rho | \alpha \rangle \langle \alpha |
        ).

    In optical setups, the measurement is performed by mixing the state :math:`\rho`
    with a vacuum state :math:`| 0 \rangle`, then subtracting the detected intensities
    of the two outputs.
    The mixing is performed with a 50:50 beamsplitter.

    Args:
        shots (int): The number of samples to generate.
    """

    def __init__(self, shots=1):
        super().__init__(
            params=dict(
                shots=shots,
            ),
            extra_params=dict(
                detection_covariance=np.identity(2),
            ),
        )
