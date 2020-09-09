#
# Copyright (C) 2020 by TODO - All rights reserved.
#

import pytest
from scipy.stats import unitary_group

from piquasso.operator import BaseOperator


@pytest.fixture
def dummy_unitary():
    def func(d):
        return BaseOperator(unitary_group.rvs(d))

    return func


@pytest.fixture(scope="session")
def tolerance():
    return 10E-10
