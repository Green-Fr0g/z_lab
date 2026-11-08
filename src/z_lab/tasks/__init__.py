# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Task registrations for the project."""

# These imports intentionally run after ``import_packages`` to register tasks.
# ruff: noqa: E402

from isaaclab_tasks.utils import import_packages

import_packages(__name__, ["utils", ".mdp"])
from . import (
    amp,
    locomotion,
    mimic,
)  # noqa: E402,F401
