# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Task registrations for the project."""

from isaaclab_tasks.utils import import_packages

import_packages(__name__, ["utils", ".mdp"])
from . import locomotion  # noqa: F401,E402
