# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to create curriculum for the learning environment.

The functions can be passed to the :class:`isaaclab.managers.CurriculumTermCfg` object to enable
the curriculum introduced by the function.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING
import pdb

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv



def modify_success_tolerance(env: ManagerBasedRLEnv, 
                             env_ids: Sequence[int],
                             reward_name: str, 
                             command_name: str, 
                             initial_tolerance: float,
                             target_tolerance: float,
                             tolerance_curriculum_increment: float, 
                             curriculum_interval: int,):
    """Curriculum that modifies a reward weight a given number of steps.

    Args:
        env: The learning environment.
        env_ids: Not used since all environments are affected.
        term_name: The name of the reward term.
        weight: The weight of the reward term.
        num_steps: The number of steps after which the change should be applied.
    """
    command_term = env.command_manager.get_term(command_name)
    if command_term.last_metric is not None:
        last_success_rate = command_term.last_metric["success"]
        if env.common_step_counter - env.last_curriculum_update > curriculum_interval and last_success_rate > 0.75:
            env.last_curriculum_update = env.common_step_counter
            # obtain term settings
            term_cfg = env.reward_manager.get_term_cfg(reward_name)
            # update term settings
            term_cfg.params["success_tolerance"] *= tolerance_curriculum_increment
            
            term_cfg.params["success_tolerance"] = min(term_cfg.params["success_tolerance"], initial_tolerance)
            term_cfg.params["success_tolerance"] = max(term_cfg.params["success_tolerance"], target_tolerance)
            command_term.metrics["success_tolerance"] = term_cfg.params["success_tolerance"]
            
            env.reward_manager.set_term_cfg(reward_name, term_cfg)

