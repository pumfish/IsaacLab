# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the Universal Robots.

The following configuration parameters are available:

* :obj:`UR10_CFG`: The UR10 arm without a gripper.

Reference: https://github.com/ros-industrial/universal_robot
"""
import math

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR

##
# Configuration
##

ur5e_allegro_usd_path = r"D:\WorkSpace\IsaacLab\MyAssets\ur5e_allegro\ur5e_allegro\ur5e_with_allegro\ur5e_with_allegro.usd"
# ur5e_allegro_usd_path = "/d/WorkSpace/IsaacLab/MyAssets/ur5e_allegro/ur5e_allegro/ur5e_with_allegro/ur5e_with_allegro.usd"

UR5e_AllEGRO_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=ur5e_allegro_usd_path,
        activate_contact_sensors=False,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            enable_gyroscopic_forces=False,
            angular_damping=0.01,
            max_linear_velocity=1000.0,
            max_angular_velocity=64 / math.pi * 180.0,
            max_depenetration_velocity=1000.0,
            max_contact_impulse=1e32,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=0,
            sleep_threshold=0.005,
            stabilization_threshold=0.0005,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={
            # UR5e Arm joint
            "shoulder_pan_joint": 0.0,
            "shoulder_lift_joint": 1.57,
            "elbow_joint": -1.57,
            "wrist_1_joint": 0.0,
            "wrist_2_joint": 0.0,
            "wrist_3_joint": 3.14,
            # Allegro Hand joint
            "^index_.*": 0.0,
            "^middle_.*": 0.0,
            "^ring_.*": 0.0,
            "^thumb_joint_0$": 0.28,
        },
    ),
    actuators={
        "arm": ImplicitActuatorCfg(
            joint_names_expr=["shoulder_.*", "elbow_.*", "wrist_.*"],
            velocity_limit=3.14,
            effort_limit=150.0,
            stiffness=60.0,
            damping=12.0,
        ),
        "fingers": ImplicitActuatorCfg(
            joint_names_expr=["(index|middle|ring|thumb)_joint_[0-3]"],
            effort_limit=0.5,
            velocity_limit=100.0,
            stiffness=3.0,
            damping=0.1,
            friction=0.01,
        ),
    },
    soft_joint_pos_limit_factor=1.0,
)
""" Configuration of UR5e with Allegro hand robot  """
