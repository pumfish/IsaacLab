# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the Universal Robots.

The following configuration parameters are available:

* :obj:`UR10_CFG`: The UR10 arm without a gripper.

Reference: https://github.com/ros-industrial/universal_robot
"""

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR

##
# Configuration
##

ur5e_inspire_usd_path = "/mnt/ssd-2t/heguanhua/assets/robots/urdf/ur5e_with_inspire/ur5e_with_inspire.usd"

UR5e_INSPIRE_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=ur5e_inspire_usd_path,
        activate_contact_sensors=False,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_postion_iteration_count=8,
            solver_velocity_iteration_count=0,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={
            # UR5e Arm
            "shoulder_pan_joint": 0.0,
            "shoulder_lift_joint": 1.57,
            "elbow_joint": -1.57,
            "wrist_1_joint": 0.0,
            "wrist_2_joint": 0.0,
            "wrist_3_joint": 3.14,
            #TODO:
            # Inspire Hand
            "thumb_proximal_yaw_joint": 0.2,
            "thumb_proximal_pitch_joint": 0.5,
            "thumb_intermediate_joint": 0.8,
            "thumb_distal_joint": 0.0,

            "index_proximal_joint": 0.7,
            "index_intermediate_joint": 1.0,

            "middle_proximal_joint": 0.7,
            "middle_intermediate_joint": 0.0,

            "ring_proximal_joint": 0.3,
            "ring_intermediate_joint": 0.0,

            "pinky_proximal_joint": 0.3,
            "pinky_intermediate_joint": 0.0,
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
        "hand": ImplicitActuatorCfg(
            joint_names_expr=["thumb_.*", "index_.*", "middle_.*", "ring_.*", "pinky_.*"],
            velocity_limit=1.57,
            effort_limit=1.5,
            stiffness=300.0,
            damping=30.0,
        ),
    },
    soft_joint_pos_limit_factor=1.0,
)
""" Configuration of UR5e with Inspire hand robot  """
