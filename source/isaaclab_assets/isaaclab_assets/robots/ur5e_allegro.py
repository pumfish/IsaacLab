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

# ['base_link', 'shoulder_link', 'upper_arm_link', 'forearm_link', 'wrist_1_link', 'wrist_2_link', 'wrist_3_link', 'index_link_0', 'middle_link_0', 'ring_link_0', 'thumb_link_0', 'index_link_1', 'middle_link_1', 'ring_link_1', 'thumb_link_1', 'index_link_2', 'middle_link_2', 'ring_link_2', 'thumb_link_2', 'index_link_3', 'middle_link_3', 'ring_link_3', 'thumb_link_3']
# ur5e_allegro_usd_path = r"D:\WorkSpace\IsaacLab\MyAssets\ur5e_allegro\ur5e_allegro\ur5e_with_allegro\ur5e_with_allegro.usd"
ur5e_allegro_usd_path = r"D:\WorkSpace\IsaacLab\MyAssets\ur5e_with_allegro\ur5e_with_allegro\ur5e_with_allegro.usd"

UR5E_ALLEGRO_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=ur5e_allegro_usd_path,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=5.0,
        ),
        activate_contact_sensors=True,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={
            "shoulder_pan_joint": -1.571,
            "shoulder_lift_joint": 0.0,
            "elbow_joint": 0.785,
            "wrist_1_joint": 0.785,
            "wrist_2_joint": 0.0,
            "wrist_3_joint": 2.358,
            "index_joint_0": 0.0,  #hand
            "index_joint_1": 0.0,
            "index_joint_2": 0.0,
            "index_joint_3": 0.0,
            "middle_joint_0": 0.0,
            "middle_joint_1": 0.0,
            "middle_joint_2": 0.0,
            "middle_joint_3": 0.0,
            "ring_joint_0": 0.0,
            "ring_joint_1": 0.0,
            "ring_joint_2": 0.0,
            "ring_joint_3": 0.0,
            "thumb_joint_0": 0.3,
            "thumb_joint_1": 0.0,
            "thumb_joint_2": 0.0,
            "thumb_joint_3": 0.0,

        },
    ),
    # 全零初始状态+旧的ur5e_allegro的usd，就是目前真机的配置
    # 但是仿真上先用容易训练的初始姿态训练
    # init_state=ArticulationCfg.InitialStateCfg(
    #     joint_pos={
    #         "shoulder_pan_joint": 0.0,
    #         "shoulder_lift_joint": 0.0,
    #         "elbow_joint": 0.0,
    #         "wrist_1_joint": 0.0,
    #         "wrist_2_joint": 0.0,
    #         "wrist_3_joint": 0.0,
    #         "index_joint_0": 0.0,  #hand
    #         "index_joint_1": 0.0,
    #         "index_joint_2": 0.0,
    #         "index_joint_3": 0.0,
    #         "middle_joint_0": 0.0,
    #         "middle_joint_1": 0.0,
    #         "middle_joint_2": 0.0,
    #         "middle_joint_3": 0.0,
    #         "ring_joint_0": 0.0,
    #         "ring_joint_1": 0.0,
    #         "ring_joint_2": 0.0,
    #         "ring_joint_3": 0.0,
    #         "thumb_joint_0": 0.0,
    #         "thumb_joint_1": 0.0,
    #         "thumb_joint_2": 0.0,
    #         "thumb_joint_3": 0.0,

    #     },
    # ),
    actuators={
        "arm1": ImplicitActuatorCfg(
            joint_names_expr=["shoulder.*","elbow_joint"],
            velocity_limit_sim=3.141592653589793,
            effort_limit_sim=150.0,
            stiffness=150.0,
            damping=15.0,
            # friction= -1.0,
        ),
        "arm2": ImplicitActuatorCfg(
            joint_names_expr=["wrist.*"],
            velocity_limit_sim=3.141592653589793,
            effort_limit_sim=28.0,
            stiffness=50.0,
            damping=5.0,
            # friction= -1.0,
        ),
        "hand": ImplicitActuatorCfg(
            joint_names_expr=["index.*","middle.*","ring.*","thumb.*"],
            velocity_limit_sim=6.283,
            effort_limit_sim=0.35,
            stiffness=40.0,
            damping=5.0,
            # friction= -1.0,
        ),
    },
)
"""Configuration of UR-10 arm using implicit actuator models."""
