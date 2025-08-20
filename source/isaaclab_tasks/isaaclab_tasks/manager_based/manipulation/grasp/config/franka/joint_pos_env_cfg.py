# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.assets import RigidObjectCfg
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from isaaclab_tasks.manager_based.manipulation.grasp import mdp
from isaaclab_tasks.manager_based.manipulation.grasp.grasp_env_cfg import GraspEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg

##
# Pre-defined configs
##
from isaaclab.markers.config import FRAME_MARKER_CFG  # isort: skip
# from isaaclab_assets.robots.franka import FRANKA_PANDA_CFG  # isort: skip
from isaaclab_assets.robots.ur5e_allegro import UR5E_ALLEGRO_CFG  # isort: skip


@configclass
class FrankaCubeGraspEnvCfg(GraspEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # Set Franka as robot
        self.scene.robot = UR5E_ALLEGRO_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # Set actions for the specific robot type (franka)
        self.actions.arm_action = mdp.JointPositionActionCfg(
            asset_name="robot", joint_names=["shoulder.*","elbow_joint","wrist.*"], scale=0.5, use_default_offset=True
        )
        # self.actions.hand_action = mdp.JointPositionActionCfg(
        #     asset_name="robot",
        #     joint_names=["index.*","middle.*","ring.*","thumb.*"], use_default_offset=True)
        
        self.actions.hand_action = mdp.EMAJointPositionToLimitsActionCfg(
                asset_name="robot",
                joint_names=["index.*","middle.*","ring.*","thumb.*"],
                alpha=0.95,
                rescale_to_limits=True,
        )
        
        # Set the body name for the end effector
        self.commands.object_pose.body_name = "wrist_3_link"

        # Set Cube as object
        self.scene.object = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/object",
            init_state=RigidObjectCfg.InitialStateCfg(pos=[0, -0.8, 0.58], rot=[1, 0, 0, 0]), # type: ignore
            spawn=UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/DexCube/dex_cube_instanceable.usd",
                # usd_path="/home/ws3/lansiming/cube_multicolor/cube_multicolor.usd",
                scale=(1.4, 1.4, 1.4),
                rigid_props=RigidBodyPropertiesCfg(
                    solver_position_iteration_count=16,
                    solver_velocity_iteration_count=1,
                    max_angular_velocity=1000.0,
                    max_linear_velocity=1000.0,
                    max_depenetration_velocity=5.0,
                    disable_gravity=False,
                ),
            ),
        )

        # Listens to the required transforms
        marker_cfg = FRAME_MARKER_CFG.copy()
        marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
        marker_cfg.prim_path = "/Visuals/FrameTransformer"
        
        self.scene.palm_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/base_link",
            debug_vis=False,
            visualizer_cfg=marker_cfg,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/wrist_3_link",
                    name="palm_center",
                    offset=OffsetCfg(
                        # pos=[-0.05, -0.05, 0.1],
                        pos=[-0.0, -0.0, 0.1],
                    ),
                ),
            ],
        )

        self.scene.finger_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/base_link",
            debug_vis=False,
            visualizer_cfg=marker_cfg,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/index_link_3",
                    name="finger1",
                    offset=OffsetCfg(pos=[0.05, 0.005, 0],),
                ),
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/middle_link_3",
                    name="finger2",
                    offset=OffsetCfg(pos=[0.05, 0.005, 0],),
                ),
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/ring_link_3",
                    name="finger3",
                    offset=OffsetCfg(pos=[0.05, 0.005, 0],),
                ),
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/thumb_link_3",
                    name="finger4",
                    offset=OffsetCfg(pos=[0.06, 0.005, 0],),
                ),
            ],
        )


@configclass
class FrankaCubeGraspEnvCfg_PLAY(FrankaCubeGraspEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # disable randomization for play
        self.observations.policy.enable_corruption = False


# 添加DR功能的子类，在reset时机随机化参数
@configclass
class FrankaCubeGraspDREnvCfg(FrankaCubeGraspEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # reset
        # -- robot
        self.events.robot_physics_material = EventTerm(
            func=mdp.randomize_rigid_body_material,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
                "static_friction_range": (0.7, 1.3),
                "dynamic_friction_range": (0.7, 1.3),
                "restitution_range": (0.0, 0.0),
                "num_buckets": 250,
            },
        )
        # self.events.robot_scale_mass = EventTerm(
        #     func=mdp.randomize_rigid_body_mass,
        #     mode="reset",
        #     params={
        #         "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
        #         "mass_distribution_params": (0.95, 1.05),
        #         "operation": "scale",
        #     },
        # )
        self.events.robot_joint_stiffness_and_damping = EventTerm(
            func=mdp.randomize_actuator_gains,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
                "stiffness_distribution_params": (0.3, 3.0),  # default: 3.0
                "damping_distribution_params": (0.75, 1.5),  # default: 0.1
                "operation": "scale",
                "distribution": "log_uniform",
            },
        )

# 添加DR功能的子类，在meta_reset时机随机化参数
@configclass
class FrankaCubeGraspMetaEnvCfg(FrankaCubeGraspEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # reset
        # -- robot
        self.events.robot_physics_material = EventTerm(
            func=mdp.randomize_rigid_body_material,
            mode="meta_reset",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
                "static_friction_range": (0.7, 1.3),
                "dynamic_friction_range": (0.7, 1.3),
                "restitution_range": (0.0, 0.0),
                "num_buckets": 250,
            },
        )
        # self.events.robot_scale_mass = EventTerm(
        #     func=mdp.randomize_rigid_body_mass,
        #     mode="reset",
        #     params={
        #         "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
        #         "mass_distribution_params": (0.95, 1.05),
        #         "operation": "scale",
        #     },
        # )
        self.events.robot_joint_stiffness_and_damping = EventTerm(
            func=mdp.randomize_actuator_gains,
            mode="meta_reset",
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
                "stiffness_distribution_params": (0.3, 3.0),  # default: 3.0
                "damping_distribution_params": (0.75, 1.5),  # default: 0.1
                "operation": "scale",
                "distribution": "log_uniform",
            },
        )