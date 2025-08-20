# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, DeformableObjectCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import FrameTransformerCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg, UsdFileCfg
from isaaclab.utils import configclass
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
import pdb

from . import mdp

##
# Scene definition
##


@configclass
class ObjectTableSceneCfg(InteractiveSceneCfg):
    """Configuration for the lift scene with a robot and a object.
    This is the abstract base implementation, the exact scene is defined in the derived classes
    which need to set the target object, robot and end-effector frames
    """

    # robots: will be populated by agent env cfg
    robot: ArticulationCfg = MISSING # type: ignore
    # end-effector sensor: will be populated by agent env cfg
    palm_frame: FrameTransformerCfg = MISSING
    finger_frame: FrameTransformerCfg = MISSING
    # target object: will be populated by agent env cfg
    object: RigidObjectCfg | DeformableObjectCfg = MISSING # type: ignore
    
    contact_forces_index = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/index_link_3",
        update_period=0.0,
        debug_vis=True,
        filter_prim_paths_expr=["{ENV_REGEX_NS}/object"],
    )
    contact_forces_middle = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/middle_link_3",
        update_period=0.0,
        debug_vis=True,
        filter_prim_paths_expr=["{ENV_REGEX_NS}/object"],
    )
    contact_forces_ring = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/ring_link_3",
        update_period=0.0,
        debug_vis=True,
        filter_prim_paths_expr=["{ENV_REGEX_NS}/object"],
    )
    contact_forces_thumb = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/thumb_link_3",
        update_period=0.0,
        debug_vis=True,
        filter_prim_paths_expr=["{ENV_REGEX_NS}/object"],
    )

    replicate_physics=False

    # Table
    # table = AssetBaseCfg(
    #     prim_path="{ENV_REGEX_NS}/Table",
    #     init_state=AssetBaseCfg.InitialStateCfg(pos=[0.5, 0, 0], rot=[0.707, 0, 0, 0.707]), # type: ignore
    #     spawn=UsdFileCfg(usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd"),
    # )

    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0, -0.8, 0.38], rot=[1, 0, 0, 0]), # type: ignore
        spawn=UsdFileCfg(usd_path=r"D:\WorkSpace\IsaacLab\MyAssets\table_narrow\table_narrow.usd"),
    )

    # plane
    plane = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0, 0, 0]), # type: ignore
        spawn=GroundPlaneCfg(),
    )

    # lights
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )


##
# MDP settings
##


@configclass
class CommandsCfg:
    """Command terms for the MDP."""

    object_pose = mdp.GraspCommandCfg(
        asset_name="robot",
        body_name=MISSING,  # will be set by agent env cfg
        resampling_time_range=(55.0, 55.0),
        # resampling_time_range=(1e6, 1e6),
        debug_vis=True,
        ranges=mdp.GraspCommandCfg.Ranges(
            pos_x=(-0.4, 0.4), pos_y=(-0.8, -0.55), pos_z=(0.68, 0.9), roll=(0.0, 0.0), pitch=(0.0, 0.0), yaw=(0.0, 0.0)
        ),
    )


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    # will be set by agent env cfg
    arm_action: mdp.JointPositionActionCfg | mdp.DifferentialInverseKinematicsActionCfg = MISSING # type: ignore
    hand_action: mdp.EMAJointPositionToLimitsActionCfg = MISSING
    


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # shape = (num_env, obs_dim)

        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        object_position = ObsTerm(func=mdp.object_position_in_robot_root_frame)

        palm_position = ObsTerm(func=mdp.palm_position_in_robot_root_frame)
        palm_rot_vel_angvel = ObsTerm(func=mdp.palm_rot_vel_angvel_in_robot_root_frame)
        obj_rot_vel_angvel = ObsTerm(func=mdp.obj_rot_vel_angvel_in_robot_root_frame)
        fingertip_pos_rel = ObsTerm(func=mdp.fingertip_pos_rel)
        obj_palm_pos = ObsTerm(func=mdp.obj_palm_pos)
        obj_goal_pos = ObsTerm(func=mdp.obj_goal_pos)
        flag_object_is_lifted = ObsTerm(func=mdp.flag_object_is_lifted,params={"minimal_height": 0.63})
        closest_obj_goal_dist = ObsTerm(func=mdp.closest_obj_goal_dist)
        closest_fingertip_dist = ObsTerm(func=mdp.closest_fingertip_dist)
        
        target_object_position = ObsTerm(func=mdp.generated_commands, params={"command_name": "object_pose"})
        actions = ObsTerm(func=mdp.last_action)
        object_scale = ObsTerm(func=mdp.object_scale)
        contact_flag = ObsTerm(func=mdp.contact_forces)
        
        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Configuration for events."""

    # 改为在子类实现DR变化
    # # startup
    # # -- robot
    # robot_physics_material = EventTerm(
    #     func=mdp.randomize_rigid_body_material,
    #     mode="startup",
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
    #         "static_friction_range": (0.7, 1.3),
    #         "dynamic_friction_range": (0.7, 1.3),
    #         "restitution_range": (0.0, 0.0),
    #         "num_buckets": 250,
    #     },
    # )
    # robot_scale_mass = EventTerm(
    #     func=mdp.randomize_rigid_body_mass,
    #     mode="startup",
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
    #         "mass_distribution_params": (0.95, 1.05),
    #         "operation": "scale",
    #     },
    # )
    # robot_joint_stiffness_and_damping = EventTerm(
    #     func=mdp.randomize_actuator_gains,
    #     mode="startup",
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
    #         "stiffness_distribution_params": (0.3, 3.0),  # default: 3.0
    #         "damping_distribution_params": (0.75, 1.5),  # default: 0.1
    #         "operation": "scale",
    #         "distribution": "log_uniform",
    #     },
    # )
    
    # # -- object
    # randomize_cube_scale = EventTerm(
    #     func=mdp.randomize_rigid_body_scale,
    #     mode="prestartup",
    #     params={
    #         "scale_range": {"x": (1.2, 1.7), "y": (1.2, 1.7), "z": (1.2, 1.7)},
    #         "asset_cfg": SceneEntityCfg("object"),
    #     },
    # )
    
    # object_physics_material = EventTerm(
    #     func=mdp.randomize_rigid_body_material,
    #     mode="startup",
    #     params={
    #         "asset_cfg": SceneEntityCfg("object", body_names=".*"),
    #         "static_friction_range": (0.7, 1.3),
    #         "dynamic_friction_range": (0.7, 1.3),
    #         "restitution_range": (0.0, 0.0),
    #         "num_buckets": 250,
    #     },
    # )
    # object_scale_mass = EventTerm(
    #     func=mdp.randomize_rigid_body_mass,
    #     mode="startup",
    #     params={
    #         "asset_cfg": SceneEntityCfg("object"),
    #         "mass_distribution_params": (0.4, 1.6),
    #         "operation": "scale",
    #     },
    # )

    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")

    reset_object_position = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.1, 0.1), "y": (-0.1, 0.1), "z": (0.0, 0.0)},
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("object", body_names="object"),
        },
    )
    
    # reset_robot_joints = EventTerm(
    #     func=mdp.reset_joints_within_limits_range,
    #     mode="reset",
    #     params={
    #         "position_range": {".*": [0.1, 0.1]},
    #         "velocity_range": {".*": [0.0, 0.0]},
    #         "use_default_offset": True,
    #         "operation": "scale",
    #     },
    # )




@configclass
class RewardsCfg:
    """Reward terms for the MDP."""
    # reward shape = [env_nums]

    # reaching_object = RewTerm(func=mdp.object_ee_distance, params={"std": 0.1}, weight=1.0)
    # reaching_object = RewTerm(func=mdp.fingertip_reaching_rew, params={"std": 0.1}, weight=1)

    # lifting_object = RewTerm(func=mdp.object_is_lifted, params={"minimal_height": 0.63}, weight=15.0)


    fingertip_delta_rew = RewTerm(
        func=mdp.fingertip_delta_rew,
        params={"minimal_height": 0.63, "command_name": "object_pose"},
        # weight=0.01/,
        weight=50.0,
    )

    lifting_object = RewTerm(func=mdp.lifting_rew, params={"minimal_height": 0.63}, weight=20.0)

    lifting_bonus = RewTerm(func=mdp.lift_bonus_rew, params={"minimal_height": 0.63}, weight=300.0)

    object_goal_tracking = RewTerm(
        func=mdp.object_goal_distance,
        params={"std": 0.3, "command_name": "object_pose"},
        weight=16.0,
    )

    obj_goal_deltas_reward = RewTerm(
        func=mdp.obj_goal_deltas_reward,
        params={"command_name": "object_pose"},
        # weight=200.0,
        weight=0.01,
    )

    success_bouns = RewTerm(
        func=mdp.success_bouns,
        params={"command_name": "object_pose", "success_tolerance": 0.075},
        weight=6000.0,
    )
    
    contact_reward = RewTerm(
        func=mdp.contact_reward,
        weight=10.0,
    )

    # action penalty
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-1e-4)

    joint_vel = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-1e-4,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    # the heigth of object on the table = 0.56
    object_dropping = DoneTerm(
        func=mdp.root_height_below_minimum, params={"minimum_height": 0.5, "asset_cfg": SceneEntityCfg("object")}
    )

    success = DoneTerm(
        func=mdp.success, params={"command_name": "object_pose"},
    )


@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""
    
    success_tolerance = CurrTerm(
        func=mdp.modify_success_tolerance, params={"reward_name": "success_bouns", 
                                                   "command_name": "object_pose", 
                                                   "initial_tolerance": 0.075,
                                                   "target_tolerance": 0.01,
                                                   "tolerance_curriculum_increment": 0.8,
                                                   "curriculum_interval": 2000}
    )

    # joint_vel = CurrTerm(
    #     func=mdp.modify_reward_weight, params={"term_name": "joint_vel", "weight": -1e-1, "num_steps": 10000}
    # )


##
# Environment configuration
##


@configclass
class GraspEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the lifting environment."""

    # Scene settings
    # scene: ObjectTableSceneCfg = ObjectTableSceneCfg(num_envs=4096, env_spacing=2.5)
    scene: ObjectTableSceneCfg = ObjectTableSceneCfg(num_envs=1, env_spacing=2.5)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        """Post initialization."""
        
        # general settings
        self.decimation = 2
        self.episode_length_s = 5.0
        # simulation settings
        self.sim.dt = 0.01  # 100Hz
        self.sim.render_interval = self.decimation

        self.sim.physx.bounce_threshold_velocity = 0.2
        self.sim.physx.bounce_threshold_velocity = 0.01
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 1024 * 1024 * 4
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 16 * 1024
        self.sim.physx.friction_correlation_distance = 0.00625
        self.object_scale = None
