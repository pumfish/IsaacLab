# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformer
from isaaclab.utils.math import subtract_frame_transforms
from isaaclab.utils.math import combine_frame_transforms
import pdb

import omni.usd
import isaaclab.sim as sim_utils
from pxr import Gf, Sdf, UsdGeom, Vt

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def object_position_in_robot_root_frame(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """The position of the object in the robot's root frame."""
    robot: RigidObject = env.scene[robot_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    object_pos_w = object.data.root_pos_w[:, :3]
    object_pos_b, _ = subtract_frame_transforms(robot.data.root_pos_w, robot.data.root_quat_w, object_pos_w)
    return object_pos_b

def palm_position_in_robot_root_frame(
    env: ManagerBasedRLEnv,
    palm_frame_cfg: SceneEntityCfg = SceneEntityCfg("palm_frame"),
) -> torch.Tensor:
    """The position of the object in the robot's root frame."""
    palm_frame: FrameTransformer = env.scene[palm_frame_cfg.name]
    return palm_frame.data.target_pos_source[..., 0, :]

def palm_rot_vel_angvel_in_robot_root_frame(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """The position of the object in the robot's root frame."""
    robot: RigidObject = env.scene[robot_cfg.name]

    palm_index = robot.body_names.index('wrist_3_link')

    # quaternion(4) + liner vel(3) + angle vel(3)
    # robot orientation = world orientation, so vel_w = vel_robot
    # robot.data.body_state_w.shape = (num_envs, num_bodies, 13)
    palm_rot_vel_angvel = robot.data.body_state_w[...,palm_index,3:13]
    return palm_rot_vel_angvel    

def obj_rot_vel_angvel_in_robot_root_frame(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """The position of the object in the robot's root frame."""
    object: RigidObject = env.scene[object_cfg.name]

    # quaternion(4) + liner vel(3) + angle vel(3)
    # robot orientation = world orientation, so vel_w = vel_robot
    obj_rot_vel_angvel = object.data.body_state_w[...,0,3:13]

    return obj_rot_vel_angvel    


def fingertip_pos_rel(
    env: ManagerBasedRLEnv,
    palm_frame_cfg: SceneEntityCfg = SceneEntityCfg("palm_frame"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    finger_frame_cfg: SceneEntityCfg = SceneEntityCfg("finger_frame"),
) -> torch.Tensor:
    """The position of the object in the robot's root frame."""
    palm_frame: FrameTransformer = env.scene[palm_frame_cfg.name]
    robot: RigidObject = env.scene[robot_cfg.name]
    finger_frame: FrameTransformer = env.scene[finger_frame_cfg.name]

    fingertip_pos_w = finger_frame.data.target_pos_w
    num_allegro_fingertips = fingertip_pos_w.shape[1]
    num_env = fingertip_pos_w.shape[0]

    palm_center_repeat = palm_frame.data.target_pos_w[..., 0, :].unsqueeze(1).repeat(1, num_allegro_fingertips, 1)
    fingertip_pos_rel = fingertip_pos_w - palm_center_repeat
    
    # root_repeat = robot.data.root_pos_w.unsqueeze(1).repeat(1, num_allegro_fingertips, 1)

    # fingertip_pos_rel = fingertip_pos_w - root_repeat

    return fingertip_pos_rel.reshape(num_env, -1)  # (num_envs, num_allegro_fingertips * 3)

def obj_palm_pos(
        env: ManagerBasedRLEnv,
        palm_frame_cfg: SceneEntityCfg = SceneEntityCfg("palm_frame"),
        object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ) -> torch.Tensor:
        """The position of the object in the palm frame."""
        palm_frame: FrameTransformer = env.scene[palm_frame_cfg.name]
        object: RigidObject = env.scene[object_cfg.name]

        palm_pos_w = palm_frame.data.target_pos_w[..., 0, :]
        obj_pos_w = object.data.root_pos_w[:, :3]

        return obj_pos_w - palm_pos_w

def obj_goal_pos(
        env: ManagerBasedRLEnv,
        object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
        robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
        command_name: str = "object_pose",
) -> torch.Tensor:
    """The position of the goal in the palm frame."""
    object: RigidObject = env.scene[object_cfg.name]
    robot: RigidObject = env.scene[robot_cfg.name]
    command = env.command_manager.get_command(command_name)
    des_pos_b = command[:, :3]
    des_pos_w, _ = combine_frame_transforms(robot.data.root_pos_w, robot.data.root_quat_w, des_pos_b)
    # print(robot.data.root_pos_w+des_pos_b)
    # print(des_pos_b)
    obj_pos_w = object.data.root_pos_w[:, :3]
    return obj_pos_w - des_pos_w

def closest_obj_goal_dist(
        env: ManagerBasedRLEnv,
        command_name: str = "object_pose",
) -> torch.Tensor:
    """The closest distance between the obj and object goal."""
    command_term = env.command_manager.get_term(command_name)

    return command_term.metrics["closest_obj_goal_distance"].unsqueeze(-1)

def closest_fingertip_dist(
        env: ManagerBasedRLEnv,
        command_name: str = "object_pose",
) -> torch.Tensor:
    """The closest distance between the obj and object goal."""
    command_term = env.command_manager.get_term(command_name)

    return command_term.metrics["closest_fingertip_dist"]

def flag_object_is_lifted(
    env: ManagerBasedRLEnv, minimal_height: float, object_cfg: SceneEntityCfg = SceneEntityCfg("object"),command_name: str = "object_pose",
) -> torch.Tensor:
    """Reward the agent for lifting the object above the minimal height."""
    # object: RigidObject = env.scene[object_cfg.name]
    command_term = env.command_manager.get_term(command_name)
    # lifted_object = (object.data.root_pos_w[:, 2] > minimal_height) | command_term.metrics["lifted_object"]
    lifted_object = command_term.metrics["lifted_object"]
    return lifted_object.unsqueeze(-1)


def contact_forces(
    env: ManagerBasedRLEnv,
    contact_sensor_cfg1: SceneEntityCfg = SceneEntityCfg("contact_forces_index"),
    contact_sensor_cfg2: SceneEntityCfg = SceneEntityCfg("contact_forces_middle"),
    contact_sensor_cfg3: SceneEntityCfg = SceneEntityCfg("contact_forces_ring"),
    contact_sensor_cfg4: SceneEntityCfg = SceneEntityCfg("contact_forces_thumb"),
) -> torch.Tensor:
    """The contact forces from the contact sensor."""
    contact_sensor1 = env.scene[contact_sensor_cfg1.name]
    contact_sensor2 = env.scene[contact_sensor_cfg2.name]
    contact_sensor3 = env.scene[contact_sensor_cfg3.name]
    contact_sensor4 = env.scene[contact_sensor_cfg4.name]
    
    sensors = [contact_sensor1, contact_sensor2, contact_sensor3, contact_sensor4]

    # # 对每个传感器数据重塑后在第0维度拼接
    # concatenated_force = torch.cat(
    #     [sensor.data.force_matrix_w.reshape(-1, 3) for sensor in sensors],
    #     dim=1  # 在第0维度（行方向）拼接
    # )
    
    contact_flag = torch.cat(
        [(sensor.data.force_matrix_w.reshape(-1, 3)== 0.0).all(dim=-1, keepdim=True) for sensor in sensors],
        dim=1  # 在第0维度（行方向）拼接
    )
    
    # if not torch.all(contact_sensor.data.force_matrix_w  == 0):
    # # if True:
    #     print("Received contact force of: ", contact_sensor.data.net_forces_w)
    #     print("Received force matrix of: ", contact_sensor.data.force_matrix_w)
    #     pdb.set_trace()
    # contact forces are in world frame
    return ~contact_flag


def object_scale(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    if env.cfg.object_scale is not None:
        return env.cfg.object_scale
    else: 
        """The scale of the object."""
        object: RigidObject = env.scene[object_cfg.name]
        
        stage = omni.usd.get_context().get_stage()
        prim_paths = sim_utils.find_matching_prim_paths(object.cfg.prim_path)
        
        num_envs = object.data.root_pos_w.shape[0]
        
        scale = torch.zeros((num_envs, 3), dtype=torch.float32, device=object.data.root_pos_w.device)
        
        with Sdf.ChangeBlock():
            for env_id in range(num_envs):
                # path to prim to randomize
                prim_path = prim_paths[env_id] 
                # spawn single instance
                prim_spec = Sdf.CreatePrimInLayer(stage.GetRootLayer(), prim_path)
    
                # get the attribute to randomize
                scale_spec = prim_spec.GetAttributeAtPath(prim_path + ".xformOp:scale")
                # if the scale attribute does not exist, create it
                has_scale_attr = scale_spec is not None
                if not has_scale_attr:
                    scale_spec = Sdf.AttributeSpec(prim_spec, prim_path + ".xformOp:scale", Sdf.    ValueTypeNames.Double3)
                
                curr_scale = torch.tensor([scale_spec.default[0],scale_spec.default[0],scale_spec.default[0]])
                scale[env_id, :] = curr_scale  
        env.cfg.object_scale = scale         
        return scale