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
import pdb
from isaaclab.utils.math import combine_frame_transforms

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def object_is_lifted(
    env: ManagerBasedRLEnv, minimal_height: float, object_cfg: SceneEntityCfg = SceneEntityCfg("object")
) -> torch.Tensor:
    """Reward the agent for lifting the object above the minimal height."""
    object: RigidObject = env.scene[object_cfg.name]
    command_term = env.command_manager.get_term("object_pose")
    command_term.metrics["lifted_object"] = object.data.root_pos_w[:, 2] > minimal_height

    return torch.where(object.data.root_pos_w[:, 2] + 0.03 > minimal_height, 1.0, 0.0)

def lifting_rew(
    env: ManagerBasedRLEnv, minimal_height: float, object_cfg: SceneEntityCfg = SceneEntityCfg("object")
) -> torch.Tensor:
    """Reward the agent for lifting the object above the minimal height."""
    object: RigidObject = env.scene[object_cfg.name]
    command_term = env.command_manager.get_term("object_pose")

    z_lift = 0.05 + object.data.root_pos_w[:, 2] - minimal_height 
    lifting_rew = torch.clip(z_lift, 0, 0.5)
    lifted_object = (z_lift > 0.15) | command_term.metrics["lifted_object"]
    # stop giving lifting reward once we crossed the threshold - now the agent can focus entirely on the keypoint reward 
    lifting_rew *= ~lifted_object

    return lifting_rew

def lift_bonus_rew(
    env: ManagerBasedRLEnv, minimal_height: float, object_cfg: SceneEntityCfg = SceneEntityCfg("object")
) -> torch.Tensor:
    """Reward the agent for lifting the object above the minimal height."""
    object: RigidObject = env.scene[object_cfg.name]
    command_term = env.command_manager.get_term("object_pose")

    z_lift = 0.05 + object.data.root_pos_w[:, 2] - minimal_height 
    lifted_object = (z_lift > 0.15) | command_term.metrics["lifted_object"]
    # give bonus only once when the object is lifted above the threshold
    just_lifted_above_threshold = lifted_object & ~command_term.metrics["lifted_object"]

    command_term.metrics["lifted_object"] = lifted_object

    # print(just_lifted_above_threshold,lifted_object)
    # pdb.set_trace()

    return just_lifted_above_threshold


def object_ee_distance(
    env: ManagerBasedRLEnv,
    std: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    palm_frame_cfg: SceneEntityCfg = SceneEntityCfg("palm_frame"),
) -> torch.Tensor:
    """Reward the agent for reaching the object using tanh-kernel."""
    # extract the used quantities (to enable type-hinting)
    object: RigidObject = env.scene[object_cfg.name]
    palm_frame: FrameTransformer = env.scene[palm_frame_cfg.name]
    # Target object position: (num_envs, 3)
    cube_pos_w = object.data.root_pos_w
    # End-effector position: (num_envs, 3)
    ee_w = palm_frame.data.target_pos_w[..., 0, :]
    # Distance of the end-effector to the object: (num_envs,)
    object_ee_distance = torch.norm(cube_pos_w - ee_w, dim=1)

    return 1 - torch.tanh(object_ee_distance / std)

def fingertip_reaching_rew(env: ManagerBasedRLEnv,
    std: float,
    command_name: str = "object_pose",
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    finger_frame_cfg: SceneEntityCfg = SceneEntityCfg("finger_frame"),
) -> torch.Tensor:
    """Rewards for fingertips approaching the object or penalty for hand getting further away from the object."""

    object: RigidObject = env.scene[object_cfg.name]
    finger_frame: FrameTransformer = env.scene[finger_frame_cfg.name]

    command_term = env.command_manager.get_term(command_name)

    fingertip_pos_w = finger_frame.data.target_pos_w
    num_allegro_fingertips = fingertip_pos_w.shape[1]
    obj_pos_repeat_w = object.data.root_pos_w[:, :3].unsqueeze(1).repeat(1, num_allegro_fingertips, 1)
    fingertip_pos_rel_object = fingertip_pos_w - obj_pos_repeat_w

    curr_fingertip_distances = torch.norm(fingertip_pos_rel_object, dim=-1)

    # lifted_object = command_term.metrics["lifted_object"]

    # clip between zero and +inf to turn deltas into rewards
    curr_fingertip_distances_sum = torch.mean(curr_fingertip_distances, dim=-1) 
    # print(curr_fingertip_distances_sum)
    # add this reward only before the object is lifted off the table
    # after this, we should be guided only by keypoint and bonus rewards
    # curr_fingertip_distances_sum *= ~lifted_object

    # command_term.metrics["closest_fingertip_dist"] = torch.minimum(command_term.metrics["closest_fingertip_dist"], curr_fingertip_distances)

    # rew = 1 - torch.tanh(curr_fingertip_distances_sum)
    # print('jjjjjjj',rew)
    # pdb.set_trace()

    return ~command_term.metrics["lifted_object"] *(1 - torch.tanh(curr_fingertip_distances_sum/std))


def object_goal_distance(
    env: ManagerBasedRLEnv,
    std: float,
    command_name: str,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Reward the agent for tracking the goal pose using tanh-kernel."""
    # extract the used quantities (to enable type-hinting)
    robot: RigidObject = env.scene[robot_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    command = env.command_manager.get_command(command_name)
    command_term = env.command_manager.get_term("object_pose")
    # compute the desired position in the world frame
    des_pos_b = command[:, :3]
    des_pos_w, _ = combine_frame_transforms(robot.data.root_pos_w, robot.data.root_quat_w, des_pos_b)
    # distance of the end-effector to the object: (num_envs,)
    distance = torch.norm(des_pos_w - object.data.root_pos_w, dim=1)
    # rewarded if the object is lifted above the threshold
    return command_term.metrics["lifted_object"] * (1 - torch.tanh(distance / std))

def fingertip_delta_rew(env: ManagerBasedRLEnv,
    minimal_height: float,
    command_name: str = "object_pose",
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    finger_frame_cfg: SceneEntityCfg = SceneEntityCfg("finger_frame"),
) -> torch.Tensor:
    """Rewards for fingertips approaching the object or penalty for hand getting further away from the object."""

    object: RigidObject = env.scene[object_cfg.name]
    finger_frame: FrameTransformer = env.scene[finger_frame_cfg.name]

    command_term = env.command_manager.get_term(command_name)

    fingertip_pos_w = finger_frame.data.target_pos_w
    num_allegro_fingertips = fingertip_pos_w.shape[1]
    obj_pos_repeat_w = object.data.root_pos_w[:, :3].unsqueeze(1).repeat(1, num_allegro_fingertips, 1)
    fingertip_pos_rel_object = fingertip_pos_w - obj_pos_repeat_w

    curr_fingertip_distances = torch.norm(fingertip_pos_rel_object, dim=-1)

    #  reset env will set all command_term.metrics to 0
    
    command_term.metrics["closest_fingertip_dist"] = torch.where(
        command_term.metrics["closest_fingertip_dist"] == 0,
        curr_fingertip_distances,
        command_term.metrics["closest_fingertip_dist"],
    )
    # if torch.all(command_term.metrics["closest_fingertip_dist"] == 0):
    #     pdb.set_trace()
    #     command_term.metrics["closest_fingertip_dist"] = curr_fingertip_distances

    lifted_object = command_term.metrics["lifted_object"]

    # this is positive if we got closer, negative if we're further away than the closest we've gotten
    fingertip_deltas_closest = command_term.metrics["closest_fingertip_dist"] - curr_fingertip_distances
    # print(f"fingertip_deltas_closest: {fingertip_deltas_closest}", command_term.metrics["closest_fingertip_dist"],curr_fingertip_distances)
    # update the values if finger tips got closer to the object
    command_term.metrics["closest_fingertip_dist"] = torch.minimum(command_term.metrics["closest_fingertip_dist"], curr_fingertip_distances)
    # clip between zero and +inf to turn deltas into rewards
    fingertip_deltas = torch.clip(fingertip_deltas_closest, 0, 10)
    fingertip_delta_rew = torch.sum(fingertip_deltas, dim=-1)
    # add this reward only before the object is lifted off the table
    # after this, we should be guided only by keypoint and bonus rewards
    fingertip_delta_rew *= ~lifted_object

    return fingertip_delta_rew


def obj_goal_deltas_reward(env: ManagerBasedRLEnv,
    command_name: str = "object_pose",
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    
    object: RigidObject = env.scene[object_cfg.name]
    robot: RigidObject = env.scene[robot_cfg.name]
    command_term = env.command_manager.get_term(command_name)
    command = env.command_manager.get_command(command_name)
    des_pos_b = command[:, :3]
    des_pos_w, _ = combine_frame_transforms(robot.data.root_pos_w, robot.data.root_quat_w, des_pos_b)
    obj_pos_w = object.data.root_pos_w[:, :3]

    curr_obj_goal_distance = torch.norm(des_pos_w - obj_pos_w, dim=-1)
    
    command_term.metrics["closest_obj_goal_distance"]= torch.where(
        command_term.metrics["closest_obj_goal_distance"] == 0,
        curr_obj_goal_distance,      
        command_term.metrics["closest_obj_goal_distance"])
 
    # if torch.all(command_term.metrics["closest_obj_goal_distance"] == 0):
    #     command_term.metrics["closest_obj_goal_distance"] = curr_obj_goal_distance

    obj_goal_distance_deltas = command_term.metrics["closest_obj_goal_distance"] - curr_obj_goal_distance    

    command_term.metrics["closest_obj_goal_distance"] = torch.minimum(command_term.metrics["closest_obj_goal_distance"], curr_obj_goal_distance)

    max_distance_deltas = torch.clip(obj_goal_distance_deltas, 0, 100)

    return max_distance_deltas * command_term.metrics["lifted_object"]


def success_bouns(
    env: ManagerBasedRLEnv,
    command_name: str,
    success_tolerance: float = 0.075,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Reward the agent for tracking the goal pose using tanh-kernel."""
    # extract the used quantities (to enable type-hinting)
    robot: RigidObject = env.scene[robot_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    command = env.command_manager.get_command(command_name)
    command_term = env.command_manager.get_term("object_pose")
    # compute the desired position in the world frame
    des_pos_b = command[:, :3]
    des_pos_w, _ = combine_frame_transforms(robot.data.root_pos_w, robot.data.root_quat_w, des_pos_b)
    # distance of the end-object to the goal: (num_envs,)
    distance = torch.norm(des_pos_w - object.data.root_pos_w, dim=1)
    success = distance < success_tolerance

    # command_term.metrics["distance"] = distance
    command_term.metrics["success"] = torch.where(success, success, command_term.metrics["success"])
    
    current_time = env.episode_length_buf * env.step_dt
        
    remaining_time_part = 1 - (current_time / env.max_episode_length_s)
    
    return success * remaining_time_part


def contact_reward(
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
    
    contact_index = ~(contact_sensor1.data.force_matrix_w.reshape(-1, 3) == 0.0).all(dim=-1)
    contact_middle = ~(contact_sensor2.data.force_matrix_w.reshape(-1, 3) == 0.0).all(dim=-1)
    contact_ring = ~(contact_sensor3.data.force_matrix_w.reshape(-1, 3) == 0.0).all(dim=-1)
    contact_thumb = ~(contact_sensor4.data.force_matrix_w.reshape(-1, 3) == 0.0).all(dim=-1)
    
    # print(contact_sensor4.data.force_matrix_w.reshape(-1, 3))
    # print(contact_sensor4.data.net_forces_w.reshape(-1, 3))

    # 条件：拇指有接触 且 其他三个（index/middle/ring）中至少有一个有接触
    both_condition = contact_thumb & (contact_index | contact_middle | contact_ring)
    # print(both_condition)

    return both_condition