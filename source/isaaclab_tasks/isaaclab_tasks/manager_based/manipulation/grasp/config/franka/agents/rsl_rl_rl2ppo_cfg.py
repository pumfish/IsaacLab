# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

# from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticRL2Cfg, RslRlPpoRL2AlgorithmCfg


@configclass
class GraspCubeRL2PPORunnerCfg(RslRlOnPolicyRunnerCfg):
    seed = 42
    num_steps_per_env = 48        # 96 num_envs=512         # lsm 24
    max_iterations = 3001                  # lsm 3000
    save_interval = 100                    # lsm 100
    experiment_name = "Allegro_Grasp_DR"
    empirical_normalization = False
    policy = RslRlPpoActorCriticRL2Cfg(
        init_noise_std=1.0,
        actor_hidden_dims=[256, 128, 64],
        critic_hidden_dims=[256, 128, 64],
        activation="elu",
        # rnn cfg
        rnn_type="lstm",
        rnn_hidden_dim=64,
        rnn_num_layers=1,
    )
    algorithm = RslRlPpoRL2AlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.006,
        num_learning_epochs=5,
        num_mini_batches=4,             # lsm 4
        learning_rate=1.0e-4,           # lsm 1.0e-4
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
