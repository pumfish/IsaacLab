# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import isaaclab.sim as sim_utils
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import TiledCameraCfg
from isaaclab.utils import configclass

import isaaclab_tasks.manager_based.manipulation.lift.mdp as mdp

from .lift_env_cfg import LiftEnvCfg, ObjectTableSceneCfg

##
# Scene definition
##

## Add CameraCfg to origin scene configuration
@configclass
class LiftRGBCameraSceneCfg(ObjectTableSceneCfg):

    # add camera to the scene
    # TODO: the ex and in settings need to be set
    tiled_camera: TiledCameraCfg = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/Camera",
        offset=TiledCameraCfg.OffsetCfg(
            pos=(2.5, 0, 1.5),
            rot=(0.61237, 0.35355, 0.35355, 0.61237),
            convention="opengl",
        ),
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=36.0,
            focus_distance=200.0,
            horizontal_aperture=20.955,
            clipping_range=(0.1, 20.0),
        ),
        width=128,
        height=128,
    )

##
# MDP settings
##

## Expand the RGB observations configuration to include
## the camera observations
@configclass
class RGBObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class RGBCameraPolicyCfg(ObsGroup):
        """Observations for policy group with RGB images."""

        image = ObsTerm(
            func=mdp.image,
            params={
                "sensor_cfg": SceneEntityCfg("tiled_camera"),
                "data_type": "rgb",
            }
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: ObsGroup = RGBCameraPolicyCfg()


# Extract features from RGB images using a frozen ResNet18 model
@configclass
class ResNet18ObservationCfg:
    """Observation specifications for the MDP with ResNet18."""

    @configclass
    class ResNet18FeaturesCameraPolicyCfg(ObsGroup):
        """Observations for policy group with ResNet18 images
        with a frozen ResNet18 model.
        """

        image = ObsTerm(
            func=mdp.image_features,
            params={
                "sensor_cfg": SceneEntityCfg("tiled_camera"),
                "data_type": "rgb",
                "model_name": "resnet18",
            }
        )

    policy: ObsGroup = ResNet18FeaturesCameraPolicyCfg()

# # Extract features from RGB images using a frozen ResNet18 model
# @configclass
# class TheiaTinyObservationCfg:
#     """Observation specifications for the MDP."""

#     @configclass
#     class TheiaTinyFeaturesCameraPolicyCfg(ObsGroup):
#         """Observations for policy group with features extracted from RGB images with a frozen Theia-Tiny Transformer"""

#         image = ObsTerm(
#             func=mdp.image_features,
#             params={
#                 "sensor_cfg": SceneEntityCfg("tiled_camera"),
#                 "data_type": "rgb",
#                 "model_name": "theia-tiny-patch16-224-cddsv",
#                 "model_device": "cuda:0",
#             },
#         )

#     policy: ObsGroup = TheiaTinyFeaturesCameraPolicyCfg()


##
# Environment configuration
##

@configclass
class LiftRGBCameraEnvCfg(LiftEnvCfg):
    """Environment configuration for the Lift task with RGB camera."""

    # scene configuration
    scene: LiftRGBCameraSceneCfg = LiftRGBCameraSceneCfg(num_envs=512, env_spacing=20)
    observations: RGBObservationsCfg = RGBObservationsCfg()

    def __post_init__(self):
        super().__post_init__()
        # viewer settings
        self.viewer.eye = (3.0, 1.5, 1.5)
        self.viewer.lookat = (0.0, 0.0, 0.0)


@configclass
class LiftResNet18CameraEnvCfg(LiftRGBCameraEnvCfg):
    """Environment configuration for the Lift task
    with ResNet18 features as observations.
    """

    # scene configuration
    observations: ResNet18ObservationCfg = ResNet18ObservationCfg()
