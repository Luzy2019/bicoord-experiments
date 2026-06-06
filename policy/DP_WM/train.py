"""
Usage:
Training:
python train.py --config-name=robot_dp_14
"""

import sys

# use line-buffering for both stdout and stderr
sys.stdout = open(sys.stdout.fileno(), mode="w", buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode="w", buffering=1)

import hydra, pdb
from omegaconf import OmegaConf
import pathlib, yaml
from flow_matching_policy.workspace.base_workspace import BaseWorkspace

import os

current_file_path = os.path.abspath(__file__)
parent_directory = os.path.dirname(current_file_path)


def get_camera_config(camera_type):
    camera_config_path = os.path.join(parent_directory, "../../task_config/_camera_config.yml")

    assert os.path.isfile(camera_config_path), "task config file is missing"

    with open(camera_config_path, "r", encoding="utf-8") as f:
        args = yaml.load(f.read(), Loader=yaml.FullLoader)

    assert camera_type in args, f"camera {camera_type} is not defined"
    return args[camera_type]


# allows arbitrary python code execution in configs using the ${eval:''} resolver
OmegaConf.register_new_resolver("eval", eval, replace=True)


def set_rgb_obs_shape(shape_meta, image_shape):
    for obs_name, obs_meta in shape_meta.obs.items():
        if obs_meta.get("type") == "rgb":
            obs_meta.shape = image_shape


@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.joinpath("flow_matching_policy", "config")),
)
def main(cfg: OmegaConf):
    # resolve immediately so all the ${now:} resolvers
    # will use the same time.
    head_camera_type = cfg.head_camera_type
    head_camera_cfg = get_camera_config(head_camera_type)
    image_shape = [
        3,
        head_camera_cfg["h"],
        head_camera_cfg["w"],
    ]
    cfg.task.image_shape = image_shape
    set_rgb_obs_shape(cfg.task.shape_meta, image_shape)
    OmegaConf.resolve(cfg)
    cfg.task.image_shape = image_shape
    set_rgb_obs_shape(cfg.task.shape_meta, image_shape)

    cls = hydra.utils.get_class(cfg._target_)
    workspace: BaseWorkspace = cls(cfg)
    print(cfg.task.dataset.zarr_path, cfg.task_name)
    workspace.run()


if __name__ == "__main__":
    main()
