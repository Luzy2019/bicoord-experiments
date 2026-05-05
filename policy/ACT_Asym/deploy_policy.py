import sys
import numpy as np
import torch
import os
import pickle
import cv2
import time  # Add import for timestamp
import h5py  # Add import for HDF5
from datetime import datetime  # Add import for datetime formatting
from .act_policy import ACT
import copy
from argparse import Namespace


def encode_obs(observation):
    head_cam = observation["observation"]["head_camera"]["rgb"]
    left_cam = observation["observation"]["left_camera"]["rgb"]
    right_cam = observation["observation"]["right_camera"]["rgb"]
    head_cam = np.moveaxis(head_cam, -1, 0) / 255.0
    left_cam = np.moveaxis(left_cam, -1, 0) / 255.0
    right_cam = np.moveaxis(right_cam, -1, 0) / 255.0
    qpos = (observation["joint_action"]["left_arm"] + [observation["joint_action"]["left_gripper"]] +
            observation["joint_action"]["right_arm"] + [observation["joint_action"]["right_gripper"]])
    return {
        "head_cam": head_cam,
        "left_cam": left_cam,
        "right_cam": right_cam,
        "qpos": qpos,
    }


def get_model(usr_args):
    return ACT(usr_args, Namespace(**usr_args))


def dispatch_asymmetric_action(TASK_ENV, action_packet):
    """
    Sends arm actions separately when the environment exposes per-arm APIs.

    Fallback keeps compatibility with existing RoboTwin environments by sending
    the full action vector, whose non-update arm has already been held by the
    policy.
    """
    if hasattr(TASK_ENV, "take_action_asym"):
        TASK_ENV.take_action_asym(
            left_action=action_packet["left_action"],
            right_action=action_packet["right_action"],
            left_update=action_packet["left_update"],
            right_update=action_packet["right_update"],
        )
        return

    has_split_api = hasattr(TASK_ENV, "take_left_action") and hasattr(TASK_ENV, "take_right_action")
    if has_split_api:
        if action_packet["left_update"]:
            TASK_ENV.take_left_action(action_packet["left_action"])
        if action_packet["right_update"]:
            TASK_ENV.take_right_action(action_packet["right_action"])
        return

    TASK_ENV.take_action(action_packet["action"])


def eval(TASK_ENV, model, observation):
    obs = encode_obs(observation)
    if hasattr(TASK_ENV, "get_asym_role"):
        obs["asym_role"] = TASK_ENV.get_asym_role()
    elif hasattr(TASK_ENV, "get_instruction"):
        # Optional user-side hook: environments may map language to a role.
        obs["instruction"] = TASK_ENV.get_instruction()

    # Get action from model
    action_packet = model.get_action(obs, return_packet=True)
    dispatch_asymmetric_action(TASK_ENV, action_packet)
    observation = TASK_ENV.get_obs()
    return observation


def reset_model(model):
    # Reset temporal aggregation state if enabled
    if model.temporal_agg:
        model.all_time_actions = torch.zeros([
            model.max_timesteps,
            model.max_timesteps + model.num_queries,
            model.state_dim,
        ]).to(model.device)
        model.t = 0
        model.last_asym_info = None
        print("Reset temporal aggregation state")
    else:
        model.t = 0
        model.last_asym_info = None
