from ._base_task import Base_Task
from .utils import *
import sapien
import math
import numpy as np


class build_bridge(Base_Task):

    def setup_demo(self, **kwags):
        super()._init_task_env_(**kwags)

    def load_actors(self):
        block_half_size = np.random.uniform(0.02, 0.04)
        self.size = block_half_size * 2
        block_half_length = np.random.uniform(0.12, 0.16)
        self.length = block_half_length * 2
        self.y_pose = np.random.uniform(-0.05, -0.02)

        self.target_pose_1 = [-block_half_length + block_half_size, self.y_pose, 0.741 + block_half_size, 0, 1, 0, 0]
        self.target_pose_2 = [block_half_length - block_half_size, self.y_pose, 0.741 + block_half_size, 0, 1, 0, 0]
        
        self.pose_0 = rand_pose(
            xlim=[-0.05, 0.05],
            ylim=[0.06, 0.10],
            zlim=[0.741],
            qpos=[1, 0, 0, 0],
            rotate_rand=False,
        )
        self.pose_1 = rand_pose(
            xlim=[-0.15, -0.05],
            ylim=[-0.1, -0.05],
            zlim=[0.741],
            qpos=[1, 0, 0, 0],
            rotate_rand=True,
            rotate_lim=[0, 0, 0.75],
        )
        self.pose_2 = rand_pose(
            xlim=[0.05, 0.15],
            ylim=[-0.1, -0.05],
            zlim=[0.741],
            qpos=[1, 0, 0, 0],
            rotate_rand=True,
            rotate_lim=[0, 0, 0.75],
        )

        self.block0 = create_box(
            scene=self,
            pose=self.pose_0,
            half_size=(block_half_length, block_half_size, block_half_size),
            color=(1, 0, 0),
            name="box0",
        )
        self.block1 = create_box(
            scene=self,
            pose=self.pose_1,
            half_size=(block_half_size, block_half_size, block_half_size),
            color=(0, 0, 1),
            name="box1",
        )
        self.block2 = create_box(
            scene=self,
            pose=self.pose_2,
            half_size=(block_half_size, block_half_size, block_half_size),
            color=(0, 0, 1),
            name="box2",
        )

        self.block0.set_mass(0.01)
        self.block1.set_mass(0.01)
        self.block2.set_mass(0.01)

    def play_once(self):
        left = ArmTag("left")
        right = ArmTag("right")

        blk_pose = self.block0.get_pose().p

        pose_1 = [blk_pose[0] - self.length / 2 - 0.05, blk_pose[1], blk_pose[2] + 0.15, 0.924, 0, 0.383, 0]
        pose_2 = [blk_pose[0] + self.length / 2 + 0.05, blk_pose[1], blk_pose[2] + 0.15, 0, -0.383, 0, 0.924]

        y_delta = self.y_pose - blk_pose[1]
        x_delta = -blk_pose[0]

        self.move(
            self.grasp_actor(self.block1, arm_tag=left, pre_grasp_dis=0.09),
            self.grasp_actor(self.block2, arm_tag=right, pre_grasp_dis=0.09),
        )
        self.move(
            self.move_by_displacement(arm_tag=left, z=0.07),
            self.move_by_displacement(arm_tag=right, z=0.07),
        )
        self.move(
            self.place_actor(self.block1, target_pose=self.target_pose_1, arm_tag=left, functional_point_id=0, pre_dis=0.09, dis=0.02, constrain="align"),
            self.place_actor(self.block2, target_pose=self.target_pose_2, arm_tag=right, functional_point_id=0, pre_dis=0.09, dis=0.02, constrain="align"),
        )
        self.move(
            self.move_by_displacement(arm_tag=left, z=0.1),
            self.move_by_displacement(arm_tag=right, z=0.1),
        )

        self.move(
            self.move_to_pose(left, pose_1),
            self.move_to_pose(right, pose_2),
        )
        self.move(
            self.move_by_displacement(arm_tag=left, z=-0.05),
            self.move_by_displacement(arm_tag=right, z=-0.05),
        )
        self.move(
            self.close_gripper(left),
            self.close_gripper(right),
        )
        self.move(
            self.move_by_displacement(arm_tag=left, z=0.1),
            self.move_by_displacement(arm_tag=right, z=0.1),
        )
 
        self.move(
            self.move_by_displacement(arm_tag=left, x=x_delta, y=y_delta),
            self.move_by_displacement(arm_tag=right, x=x_delta, y=y_delta),
        )
        z_delta = (self.block1.get_pose().p[2] + self.size / 2 + 0.01) - (self.block0.get_pose().p[2] - self.size / 2)
        self.move(
            self.move_by_displacement(arm_tag=left, z=z_delta),
            self.move_by_displacement(arm_tag=right, z=z_delta),
        )
        self.move(
            self.open_gripper(left),
            self.open_gripper(right),
        )
        self.move(
            self.move_by_displacement(arm_tag=left, z=0.05),
            self.move_by_displacement(arm_tag=right, z=0.05),
        )
        
        return self.info

    def check_success(self):
        pose_0 = self.block0.get_pose().p
        pose_1 = self.block1.get_pose().p
        pose_2 = self.block2.get_pose().p
        score = 0.0

        if (pose_0[2] - pose_1[2]) >= self.size - 0.01 and (pose_0[2] - pose_2[2]) >= self.size - 0.01:
            score += 0.5
            for pose in [pose_1, pose_2]:
                dis = np.linalg.norm(pose[:2] - pose_0[:2])
                if abs(self.length / 2 - self.size / 2 - dis) <= 0.04:
                    score += 0.25

        self.stage_eval_score = score
        return score == 1.0