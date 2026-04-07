from ._base_task import Base_Task
from .utils import *
import sapien
import math
import numpy as np


class extract_bottom_block_to_top(Base_Task):

    def setup_demo(self, **kwags):
        super()._init_task_env_(**kwags)

    def load_actors(self):
        block_half_size = np.random.uniform(0.02, 0.03)
        self.size = block_half_size * 2
        base_half_size = 0.03

        block_pose_lst = []
        block_pose = rand_pose(
            xlim=[-0.05, 0],
            ylim=[-0.05, 0.1],
            zlim=[0.741 + base_half_size],
            qpos=[1, 0, 0, 0],
            rotate_rand=False,
        )
        for i in range(3):
            new_pose = deepcopy(block_pose)
            new_pose.p += [0, 0, base_half_size + self.size * (i + 0.5)]
            block_pose_lst.append(new_pose)

        self.block0 = create_box(
            scene=self,
            pose=block_pose,
            half_size=(0.04, 0.04, base_half_size),
            color=(0, 0, 0),
            name="box0",
            is_static=True,
        )
        self.block1 = create_box(
            scene=self,
            pose=block_pose_lst[0],
            half_size=(block_half_size, block_half_size, block_half_size),
            color=(np.random.uniform(0, 1), np.random.uniform(0, 1), np.random.uniform(0, 1)),
            name="box1",
        )
        self.block2 = create_box(
            scene=self,
            pose=block_pose_lst[1],
            half_size=(block_half_size, block_half_size, block_half_size),
            color=(np.random.uniform(0, 1), np.random.uniform(0, 1), np.random.uniform(0, 1)),
            name="box2",
        )
        self.block3 = create_box(
            scene=self,
            pose=block_pose_lst[2],
            half_size=(block_half_size, block_half_size, block_half_size),
            color=(np.random.uniform(0, 1), np.random.uniform(0, 1), np.random.uniform(0, 1)),
            name="box3",
        )

    def play_once(self):
        self.move(
            self.move_to_pose(ArmTag("left"), [-0.25, 0, 0.85, 0, 1, 0, 0]),
            self.move_to_pose(ArmTag("right"), [0.3, 0, 0.9, 0, 0, 0, 1]),
        )
        arm_tag = ArmTag(["left", "right"][np.random.randint(0, 2)])
        if arm_tag == ArmTag("left"):
            f = 1
            a_q = np.array([0, 1, 0, 0])
            b_q = np.array([0, 0, 0, 1])
        else:
            f = -1
            a_q = np.array([0, 0, 0, 1])
            b_q = np.array([0, 1, 0, 0])

        pose_a = self.block1.get_pose()
        pose_a.p += [-0.2 * f, 0, 0]
        pose_a.q = a_q

        pose_b = self.block2.get_pose()
        pose_b.p += [0.2 * f, 0, 0]
        pose_b.q = b_q
        self.move(
            self.move_to_pose(arm_tag, pose_a),
            self.move_to_pose(arm_tag.opposite, pose_b),
        )

        pose_a.p += [0.05 * f, 0, 0]
        pose_b.p += [-0.05 * f, 0, 0]
        self.move(
            self.move_to_pose(arm_tag, pose_a),
            self.move_to_pose(arm_tag.opposite, pose_b),
        )
        self.move(
            self.close_gripper(arm_tag),
            self.close_gripper(arm_tag.opposite)
        )


        self.move(
            self.move_by_displacement(arm_tag=arm_tag, x=-0.1 * f),
            self.move_by_displacement(arm_tag=arm_tag.opposite, z=0.02),
        )
        

        self.move(
            self.move_by_displacement(arm_tag=arm_tag, z=0.02 + self.size * 2),
            self.move_by_displacement(arm_tag=arm_tag.opposite, z=-0.02 - self.size),
        )
    

        target_pose = self.block3.get_pose().p
        current_pose = self.block1.get_pose().p
        self.move(
            self.move_by_displacement(arm_tag=arm_tag, x=target_pose[0] - current_pose[0], y=target_pose[1] - current_pose[1]),
            self.open_gripper(arm_tag.opposite),
        )
        target_pose = self.block3.get_pose().p
        current_pose = self.block1.get_pose().p
        self.move(
            self.move_by_displacement(arm_tag=arm_tag, x=target_pose[0] - current_pose[0], y=target_pose[1] - current_pose[1], z=target_pose[2] - current_pose[2] + self.size)
        )

        self.move(
            self.open_gripper(arm_tag)
        )
        
        return self.info

    def check_success(self):
        block0_pose = self.block0.get_pose().p
        block1_pose = self.block1.get_pose().p
        block2_pose = self.block2.get_pose().p
        block3_pose = self.block3.get_pose().p

        eps1 = [0.04, 0.04]
        eps2 = [self.size / 2, self.size / 2]
        score = 0.0
        if np.all(abs(block0_pose[:2] - block2_pose[:2]) < eps1) \
            and np.all(abs(block2_pose[:2] - block3_pose[:2]) < eps2) \
            and block3_pose[2] > block2_pose[2] + 0.03:
            score += 0.5
            if np.all(abs(block3_pose[:2] - block1_pose[:2]) < eps2) \
                and block1_pose[2] > block3_pose[2] + 0.03:
                score += 0.5

        if not self.is_left_gripper_open() or not self.is_right_gripper_open():
            score = min(score, 0.5)

        self.stage_eval_score = score

        return score == 1.0