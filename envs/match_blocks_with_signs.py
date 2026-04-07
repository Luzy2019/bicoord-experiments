from ._base_task import Base_Task
from .utils import *
import sapien
import math
import numpy as np


class match_blocks_with_signs(Base_Task):

    def setup_demo(self, **kwags):
        super()._init_task_env_(**kwags)

    def load_actors(self):
        self.block_half_sizes = [0.02, 0.025, 0.03]
        self.block_half_height = 0.06
        colors = [
            (1.0, 0.0, 0.0),  # Red
            (0.0, 1.0, 0.0),  # Green
            (0.0, 0.0, 1.0),  # Blue
            (1.0, 1.0, 0.0),  # Yellow
            (1.0, 0.0, 1.0),  # Magenta
            (0.0, 1.0, 1.0),  # Cyan
        ]
        color_ids = np.random.choice(len(colors), 3, replace=False)
        self.colors = [colors[i] for i in color_ids]

        target_poses = []
        target_poses.append(rand_pose(
            xlim=[0.15, 0.15],
            ylim=[0.15, 0.15],
            rotate_rand=False,
        ))
        target_poses.append(rand_pose(
            xlim=[0.15, 0.15],
            ylim=[0.03, 0.03],
            rotate_rand=False,
        ))
        target_poses.append(rand_pose(
            xlim=[0.15, 0.15],
            ylim=[-0.1, -0.1],
            rotate_rand=False,
        ))

        block_poses = []
        block_poses.append(rand_pose(
            xlim=[-0.3, -0.3],
            ylim=[-0.05, -0.05],
            zlim = [0.741 + self.block_half_height],
            qpos=[1, 0, 0, 0],
            rotate_rand=False,
        ))
        block_poses.append(rand_pose(
            xlim=[-0.2, -0.2],
            ylim=[-0.05, -0.05],
            zlim = [0.741 + self.block_half_height],
            qpos=[1, 0, 0, 0],
            rotate_rand=False,
        ))
        block_poses.append(rand_pose(
            xlim=[-0.1, -0.1],
            ylim=[-0.05, -0.05],
            zlim = [0.741 + self.block_half_height],
            qpos=[1, 0, 0, 0],
            rotate_rand=False,
        ))
        
        
        np.random.shuffle(self.block_half_sizes)
        np.random.shuffle(block_poses)
        self.targets = []
        self.blocks = []
        for i in range(3):
            self.targets.append(create_box(
                scene=self,
                pose=target_poses[i],
                half_size=(self.block_half_sizes[i], self.block_half_sizes[i], 0.00005),
                color=(0.5, 0.5, 0.5),
                is_static=True,
            ))
            self.blocks.append(create_box(
                scene=self,
                pose=block_poses[i],
                half_size=(self.block_half_sizes[i], self.block_half_height, self.block_half_sizes[i]),
                color=self.colors[i],
            ))

    def get_left_target_pose(self, block_id):
        block_pose = self.blocks[block_id].get_pose()
        pose = list(block_pose.p + np.array([0.01, -self.block_half_height + 0.02, 0.25]))
        quat = [0.5, -0.5, 0.5, 0.5]
        return pose + quat

    def get_right_target_pose(self, target_id):
        target_pose = self.targets[target_id].get_pose()
        pose = list(target_pose.p + np.array([0.15, -0.01, 0.1]))
        quat = [0, 0, 0, 1]
        return pose + quat

    def move_right_to_target(self, target_id):
        right = ArmTag('right')
        block_pose = self.blocks[target_id].get_pose().p
        target_pose = self.targets[target_id].get_pose().p
        target_pose += np.array([0, -target_id * 0.01, 0.09])
        delta = target_pose - block_pose
        return self.move_by_displacement(right, x=delta[0], y=delta[1], z=delta[2])
    
    def handover_block(self):
        left = ArmTag('left')
        right = ArmTag('right')
        self.move(
            self.move_to_pose(left, [-0.18, -0.1, 0.95, 1, 0, 0, 0]),
            self.move_to_pose(right, [0.18, -0.09, 1.0, 0, 0, 0, 1])
        )
        self.move(
            self.move_by_displacement(left, x=0.04),
            self.move_by_displacement(right, x=-0.04)
        )
        self.move(
            self.close_gripper(right)
        )
        self.move(
            self.open_gripper(left)
        )
        self.move(
            self.move_by_displacement(left, x=-0.04),
            self.move_by_displacement(right, x=0.04)
        )

    def work_both(self, left_id, right_id):
        left = ArmTag('left')
        right = ArmTag('right')
        self.move(
            self.move_to_pose(left, self.get_left_target_pose(left_id)),
            self.move_right_to_target(right_id)
        )
        self.move(
            self.move_by_displacement(left, z=-0.1),
            self.move_by_displacement(right, z=-0.03)
        )
        self.move(
            self.close_gripper(left),
            self.open_gripper(right)
        )
        self.move(
            self.move_by_displacement(left, z=0.1),
            self.move_by_displacement(right, z=0.1)
        )


    def play_once(self):
        left = ArmTag('left')
        right = ArmTag('right')
        self.move(
            self.move_to_pose(left, self.get_left_target_pose(0))
        )
        self.move(
            self.move_by_displacement(left, z=-0.1)
        )
        self.move(
            self.close_gripper(left)
        )
        self.move(
            self.move_by_displacement(left, z=0.1)
        )
        self.handover_block()
        self.work_both(1, 0)
        self.handover_block()
        self.work_both(2, 1)
        self.handover_block()
        self.move(
            self.back_to_origin(left),
            self.move_right_to_target(2)
        )
        self.move(
            self.move_by_displacement(right, z=-0.03)
        )
        self.move(
            self.open_gripper(right)
        )

        return self.info

    def check_success(self):
        cnt = 0
        for i in range(3):
            block_pose = self.blocks[i].get_pose().p
            target_pose = self.targets[i].get_pose().p
            # print(block_pose, target_pose)
            if np.all(abs(block_pose[:2] - target_pose[:2]) <= 0.03) and abs(block_pose[2] - target_pose[2]) > self.block_half_height - 0.01:
                cnt += 1
        self.stage_eval_score = cnt / 3.0
        return cnt == 3