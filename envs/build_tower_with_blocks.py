from ._base_task import Base_Task
from .utils import *
import sapien
import math
import numpy as np
import copy

class build_tower_with_blocks(Base_Task):

    def setup_demo(self, **kwags):
        super()._init_task_env_(**kwags)

    def load_actors(self):
        self.half_sizes = [np.random.uniform(0.015, 0.019), np.random.uniform(0.02, 0.024), np.random.uniform(0.026, 0.03)]

        self.target_pose = rand_pose(
            xlim=[0, 0],
            ylim=[-0.15, -0.1],
            qpos=[0, 1, 0, 0],
            rotate_rand=False,
        )


        while True:
            block_poses = []
            cnt = 0
            for i in range(3):
                pose = rand_pose(
                    xlim=[-0.25, 0.25],
                    ylim=[-0.1, 0.15],
                    zlim=[0.741 + self.half_sizes[i]],
                    qpos=[1, 0, 0, 0],
                    rotate_rand=True,
                    rotate_lim=[0, 0, 0.75],
                )
                block_poses.append(pose)
                if pose.p[0] < 0:
                    cnt += 1
            if cnt == 0 or cnt == 3:
                continue

            flag = True
            for i in range(3):
                if np.any(np.abs(block_poses[i].p[:2] - self.target_pose.p[:2]) < 0.05):
                    flag = False
                for j in range(i):
                    if np.all(np.abs(block_poses[i].p[:2] - block_poses[j].p[:2]) < 0.06):
                        flag = False
            if not flag:
                continue
            break
        
        self.blocks = []
        for i in range(3):
            self.blocks.append(create_box(
                scene=self,
                pose=block_poses[i],
                half_size=(self.half_sizes[i], self.half_sizes[i], self.half_sizes[i]),
                color=(np.random.uniform(0, 1), np.random.uniform(0, 1), np.random.uniform(0, 1)),
                name=f"block{i}",
            ))

        self.target_poses = []
        for i in range(2, -1, -1):
            self.target_pose.p += [0, 0, self.half_sizes[i]]
            self.target_poses.append(copy.deepcopy(self.target_pose))
            self.target_pose.p += [0, 0, self.half_sizes[i]]
        self.target_poses = self.target_poses[::-1]

    def play_once(self):
        cnt = [[], []]
        for i in range(3):
            if self.blocks[i].get_pose().p[0] < 0:
                cnt[0].append(i)
            else:
                cnt[1].append(i)

        
        if len(cnt[0]) == 1:
            armtag = ArmTag('right')
            f = 1
            change = cnt[0][0]
        else:
            armtag = ArmTag('left')
            f = -1
            change = cnt[1][0]

        if change == 2:
            self.move(
                self.grasp_actor(arm_tag=armtag.opposite, actor=self.blocks[2]),
                self.grasp_actor(arm_tag=armtag, actor=self.blocks[1])
            )
            self.move(
                self.place_actor(arm_tag=armtag.opposite, actor=self.blocks[2], target_pose=self.target_poses[2], functional_point_id=0, pre_dis=0.05, dis=0.01)
            )
            self.move(
                self.move_by_displacement(arm_tag=armtag.opposite, z=0.05)
            )
            self.move(
                self.back_to_origin(arm_tag=armtag.opposite),
                self.place_actor(arm_tag=armtag, actor=self.blocks[1], target_pose=self.target_poses[1], functional_point_id=0, pre_dis=0.05, dis=0.01)
            )
            self.move(
                self.move_by_displacement(arm_tag=armtag, z=0.05)
            )
            self.move(
                self.grasp_actor(arm_tag=armtag, actor=self.blocks[0])
            )
            self.move(
                self.place_actor(arm_tag=armtag, actor=self.blocks[0], target_pose=self.target_poses[0], functional_point_id=0, pre_dis=0.05, dis=0.01)
            )
        elif change == 1:
            self.move(
                self.grasp_actor(arm_tag=armtag, actor=self.blocks[2]),
                self.grasp_actor(arm_tag=armtag.opposite, actor=self.blocks[1])
            )
            self.move(
                self.place_actor(arm_tag=armtag, actor=self.blocks[2], target_pose=self.target_poses[2], functional_point_id=0, pre_dis=0.05, dis=0.01)
            )
            self.move(
                self.move_by_displacement(arm_tag=armtag, z=0.05)
            )
            self.move(
                self.grasp_actor(arm_tag=armtag, actor=self.blocks[0]),
                self.place_actor(arm_tag=armtag.opposite, actor=self.blocks[1], target_pose=self.target_poses[1], functional_point_id=0, pre_dis=0.05, dis=0.01)
            )
            self.move(
                self.move_by_displacement(arm_tag=armtag.opposite, z=0.05)
            )
            self.move(
                self.place_actor(arm_tag=armtag, actor=self.blocks[0], target_pose=self.target_poses[0], functional_point_id=0, pre_dis=0.05, dis=0.01),
                self.back_to_origin(arm_tag=armtag.opposite)
                
            )
        else:
            self.move(
                self.grasp_actor(arm_tag=armtag, actor=self.blocks[2])
            )
            self.move(
                self.place_actor(arm_tag=armtag, actor=self.blocks[2], target_pose=self.target_poses[2], functional_point_id=0, pre_dis=0.05, dis=0.01)
            )
            self.move(
                self.move_by_displacement(arm_tag=armtag, z=0.05)
            )
            self.move(
                self.grasp_actor(arm_tag=armtag, actor=self.blocks[1]),
                self.grasp_actor(arm_tag=armtag.opposite, actor=self.blocks[0])
            )
            self.move(
                self.place_actor(arm_tag=armtag, actor=self.blocks[1], target_pose=self.target_poses[1], functional_point_id=0, pre_dis=0.05, dis=0.01)
            )
            self.move(
                self.move_by_displacement(arm_tag=armtag, z=0.05)
            )
            self.move(
                self.back_to_origin(arm_tag=armtag),
                self.place_actor(arm_tag=armtag.opposite, actor=self.blocks[0], target_pose=self.target_poses[0], functional_point_id=0, pre_dis=0.05, dis=0.01)
            )

        return self.info

    def check_success(self):
        cnt = 0
        for i in range(2):
            if np.any(np.abs(self.blocks[i].get_pose().p[:2] - self.blocks[i + 1].get_pose().p[:2]) > self.half_sizes[i + 1]):
                continue
            if abs(self.blocks[i].get_pose().p[2] - self.blocks[i + 1].get_pose().p[2]) < self.half_sizes[i] + self.half_sizes[i + 1] - 0.01:
                continue
            cnt += 1
        
        if not self.is_left_gripper_open() or not self.is_right_gripper_open():
            cnt = min(cnt, 1)

        self.stage_eval_score = cnt / 2.0
        return cnt == 2