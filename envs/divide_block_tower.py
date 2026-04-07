from ._base_task import Base_Task
from .utils import *
import sapien
import math
import numpy as np


class divide_block_tower(Base_Task):

    def setup_demo(self, **kwags):
        super()._init_task_env_(**kwags)

    def load_actors(self):
        colors = [
            (1.0, 0.0, 0.0),  # Red
            (0.0, 1.0, 0.0),  # Green
            (0.0, 0.0, 1.0),  # Blue
            (1.0, 1.0, 0.0),  # Yellow
            (1.0, 0.0, 1.0),  # Magenta
            (0.0, 1.0, 1.0),  # Cyan
        ]
        color_ids = np.random.choice(len(colors), 2, replace=False)
        self.color_1 = colors[color_ids[0]]
        self.color_2 = colors[color_ids[1]]
        block_color = [self.color_1, self.color_2] * 2
        np.random.shuffle(block_color)
        self.blocks = []
        self.blocks_1 = []
        self.blocks_2 = []
        self.blocks_belong = {}

        block_half_size = np.random.uniform(0.02, 0.025)
        self.size = block_half_size * 2
        base_half_size = 0.03

        block_pose = rand_pose(
            xlim=[-0.05, 0],
            ylim=[0.05, 0.1],
            zlim=[0.741 + base_half_size],
            qpos=[1, 0, 0, 0],
            rotate_rand=False,
        )

        self.block_base = create_box(
            scene=self,
            pose=block_pose,
            half_size=(0.04, 0.04, base_half_size),
            color=(0, 0, 0),
            name="base",
            is_static=True,
        )
        for i in range(4):
            new_pose = deepcopy(block_pose)
            new_pose.p += [0, 0, base_half_size + self.size * (i + 0.5)]
            self.blocks.append(create_box(
                scene=self,
                pose=new_pose,
                half_size=(block_half_size, block_half_size, block_half_size),
                color=block_color[i],
                name=f"block{i}",
            ))
            if block_color[i] == self.color_1:
                self.blocks_belong[self.blocks[i].get_name()] = 1
                self.blocks_1.append(self.blocks[i])
            else:
                self.blocks_belong[self.blocks[i].get_name()] = 2
                self.blocks_2.append(self.blocks[i])
        
        
        target_pos_1 = rand_pose(
            xlim=[-0.2, -0.15],
            ylim=[-0.1, -0.05],
            rotate_rand=False,
        )
        target_pos_2 = rand_pose(
            xlim=[0.10, 0.15],
            ylim=[-0.1, -0.05],
            rotate_rand=False,
        )
        self.target_1 = create_box(
            scene=self,
            pose=target_pos_1,
            half_size=(0.05, 0.05, 0.00005),
            color=self.color_1,
            is_static=True,
        )
        create_box(
            scene=self,
            pose=target_pos_1,
            half_size=(0.045, 0.045, 0.00005),
            color=(1, 1, 1),
            is_static=True,
        )
        self.target_2 = create_box(
            scene=self,
            pose=target_pos_2,
            half_size=(0.05, 0.05, 0.00005),
            color=self.color_2,
            is_static=True,
        )
        create_box(
            scene=self,
            pose=target_pos_2,
            half_size=(0.045, 0.045, 0.00005),
            color=(1, 1, 1),
            is_static=True,
        )


    def pick_and_place_block(self, step, targets):
        left = ArmTag("left")
        right = ArmTag("right")
        pre_dis = 0.25
        pre_dis_offset = 0.1
        lift_height = 0.03
        pull_dis = 0.1
        crawl_height = 0.25
        if step[0] is None or step[1] is None:
            if step[0] is None:
                arm_tag = right
                self.move(
                    self.move_to_pose(right, [0.3, 0, 0.9, 0, 0, 0, 1]),
                )
                pose = step[1].get_pose()
                pose.p += [pre_dis, 0, 0]
                pose.q = np.array([0, 0, 0, 1])
                f = -1
                target_pose = targets[1].get_pose()
                target_pose.p += [0, 0, crawl_height]
                target_pose.q = np.array(np.array([0.707, 0, 0.707, 0]))
            else:
                arm_tag = left
                self.move(
                    self.move_to_pose(left, [-0.3, 0, 0.9, 1, 0, 0, 0]),
                )
                pose = step[0].get_pose()
                pose.p += [-pre_dis, 0, 0]
                pose.q = np.array([1, 0, 0, 0])
                f = 1
                target_pose = targets[0].get_pose()
                target_pose.p += [0.01, 0, crawl_height]
                target_pose.q = np.array(np.array([0, -0.707, 0, 0.707]))

            self.move(self.move_to_pose(arm_tag, pose))

            pose.p += [pre_dis_offset * f, 0, 0]
            self.move(self.move_to_pose(arm_tag, pose))
            self.move(self.close_gripper(arm_tag))
            self.move(self.move_by_displacement(arm_tag, x=-pull_dis * f, z=lift_height))
            self.move(self.move_to_pose(arm_tag, target_pose))
            self.move(self.open_gripper(arm_tag))
        else:
            self.move(
                self.move_to_pose(left, [-0.3, 0, 0.9, 1, 0, 0, 0]),
                self.move_to_pose(right, [0.3, 0, 0.9, 0, 0, 0, 1]),
            )

            pose_a = step[0].get_pose()
            pose_a.p += [-pre_dis, 0, 0]
            pose_a.q = np.array([1, 0, 0, 0])

            pose_b = step[1].get_pose()
            pose_b.p += [pre_dis, 0, 0]
            pose_b.q = np.array([0, 0, 0, 1])
            self.move(
                self.move_to_pose(left, pose_a),
                self.move_to_pose(right, pose_b),
            )


            pose_a.p += [pre_dis_offset, 0, 0]
            pose_b.p += [-pre_dis_offset, 0, 0]
            self.move(
                self.move_to_pose(left, pose_a),
                self.move_to_pose(right, pose_b),
            )
            self.move(
                self.close_gripper(left),
                self.close_gripper(right)
            )

            self.move(
                self.move_by_displacement(left, x=-pull_dis, z=lift_height),
                self.move_by_displacement(right, x=pull_dis, z=lift_height),
            )
            
            
            target_pose_a = targets[0].get_pose()
            target_pose_a.p += [0.01, 0, crawl_height]
            target_pose_a.q = np.array(np.array([0, -0.707, 0, 0.707]))
            target_pose_b = targets[1].get_pose()
            target_pose_b.p += [0, 0, crawl_height]
            target_pose_b.q = np.array(np.array([0.707, 0, 0.707, 0]))
            self.move(
                self.move_to_pose(left, target_pose_a),
                self.move_to_pose(right, target_pose_b),
            )

            self.move(
                self.open_gripper(left),
                self.open_gripper(right)
            )

        return self.info

    def play_once(self):
        steps = []
        if self.blocks_belong[self.blocks[2].get_name()] == self.blocks_belong[self.blocks[3].get_name()]:
            steps.append((self.blocks[3], None))
            steps.append((self.blocks[1], self.blocks[2]))
            steps.append((self.blocks[0], None))
        else:
            steps.append((self.blocks[2], self.blocks[3]))
            steps.append((self.blocks[0], self.blocks[1]))
        
        targets = [self.target_1, self.target_2]
        for i in range(len(steps)):
            if self.blocks_belong[steps[i][0].get_name()] == 2:
                steps[i] = (steps[i][1], steps[i][0])
            self.pick_and_place_block(steps[i], targets)
            old_targets = targets
            targets = []
            for j in range(2):
                if steps[i][j] is not None:
                    targets.append(steps[i][j])
                else:
                    targets.append(old_targets[j])

        return self.info

    def check_success(self):
        eps = 0.05
        
        def check(blocks, target_pos):
            block_lst = sorted(blocks, key=lambda x: x.get_pose().p[2])
            pose0 = block_lst[0].get_pose().p
            pose1 = block_lst[1].get_pose().p
            divide = int(np.all(abs(pose0[:2] - target_pos[:2]) <= eps)) + int(np.all(abs(pose1[:2] - target_pos[:2]) <= eps))
            stack = int(np.all(abs(pose1[:2] - pose0[:2]) <= self.size / 2) and pose1[2] > pose0[2] + self.size - 0.01)
            return divide, stack

        devide_1, stack_1 = check(self.blocks_1, self.target_1.get_pose().p)
        devide_2, stack_2 = check(self.blocks_2, self.target_2.get_pose().p)
        divide_cnt = devide_1 + devide_2
        stack_cnt = stack_1 + stack_2
        if not self.is_left_gripper_open() or not self.is_right_gripper_open():
            stack_cnt = 0

        self.stage_eval_score = divide_cnt * 0.125 + stack_cnt * 0.25
        return divide_cnt == 4 and stack_cnt == 2