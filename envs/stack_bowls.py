from ._base_task import Base_Task
from .utils import *
import sapien
import math


class stack_bowls(Base_Task):

    def setup_demo(self, **kwags):
        super()._init_task_env_(**kwags)

    def load_actors(self):
        bowl_pose_lst = []
        for i in range(4):
            bowl_pose = rand_pose(
                xlim=[-0.3, 0.3],
                ylim=[-0.15, 0.15],
                qpos=[0.5, 0.5, 0.5, 0.5],
                ylim_prop=True,
                rotate_rand=False,
            )

            def check_bowl_pose(bowl_pose):
                cnt = 0
                for j in range(len(bowl_pose_lst)):
                    if (np.sum(pow(bowl_pose.p[:2] - bowl_pose_lst[j].p[:2], 2)) < 0.0169):
                        return False
                    if bowl_pose_lst[j].p[0] < 0:
                        cnt += 1
                if bowl_pose.p[0] < 0:
                    cnt += 1
                if cnt > 2 or len(bowl_pose_lst) + 1 - cnt > 2:
                    return False
                return True

            while (abs(bowl_pose.p[0]) < 0.09 or np.sum(pow(bowl_pose.p[:2] - np.array([0, -0.1]), 2)) < 0.0169
                   or not check_bowl_pose(bowl_pose)):
                bowl_pose = rand_pose(
                    xlim=[-0.3, 0.3],
                    ylim=[-0.15, 0.15],
                    qpos=[0.5, 0.5, 0.5, 0.5],
                    ylim_prop=True,
                    rotate_rand=False,
                )
            bowl_pose_lst.append(deepcopy(bowl_pose))

        bowl_pose_lst = sorted(bowl_pose_lst, key=lambda x: x.p[0])

        def create_bowl(bowl_pose, id):
            return create_actor(self, pose=bowl_pose, modelname="002_bowl", model_id=id, convex=True)

        model_id = 3 #np.random.randint(1, 8)
        self.bowl1 = create_bowl(bowl_pose_lst[0], model_id)
        self.bowl2 = create_bowl(bowl_pose_lst[1], model_id)
        self.bowl3 = create_bowl(bowl_pose_lst[2], model_id)
        self.bowl4 = create_bowl(bowl_pose_lst[3], model_id)

        self.add_prohibit_area(self.bowl1, padding=0.07)
        self.add_prohibit_area(self.bowl2, padding=0.07)
        self.add_prohibit_area(self.bowl3, padding=0.07)
        self.add_prohibit_area(self.bowl4, padding=0.07)

        target_pose = [-0.1, -0.15, 0.1, -0.05]
        self.prohibited_area.append(target_pose)
        self.bowl_target_pose = np.array([0, -0.1, 0.76])
        self.quat_of_target_pose =  [0, 0.707, 0.707, 0]

    def play_once(self):
        arm_tag = ArmTag('left')
        move_list_a = [self.bowl1, self.bowl2]
        move_list_b = [self.bowl3, self.bowl4]

        if np.random.rand() < 0.5:
            arm_tag = arm_tag.opposite
            move_list_a, move_list_b = move_list_b, move_list_a
        if np.random.rand() < 0.5:
            move_list_a = move_list_a[::-1]
        if np.random.rand() < 0.5:
            move_list_b = move_list_b[::-1]
        
        last_pose = self.bowl_target_pose
        place_height = 0.06
        lift_height = 0.05
        back_height = 0.1

        self.move(
            self.grasp_actor(
                move_list_a[0],
                arm_tag=arm_tag,
                contact_point_id=[0, 2][int(arm_tag == "left")],
            ),
            self.grasp_actor(
                move_list_b[0],
                arm_tag=arm_tag.opposite,
                contact_point_id=[0, 2][int(arm_tag.opposite == "left")],
            )
        )

        self.move(
            self.move_by_displacement(arm_tag, z=lift_height),
            self.move_by_displacement(arm_tag.opposite, z=lift_height),
        )

        self.move(
            self.place_actor(
                move_list_a[0],
                target_pose=last_pose.tolist() + self.quat_of_target_pose,
                arm_tag=arm_tag,
                functional_point_id=0,
                pre_dis=0.09,
                dis=0,
                constrain="align",
            )
        )
        last_pose = move_list_a[0].get_pose().p + [0, 0, place_height]
        self.move(self.move_by_displacement(arm_tag, z=back_height))

        self.move(
            self.grasp_actor(
                move_list_a[1],
                arm_tag=arm_tag,
                contact_point_id=[0, 2][int(arm_tag == "left")],
            ),
            self.place_actor(
                move_list_b[0],
                target_pose=last_pose.tolist() + self.quat_of_target_pose,
                arm_tag=arm_tag.opposite,
                functional_point_id=0,
                pre_dis=0.09,
                dis=0,
                constrain="align",
            )
        )
        last_pose = move_list_b[0].get_pose().p + [0, 0, place_height]

        self.move(
            self.move_by_displacement(arm_tag, z=lift_height),
            self.move_by_displacement(arm_tag.opposite, z=back_height),
        )

        self.move(
            self.place_actor(
                move_list_a[1],
                target_pose=last_pose.tolist() + self.quat_of_target_pose,
                arm_tag=arm_tag,
                functional_point_id=0,
                pre_dis=0.09,
                dis=0,
                constrain="align",
            ),
            self.grasp_actor(
                move_list_b[1],
                arm_tag=arm_tag.opposite,
                contact_point_id=[0, 2][int(arm_tag.opposite == "left")],
            )
        )
        last_pose = move_list_a[1].get_pose().p + [0, 0, place_height]
        self.move(
            self.move_by_displacement(arm_tag, z=back_height),
            self.move_by_displacement(arm_tag.opposite, z=lift_height),
        )

        self.move(
            self.back_to_origin(arm_tag=arm_tag),
            self.place_actor(
                move_list_b[1],
                target_pose=last_pose.tolist() + self.quat_of_target_pose,
                arm_tag=arm_tag.opposite,
                functional_point_id=0,
                pre_dis=0.09,
                dis=0,
                constrain="align",
            ),
        )

        self.move(self.move_by_displacement(arm_tag.opposite, z=back_height))

        return self.info

    def check_success(self):
        poses = [self.bowl1.get_pose().p, self.bowl2.get_pose().p, self.bowl3.get_pose().p, self.bowl4.get_pose().p]

        poses = sorted(poses, key=lambda x: x[2])

        eps1 = 0.04
        eps2 = 0.02

        max_stack = [1, 1, 1, 1]

        for i in range(4):
            for j in range(i):
                if np.all(abs(poses[i][:2] - poses[j][:2]) <= eps1) and abs(poses[i][2] - poses[j][2]) > eps2:
                    max_stack[i] = max(max_stack[i], max_stack[j] + 1)
        res = max(max_stack)
        if not self.is_left_gripper_open() or not self.is_right_gripper_open():
            res = min(res, 3)

        self.stage_eval_score = (res - 1) / 3.0
        return res == 4