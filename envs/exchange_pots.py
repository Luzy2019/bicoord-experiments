from ._base_task import Base_Task
from .utils import *
import sapien
import math


class exchange_pots(Base_Task):

    def setup_demo(self, is_test=False, **kwags):
        super()._init_task_env_(**kwags)

    def load_actors(self):
        x = 0.15
        target_rand_pos = rand_pose(
            xlim=[-x, -x],
            ylim=[-0.169, -0.171],
            qpos=[0.5, 0.5, 0.5, 0.5],
        )
        target_rand_pos_2 = rand_pose(
            xlim=[x, x],
            ylim=[0.045, 0.047],
            qpos=[0.5, 0.5, 0.5, 0.5],
        )

        self.target_rand_pos = target_rand_pos
        self.target_rand_pos_2 = target_rand_pos_2


        self.model_name = "060_kitchenpot"
        self.model_id = 0
        self.pot = rand_create_sapien_urdf_obj(
            scene=self,
            modelname=self.model_name,
            modelid=self.model_id,
            xlim=[target_rand_pos.p[0], target_rand_pos.p[0]],
            ylim=[target_rand_pos.p[1], target_rand_pos.p[1]],
            qpos=[0.704141, 0, 0, 0.71006],
        )
        self.pot_2 = rand_create_sapien_urdf_obj(
            scene=self,
            modelname=self.model_name,
            modelid=self.model_id,
            xlim=[target_rand_pos_2.p[0], target_rand_pos_2.p[0]],
            ylim=[target_rand_pos_2.p[1], target_rand_pos_2.p[1]],
            qpos=[0.704141, 0, 0, 0.71006],
        )

        x, y = self.pot.get_pose().p[0], self.pot.get_pose().p[1]
        self.prohibited_area.append([x - 0.3, y - 0.1, x + 0.3, y + 0.1])
        x, y = self.pot_2.get_pose().p[0], self.pot_2.get_pose().p[1]
        self.prohibited_area.append([x - 0.3, y - 0.1, x + 0.3, y + 0.1])
        



    def play_once(self):
        left_arm_tag = ArmTag("left")
        right_arm_tag = ArmTag("right")

        self.move(
            self.close_gripper(left_arm_tag, pos=0.5),
            self.close_gripper(right_arm_tag, pos=0.5),
        )

        self.move(
            self.grasp_actor(self.pot, left_arm_tag, pre_grasp_dis=0.035, contact_point_id=0),
            self.grasp_actor(self.pot, right_arm_tag, pre_grasp_dis=0.035, contact_point_id=1),
        )

        self.move(
            self.move_by_displacement(left_arm_tag, x=0.4),
            self.move_by_displacement(right_arm_tag, x=0.4),
        )

        self.move(
            self.open_gripper(left_arm_tag),
            self.open_gripper(right_arm_tag),
        )

        self.move(
            self.grasp_actor(self.pot_2, left_arm_tag, pre_grasp_dis=0.035, contact_point_id=0),
            self.grasp_actor(self.pot_2, right_arm_tag, pre_grasp_dis=0.035, contact_point_id=1),
        )

        delta_x = self.target_rand_pos.p[0] - self.target_rand_pos_2.p[0]
        delta_y = self.target_rand_pos.p[1] - self.target_rand_pos_2.p[1]

        self.move(
            self.move_by_displacement(left_arm_tag, x=delta_x / 2, y=delta_y / 2, z=0.2),
            self.move_by_displacement(right_arm_tag, x=delta_x / 2, y=delta_y / 2, z=0.2),
        )

        self.move(
            self.move_by_displacement(left_arm_tag, x=delta_x / 2, y=delta_y / 2, z=-0.2),
            self.move_by_displacement(right_arm_tag, x=delta_x / 2, y=delta_y / 2, z=-0.2),
        )

        self.move(
            self.open_gripper(left_arm_tag),
            self.open_gripper(right_arm_tag),
        )

        self.move(
            self.move_by_displacement(left_arm_tag, x=-0.05),
            self.move_by_displacement(right_arm_tag, x=0.05),
        )

        self.move(
            self.move_by_displacement(left_arm_tag, z=0.15),
            self.move_by_displacement(right_arm_tag, z=0.15),
        )

        self.move(
            self.move_by_displacement(left_arm_tag, x=0.15),
            self.move_by_displacement(right_arm_tag, x=0.3),
        )


        self.move(
            self.grasp_actor(self.pot, left_arm_tag, pre_grasp_dis=0.035, contact_point_id=0),
            self.grasp_actor(self.pot, right_arm_tag, pre_grasp_dis=0.035, contact_point_id=1),
        )

        pos = self.pot.get_pose().p
        delta_x = self.target_rand_pos_2.p[0] - pos[0]
        delta_y = self.target_rand_pos_2.p[1] - pos[1]

        self.move(
            self.move_by_displacement(left_arm_tag, x=delta_x, y=delta_y),
            self.move_by_displacement(right_arm_tag, x=delta_x, y=delta_y),
        )

        return self.info

    def check_success(self):
        pos = self.pot.get_pose().p[:2]
        pos_2 = self.pot_2.get_pose().p[:2]
        target_pos = self.target_rand_pos_2.p[:2]
        target_pos_2 = self.target_rand_pos.p[:2]
        eps = np.array([0.05, 0.05])
        tag = np.all(abs(target_pos - pos) < eps)
        tag_2 = np.all(abs(target_pos_2 - pos_2) < eps)
        self.stage_eval_score = (int(tag) + int(tag_2)) / 2.0

        return self.stage_eval_score == 1.0
