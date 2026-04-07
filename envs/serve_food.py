from ._base_task import Base_Task
from .utils import *
from ._GLOBAL_CONFIGS import *


class serve_food(Base_Task):

    def setup_demo(self, **kwags):
        super()._init_task_env_(**kwags)

    def load_actors(self):

        self.tray_pose = rand_pose(
            xlim=[-0.02, 0.02],
            ylim=[-0.18, -0.22],
            qpos=[0.707, 0.707, 0.0, 0.0],
            rotate_rand=False,
        )

        self.tray = create_actor(
            self,
            pose=self.tray_pose,
            modelname="008_tray",
            model_id=2,
            convex=True,
        )

        self.stand_pose = rand_pose(
            xlim=[-0.02, 0.02],
            ylim=[0.00, 0.02],
            qpos=[0.707, 0.707, 0.0, 0.0],
            rotate_rand=False,
        )

        self.stand = create_actor(
            self,
            pose=self.stand_pose,
            modelname="074_displaystand",
            model_id=1,
            convex=True,
            is_static=True,
        )

        self.hamburg_num = np.random.randint(1,3)
        self.hamburg_success = [False] * self.hamburg_num
        self.fries_num = np.random.randint(1,3)
        self.fries_success = [False] * self.fries_num

        hamburg_pose = rand_pose(
            xlim=[-0.30, -0.15],
            ylim=[-0.05, 0.05],
            qpos=[0.5, 0.5, 0.5, 0.5],
            rotate_rand=True,
            rotate_lim=[0, np.pi / 4, 0])
        
        self.hamburg = create_actor(
            self,
            pose=hamburg_pose,
            modelname="006_hamburg_small",
            model_id=np.random.randint(0,6),
            convex=True,
        )

        if self.hamburg_num == 2:
            hamburg_pose_2 = rand_pose(
                xlim=[-0.30, -0.15],
                ylim=[-0.05, 0.05],
                qpos=[0.5, 0.5, 0.5, 0.5],
                rotate_rand=True,
                rotate_lim=[0, np.pi / 4, 0])
            while np.linalg.norm(hamburg_pose.p - hamburg_pose_2.p) < 0.075:
                hamburg_pose_2 = rand_pose(
                    xlim=[-0.30, -0.15],
                    ylim=[-0.05, 0.05],
                    qpos=[0.5, 0.5, 0.5, 0.5],
                    rotate_rand=True,
                    rotate_lim=[0, np.pi / 4, 0])
            self.hamburg_2 = create_actor(
                self,
                pose=hamburg_pose_2,
                modelname="006_hamburg_small",
                model_id=np.random.randint(0,6),
                convex=True,
            )

        fries_pose = rand_pose(
            xlim=[0.15, 0.30],
            ylim=[-0.05, 0.05],
            qpos=[1, 0, 0, 0],
            rotate_rand=True,
            rotate_lim=[0, 0, 0])
        
        self.fries = create_actor(
            self,
            pose=fries_pose,
            modelname="005_french-fries_small",
            model_id=np.random.randint(0,2),
            convex=True,
        )
        if self.fries_num == 2:
            fries_pose_2 = rand_pose(
                xlim=[0.15, 0.30],
                ylim=[-0.05, 0.05],
                qpos=[1, 0, 0, 0],
                rotate_rand=True,
                rotate_lim=[0, 0, 0])
            while np.linalg.norm(fries_pose.p - fries_pose_2.p) < 0.075:
                fries_pose_2 = rand_pose(
                    xlim=[0.15, 0.30],
                    ylim=[-0.05, 0.05],
                    qpos=[1, 0, 0, 0],
                    rotate_rand=True,
                    rotate_lim=[0, 0, 0])
            self.fries_2 = create_actor(
                self,
                pose=fries_pose_2,
                modelname="005_french-fries_small",
                model_id=np.random.randint(0,2),
                convex=True,
            )

    def play_once(self):
        left = ArmTag("left")
        right = ArmTag("right")
        self.move(
            self.grasp_actor(self.hamburg, arm_tag=left),
            self.grasp_actor(self.fries, arm_tag=right)
        )
        self.move(
            self.move_by_displacement(left, z=0.12),
            self.move_by_displacement(right, z=0.12)
        )

        tray_place_pose_left = self.tray.get_functional_point(0)
        tray_place_pose_left[0] += 0.02
        tray_place_pose_left[1] += 0.06
        tray_place_pose_right = self.tray.get_functional_point(1)
        tray_place_pose_right[0] -= 0.02
        tray_place_pose_right[1] -= 0.06

        self.move(self.place_actor(self.hamburg, arm_tag=left, target_pose=tray_place_pose_left, functional_point_id=0, constrain="free", pre_dis=0.1, pre_dis_axis='fp'), 
                  self.place_actor(self.fries, arm_tag=right, target_pose=tray_place_pose_right, functional_point_id=0, constrain="free", pre_dis=0.1, pre_dis_axis='fp'))

        self.move(
            self.move_by_displacement(left, z=0.12),
            self.move_by_displacement(right, z=0.12)
        )

        tray_place_pose_left = self.tray.get_functional_point(0)
        tray_place_pose_left[0] += 0.02
        tray_place_pose_left[1] -= 0.06
        tray_place_pose_right = self.tray.get_functional_point(1)
        tray_place_pose_right[0] -= 0.02
        tray_place_pose_right[1] += 0.06

        if self.hamburg_num == 2 and self.fries_num == 2:
            self.move(
                self.grasp_actor(self.hamburg_2, arm_tag=left),
                self.grasp_actor(self.fries_2, arm_tag=right)
            )
            self.move(
                self.move_by_displacement(left, z=0.12),
                self.move_by_displacement(right, z=0.12)
            )

            self.move(self.place_actor(self.hamburg_2, arm_tag=left, target_pose=tray_place_pose_left, functional_point_id=0, constrain="free", pre_dis=0.1, pre_dis_axis='fp'), 
                    self.place_actor(self.fries_2, arm_tag=right, target_pose=tray_place_pose_right, functional_point_id=0, constrain="free", pre_dis=0.1, pre_dis_axis='fp'))
            self.move(
                self.move_by_displacement(left, x=-0.075, z=0.12),
                self.move_by_displacement(right, x=0.075, z=0.12)
            )
        elif self.hamburg_num == 2 and self.fries_num == 1:
            self.move(
                self.grasp_actor(self.hamburg_2, arm_tag=left),
            )
            self.move(
                self.move_by_displacement(left, z=0.12),
                self.move_by_displacement(right, x=0.075)
            )

            self.move(self.place_actor(self.hamburg_2, arm_tag=left, target_pose=tray_place_pose_left, functional_point_id=0, constrain="free", pre_dis=0.1, pre_dis_axis='fp'))
            self.move(
                self.move_by_displacement(left, x=-0.075, z=0.12),
            )
        elif self.hamburg_num == 1 and self.fries_num == 2:
            self.move(
                self.grasp_actor(self.fries_2, arm_tag=right),
            )
            self.move(
                self.move_by_displacement(right, z=0.12),
                self.move_by_displacement(left, x=-0.075)
            )

            self.move(self.place_actor(self.fries_2, arm_tag=right, target_pose=tray_place_pose_right, functional_point_id=0, constrain="free", pre_dis=0.1, pre_dis_axis='fp'))
            self.move(
                self.move_by_displacement(left, x=-0.075),
                self.move_by_displacement(right, x=0.075, z=0.12)
            )
        else:
            self.move(
                self.move_by_displacement(left, x=-0.075),
                self.move_by_displacement(right, x=0.075)
            )

        left_pose=np.array(self.get_arm_pose(left))
        left_pose[:3] = self.tray_pose.p + np.array([-0.125, 0, 0.14])
        left_pose[3:] = np.array([0.5, -0.5, 0.5, 0.5])
        right_pose=np.array(self.get_arm_pose(right))
        right_pose[:3] = self.tray_pose.p + np.array([0.125, 0, 0.14])
        right_pose[3:] = np.array([0.5, -0.5, 0.5, 0.5])

        self.move(
            self.move_to_pose(left, left_pose),
            self.move_to_pose(right, right_pose)
        )

        self.move(
            self.close_gripper(left),
            self.close_gripper(right)
        )

        self.move(
            self.move_by_displacement(left, z=0.12),
            self.move_by_displacement(right, z=0.12)
        )

        stand_p = self.stand.get_pose().p
        tray_p = self.tray.get_pose().p

        self.move(
            self.move_by_displacement(left, x=stand_p[0] - tray_p[0], y=stand_p[1] - tray_p[1]),
            self.move_by_displacement(right, x=stand_p[0] - tray_p[0], y=stand_p[1] - tray_p[1])
        )
        self.move(
            self.open_gripper(left),
            self.open_gripper(right)
        )

        return self.info
    
    def in_tray(self, obj):
        obj_p = obj.get_pose().p
        tray_p = self.tray.get_pose().p
        return np.linalg.norm(obj_p - tray_p) < 0.2


    def check_success(self):
        tray_p = self.tray.get_pose().p
        stand_p = self.stand.get_pose().p
        hamburg_p = self.hamburg.get_pose().p
        fries_p = self.fries.get_pose().p
        total_num = self.hamburg_num + self.fries_num
        success_obj_num = 0
        if self.in_tray(self.hamburg):
            success_obj_num += 1
        if self.hamburg_num == 2 and self.in_tray(self.hamburg_2):
            success_obj_num += 1
        if self.in_tray(self.fries):
            success_obj_num += 1
        if self.fries_num == 2 and self.in_tray(self.fries_2):
            success_obj_num += 1
        self.stage_eval_score = success_obj_num / total_num * 0.75

        if success_obj_num == total_num and np.linalg.norm(stand_p - tray_p) < 0.1:
            self.stage_eval_score = 1

        return self.stage_eval_score == 1