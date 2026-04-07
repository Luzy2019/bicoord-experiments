from ._base_task import Base_Task
from .utils import *
import sapien
import math


class heat_food(Base_Task):

    def setup_demo(self, is_test=False, **kwags):
        super()._init_task_env_(**kwags)

    def load_actors(self):
        self.model_name = "044_microwave_big"
        self.model_id = 0
        self.microwave = rand_create_sapien_urdf_obj(
            scene=self,
            modelname=self.model_name,
            modelid=self.model_id,
            xlim=[-0.081655845, -0.081655845],
            ylim=[0.18958625, 0.18958625],
            zlim=[0.8, 0.8],
            qpos=[0.707, 0, 0, 0.707],
            fix_root_link=True,
        )

        block_pose = rand_pose(
            xlim=[-0.081655845, -0.081655845],
            ylim=[0.18958625, 0.18958625],
        )

        self.block = create_box(
            scene=self,
            pose=block_pose,
            half_size=(0.1, 0.1, 0.01),
            color=(0,0,0),
        )

        hamburg_pose = rand_pose(
            xlim=[0.2, 0.3],
            ylim=[-0.1, -0.2],
            qpos=[0.5, 0.5, 0.5, 0.5],
            rotate_rand=True,
            rotate_lim=[0, np.pi / 4, 0])
        
        self.hamburg = create_actor(
            self,
            pose=hamburg_pose,
            modelname="006_hamburg",
            model_id=np.random.randint(0,4),
            convex=True,
        )

        self.microwave.set_mass(0.01)
        self.is_open = False

    def play_once(self):
        start_pose = [0.03688782826066017, -0.10763275623321533, 1.0231245756149292, 0.6875327554359507, -0.15794860162099558, 0.1554582490465387, 0.691508266221023]
        end_pose = [-0.29459068179130554, -0.3100886642932892, 1.0761722326278687, 0.8353245219182006, -0.1295189068694262, 0.4807629309736423, 0.23307681147447937]
        left = ArmTag("left")
        right = ArmTag("right")

        microwave_pose = self.microwave.get_pose()

        self.move(self.grasp_actor(self.microwave, arm_tag=left, pre_grasp_dis=0.08, contact_point_id=0),
                  self.grasp_actor(self.hamburg, arm_tag=right, pre_grasp_dis=0.08, contact_point_id=0))

        self.move(
            self.move_to_pose(left, end_pose),
            self.move_by_displacement(right, z=0.08)
        )

        start_qpos = self.microwave.get_qpos()[0]
        for _ in range(2):
            self.move(
                self.grasp_actor(
                    self.microwave,
                    arm_tag=left,
                    pre_grasp_dis=0.0,
                    grasp_dis=0.0,
                    contact_point_id=4,
                ))

            new_qpos = self.microwave.get_qpos()[0]
            if new_qpos - start_qpos <= 0.001:
                break
            start_qpos = new_qpos
            if not self.plan_success:
                break

        limits = self.microwave.get_qlimits()
        qpos = self.microwave.get_qpos()
        if qpos[0] >= limits[0][1] * 0.6:
            self.is_open = True

        self.move(self.move_by_displacement(right, x=-0.3, z=-0.1))
        self.move(self.move_by_displacement(right, quat=[0.707, 0, 0, 0.707]))
        delta_x = self.get_arm_pose(right)[0] + 0.107 
        self.move(self.move_by_displacement(right, x=-delta_x))
        delta_y = -0.017 - self.get_arm_pose(right)[1]
        self.move(self.move_by_displacement(right, y=delta_y))
        self.move(self.open_gripper(right))
        self.move(self.move_by_displacement(right, y=-delta_y))
        self.move(self.move_by_displacement(right, x=0.3+delta_x))

        self.move(
            self.move_to_pose(left, start_pose)
        )



        return self.info

    def in_microwave(self):
        obj_p = self.hamburg.get_pose().p
        block_p = self.block.get_pose().p

        return np.linalg.norm(obj_p - block_p) < 0.15

    def check_success(self):
        limits = self.microwave.get_qlimits()
        qpos = self.microwave.get_qpos()
        if qpos[0] >= limits[0][1] * 0.6:
            self.is_open = True
            self.stage_eval_score = 1 / 3
        if self.is_open and self.in_microwave():
            self.stage_eval_score = 2 / 3

        if self.is_open and self.in_microwave() and qpos[0] < limits[0][1] * 0.05:
            self.stage_eval_score = 1.0
        return self.stage_eval_score == 1.0
