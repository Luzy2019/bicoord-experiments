from ._base_task import Base_Task
from .utils import *
import sapien
import glob


class cook(Base_Task):

    def setup_demo(self, **kwags):
        super()._init_task_env_(**kwags, table_static=False)

    def load_actors(self):
        self.bread_middle_check = 0
        self.bread_middle_check_2 = 0

        id_list = [3]
        self.bread_id = np.random.choice(id_list)
        self.bread_id_2 = np.random.choice(id_list)


        rand_pos = rand_pose(
            xlim=[-0.28, -0.2],
            ylim=[-0.2, 0.05],
            qpos=[0.707, 0.707, 0.0, 0.0],
            rotate_rand=True,
            rotate_lim=[0, np.pi / 4, 0],
        )

        self.bread = create_actor(
            self,
            pose=rand_pos,
            modelname="075_bread",
            model_id=self.bread_id,
            convex=True,
        )

        rand_pos_2 = rand_pose(
            xlim=[-0.28, -0.2],
            ylim=[-0.2, 0.05],
            qpos=[0.707, 0.707, 0.0, 0.0],
            rotate_rand=True,
            rotate_lim=[0, np.pi / 4, 0],
        )
        while abs(rand_pos_2.p[0]) < 0.2 or np.linalg.norm(rand_pos.p - rand_pos_2.p) < 0.1 or rand_pos.p[0] * rand_pos_2.p[0] < 0:
            rand_pos_2 = rand_pose(
                xlim=[-0.28, -0.2],
                ylim=[-0.2, 0.05],
                qpos=[0.707, 0.707, 0.0, 0.0],
                rotate_rand=True,
                rotate_lim=[0, np.pi / 4, 0],
            )
        self.bread_2 = create_actor(
            self,
            pose=rand_pos_2,
            modelname="075_bread",
            model_id=self.bread_id_2,
            convex=True,
        )

        xlim = [0.15, 0.25] if rand_pos.p[0] < 0 else [-0.25, -0.15]
        self.model_id_list = [0]
        self.skillet_id = np.random.choice(self.model_id_list)
        rand_pos = rand_pose(
            xlim=xlim,
            ylim=[-0.15, -0.05],
            qpos=[0, 0, 0.707, 0.707],
            rotate_rand=False,
        )
        self.skillet = create_actor(
            self,
            pose=rand_pos,
            modelname="106_skillet",
            model_id=self.skillet_id,
            convex=True,
        )

        rand_pos_2 = rand_pose(
            xlim=[-0.05, 0],
            ylim=[0.15, 0.20],
            rotate_rand=False,
            qpos=[0.5, 0.5, 0.5, 0.5],
        )

        self.plate = create_actor(
            self,
            pose=rand_pos_2,
            modelname="003_plate_large",
            is_static=True,
            convex=True,
        )

        if self.skillet.get_pose().p[0] > 0:
            rand_pos = rand_pose(
                xlim=[-0.056, -0.056],
                ylim=[-0.056, -0.056],
                rotate_rand=False,
            )
        else:
            rand_pos = rand_pose(
                xlim=[0.049, 0.049],
                ylim=[-0.038, -0.038],
                rotate_rand=False,
            )

        self.fire = create_box(
            scene=self,
            pose=rand_pos,
            half_size=(0.05, 0.05, 0.00005),
            color=(1,0,0),
            is_static=True,
        )

        self.bread.set_mass(0.001)
        self.bread_2.set_mass(0.001)
        self.skillet.set_mass(0.01)
        self.add_prohibit_area(self.skillet, padding=0.03)
        self.add_prohibit_area(self.plate, padding=0.03)

    def play_once(self):
        arm_tag = ArmTag("right" if self.skillet.get_pose().p[0] > 0 else "left")

        self.move(
            self.grasp_actor(self.skillet, arm_tag=arm_tag, pre_grasp_dis=0.07, gripper_pos=0, contact_point_id=0),
            self.grasp_actor(self.bread, arm_tag=arm_tag.opposite, pre_grasp_dis=0.07, gripper_pos=0),
        )

        self.move(
            self.move_by_displacement(arm_tag=arm_tag, z=0.1, move_axis="arm"),
            self.move_by_displacement(arm_tag=arm_tag.opposite, z=0.15),
        )

        target_pose = self.get_arm_pose(arm_tag=arm_tag)
        if arm_tag == "left":
            target_pose[:2] = [-0.1, -0.05]
            target_pose[2] -= 0.05
            target_pose[3:] = [-0.707, 0, -0.707, 0]
            target_bread_pose = [0.049121066786788126, -0.03826127788311616, 0.8295127836414085, 0.6890379245322912, -0.06551997380107899, 0.056440864452009465, 0.7195472885149719]
        else:
            target_pose[:2] = [0.1, -0.05]
            target_pose[2] -= 0.05
            target_pose[3:] = [0, 0.707, 0, -0.707]
            target_bread_pose = [-0.05614512197655093, -0.05602579364675265, 0.8163339467224908, 0.7087255417759555, 0.05500297094322582, 0.06663775669695104, -0.7001729707752188]

        self.move(
            self.move_to_pose(arm_tag=arm_tag, target_pose=target_pose),
            self.place_actor(self.bread, target_pose=target_bread_pose, arm_tag=arm_tag.opposite, constrain="free", pre_dis=0.05, dis=0.05)
        )

        self.move(
            self.move_to_pose(arm_tag=arm_tag, target_pose=target_pose),
            self.grasp_actor(self.bread_2, arm_tag=arm_tag.opposite, pre_grasp_dis=0.07, gripper_pos=0),
        )
        self.move(
            self.move_by_displacement(arm_tag=arm_tag.opposite, z=0.1),
        )
        self.move(
            self.place_actor(self.bread_2, target_pose=target_bread_pose, arm_tag=arm_tag.opposite, constrain="free", pre_dis=0.05, dis=0.05)
        )

        self.move(
            self.move_to_pose(arm_tag=arm_tag, target_pose=target_pose),
            self.back_to_origin(arm_tag=arm_tag.opposite),
        )
        self.move(
            self.move_by_displacement(arm_tag=arm_tag, z=0.05, move_axis="arm"),
        )
        self.move(
            self.move_by_displacement(arm_tag=arm_tag, z=-0.05, move_axis="arm"),
        )
        self.move(
            self.move_by_displacement(arm_tag=arm_tag, z=0.05, move_axis="arm"),
        )
        self.move(
            self.move_by_displacement(arm_tag=arm_tag, z=-0.05, move_axis="arm"),
        )
        self.move(
            self.move_by_displacement(arm_tag=arm_tag, z=0.10, move_axis="arm"),
        )

        self.move(
            self.move_by_displacement(arm_tag=arm_tag, quat=(0.5, -0.5, 0.5, 0.5)),
        )

        if arm_tag == "left":
            self.move(
                self.move_by_displacement(arm_tag=arm_tag, quat=(0, 0, 0.707, 0.707)),
            )
        else:
            self.move(
                self.move_by_displacement(arm_tag=arm_tag, quat=(0.707, -0.707, 0, 0)),
            )


        return self.info

    def check_success(self):
        plate_pose = self.plate.get_functional_point(0)[:3]
        bread_pose = self.bread.get_pose().p
        bread_2_pose = self.bread_2.get_pose().p
        skillet_pose = self.skillet.get_functional_point(0)[:3]
        eps = np.array([0.1, 0.1, 0.05])

        if not self.bread_middle_check:
            self.bread_middle_check =  int(np.all(abs(bread_pose - skillet_pose) < eps))
        if not self.bread_middle_check_2:
            self.bread_middle_check_2 = int(np.all(abs(bread_2_pose - skillet_pose) < eps))
        self.stage_eval_score = self.bread_middle_check + self.bread_middle_check_2 + int(np.all(abs(plate_pose - bread_pose) < eps)) + int(np.all(abs(plate_pose - bread_2_pose) < eps))
        self.stage_eval_score = self.stage_eval_score / 4.0
        return (np.all(abs(plate_pose - bread_pose) < eps) and np.all(abs(plate_pose - bread_2_pose) < eps))
