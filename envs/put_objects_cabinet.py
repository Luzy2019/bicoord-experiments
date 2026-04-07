from ._base_task import Base_Task
from .utils import *
import sapien
import glob


class put_objects_cabinet(Base_Task):

    def setup_demo(self, **kwags):
        super()._init_task_env_(**kwags, table_static=False)

    def load_actors(self):
        self.model_name = "036_cabinet"
        self.model_id = 46653
        self.cabinet = rand_create_sapien_urdf_obj(
            scene=self,
            modelname=self.model_name,
            modelid=self.model_id,
            xlim=[-0.05, 0.05],
            ylim=[0.155, 0.155],
            rotate_rand=False,
            rotate_lim=[0, 0, np.pi / 16],
            qpos=[1, 0, 0, 1],
            fix_root_link=True,
        )
        rand_pos = rand_pose(
            xlim=[-0.32, -0.20],
            ylim=[-0.15, -0.1],
            qpos=[0.707, 0.707, 0.0, 0.0],
            rotate_rand=False,
            rotate_lim=[0, np.pi / 3, 0],
        )

        rand_pos_2 = rand_pose(
            xlim=[-0.32, -0.20],
            ylim=[-0.2, -0.15],
            qpos=[0.707, 0.707, 0.0, 0.0],
            rotate_rand=False,
            rotate_lim=[0, np.pi / 3, 0],
        )

        t = 0
        while np.linalg.norm(np.array(rand_pos.p[:2]) - np.array(rand_pos_2.p[:2])) < 0.10 and t<=100:
            rand_pos_2 = rand_pose(
                xlim=[-0.32, -0.20],
                ylim=[-0.2, -0.15],
                qpos=[0.707, 0.707, 0.0, 0.0],
                rotate_rand=False,
                rotate_lim=[0, np.pi / 3, 0],
            )
            t += 1
        def get_available_model_ids(modelname):
            asset_path = os.path.join("assets/objects", modelname)
            json_files = glob.glob(os.path.join(asset_path, "model_data*.json"))
            available_ids = []
            for file in json_files:
                base = os.path.basename(file)
                try:
                    idx = int(base.replace("model_data", "").replace(".json", ""))
                    available_ids.append(idx)
                except ValueError:
                    continue
            return available_ids

        object_list = [
            "081_playingcards",
        ]
        self.selected_modelname = np.random.choice(object_list)
        available_model_ids = [1,2]
        if not available_model_ids:
            raise ValueError(f"No available model_data.json files found for {self.selected_modelname}")
        self.selected_model_id = np.random.choice(available_model_ids)
        self.object = create_actor(
            scene=self,
            pose=rand_pos,
            modelname=self.selected_modelname,
            convex=True,
            model_id=self.selected_model_id,
        )
        self.object_2 = create_actor(
            scene=self,
            pose=rand_pos_2,
            modelname=self.selected_modelname,
            convex=True,
            model_id=self.selected_model_id,
        )
        self.object.set_mass(0.01)
        self.object_2.set_mass(0.01)
        self.add_prohibit_area(self.object, padding=0.01)
        self.add_prohibit_area(self.object_2, padding=0.01)
        self.add_prohibit_area(self.cabinet, padding=0.01)
        self.prohibited_area.append([-0.15, -0.3, 0.15, 0.3])
        self.origin_y = self.cabinet.get_functional_point(0)[1]

    def play_once(self):
        arm_tag = ArmTag("right" if self.object.get_pose().p[0] > 0 else "left")
        self.arm_tag = arm_tag
        self.origin_z = self.object.get_pose().p[2]

        self.move(
            self.grasp_actor(self.object, arm_tag=arm_tag, pre_grasp_dis=0.1),
            self.grasp_actor(self.cabinet, arm_tag=arm_tag.opposite, pre_grasp_dis=0.05)
        )

        for _ in range(4):
            self.move(self.move_by_displacement(arm_tag=arm_tag.opposite, y=-0.04))
            self.move(self.move_by_displacement(arm_tag=arm_tag, z=0.05))

        target_pose = self.cabinet.get_functional_point(0)
        self.move(self.place_actor(
            self.object,
            arm_tag=arm_tag,
            target_pose=target_pose,
            pre_dis=0.13,
            dis=0.1,
        ))

        self.move(
            self.grasp_actor(self.object_2, arm_tag=arm_tag, pre_grasp_dis=0.1),
        )
        self.move(self.move_by_displacement(arm_tag=arm_tag, z=0.2))
        self.move(self.place_actor(
            self.object_2,
            arm_tag=arm_tag,
            target_pose=target_pose,
            pre_dis=0.13,
            dis=0.1,
        ))

        self.move(self.move_by_displacement(arm_tag=arm_tag, x=-0.20))

        for _ in range(4):
            self.move(self.move_by_displacement(arm_tag=arm_tag.opposite, y=0.04))

        return self.info

    def check_success(self):
        object_pose = self.object.get_pose().p
        object_pose_2 = self.object_2.get_pose().p
        target_pose = self.cabinet.get_functional_point(0)[:3]
        tag = np.all(abs(object_pose - target_pose) < np.array([0.05, 0.05, 0.05]))
        tag_2 = np.all(abs(object_pose_2 - target_pose) < np.array([0.05, 0.05, 0.05]))

        self.stage_eval_score = (int(tag) + int(tag_2)) / 2.0
        return (tag and tag_2 and abs(self.origin_y - target_pose[1]) < 0.05)
