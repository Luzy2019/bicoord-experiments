from ._base_task import Base_Task
from .utils import *
import sapien
import math
import numpy as np
import copy
import os
import glob

class clean_table(Base_Task):

    def setup_demo(self, **kwags):
        super()._init_task_env_(**kwags)

    def load_actors(self):
        self.target_pose = rand_pose(
            xlim=[-0.03, -0.03],
            ylim=[-0.05, -0.05],
            qpos=[0.5, 0.5, 0.5, 0.5],
            rotate_rand=False,
        )
        self.bin = create_actor(
            self.scene,
            pose=self.target_pose,
            modelname="063_tabletrashbin_jlk",
            model_id=8,
            convex=True,
            is_static=True,
        )
        self.target_pose.p += np.array([0.0, 0.0, 0.15])
        
        while True:
            object_poses = []
            cnt = 0
            for i in range(4):
                pose = rand_pose(
                    xlim=[-0.3, 0.25],
                    ylim=[-0.1, 0.1],
                    qpos=[0.707, 0.707, 0.0, 0.0],
                    rotate_rand=True,
                    rotate_lim=[0, np.pi / 3, 0],
                )
                object_poses.append(pose)
                if pose.p[0] < 0:
                    cnt += 1
            if cnt != 2:
                continue

            flag = True
            for i in range(4):
                if np.abs(object_poses[i].p[0] - self.target_pose.p[0]) < 0.18:
                    flag = False
                for j in range(i):
                    if np.all(np.abs(object_poses[i].p[:2] - object_poses[j].p[:2]) < 0.1):
                        flag = False
            if not flag:
                continue
            break

        def get_available_model_ids(modelname):
            asset_path = os.path.join("assets/objects", modelname)
            json_files = glob.glob(os.path.join(asset_path, "model_data*.json"))
            available_ids = []
            for file in json_files:
                base = os.path.basename(file)
                try:
                    if 'soap' in modelname and 'data1.json' in file:
                        continue
                    with open(file, "r") as f:
                        data = json.load(f)
                    if data["extents"][0] * data["extents"][1] * data["extents"][2] * data["scale"][0] * data["scale"][1] * data["scale"][2] > 0.000125:
                        continue
                    idx = int(base.replace("model_data", "").replace(".json", ""))
                    available_ids.append(idx)
                except ValueError:
                    continue
            return available_ids

        object_list = [
            "047_mouse",
            "048_stapler",
            "057_toycar",
            "073_rubikscube",
            "075_bread",
            "081_playingcards",
            "112_tea-box",
            "113_coffee-box",
            "107_soap",
        ]
        available_objects = []
        for obj in object_list:
            available_model_ids = get_available_model_ids(obj)
            if len(available_model_ids) == 0:
                continue
            available_objects.append((obj, available_model_ids))
        self.objects = []
        for i in range(4):
            selected_modelname, available_model_ids = available_objects[np.random.randint(len(available_objects))]
            selected_model_id = np.random.choice(available_model_ids)
            self.objects.append(create_actor(
                scene=self,
                pose=object_poses[i],
                modelname=selected_modelname,
                convex=True,
                model_id=selected_model_id,
            ))
        self.objects = sorted(self.objects, key=lambda x: x.get_pose().p[0])
            


    def play_once(self):
        self.target_pose_left = [-0.25+self.target_pose.p[0], self.target_pose.p[1], 0.9, 0, 1, 0, 0]
        self.target_pose_right = [0.25+self.target_pose.p[0], self.target_pose.p[1], 0.95, 0, 0, 0, 1]
        arm_tag = ArmTag('left')
        move_list_a = self.objects[:2]
        move_list_b = self.objects[2:]
        target_pose_a = self.target_pose_left
        target_pose_b = self.target_pose_right
        f = 1
        push_dis = 0.1
        pull_dis = 0.2

        if np.random.rand() < 0.5:
            arm_tag = arm_tag.opposite
            move_list_a, move_list_b = move_list_b, move_list_a
            target_pose_a, target_pose_b = target_pose_b, target_pose_a
            f = -1
        if np.random.rand() < 0.5:
            move_list_a = move_list_a[::-1]
        if np.random.rand() < 0.5:
            move_list_b = move_list_b[::-1]

        self.move(
            self.grasp_actor(move_list_a[0], arm_tag=arm_tag, pre_grasp_dis=0.1),
            self.grasp_actor(move_list_b[0], arm_tag=arm_tag.opposite, pre_grasp_dis=0.1),
        )
        self.move(
            self.move_to_pose(arm_tag, target_pose_a),
            self.move_to_pose(arm_tag.opposite, target_pose_b)
        )
        self.move(
            self.move_by_displacement(arm_tag, x=f * push_dis)
        )
        self.move(
            self.open_gripper(arm_tag)
        )
        self.move(
            self.move_by_displacement(arm_tag, x=-f * pull_dis)
        )
        self.move(
            self.grasp_actor(move_list_a[1], arm_tag=arm_tag, pre_grasp_dis=0.1),
            self.move_by_displacement(arm_tag.opposite, x=-f * push_dis)
        )
        self.move(
            self.move_to_pose(arm_tag, target_pose_a),
            self.open_gripper(arm_tag.opposite)
        )
        self.move(
            self.move_by_displacement(arm_tag.opposite, x=f * pull_dis)
        )
        self.move(
            self.move_by_displacement(arm_tag, x=f * push_dis),
            self.grasp_actor(move_list_b[1], arm_tag=arm_tag.opposite, pre_grasp_dis=0.1)
        )
        self.move(
            self.open_gripper(arm_tag),
            self.move_to_pose(arm_tag.opposite, target_pose_b)
        )
        self.move(
            self.back_to_origin(arm_tag=arm_tag),
            self.move_by_displacement(arm_tag.opposite, x=-f * push_dis)
        )
        self.move(
            self.open_gripper(arm_tag.opposite)
        )

        return self.info

    def check_success(self):
        eps = 0.09
        cnt = 0
        
        for obj in self.objects:
            if np.all(np.abs(obj.get_pose().p[:2] - self.target_pose.p[:2]) <= eps):
                cnt += 1
        if not self.is_left_gripper_open() or not self.is_right_gripper_open():
            cnt = min(cnt, 3)
        self.stage_eval_score = cnt / 4.0
        return cnt == 4