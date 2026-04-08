# BiCoord: A Bimanual Manipulation Benchmark towards Long-Horizon Spatial-Temporal Coordination

**Xingyu Peng<sup>\*,1,2</sup>, Chen Gao<sup>\*,1,3</sup>,  Liankai Jin<sup>\*,1</sup>, Annan Li<sup>1</sup>, Si Liu<sup>†,1</sup>**

Beihang University<sup>1</sup>, Zhongguancun Academy<sup>2</sup>, National University of Singapore<sup>3</sup>

**Website:** https://buaa-colalab.github.io/BiCoord/

<img src="./assets/fig1.png" alt="description" style="display: block; margin: auto; width: 100%;">

# Overview, Tasks & LeaderBoard

Please see https://buaa-colalab.github.io/BiCoord/ for details.

# Installation

【Step 1】Clone the Repository

```
git clone https://github.com/buaa-colalab/BiCoord-Bench.git
```

【Step 2】Install the RoboTwin 2.0 Environment

Please follow [RoboTwin 2.0 Document (Usage - Install & Download)](https://robotwin-platform.github.io/doc/usage/robotwin-install.html) to install the RoboTwin 2.0 environment. 

【Step 3】Download Additional Assets

We have modified some objects in the original RoboTwin-OD dataset. Please download the additional objects at https://huggingface.co/datasets/GradiusTwinbee/BiCoord/blob/main/objects.zip and extract them to `BiCoord-Bench/assets/objects`.

【Step 4】For the env installation of specific policies, please refer to [RoboTwin 2.0 Usage Guide - RoboTwin 2.0 Offical Document](https://robotwin-platform.github.io/doc/usage/index.html)

In our practice, we install the RoboTwin 2.0 Environment first, and then install the policies' envs respectively based on this basic environment. 

# Data Preparation

We recommend to download official training data at https://huggingface.co/datasets/GradiusTwinbee/BiCoord and put data under `BiCoord-Bench/data`. Otherwise, you can generate data following the guidance at [Collect Data - RoboTwin 2.0 Offical Document](https://robotwin-platform.github.io/doc/usage/collect-data.html), and run correspnding  `split_stages.py` to obtain stage information.

After preparation,  `BiCoord-Bench/data` should look like

```
data/
├── balance_roller/
│   └── demo_clean/
│       ├── _traj_data/
│       ├── .ipynb_checkpoints/
│       ├── data/					#Trajectory data, including endpose, joint_action and observation
│       ├── instructions/			#Task instruction
│       ├── stages/					#Stage information for each episode
│       ├── video/					#Video
│       ├── scene_info.json
│       ├── seed.txt				#The corresponding random seeds for producing the trajectory data
│       └── split_stages.py			#Produce stage information based on trajectory data
├── build_tower_with_blocks/
├── clean_table/
├── ......
```

After get data prepared under  `BiCoord-Bench/data`, you need to convert the data into the training format according to the policy you choose. For data conversion of baseline policies, please follow the guidance at [RoboTwin 2.0 Usage Guide - RoboTwin 2.0 Offical Document](https://robotwin-platform.github.io/doc/usage/index.html).

# Training & Evaluation

Please follow [RoboTwin 2.0 Usage Guide - RoboTwin 2.0 Offical Document](https://robotwin-platform.github.io/doc/usage/index.html).

For convenience, we also provide all checkpoints of baseline policies (DP, RDT, OpenVLA-OFT, Pi0 under single&multi task setting) at https://huggingface.co/Oshwiciqwq/BiCoord-checkpoints.

# Acknowledgement

We sincerely thank [RoboTwin 2.0](https://github.com/robotwin-Platform/RoboTwin) for their outstanding contributions to bimanual manipulation simulation, convenient action APIs and open-source release.

We also thank [DP](https://github.com/real-stanford/diffusion_policy), [RDT](https://github.com/thu-ml/RoboticsDiffusionTransformer), [OpenVLA-OFT](https://github.com/moojink/openvla-oft) and [Pi0](https://github.com/Physical-Intelligence/openpi) for their outstanding and representive contributions in manipulation and VLAs.

# License
This benchmark is released under the **[MIT License](https://opensource.org/licenses/MIT)**.
