# SPINE

SPINE: Synthesizing Physics Infill for Neural Execution in Contact-Rich Manipulation. A hybrid data engine that synthesizes physics from kinematic skeletons, enabling safe, contact-rich manipulation via 3D diffusion policies and neural CBFs.

## 开发环境与外部依赖

子任务 1.1 依赖外部仓库（仅作只读参考，不要修改 upstream），推荐放在 `external/` 或以 submodule 方式挂载：

1. 拉取依赖仓库（任选其一）：
   * submodule：已包含 `external/{real2render2real,mimicgen,DexCap,robocasa,cito}`，首次使用请执行 `git submodule update --init --recursive`。
   * 或手动 clone：`git clone https://github.com/NVlabs/mimicgen external/mimicgen` 等。
2. 安装依赖（本地可编辑）：`pip install -e external/real2render2real`, `pip install -e external/mimicgen`, `pip install -e external/DexCap`, `pip install -e external/robocasa`, `pip install -e external/cito`。开发依赖见 `requirements-dev.txt`。
3. 运行运动学生成器：`python scripts/run_kinematic_generator.py --config configs/kinematics/kinematic_generator.yaml`。

## 真实数据快速验证

- MimicGen 样例（隔离环境下载）：在 `.venv_mgdl` 安装 mimicgen 及依赖（robosuite 1.4.0，robomimic 简单 stub），运行
  `python external/mimicgen/mimicgen/scripts/download_datasets.py --dataset_type core --tasks square_d0 --download_dir /tmp/mimicgen_core_square_d0`
  获取 `core/square_d0.hdf5`（约 1.6GB）。
- 生成器实测：配置 `/tmp/mg_run_config.yaml`（max_trajectories=2，augmentations=0，dataset_root=/tmp/mimicgen_core_square_d0/core），执行
  `python scripts/run_kinematic_generator.py --config /tmp/mg_run_config.yaml`，输出 2 条轨迹到 `/tmp/mg_out/mimicgen`，帧数约 135/136，末端姿态长度为 7。
- DexCap / R2R2R：请将真实数据放到 `configs/kinematics/kinematic_generator.yaml` 对应的 `dataset_root` / `capture_root`，并先 `pip install -e external/DexCap`、`pip install -e external/real2render2real`。然后运行生成器即可落盘 JSON 轨迹。若仅做快速干跑可设置 `SPINE_SKIP_DEP_CHECK=1` 并使用本仓构造的最小样例运行 `pytest tests/perception/test_adapters_integration.py -k dexcap -k r2r2r`。

## 当前优先级 / 进度小结
- FR3/DexCap：缺手套标记语义/臂 IK，无法构建真实映射；仅保留 7DOF 占位 IK（基座对齐）作流程测试，已知风险。PDF、FBX、硬件教程均未给出标记表/臂关节。
- MimicGen：已打通，数据 `data/mimicgen/{pick_place,square_d0}.hdf5`，生成 10 条轨迹到 `data/kinematics/mimicgen/`；静态检查（帧数约 595–631，姿态长度 7）无 NaN/Inf；可视化在 `results/mimicgen_vis/`；Panda/robosuite 无 GUI 回放两条样本各 500 步，运行正常（未做碰撞/限位检测，环境与任务不完全匹配）。
- MimicGen 批量严检（无 GUI）：`MUJOCO_GL=osmesa`、robosuite Panda、`--horizon 800` 回放全部 10 条，未见限位/碰撞/NaN/警告；日志位于 `logs/mimicgen_playback/`。步数汇总：

  | 轨迹 | 回放步数 |
  | --- | --- |
  | pick_place_demo_0_aug_mimicgen_0001.json | 605 |
  | pick_place_demo_0_aug_mimicgen_0002.json | 605 |
  | pick_place_demo_0_mimicgen_0000.json | 605 |
  | pick_place_demo_1_aug_mimicgen_0004.json | 595 |
  | pick_place_demo_1_aug_mimicgen_0005.json | 595 |
  | pick_place_demo_1_mimicgen_0003.json | 595 |
  | pick_place_demo_2_aug_mimicgen_0007.json | 631 |
  | pick_place_demo_2_aug_mimicgen_0008.json | 631 |
  | pick_place_demo_2_mimicgen_0006.json | 631 |
  | pick_place_demo_3_mimicgen_0009.json | 612 |

- MimicGen 数据协议检查：全部 10 条 JSON 的 `frames` 均含 7D `joint_positions`、7D `end_effector_pose`，无 NaN/Inf；帧数如上表。
- MimicGen 实时回放观察：在默认 robosuite Lift 环境渲染播放时，机械臂未与桌面方块发生互动或抓取，这是预期的原因有：① 未复现 MimicGen 采集时的物体初始位姿/世界系，默认方块位置与轨迹不对齐；② 脚本仅回放 7 个臂关节，未控制夹爪/力矩；③ 未加载数据中的目标/物体状态。当前回放仅用于格式和轨迹可跑通检查，非任务成功验证。
- MimicGen 对齐回放（技术栈与意图）：
  - 技术栈：robosuite（Panda+Lift，MuJoCo）关节层回放与渲染；读取 MimicGen hdf5 的 `robot0_joint_pos`、`robot0_gripper_qpos`、`obs["object"]`；自定义脚本 `scripts/playback_mimicgen_panda.py` 支持 JSON/hdf5 轨迹、方块初始 pose、基座对齐参数（`--base-yaw-deg`、`--base-xyz`）、回放夹爪，渲染/录像。渲染可用 `MUJOCO_GL=glfw`（有显示）或 `osmesa`（离屏），禁用 GPU/EGL 规避权限问题。
  - 验证意图：1）场景对齐，避免镜像/错位；2）轨迹可执行性，无 NaN/限位/碰撞；3）产出对齐良好的渲染/录像，作为“运动学生成→仿真回放”链路的 sanity check；4）为后续 CITO/物理投影打基础，初始对齐越好，收敛越稳。
  - 结论：当前回放仍属对齐+可执行的快速验证与素材生成，不代表任务成功率。

- R2R2R：未安装/未启用（配置设为 null），后续如需使用需安装 `external/real2render2real` 并提供有效路径。
- 物理填充：目前仅有简化 IK/VSCM 占位，尚未引入真实动力学、碰撞、关节/力矩约束；待映射/多源稳定后再做统一重采样/物理投影。

### 物理填充（CasADi+Pinocchio CITO）阶段性结果

- 环境：`conda create -n spine_opt python=3.10 && conda install -c conda-forge pinocchio casadi`，运行前确保 `PYTHONPATH=""`。
- 核心脚本：`tools/casadi_cito_mvp.py`（Pinocchio.casadi + finite-diff 动力学 + 互补约束）；参考轨迹转换 `tools/convert_mimicgen_to_npy.py`（7DOF → 9DOF 补齐夹爪）；可视化 `tools/plot_results.py`。
- 实验设定（关键一次）：对 MimicGen pick_place 轨迹切片 200–300 帧，桌面高度 0.20 m，互补松弛 1e-2，跟踪权重降低，允许起点微调，强制非穿透 `phi>=0`。
- 结果：Ipopt 收敛（Optimal Solution Found），接触力区间约 0.02–5.17 N，优化后 Z 轨迹在接触段被“夹”在桌面高度，力曲线与几何曲线同步跃升，表明互补与省力项发挥作用。可视化见 `data/spine_validation.png`（蓝虚线原始穿模至 ~0.12 m，蓝实线被抬至 0.20 m 平台，红线接触力在接触段跃升）。
- 运行示例：
  ```bash
  export PYTHONPATH=""
  conda run -n spine_opt python tools/convert_mimicgen_to_npy.py
  conda run -n spine_opt python tools/casadi_cito_mvp.py \
    --urdf models/fr3.urdf --q-ref data/fr3_q_ref.npy \
    --dt 0.05 --table-height 0.20 --comp-penalty 0.01
  conda run -n spine_opt python tools/plot_results.py
  ```
  产物：`data/fr3_opt_result.npy`（修正后的 q），`data/fr3_opt_forces.npy`（接触力），`data/spine_validation.png`。
- 关键观察（“Money Shot”）：在切片 200–300 帧的事故窗中，原始 MimicGen 轨迹深潜到 ~0.12 m；CITO 将轨迹截停在 0.20 m，接触段接触力跃升到 1.5–5 N，且力曲线与几何曲线同步爆发，证明互补约束和省力诱导生效。
- 失败/对比记录：  
  1) `table_height=0.25` 且严格非穿透时，深穿模导致可行域为空，Ipopt 直接 infeasible；  
  2) `table_height=0.20`、允许穿透 `phi>=-0.10`、取前 0–100 帧时，轨迹整体高于桌面（未进入事故段），接触力仅 ~0.1 N，呈“空挥”现象；  
  这两次失败促使我们切片事故窗 + 严格非穿透，才得到当前 5 N 力峰的“Money Shot”。
- 后续计划：编写批处理脚本（例如 `scripts/batch_spine_process.py`）批量处理 MimicGen 轨迹（推荐每任务 ≥50–200 条，用于策略训练对比：原始 MimicGen vs Free-space IK vs SPINE Inpainted），统计修复成功率并产出物理完备数据集。

#### 实验日志：失败与改良对比

- 阶段一（数值噪声/不稳定，失败，图：`image_8203b7.png`、`image_66c8f5.png`）：蓝实线高频抖动，接触力 0.002–0.1 N。原因：初始点穿模且被硬锁 (`Q[:,0]==q_ref[0]`)，互补松弛在穿模区泄漏，长窗口稀释碰撞段权重，求解器“潜入地下+零力”。
- 阶段二（物理修补成功，Money Shot，图：`image_66ad92.png`）：切片事故窗（200–300）、初始松弛（软约束 1000‖Q0−qref‖²）、严格非穿透 (`phi>=0`) 后，蓝线在 0.20 m 平台截停，红线接触力同步跃升至 1.5–5.2 N 并随离开归零，体现物理一致性。

## CITO 纯 Python 原型

`spine/planning/cito` 提供了基于 VSCM 的轻量级 Python 版本，用于在未安装 C++/SNOPT 的环境中预优化轨迹：

- 使用示例：参考 `tests/planning/test_cito.py`，构造 `CITOParameters`、`CITOPlanner`，传入动力学函数和接触距离函数。
- 该原型仅用于快速验证/暖启动，性能与原 C++ 版本不同，重物理任务仍建议使用官方 CITO 构建。
- 将子任务 1.1 输出的运动学 JSON 做最小物理投影：`python scripts/run_cito_projection.py --input <traj.json> --output /tmp/proj.json --table-height 0.0 --dt 0.05 --max-iters 5`。内部以末端位姿 (x,y,z) 运行简化 VSCM/接触惩罚，生成修正后的轨迹，便于后续在 robosuite 中回放/验证。

### 实验记录：简化 VSCM 投影可视化（/tmp/proj.json）

- 结果：末端 3D 轨迹在 x/y 内约 0.3 m 范围，z 维保持 ~0.8–1.05 m，无穿透 table_height=0 以下，说明 VSCM 惩罚将 z 向上抬升；右侧 z 曲线在 0.8–1.05 m 间起伏，x/y 小幅振荡，未出现负 z。
- 机制解析：仅有平面接触惩罚作用在末端，软惩罚推动 z 升高，但对 x/y 约束弱；无物体几何/多接触体，无法捕捉侧向穿透或复杂法向。
- 当前局限：点质量 + 平面接触的占位投影，缺少关节空间动力学、关节限位、力矩/速度/加速度约束，未重定向到 FR3/Panda，未考虑夹爪姿态/抓取约束。轨迹抖动源于未滤波、未优化控制；无物体/障碍检测仅在 z 维“防穿透”，无法代表真实 CITO。
- 结论：可视化表明简化 VSCM 能把末端 z 从潜在穿透区域抬至桌面以上，但距离预期的物理填充还需引入完整动力学、约束、多接触几何、力矩输出与平滑控制，并对目标机器人重定向后再验证。

### CITO 升级方向：从“美图”到“物理修补”

**TLDR**：当前 Python 脚本（仅做 z 轴限位/VSCM 占位）只是“给视频美颜”，不是物理修补。要达成 RSS/CoRL 水准，必须实现 **接触隐式轨迹优化 (CITO)**，让优化器凭借“省力”原理自动发现“撑在桌子上比悬空更省力”。

- 直觉类比：传统规划把环境视作禁区（Distance>0），而“撑持”要把环境当作支撑点——像焊电路板时用小拇指抵桌，形成闭环分担力矩。CITO 就是让优化器主动选择这种“借力”。
- 数学建模（直接配点式）：
  - 决策变量：\(\{q_{1:T}, u_{1:T}, \lambda_{1:T}\}\)（关节、力矩、接触力）。
  - 代价：\(J = \sum_t w_{track}\|q_t-q_t^{ref}\|^2 + w_{effort}\|u_t\|^2 + w_{reg}\|\lambda_t\|^2\)。跟踪松弛、最小力矩诱导“撑持”、接触力正则。
  - 约束：动力学 \(M\dot v+C+g = B u + J^T\lambda\)；互补 \(0 \le \phi(q_t) \perp \lambda_t \ge 0\)（或 \(\phi \cdot \lambda \le \epsilon\)）；摩擦锥 \(\|\lambda_t^{tan}\|\le \mu \lambda_t^{n}\)。含接触力变化平滑可抑制“敲击”。
- 论文故事：输入是物理破碎的运动学骨架；算法是带互补/省力代价的 CITO；涌现行为是自动生成接触力和可执行轨迹；输出是包含 \([RGB, q, \tau, \lambda]\) 的物理完备数据。
- 实施建议：优先用 **Drake/CasADi** 写“多接触 + 互补 + 摩擦锥”的数学规划；MuJoCo 可作 MVP 逆动力学验证但论文严谨性不如 Drake。先在短地平线、少量接触对上验证收敛，再扩展到任务级。

### Drake 导入 FR3 STL 碰撞网格的兼容性问题
- 现象：Drake 读取 `fr3.urdf` 时，对 STL 碰撞网格默认尝试 MakeConvexHull，报错“unsupported extension '.stl'”。  
- 临时绕过：将碰撞几何的 STL 替换为对应的视觉 DAE/OBJ，Drake 可直接加载，不触发凸包转换（例如 `collision/linkX.stl -> visual/linkX.dae`）。
- 影响：这是工程层面的几何近似，不改变 CITO 数学框架。但会改变碰撞几何精度/形状，可能影响接触位置/穿透深度或性能；后续可恢复高保真几何（如转 STL->OBJ 并保留原碰撞网格，或使用 Drake 推荐的 convex decomposition）。
- 下一步：在替换视觉 DAE 后先跑通短时域 CITO 验证，再视需要换回更精确的碰撞网格并重跑。
- 补充：为继续跑通短时域 CITO，已接受“用视觉网格（OBJ/DAE）替换/移除碰撞 STL”这一工程折衷。需在实验解读中注明：碰撞几何与真实/高保真模型存在偏差，结果仅用于算法可行性验证，后续对接真机或高保真仿真需恢复或转换碰撞网格并重跑。
- 最新折衷（FR3 Drake 导入）：为避免 Drake 对 STL/DAE 做凸包失败，暂时移除 URDF 中的 collision 网格，仅保留视觉网格加载，以便跑通短时域 CITO。此举仅为算法可行性验证，接触几何不高保真，后续需恢复或替换为可解析的碰撞网格并重验。
- Drake 网格受阻 → 计划切换 CasADi + Pinocchio 路线：Drake 对 mesh/凸包的工程要求较高，短期验证 CITO 阻力大。算法核心（互补/动力学/最小力矩）与网格无关，后续将尝试用 CasADi+Pinocchio 搭建 CITO，使用代理几何（平面+关键点碰撞球/胶囊），确保短时域可收敛，先验证“物理填充”算法有效性，再视需要回到高保真网格。

### 实验记录：关节平滑（/tmp/proj_joints.json）

- EE 图与原始几乎一致是正常的：`run_joint_projection.py` 只对关节序列做限幅/平滑，未重算前向运动学更新末端位姿；若原轨迹未触发限位/限速，平滑结果也与原始相近。
- 占位性质：仅对关节向量数值夹紧，未将约束反馈到末端 pose。
- 如何确认改动：比较关节序列差异 `np.max(np.abs(q_proj - q_orig))`；如需强制触发约束，可将 `--vel-limit` 调小（如 0.1 rad/s）后重跑并可视化或打印关节差异。
- 交互对比（`results/compare_orig_vs_fk.html`）：原始 MimicGen 末端（蓝）与关节平滑+FK 末端（橙）在不同高度/平移，是因为 FK 使用了 robosuite Panda 默认基座（Lift 原点），未对齐原始世界系。需要在 FK 时应用原数据的基座位姿（或在重算后加回初始偏移）才能获得对齐的 EE 轨迹；关节平滑本身不是造成大偏移的原因。

## 依赖兼容性提醒

安装 robocasa 会将 `protobuf` 降到 3.19.x，可能与 `tensorboardX` (>=3.20) 或 `mcap-protobuf-support` (>=4.25) 冲突。建议在隔离环境使用 robocasa，或手动 `pip install 'protobuf>=4.25'` 后自行验证 robocasa 功能是否受影响。
robocasa 0.2.0 对依赖高度敏感：官方要求 MuJoCo==3.2.6 / numpy==1.23.3 / numba==0.56.4 / robosuite master（含 PandaOmron）。高于 3.2.6 的 MuJoCo（如 3.3.x）会触发版本检查失败，低版本 robosuite(1.4.x) 则缺失 PandaOmron，旧版 robosuite 依赖 mujoco-py 2.0 需 MuJoCo 200+license。请在独立 venv 中按 robocasa README 所列版本安装，并运行 robocasa/scripts/setup_macros.py / download_kitchen_assets 以完成配置。
3. 这些仓库只用于格式解析/重定向调用，核心代码保持解耦；缺失时适配器会提示安装路径。

### **一、 混合数据生成引擎：从运动学幻觉到物理真实 (The Hybrid Data Engine)**

1. 核心痛点深度解析：

目前的主流数据生成方法存在“二元对立”：

* **R2R2R (Real2Render2Real)** 极具可扩展性，能生成百万级轨迹，但它是“物理盲”的（Physics-Blind）。它通过运动学重定向生成轨迹，忽略了质量、摩擦和接触力，导致生成的策略在接触丰富（Contact-Rich）任务中只会“做动作”而不会“施力”。
* **纯物理模拟 (Pure Sim)** 虽然物理严谨，但资产建模昂贵，缺乏视觉多样性，且难以捕捉真实世界的人类操作细微差别（如灵巧操作的微调）。

2. 技术方案：物理填充 (Physics Infill) 与 轨迹优化 (Trajectory Optimization)

本框架提出的“混合引擎”本质上是一个物理投影算子 (Physical Projection Operator)。

* **第一步：运动学骨架提取。** 使用 **MimicGen**和 **DexCap** 从人类演示中提取运动学轨迹。这些轨迹定义了“意图”（如：手掌接近杯子），但可能包含物理上不可行的状态（如穿模、悬空抓取）。
* **第二步：接触隐式轨迹优化 (CITO)。** 这是核心算法。我们将运动学轨迹作为初值（Warm-start），在 **RoboCasa** 等高保真模拟器中运行优化算法。
  * **优化目标函数：**\$J = w\_1 ||q - q\_{ref}||^2 + w\_2 ||\\tau||^2 + w\_3 ||\\lambda\_{contact}||^2\$。即在尽可能贴近原轨迹的同时，最小化力矩和接触力，寻找最“省力且自然”的接触方式。
  * **松弛变量 (Slack Variables)：** 允许轨迹在微小范围内偏离原运动学路径，以满足物理约束（如不穿透刚体）。
  * **输出：** 这一步“合成”了原本不存在的真实接触力数据，产生了包含 `{图像, 关节状态, 力矩, 接触力}` 的完备四元组数据。

**3. 关键参考文献 (Key References)：**

* **[运动学生成]** Mandlekar, A., et al. (2023). **MimicGen: A Data Generation System for Scalable Robot Learning using Human Demonstrations**. *CoRL*.
  * *解析：提供了从少量演示生成大规模运动学数据的核心方法。*
* **[灵巧操作生成]** Wang, C., et al. (2024). **DexMimicGen: Automated Data Generation for Bimanual Dexterous Manipulation via Imitation Learning**. *arXiv*.
  * *解析：解决了灵巧手操作数据的生成难点，是“撑持”任务的基础。*
* **[物理填充核心]** Posa, M., et al. (2020). **Tuning-Free Contact-Implicit Trajectory Optimization**. *IEEE Transactions on Robotics*.
  * *解析：CITO的经典算法，支持无需预定义接触序列的优化。*
* **[最新物理驱动生成]** (2025). **Physics-Driven Data Generation for Contact-Rich Manipulation via Trajectory Optimization**. *arXiv:2502.20382*.
  * *解析：直接支持本框架核心思想的最新工作，利用优化合成动力学数据。*
* **[R2R2R范式]** (2025). **Real2Render2Real: Scaling Robot Data Without Dynamics Simulation or Robot Hardware**. *arXiv:2505.09601v1*.
  * *解析：提供了大规模视觉背景生成的基座。*

---

### **二、 共生控制架构：三维几何感知与多模态融合 (Symbiotic Control Architecture)**

1. 核心痛点深度解析：

传统的2D扩散策略（Diffusion Policy）在处理接触任务时表现不佳，因为“接触”本质上是一个三维空间几何问题（点云的距离、法向量对齐）。仅靠2D图像难以精确推断接触深度和法向力。

**2. 技术方案：3D Diffusion Policy (DP3) + VLA**

* **3D感知优先：** 控制器（Executor）采用 **DP3** 架构。它直接以点云（Point Cloud）作为输入，而非RGB图像。点云天然包含了物体的几何信息，使得策略能更精准地预测“哪里可以撑持”、“哪里可以抓取”。
* **多模态Transformer：** 我们设计一个Transformer Backbone，将 **Force Token (力触觉令牌)** 显式地注入到输入序列中。
  * 输入序列：`[Visual_Token, Proprio_Token, Force_Token, Goal_Token]`。
  * 机制：当机器人感受到阻力时（Force Token激活），Transformer的注意力机制会从关注“视觉目标”转移到关注“触觉反馈”，从而自适应地调整动作（如减速、顺应）。
* **推理加速：** 引入 **Fast-dLLM** 原理，利用 **KV Cache** 技术缓存扩散模型的中间特征，解决实时控制频率不足的问题。

**3. 关键参考文献 (Key References)：**

* **[3D策略核心]** Ze, Y., et al. (2024). **3D Diffusion Policy: Generalizable Visuomotor Policy Learning via Simple 3D Representations**. *ICLR*.
  * *解析：证明了点云在接触任务中优于2D图像，是本框架Controller的基石。*
* **[高层规划]** (2024). **OpenVLA: An Open-Source Vision-Language-Action Model**. *arXiv:2406.09246*.
  * *解析：用于解析自然语言指令（如“撑在桌子上抓取”），生成高层Goal Token。*
* **[3D世界模型]** (2024). **3D-VLA: A 3D Vision-Language-Action Generative World Model**. *arXiv:2403.09631*.
  * *解析：增强了VLA对三维物理世界的理解能力。*

---

### **三、 可验证安全层：从“硬约束”到“动态护栏” (Verifiable Safety Layer)**

1. 核心痛点深度解析：

传统的控制障碍函数 (CBF) 需要人工推导物理模型（如倒立摆模型），这在非结构化环境（如乱堆的杂物）中几乎不可能实现。而且传统CBF非常保守，往往会阻止机器人的“撑持”行为，因为它误判为“碰撞”。

**2. 技术方案：学习型神经CBF (Neural CBF) 与 动态安全集**

* **数据驱动的CBF学习：** 我们利用混合数据引擎生成的“失败案例”（如滑落、翻倒）作为负样本，训练神经网络拟合 CBF 函数 **\$h(x)\$**。
  * 训练目标：让 **\$h(x) < 0\$** 对应不安全状态， **\$h(x) \\ge 0\$** 对应安全状态。
* **撑持感知的动态安全集 (Bracing-Aware Safety)：**
  * **创新点：** 传统的安全性是静态的。本框架中，当检测到稳定的“撑持”接触（Bracing Contact）时，CBF函数 **\$h(x)\$** 的判定边界会动态扩展（Expand）。
  * **物理意义：** 就像人扶着扶手下楼梯可以走得更快一样，机器人一旦建立稳定接触，CBF就允许其使用更大的力矩和速度，而不触发急停。这是通过在包含“撑持”状态的数据上训练CBF自动习得的。

**3. 关键参考文献 (Key References)：**

* **[神经CBF基础]** Robey, A., et al. (2020). **Learning Control Barrier Functions from Expert Demonstrations**. *IEEE CDC*.
  * *解析：奠定了从数据中学习CBF的理论基础。*
* **[最新安全架构]** (2025). **Certificated Actor-Critic: Hierarchical Reinforcement Learning with Control Barrier Functions for Safe Navigation**. *arXiv:2501.17424v1*.
  * *解析：引入了分层安全控制的思想，支持动态调整安全策略。*
* **[神经控制综述]** (2024). **Learning Control Barrier Functions and their application in Reinforcement Learning: A Survey**. *arXiv:2404.16879*.
  * *解析：全面总结了学习型CBF的最新进展，用于证明技术路线的可行性。*

---

### **四、 Sim-to-Real 迁移策略：解耦与对齐**

1. 技术方案：解耦视觉与控制 (Decoupled Visuomotor Manipulation)

为了跨越“现实鸿沟”，我们不直接在真实世界训练端到端模型，而是采用解耦策略：

* **在模拟中 (Sim)：** 使用“特权信息”（真实的物体位姿、接触力）训练控制策略。因为模拟中的物理（经过SysID校准后）是可靠的。
* **在现实中 (Real)：** 仅训练/微调“视觉感知模块”（Visual Encoder）。将真实图像映射到模拟中的特权状态空间。
* **优势：** 避免了在真实机器人上进行危险且昂贵的强化学习探索，只需少量真实数据校准视觉即可。

**2. 关键参考文献：**

* **[解耦迁移]** (2025). **Decoupled Visuomotor Manipulation**. *arXiv:2509.25747v1*.
  * *解析：提供了Sim-to-Real的具体实施路径，强调在Sim学控制，在Real学感知。*

**阶段 1：构建混合数据引擎 (Hybrid Data Engine)”**

以下是详细任务规划。

**TLDR**：不要把这个阶段仅仅看作“造数据”，这其实是构建一个**“物理投影仪”**。你的核心任务是搭建一套流水线，能吞入 R2R2R 的“视觉/运动学”数据，吐出包含“接触力/力矩”的物理完备数据。为了服务后续的策略学习（Phase 2）和安全验证（Phase 3），你必须现在就设计好**多模态数据协议**和**负样本自动标注接口**，否则后续返工成本极高。

---

### **阶段 1：混合数据引擎构建与验证 (任务规划书)**

#### **一、 核心目标 (Objectives)**

1. **规模化运动学生成：** 利用 R2R2R 和 MimicGen 生成高视觉保真度、高任务多样性的基础轨迹（骨架）。
2. **高保真动力学合成：** 通过 CITO/IDTO 在 RoboCasa/Isaac Gym 中将运动学轨迹“投影”到物理可行流形，合成接触力与力矩（肌肉）。
3. **统一数据基座：** 建立 Robo-DM 存储标准，确保视觉、本体感知、触觉力流的时间同步，为下游任务预留接口。

---

#### **二、 详细任务分解 (Step-by-Step Plan)**

##### **子任务 1.1：运动学骨架生成流水线 (The Kinematic Generator)**

*目标：生成海量“看起来对”但“物理不知”的轨迹。*

* **[工程实施] 集成 R2R2R 与 MimicGen 2.0：**
  * 部署 R2R2R 管线，利用 3D Gaussian Splatting 重建多样化物体。
  * 接入 **MimicGen 2.0**，从少量人类演示中通过数据增强（变换物体位置、改变抓取姿态）生成大量运动学轨迹。
  * **关键修正：** 引入 **DexCap** 的灵巧操作数据作为补充，特别是针对复杂的“撑持”动作，因为简单的二指抓取不足以体现撑持的动力学优势。
* **[预留接口 - 针对 Phase 2 策略学习]：**
  * **视觉多样性接口：** 集成 **RoboEngine**，确保生成的 RGB 图像包含丰富的光照和背景变化。这是为了后续“解耦 Sim-to-Real 策略”中训练视觉编码器做准备。

##### **子任务 1.2：物理填充核心 (The Physics Infill Core)**

*目标：解决“运动学幻觉”，合成真实的力/力矩。*

* **[工程实施] 搭建 RoboCasa/Isaac Gym 仿真环境：**
  * 导入 R2R2R 中的物体资产，开启物理属性（摩擦、质量、碰撞）。
  * **实施 CITO (接触隐式轨迹优化)：**
    * **输入：** 将子任务 1.1 生成的运动学轨迹作为 **Warm-start (热启动)** 初值。
    * **优化器设计：** 设置松弛变量（Slack Variables），允许优化后的轨迹在微小范围内偏离原运动学轨迹，以解决原轨迹中的“穿模”或“悬空”问题。
    * **输出：** 计算并记录每一帧的**接触力 (Contact Force)** 和 **关节力矩 (Joint Torque)**。
* **[预留接口 - 针对 Phase 3 安全验证]：**
  * **负样本自动捕获 (Automatic Negative Mining)：**
    * **逻辑：** 并非所有 CITO 优化都会成功。如果优化失败（残差过大）或导致物体飞出/翻倒，**不要丢弃这些数据！**
    * **接口：** 自动将这些失败轨迹标记为 `label=unsafe`。这些是训练 **Neural CBF (神经控制障碍函数)** 最宝贵的负样本，用于定义安全边界。

##### **子任务 1.3：系统辨识与残差学习 (The Reality Check)**

*目标：缩小仿真与现实的动力学分布差异。*

* **[工程实施] 基础 SysID 与域随机化：**
  * 实施基础的系统辨识（SysID），校准摩擦系数和恢复系数。
  * 实施**域随机化 (Domain Randomization)**：在物理填充过程中，随机扰动摩擦系数、物体质量和质心位置，生成具有不同物理参数的多个轨迹副本。
* **[预留接口 - 针对 Sim-to-Real]：**
  * **残差物理占位符：** 预留一个 **Residual Physics Model** 的接口。虽然阶段 1 可能还没大量真实数据来训练它，但架构上要允许未来注入 **\$\\Delta(s, a)\$** 来修正模拟器的输出。

##### **子任务 1.4：统一数据管理 (Data Infrastructure)**

*目标：构建“唯一真实来源 (Single Source of Truth)”。*

* **[工程实施] Robo-DM 部署：**
  * 使用 **Robo-DM** (EBML格式) 存储数据，以应对多模态数据的高吞吐量。
* **[关键协议定义]：** 定义统一的数据帧结构 `Frame_t`，必须包含：
  * `Observation`: RGB (来自 R2R2R), PointCloud (用于 DP3), Proprioception.
  * `Privileged_State`: Object\_Pose\_GT, Contact\_Forces\_GT (来自物理填充，用于 Sim 训练).
  * `Action`: Joint\_Positions/Velocities.
  * `Meta_Labels`: `Is_Bracing` (布尔值，是否发生撑持), `Safety_Score` (安全/不安全), `Optimization_Residual` (物理填充的置信度).

---

#### **三、 关键风险与缓解 (Risks & Mitigation)**

* **风险点：** CITO 优化速度太慢，无法处理百万级 R2R2R 轨迹。
  * **缓解方案：** 采用分层策略。对关键的、接触复杂的“撑持”片段使用完整的 CITO；对简单的自由空间运动使用 **IDTO (逆动力学)** 快速计算。
* **风险点：** 运动学轨迹与物理环境冲突严重（如严重穿模），导致优化不收敛。
  * **缓解方案：** 在 CITO 中增加**穿透惩罚项 (Penetration Penalty)** 的权重，并允许优化器丢弃物理上完全不可行的“垃圾”轨迹（这也是对 R2R2R 生成质量的一种过滤）。

#### **四、 验收标准 (Definition of Done)**

1. **流水线跑通：** 能够自动将一条 R2R2R 视频转换为包含力矩信息的 `.hdf5` / `.ebml` 文件。
2. **物理一致性验证：** 在仿真中重放生成的力矩序列，机器人能成功复现原动作且不发生穿模或物体滑落。
3. **数据完备性：** 至少生成 1,000 条包含“撑持”动作的完整多模态轨迹，且每一帧都带有 `Force` 和 `Safety` 标签。

repo links:

R2R2R: [https://github.com/uynitsuj/real2render2real#]()

Mimicgen: [https://github.com/NVlabs/mimicgen]()

DexCap: [https://github.com/j96w/DexCap/tree/main]()

robocasa: [https://github.com/robocasa/robocasa]()

CITO: [https://github.com/aykutonol/cito]()

RoboEngine: [https://github.com/michaelyuancb/roboengine]()

codex resume 019aaaac-0630-70e2-ac53-d79401381f1b
