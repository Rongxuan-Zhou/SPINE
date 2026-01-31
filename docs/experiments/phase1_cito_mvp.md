# Phase 1 — CITO 物理修补 MVP 记录

## 目标
- 验证 SPINE 的“物理填充 (Physics Inpainting)”链路是否可收敛，并能从纯运动学轨迹中合成接触力。
- 证明互补约束 + 省力代价能够把“穿模/悬空”修正为“贴合接触”。

## 实验环境
- 优化器：CasADi + IPOPT
- 动力学：Pinocchio（RNEA + FK）
- 脚本：`tools/casadi_cito_mvp.py`、`tools/plot_results.py`
- 机器人：FR3 URDF（`models/fr3.urdf`）
- 接触平面：table height 由参数 `--table-height` 控制

## 关键步骤与演进
### 1) 初始失败：数值噪声 + 无力
- 现象：EE z 曲线高频抖动，接触力近似为 0。
- 原因：
  - 初始穿模导致可行域为空。
  - 互补约束在穿模区域泄漏（`phi < 0` 时力为 0 仍可满足松弛不等式）。
- 记录图：
  - `docs/pics/spine_validation_air_drumming.png`
  - `docs/pics/spine_validation_solver_infeasible.png`

### 2) 成功修补：几何截断 + 力觉唤醒
关键改进：
- 自适应局部窗口（自动定位穿模片段）
- 初始状态松弛（`Q0` 由硬约束改为 cost）
- 严格非穿透（`dist >= 0`）

结果：
- EE z 在桌面处形成平底（几何截断）
- 接触力在接触时间段显著抬升（1–5 N 级）
- 记录图（Money Shot）：
  - `docs/pics/spine_validation.png`

## 可复现指令（示例）
```bash
# 运行 CITO 物理修补
conda run -n spine_opt python tools/casadi_cito_mvp.py \
  --urdf models/fr3.urdf \
  --q-ref data/fr3_q_ref.npy \
  --dt 0.05 \
  --table-height 0.20 \
  --comp-penalty 0.01

# 绘制对比图
conda run -n spine_opt python tools/plot_results.py
```

## 产出
- 轨迹结果：`data/fr3_opt_result.npy`
- 接触力：`data/fr3_opt_forces.npy`
- 图像：`docs/pics/spine_validation.png`

