# FR3 关节与坐标系参考（来自 external/franka_description）

- 关节顺序（无前缀，见 `robots/fr3/fr3.urdf.xacro` → `franka_arm.xacro`）：
  `joint1`, `joint2`, `joint3`, `joint4`, `joint5`, `joint6`, `joint7`
- 基座帧：`base`（通过固定关节连接到 `link0`），默认连接世界系。
- 法兰帧：`link8`（挂载手爪/EE），TCP 默认偏移 `xyz=(0, 0, 0.1034)m`，`rpy=(0, 0, 0)`。
- 关节极限、惯性、动力学参数可在 `robots/fr3/{joint_limits,kinematics,inertials,dynamics}.yaml` 中查阅。

用途：重定向 DexCap/MimicGen 等轨迹到 FR3 时使用上述关节顺序，保持末端/基座定义与官方描述一致。***
