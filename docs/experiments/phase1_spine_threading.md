# Phase 1 — SPINE Threading 数据工厂 + 注意力证据

## 目标
- 生成 1000 条 Threading 任务物理可行轨迹（含接触力）。
- 训练基于注意力的 Diffusion Policy，并输出“Force Token 关注度”证据图。

## 数据来源与规模
- MimicGen 生成轨迹（Threading）
- 输入：`data/kinematics/mimicgen_threading_1300/mimicgen/*.json`
- 输出：`data/spine_dataset_threading/*_{q_opt,force}.npy`
- 目标规模：1000 条

## 批处理参数
- table height: `0.15`（Threading 夹具高度更低）
- comp penalty: `0.01`
- 自动切片：由 `tools/casadi_cito_mvp.py` 自动检测穿模窗口

## 批处理命令
```bash
conda run -n spine_opt python scripts/run_spine_batch.py \
  --input_dir data/kinematics/mimicgen_threading_1300/mimicgen \
  --output_dir data/spine_dataset_threading \
  --urdf models/fr3.urdf \
  --table_height 0.15 \
  --comp_penalty 0.01 \
  --limit 1000 \
  --skip-existing \
  --python_bin "conda run -n spine_opt python"
```

## 数据打包
- 打包脚本：`tools/package_dataset.py`
- 输出 HDF5：`data/spine_threading.hdf5`

```bash
conda run -n spine_opt python tools/package_dataset.py \
  --input_dir data/spine_dataset_threading \
  --output data/spine_threading.hdf5
```

## Policy 训练与注意力证据
### 训练
- 训练脚本：`train_dit_min.py`
- 数据集：`data/spine_threading.hdf5`
- Checkpoints 输出：`data/checkpoints_threading/`

### 证据图（Attention）
流程：
1. `tools/freeze_evidence.py` 扫描并冻结最佳注意力曲线（Threading）
2. `tools/plot_frozen.py` 输出论文级图像

输出：
- 图像：`docs/pics/paper_figure_5_final.png`

说明：
- Free Space 与 Hard Contact 对比显示 Force Token 关注度上升
- 该图与 Square 的 `docs/pics/paper_figure_4_final.png` 形成成对证据

