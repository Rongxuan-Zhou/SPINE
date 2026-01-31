# Phase 1 — SPINE Square 数据工厂

## 目标
- 生成 1000 条 Square 任务的物理可行轨迹（含接触力），用于后续 Policy 训练。
- 验证 CITO 修补批量化稳定性与修复率。

## 数据来源与规模
- MimicGen 生成轨迹（Square）
- 输入：`data/kinematics/mimicgen_square_1300/mimicgen/*.json`
- 输出：`data/spine_dataset_square/*_{q_opt,force}.npy`
- 目标规模：1000 条

## 批处理参数
- table height: `0.20`
- comp penalty: `0.01`
- 自动切片：由 `tools/casadi_cito_mvp.py` 内部自动检测穿模窗口

## 批处理命令
```bash
conda run -n spine_opt python scripts/run_spine_batch.py \
  --input_dir data/kinematics/mimicgen_square_1300/mimicgen \
  --output_dir data/spine_dataset_square \
  --urdf models/fr3.urdf \
  --table_height 0.20 \
  --comp_penalty 0.01 \
  --limit 1000 \
  --skip-existing \
  --python_bin "conda run -n spine_opt python"
```

## 数据打包
- 打包脚本：`tools/package_dataset.py`
- 输出 HDF5：`data/spine_square.hdf5`

```bash
conda run -n spine_opt python tools/package_dataset.py \
  --input_dir data/spine_dataset_square \
  --output data/spine_square.hdf5
```

## 质量评估
建议使用 `tools/analyze_dataset.py` 统计：
- 平均接触力（Mean Contact Force）
- 修复率（Max Force > 0.5N）
- 残留穿模（z_min >= table_height）

