# P-B-Occ-ALIGNN-full-Tc

Occupancy-aware ALIGNN-style graph neural network for superconducting critical temperature `Tc_K` regression.

本仓库只做一个任务：

```text
CIF + pressure_GPa + magnetic_field_T + field_direction + optional metadata -> Tc_K
```

不做 superconducting / non-superconducting 分类。

## 1. 模型简介

`PBOccALIGNNFullTc` 是一个纯 PyTorch 实现的 ALIGNN-style 图神经网络：

- bond graph 的 node 是 crystallographic site。
- bond graph 的 edge 是周期近邻 directed bond。
- line graph 的 node 是 directed bond。
- line graph 的 edge 是角度 `i-j-k`。
- 所有 scatter / aggregation 使用 `torch.index_add`，不依赖 `torch_scatter`、DGL、PyTorch Geometric 或官方 ALIGNN 包。
- 输出为 `logtc_mu`、`logtc_logvar` 和 `Tc_pred_K = exp(logtc_mu) - 1`。

默认 loss 是 Gaussian NLL，标签为：

```text
y = log(1 + Tc_K)
```

## 2. 为什么不修改 3DSC artificial doping CIF

3DSC artificial doping 后的 CIF 通常通过修改 occupancy 表达掺杂，例如：

```text
Sn 0.000 0.000 0.000 occupancy 0.85
Ag 0.000 0.000 0.000 occupancy 0.15
```

这里应被理解为同一个 crystallographic site 的元素分布：

```text
site_i = {Sn: 0.85, Ag: 0.15}
```

本仓库不会把它展开成 ordered supercell，也不会重写或 relax CIF。模型在内部用 occupancy-weighted embedding 和 expected pair/triplet chemistry 处理 partial occupancy。

## 3. partial occupancy 处理方式

CIF 解析规则：

- 读取 pymatgen 的 `site.species`，保留 disordered site。
- 如果 pymatgen 把同一分数坐标的多个元素读成多个 site，则按 `frac_coord_tol` 合并。
- occupancy 总和 `< 1` 时加入 `Vacancy` token。
- occupancy 总和略大于 1 时归一化。
- occupancy 总和明显大于 1 时打印 warning 并归一化。
- 每个位点最多保留 `max_species_per_site` 个 species，超出时保留 occupancy 最大的 top-K 并重新归一化。

节点表示：

- `node_species_idx: [N, K]`
- `node_species_occ: [N, K]`
- occupancy-weighted element embedding
- 元素性质均值和方差：atomic number、atomic mass、electronegativity、atomic radius、group、period
- disorder features：entropy、max occupancy、number of species、vacancy fraction

## 4. 数据 CSV 格式

建议 CSV 至少包含：

```text
sample_id,cif_path,Tc_K,pressure_GPa,magnetic_field_T,field_direction,
structure_source,match_type,fidelity,parent_cif_id,chemical_system,family,split
```

说明：

- `Tc_K` 是回归标签。
- `cif_path` 可以是绝对路径，也可以是相对 CSV 所在目录的路径。
- 如果存在 `split` 列，训练脚本直接使用。
- 如果不存在 `split`，会优先按 `parent_cif_id` 做 grouped split，避免同一 parent CIF 泄漏到不同集合。
- 如果没有 `parent_cif_id`，则用 `sample_id` 作为 group。

## 5. pressure 和 magnetic field 的 unknown 处理

`pressure_GPa`：

- 数值：使用该值，`pressure_is_unknown = 0`
- `"unknown"`、空值、`NaN`、无法转成 float：设为 `0.0`，`pressure_is_unknown = 1`
- 输入编码为 `[pressure_value, log1p_pressure, pressure_is_unknown]`

`magnetic_field_T`：

- 数值：使用该值，`field_is_unknown = 0`
- `"unknown"`、空值、`NaN`、无法转成 float：设为 `0.0`，`field_is_unknown = 1`
- 输入编码为 `[field_value, log1p_field, field_is_unknown]`

`field_direction` 支持：

```text
zero, parallel_c, parallel_ab, parallel_a, parallel_b, powder, unknown
```

## 6. 安装方法

推荐新建环境：

```bash
conda env create -f environment.yml
conda activate pb-occ-alignn-full-tc
python -m pip install -e . --no-build-isolation
```

或使用 venv：

```bash
python -m venv .venv
.venv/Scripts/activate
python -m pip install -U pip
python -m pip install -e ".[test]"
```

Linux/macOS 激活命令为：

```bash
source .venv/bin/activate
```

## 7. 运行 toy smoke test

Linux / Git Bash / WSL：

```bash
bash scripts/run_smoke_test.sh
```

Windows PowerShell 可逐条运行：

```powershell
python scripts/make_toy_dataset.py --output_dir data/toy
python scripts/train.py --config configs/small_cpu.yaml --data_csv data/toy/toy_data.csv --output_dir runs/smoke --device cpu
python scripts/evaluate.py --checkpoint runs/smoke/best.pt --data_csv data/toy/toy_data.csv --output_dir runs/smoke_eval --split test --device cpu
python scripts/predict.py --checkpoint runs/smoke/best.pt --data_csv data/toy/toy_data.csv --output_dir runs/smoke_predict --device cpu
```

## 8. 在真实数据上训练

```bash
python scripts/train.py \
  --config configs/default.yaml \
  --data_csv path/to/data.csv \
  --output_dir runs/real_exp \
  --device auto
```

继续训练：

```bash
python scripts/train.py \
  --config runs/real_exp/config.yaml \
  --data_csv path/to/data.csv \
  --output_dir runs/real_exp \
  --resume runs/real_exp/last.pt \
  --device auto
```

## 9. 评估模型

```bash
python scripts/evaluate.py \
  --checkpoint runs/real_exp/best.pt \
  --data_csv path/to/data.csv \
  --output_dir runs/real_eval \
  --split test \
  --device auto
```

输出包括：

- `predictions_test.csv`
- `metrics_test.json`
- `grouped_metrics_test.csv`
- predicted vs true Tc 图
- log predicted vs log true 图
- residual histogram
- uncertainty vs absolute error
- metrics by family / match_type 图

## 10. 对新 CIF 推理

CSV 模式：

```bash
python scripts/predict.py \
  --checkpoint runs/real_exp/best.pt \
  --data_csv candidates.csv \
  --output_dir runs/predict_candidates \
  --device auto
```

CIF 文件夹模式：

```bash
python scripts/predict.py \
  --checkpoint runs/real_exp/best.pt \
  --cif_dir path/to/cifs \
  --conditions_csv conditions.csv \
  --output_dir runs/predict_cifs \
  --device auto
```

输出：

```text
runs/predict_cifs/predictions.csv
```

## 11. 输出结果解释

主要列：

- `Tc_pred_K`：预测的临界温度，单位 K。
- `logtc_mu`：模型预测的 `log(1 + Tc)` 均值。
- `logtc_sigma`：模型学习到的 log-space 不确定性尺度。
- `pressure_is_unknown`：压力是否未知。
- `field_is_unknown`：磁场是否未知。

评价指标：

- `MAE_K`：Tc 绝对误差均值，越低越好。
- `RMSE_K`：Tc 均方根误差，越低越好。
- `R2_raw_Tc`：原始 Tc 空间 R²，越高越好。
- `MSLE`：`log(1+Tc)` 空间均方误差，越低越好。
- `MAE_log1p` / `RMSE_log1p`：log 空间误差，越低越好。

## 12. 常见错误

- `CSV must contain a cif_path column`：CSV 缺少 `cif_path`。
- `Tc_K is required for training/evaluation`：训练或评估 CSV 缺少标签；推理请用 `scripts/predict.py`。
- CIF 无法解析：检查 CIF 是否完整、元素符号是否合法、晶胞参数是否存在。
- GPU 不可用：使用 `--device cpu`。
- Windows 不能直接运行 `.sh`：用 Git Bash / WSL，或按 README 的 PowerShell 命令逐条运行。

## 13. 模型局限性

- 该模型把 partial occupancy 当作平均局域化学环境。
- 它不会恢复真实局域无序构型。
- 3DSC artificial doping CIF 只改变 occupancy，不代表真实 relax 后结构。
- 如果训练数据中压力和磁场大多是 unknown，则模型无法真正学到可靠的压力/磁场依赖，只会把它们当作条件 metadata。
- 如果要预测高压或强磁场条件，需要相应条件下的训练数据。
- toy dataset 只用于验证代码能跑通，不代表任何真实物理规律。

## 14. 运行测试

```bash
pytest -q
```

## 15. 仓库结构

```text
pb-occ-alignn-full-tc/
  configs/
  data/toy/
  scripts/
  src/occ_alignn/
  tests/
  README.md
  requirements.txt
  environment.yml
  pyproject.toml
```
