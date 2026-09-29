# 统一样本与极化失配诊断：运行说明

本轮任务是论文前置诊断，不是新模型训练。所有交付代码和结果在 PROJECT_ROOT。
原始数据和旧实验结果不改动。脚本拒绝覆盖同名结果，复跑请指定新目录。

## 代码

- build_paper_cohort.py：以原 SPM 有效域及有限值筛选规则恢复主实验样本，再按地块、日期一对一合并植被信息。
- analyze_polarization_discrepancy.py：统计、地块 Bootstrap 和四张图。
- paper_diagnostics_core.py：两入口共享的实现，三个文件需保持在同一 scripts 目录。
- test_paper_diagnostics.py：10 项小型回归测试。

依赖：numpy、pandas、scipy、matplotlib。已在 D:/miniconda3/envs/research-pilots/python.exe 的现有环境运行，未安装新依赖。
不需要 rasterio，不重新读取或下载大尺寸 GeoTIFF；本步读取现有植被审计 CSV。

## 已生成结果

根目录：PROJECT_ROOT/outputs/scattering/rough_ground/paper_diagnostics

- cohort_audit.csv：保留全部 252 个输入行，记录是否进入各子集和排除原因。
- cohort_base.csv：原模型的 186 个有效、有限值样本。
- cohort_common.csv：植被信息可用的共同子集，本次也是 186 行。
- cohort_high_quality.csv：实地植被日期差不超过 2 天的 102 行。此阈值是分析设定，不是官方质量等级。
- cohort_counts_by_field.csv：各地块的入选数量。
- cohort_manifest.json：来源绝对路径、SHA256、样本数量、代码指纹及质量口径。
- analysis_20260908/：2000 次地块 Bootstrap、随机种子 20260908 的正式本轮结果。

analysis_20260908 内：

- association_statistics.csv：总体、地块内部、地块均值的 Spearman 相关及区间。
- diagnostic_rows.csv：带共同残差、差异残差的明细。
- analysis_summary.json：核心数字与来源。
- README_results.md：图表阅读方法和统计限制。
- 01_polarization_comparison.png：HH/VV 极化差与共同强度的模拟—实测对照。
- 02_common_residual_environment.png：共同残差与环境参数。
- 03_differential_residual_pooled_within.png：极化差残差的总体、地块内部关联。
- 04_in_situ_quality_sensitivity.png：实地匹配时间差的敏感性。

## 本次实际结果

原始输入 252 行，恢复原模型子集 186 行、19 地块、12 日期。
共同子集无额外样本损失。最近日期差 <=2 天为 102 行、19 地块、10 日期；<=4 天为 178 行。
本子集的同日实地植被匹配为 0 行，图上 unavailable 是如实报告，不是程序故障。

实地植被匹配均为最近日期；不能混同于卫星产品来源标签。
卫星产品在共同子集内保留的来源标签为：direct_satellite 33、interpolated 129、extrapolated 24。
这些标签继承旧审计，不代表本轮重新验证了卫星产品质量。

| 实地 VWC 的关联对象 | 方式 | rho | 点态 95% 地块 Bootstrap 区间 |
|---|---|---:|---|
| 共同残差 | 总体 | 0.270 | [-0.116, 0.604] |
| 共同残差 | 地块内部 | 0.502 | [0.251, 0.680] |
| 极化差残差 | 总体 | 0.568 | [0.296, 0.725] |
| 极化差残差 | 地块内部 | 0.373 | [0.108, 0.588] |
| 极化差残差，<=2天子集 | 地块内部 | 0.421 | [0.143, 0.642] |

极化差残差平均 -4.295 dB：在这个子集，SPM 的 VV-HH 平均高于实测约 4.295 dB。
共同残差平均 +7.288 dB：两极化 dB 均值的模拟整体偏低。
注意正相关指有符号残差增加，不能自动解释为绝对误差随植被增加而变大。

当前代码中高斯谱、指数谱的模拟极化差最大差别约 1.78e-14 dB，属于浮点误差。
这是当前一阶共享粗糙度因子模型的代数性质，不是新发现的普遍地表散射定律。

## 复跑

在 VS Code PowerShell Terminal 执行，而不是在 Python 的 >>> 内。

```powershell
conda activate research-pilots
Set-Location 'PROJECT_ROOT'
python -B 'PROJECT_ROOT/scripts\test_paper_diagnostics.py'
```

完整复跑到新的 D 盘目录，不覆盖已有结果：

```powershell
$diagnosticRun = 'PROJECT_ROOT/outputs\scattering\rough_ground\paper_diagnostics\repeat_' + (Get-Date -Format 'yyyyMMdd_HHmmss')
python -B 'PROJECT_ROOT/scripts\build_paper_cohort.py' --output $diagnosticRun --quality-days 2
if ($LASTEXITCODE -ne 0) { throw 'Cohort build failed. Stop and inspect the error.' }
python -B 'PROJECT_ROOT/scripts\analyze_polarization_discrepancy.py' --input "$diagnosticRun\cohort_common.csv" --output "$diagnosticRun\analysis" --bootstrap 2000 --seed 20260908
```

只复跑已有共同子集的分析，脚本自动新建时间戳目录：

```powershell
python -B 'PROJECT_ROOT/scripts\analyze_polarization_discrepancy.py' --bootstrap 2000 --seed 20260908
```

如果终端环境切换仍有疑问，可以将 python 替换为：

```powershell
 python -B 'PROJECT_ROOT/scripts\analyze_polarization_discrepancy.py'
```

Matplotlib 缓存也设在相应 D 盘分析目录。-B 禁止本轮 Python 字节码缓存写入。

## 公式和统计口径

先计算 rHH=observed_HH-SPM_HH、rVV=observed_VV-SPM_VV，全部为 dB。
rc=(rHH+rVV)/2；rd=rVV-rHH。
它们是残差坐标，不能直接命名为粗糙度散射、植被散射或双次散射功率。
共同残差中的“平均”是 dB 算术平均，不是两极化线性功率的算术平均。

总体分析按地块—日期记录计算，地块均值分析每地块一条；内部分析在每对有限变量上减地块均值。
对地块内不变化的粗糙度变量，不强行计算内部相关。静态输入在残差公式中本身出现，相关也可能含数学耦合。
Bootstrap 重采样整个地块，保留块内日期依赖；跨地块共同日期效应仍未消除。
区间是未经多重比较校正的探索性点态区间，不能用其中几个不跨零的区间证明新机制。
lofo_rho_min/max 仅为逐地块删除的描述性敏感性，不是模型留出预测。

## 对论文主线的作用

本步提供“共同强度与极化差失配需要分别检查”的证据，仍然属于地表电磁散射正演方向。
不宣称创新模型已有效，也不宣称植被因果机制已识别。
下一步优先：在同一样本和地块划分下，比较增加植被输入前后的普通基线，再做物理控制变量实验。
物理模拟预训练和物理约束服务于可信响应保留，不单独作为创新声明。
本轮无需新下载；若后续强调同日验证或跨场景泛化，再核查原始采样日期和补独立验证数据。
水下目标留作后续独立场景扩展，不纳入本轮结论。
