# 参数化 SPM 预训练代理的独立域验证

该步骤只回答一个问题：用于预训练的小型 MLP 能否在未参与训练的参数点上复现指数谱一阶 SPM，以及这种复现能力在适用域边界和训练参数范围外如何变化。

它不验证 SPM 是否代表真实地表，也不构成高保真电磁求解器的替代性证明。

从仓库根目录运行：

```powershell
python scripts/evaluate_synthetic_pretraining_domain.py --output "outputs/scattering/rough_ground/synthetic_pretraining_validation_20260908" --sample-sizes "500,1000,3000,5000" --validation-samples 1000 --repeats 5 --epochs 200
```

先运行单元测试：

```powershell
python scripts/test_synthetic_pretraining_domain.py
```

三个验证域：

- `interpolation`：与预训练相同参数分布，但使用独立随机种子；
- `near_validity_boundary`：仍满足配置的 SPM 有效性不等式，但至少一个归一化有效性指标达到阈值的 80%；
- `range_extension`：仍满足 SPM 有效性不等式，但至少一个输入超出预训练参数区间。

主要输出：

- `metrics_by_repeat.csv`：每个样本量、随机种子、验证域和输出分量的结果；
- `metrics_summary.csv`：五次重复的均值和标准差；
- `largest_model_ensemble_predictions.csv`：最大训练样本量的集成预测；
- `01_synthetic_learning_curve.png`：判断 3000 到 5000 个模拟样本是否继续改善；
- `02_independent_domain_rmse.png`：比较插值、边界和参数扩展误差；
- `03_teacher_vs_surrogate.png`：最大样本模型的预测一致性；
- `summary.json`：参数、随机种子、环境版本和学术解释边界。
