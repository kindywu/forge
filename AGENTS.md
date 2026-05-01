# AGENTS.md

本文件为 AI 编码助手提供项目背景、架构约束和开发规范。阅读者被视为对本项目一无所知。

---

## 项目概述

**电石炉电极操作智能决策系统** —— 一套面向钙电石炉的闭环 AI 控制系统。系统以 30 秒为周期，基于两段 LightGBM 因果模型 + SHAP 可解释性分析 + 独立安全守卫，推荐三相电极深度调整方案，核心目标是将功率因数维持在 `[0.92, 0.93]` 区间，同时降低吨电耗。

本项目当前使用**模拟数据**运行（`data/simulator.py`），真实部署时需替换为 DCS / 罗茨线圈实时数据读取。

---

## 技术栈

| 层级 | 技术 |
|------|------|
| 语言 | Python 3 |
| 核心依赖 | `lightgbm==4.6.0`, `pandas==3.0.2`, `numpy==2.4.4`, `scikit-learn==1.8.0`, `shap==0.51.0` |
| 包管理 | `pip` / `uv`（推荐用 `uv pip install -r requirements.txt`）|
| 虚拟环境 | `.venv/`（已纳入 `.gitignore`）|

> 注：本项目没有 `pyproject.toml`、`setup.py`、`setup.cfg` 或 `package.json` 等构建配置文件，依赖直接由 `requirements.txt` 管理。

---

## 代码组织与模块划分

```
├── main.py                      # 主入口：数据生成 → 训练 → SHAP → 闭环决策演示
├── demo.py                      # 独立的 LightGBM 入门示例（加州房价数据集）
├── requirements.txt             # Python 依赖清单
├── data/
│   └── simulator.py             # 基于物理逻辑的三层数据模拟器
├── models/
│   ├── two_stage_model.py       # TwoStageModel 类：训练 / 推理 / 保存 / 加载 / SHAP
│   └── saved/                   # 模型持久化目录（.pkl 文件）
├── engine/
│   ├── decision_engine.py       # DecisionEngine：五步闭环决策引擎
│   └── safety_guard.py          # SafetyGuard：独立于 AI 的硬约束安全检查
└── utils/
    └── feature_engineering.py   # 特征工程：滞后 / 滚动 / 时间特征构造
```

### 三层因果链（核心架构约束）

这是系统最重要的设计约束，**绝不能**将三层压缩为单一模型：

```
驱动层 (drive)      →  中间状态 (intermediate)    →  结果指标 (result)
electrode depth,      current, impedance,           power_factor,
c_cao_ratio, lime,    imbalance, temp,              energy_per_ton
coke_fixed_carbon     furnace_pressure
```

- **模型一** (`model1`)：驱动层特征 → 7 个中间状态目标（每个目标一个 LightGBM）
- **模型二** (`model2`)：中间状态特征（含历史滞后/滚动） → `power_factor` + `energy_per_ton`（每个目标一个 LightGBM）

两个模型均将各目标回归器存储为 `Dict[str, lgb.LGBMRegressor]`。

---

## 构建与运行命令

```bash
# 激活虚拟环境（Windows 示例）
.venv\Scripts\activate

# 安装依赖
uv pip install -r requirements.txt

# 运行完整演示（数据生成 → 训练 → SHAP → 5 场景模拟决策）
uv run main.py
```

**当前没有自动化测试**。`demo.py` 只是一个独立的 LightGBM 入门脚本，不参与主系统。

---

## 关键设计约束

### 1. 时序切分（禁止随机打乱）

训练集和测试集必须按时间顺序切分（`iloc[:split_idx]` / `iloc[split_idx:]`），**绝不能**使用 `train_test_split(shuffle=True)`。随机打乱会导致未来信息泄漏到训练集中。

相关代码：`models/two_stage_model.py` 中的 `_train_multi_output()`。

### 2. 模型一不加滞后/滚动特征

`build_features_for_model1()` 仅使用驱动层原始值 + 时间特征，**不添加滞后/滚动**。原因：模型一承担反事实仿真职责（如“改变电极深度后电流会如何变化”），若引入历史滞后，候选动作之间的预测差异会被历史特征抹平，导致仿真失效。

### 3. 推理特征构造必须使用 `build_inference_features()`

在 `utils/feature_engineering.py` 中，`build_inference_features()` 是推理时构造特征的正确路径：

1. 将候选行追加到历史数据尾部；
2. 重新运行完整的特征管道（`build_features_for_model1/2`）；
3. 提取最后一行。

这确保滞后/滚动特征基于“历史 + 候选”的完整序列正确计算。

`decision_engine.py` 中的 `_build_simple_drive_features()` 是一个简化回退路径，用 `iloc[-1]` 填充缺失特征，**不能产生准确的滞后特征**，应避免使用。

构造候选行时务必：
1. 设置有效的 `timestamp`（如 `last_ts + timedelta(hours=1)`），否则会被 `dropna()` 丢弃；
2. 填写完整的 `DRIVE_COLS`，缺失列会导致特征矩阵出现 NaN 而被丢弃。

### 4. 安全守卫完全独立于 AI 模型

`safety_guard.py` **不导入任何模型代码**。它只检查硬物理/电气约束：

| 约束项 | 限制值 | 说明 |
|--------|--------|------|
| 电流上限 | 额定值的 90% | 防止过载 |
| 电极深度 | 0.8 ~ 1.8 m | 避免电弧闪烁 / 防烧损顶炉 |
| 三相不平衡度 | < 5% | 超过会损坏设备 |
| 单次调节幅度 | < 5 cm（演示可放宽） | 防止突变 |
| 炉压 | 80 ~ 200 Pa | 空气倒灌爆炸 / 密封损坏 |
| 功率因数紧急底线 | ≥ 0.80 | 低于此值紧急停机转人工 |

AI 推荐若未通过安全检查，**直接丢弃**；安全守卫拥有最终否决权。

### 5. LightGBM 超参数

针对工控机部署优化（小树、快训练、低内存）：

```python
num_leaves=31
learning_rate=0.05
n_estimators=500
early_stopping_rounds=50
feature_fraction=0.8
bagging_fraction=0.8
bagging_freq=5
```

相关代码：`models/two_stage_model.py` 中的 `LGBM_PARAMS`。

---

## 五步决策循环

每次调用 `engine.step()` 执行：

1. **紧急检查** — `power_factor < 0.80` 或炉压严重超限 → 立即转人工；
2. **实时比对** — 当前 PF 若在 `[0.92, 0.93]` → 保持不动；
3. **仿真试算** — 对每相电极枚举 `±1/±2/±4 cm`（及三相联动 `±4/±8/±12 cm`），共 ≤ 18 个候选动作，用两段模型预测结果；
4. **智能择优** — 综合评分：PF 接近 0.925（50%） + 吨电耗低（30%） + 调节幅度小（20%）；
5. **安全拦截** — 逐个检查候选动作，返回最优通过者，或全部拒绝则转人工。

相关代码：`engine/decision_engine.py`。

---

## 编码规范

- 注释和文档字符串使用**中文**；
- 物理量单位在注释中明确标注（如 `m` / `cm` / `kA` / `kWh/t` / `Pa`）；
- 时序相关操作优先使用 `.iloc[]` 基于位置的索引，避免标签索引带来的对齐陷阱；
- 所有 `clip()` 调用应紧跟在随机噪声叠加之后，确保物理合理性。

---

## 测试说明

**当前仓库中没有单元测试或集成测试**。验证正确性的方式：

1. 运行 `python main.py` 观察训练精度（`power_factor` MAPE 应约 0.98%）；
2. 检查 5 个模拟场景的决策行为是否符合预期（偏低→调整、区间内→保持、紧急→转人工、全被拒→转人工）。

若新增测试，建议覆盖：
- `safety_guard.py` 的各边界条件检查；
- `feature_engineering.py` 的滞后/滚动特征正确性；
- `decision_engine.py` 的候选枚举和评分逻辑。

---

## 部署替换清单

真实部署到产线时，必须替换/增加以下内容：

1. **`data/simulator.py`** → 替换为 DCS / 罗茨线圈实时数据读取模块；
2. **`engine/decision_engine.py`** → 接入 PLC/DCS 下发接口，实现 `auto_execute=True` 的物理执行；
3. **定时任务** — 每 30 秒调用一次 `engine.step()`；
4. **Web 界面** — 展示推荐方案、历史趋势、预测 vs 实测对比；
5. **在线学习** — 累积 `record_actual_result()` 的预测误差样本，周期性重训模型。

---

## 安全注意事项

- `main.py` 开头包含 Windows GBK 终端编码修复（`sys.stdout = io.TextIOWrapper(..., encoding='utf-8')`），在 Linux 上运行无影响；
- 模型文件 `models/saved/two_stage_model.pkl` 使用 `cloudpickle` 序列化，加载时需保证 `lightgbm` 版本一致；
- ` SafetyGuard` 的 `electrode_max_step_cm` 在演示场景中被放宽到 `12.0`，真实部署必须恢复为 `5.0` 或更严格；
- 所有数值约束（深度、电流、压力、不平衡度）应根据现场设备铭牌和操作规程调整，不可直接照搬代码默认值。
