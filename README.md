# 电石炉电极操作智能决策系统

基于 LightGBM 两段因果模型 + SHAP 可解释性 + 安全约束防护的闭环决策系统。

## 架构总览

```mermaid
flowchart TD
    subgraph loop["闭环决策循环 (每30秒)"]
        direction LR
        A["① 实时比对<br/>PF vs [0.92, 0.93]"] --> B["② 仿真试算<br/>枚举18个候选动作"]
        B --> C["③ 智能择优<br/>加权评分排序"]
        C --> D["④ 安全拦截<br/>硬约束检查"]
        D --> E["⑤ 执行 + 回验<br/>记录预测vs实测偏差"]
        E --> A
    end
```

```mermaid
flowchart LR
    subgraph dataflow["数据流 — 一条因果链"]
        direction LR
        X["驱动层<br/>电极操作 · 原料参数"] -->|"模型一<br/>LightGBM"| Y["中间状态<br/>电流 · 阻抗 · 不平衡度 · 炉温 · 炉压"]
        Y -->|"模型二<br/>LightGBM"| Z["结果指标<br/>功率因数 · 吨电耗"]
    end
```

### 三层数据结构

| 层级 | 字段 | 说明 |
|------|------|------|
| **驱动层** | `electrode_depth_{a,b,c}`, `electrode_speed_{a,b,c}`, `c_cao_ratio`, `lime_flow`, `coke_fixed_carbon` | 可直接控制的变量 |
| **中间状态** | `current_{a,b,c}`, `short_net_impedance`, `imbalance`, `reaction_temp`, `furnace_pressure` | 电气参数和炉况（模型一输出） |
| **结果指标** | `power_factor`, `energy_per_ton` | 最终关心的经济/安全指标（模型二输出） |

## 小白入门：术语通俗解释

如果你刚接触工业 AI 或者对电石炉不熟悉，这一节用**生活中的比喻**帮你快速理解每个术语。

---

### 场景比喻：把电石炉想象成一口"智能高压锅"

> 你正在用高压锅炖一锅牛肉。火太小 → 炖不烂；火太大 → 糊锅；火不稳 → 半生不熟。
> 电石炉比高压锅大一万倍，温度高达 2000°C，但面临的问题本质一样：**怎么把"火候"控制在最佳状态？**

---

### 设备相关

| 术语 | 小白解释 |
|------|----------|
| **电石炉** | 一个巨大的工业电炉，用电流通过原料产生 2000°C 高温，把石灰和焦炭"煮"成电石（化工原料）。你可以理解为一口三层楼高的"电磁炉火锅"。 |
| **电极**（electrode） | 三根巨大的石墨柱子，从上往下插入炉内，电流通过它们加热原料。**比喻**：就像你把三根金属筷子插进火锅里通电加热——筷子插得越深，电流越大，温度越高。共有 A、B、C 三相。 |
| **三相电流** | 三根电极各自通的电流大小（单位：千安培 kA）。正常约 100~140 kA。**比喻**：就像三根筷子上各自通过的"火力"，最好三根火力均匀。 |
| **原料** | 石灰（CaO）和焦炭（C）。**比喻**：火锅底料和食材。比例不对，味道就差。 |

### 状态相关

| 术语 | 小白解释 |
|------|----------|
| **功率因数**（Power Factor, PF） | **最核心的指标！** 衡量电能被"有效利用"的比例，范围 0~1。1.0 = 完美利用，0.8 = 浪费了20%的电。**比喻**：就像你花100块钱买食材，最后吃到嘴里的只有80块的价值——功率因数就是"吃到嘴里的比例"。目标区间 [0.92, 0.93]。 |
| **吨电耗**（energy_per_ton） | 生产一吨电石要消耗多少度电（kWh/t）。正常约 2800~3300。**比喻**：就像汽车的百公里油耗——越低越省钱。这是老板最关心的数字。 |
| **三相不平衡度**（imbalance） | 三根电极电流不一致的程度。**比喻**：三个人抬轿子，有人出力多有人出力少——不仅不稳，还容易"闪了腰"（损坏设备）。要控制在 5% 以内。 |
| **短网阻抗**（short_net_impedance） | 从变压器到电极这段"电线"的电阻。阻抗越小，电损耗越低。**比喻**：就像水管的粗细——粗管子水流顺畅，细管子水压损失大。 |
| **反应区温度**（reaction_temp） | 炉内发生化学反应区域的温度，约 1600~2000°C。太低反应不充分，太高浪费能量还可能损坏炉子。 |
| **炉内气相压力**（furnace_pressure） | 炉子里的气体压力（单位 Pa）。**比喻**：就像高压锅的气压——太低空气倒灌有爆炸风险，太高会损坏密封。 |

### AI / 机器学习相关

| 术语 | 小白解释 |
|------|----------|
| **模型** | 一个能从历史数据中"学到规律"的数学函数。你给它输入，它给你输出。**比喻**：就像你请了一个老厨师，告诉他"以前 1000 次做了什么操作 → 出了什么结果"，他就能预测"下次你这样操作会出什么结果"。 |
| **LightGBM** | 一种**决策树集成算法**，特点是训练快、内存小、精度高。**比喻**：不是请一个专家，而是请 500 个普通人各自给出判断，然后综合投票——结果往往比单一专家更准。本系统用它来模拟"操作 → 结果"的因果链。 |
| **SHAP** | 一个**解释器**，告诉你"模型的预测是依据哪些因素做出的"。**比喻**：你问老厨师"为什么建议火开大一点？"他回答："因为当前温度偏低（占 40%）、原料湿度大（占 30%）、上次小火效果不好（占 20%）……"SHAP 就是让 AI 模型也能给出这样的解释。 |
| **特征工程** | 把原始数据"打扮"成模型更容易理解的形式。**比喻**：超市小票是原始数据，你把它整理成"每周蔬菜支出趋势图"就是特征工程。本系统主要做三件事：① 加上历史记录（滞后特征）② 计算近期波动（滚动统计）③ 标注时间规律（班次/星期几）。 |
| **训练 / 推理** | 训练 = 用历史数据"教"模型（像老师用练习册教学生）。推理 = 训练好的模型做预测（像学生考试做题）。**比喻**：训练是你给厨师 1000 份菜谱和评分让他学习；推理是你告诉他原料，让他预测好不好吃。 |
| **R²**（R方） | 衡量预测有多准的指标。0 = 瞎猜，0.5 = 有点准，0.8+ = 很准，1.0 = 百分百准确（现实中不可能）。**比喻**：R²=0.8 意味着模型"解释"了 80% 的变化规律，剩下 20% 是意外因素（噪音）。 |
| **MAPE**（平均绝对百分比误差） | 另一种衡量预测精度的指标，数字越小越好。MAPE=2% 意思是预测值和真实值平均只差 2%。功率因数预测的 MAPE 只有 0.98%，说明非常准。 |

### 系统设计相关

| 术语 | 小白解释 |
|------|----------|
| **因果链** | 把复杂问题拆成"因为…所以…"的小步骤。**比喻**：不是一步从"买菜"跳到"好不好吃"，而是：买菜 → 切菜 → 下锅 → 调味 → 品尝。本系统的因果链：**电极操作 → 电流/温度变化 → 功率因数/吨电耗**。 |
| **两段模型** | 把因果链拆成两段，各自训练一个 AI 模型。模型一：操作 → 中间状态；模型二：中间状态 → 最终结果。**比喻**：第一个师傅管"火候控制"，第二个师傅管"食材是否煮到位"——比一个师傅从头管到尾更专注、更准。 |
| **闭环决策** | 不靠人手动调整，系统自动"看 → 想 → 动 → 回头看 → 再看 → ……"无限循环。每30秒跑一次。**比喻**：不是定好闹钟 10 分钟后关火，而是每 5 秒尝一口汤，觉得淡了就加盐，咸了就加水——实时调整。 |
| **安全守卫** | **完全独立于 AI** 的硬规则检查。AI 说了不算，安全守卫说了才算。**比喻**：AI 是老厨师，安全守卫是食品安检员——不管厨师多厉害，安检员说"这盘菜细菌超标不能上桌"，那就绝对不能上桌。 |
| **滞后特征**（lag） | 把过去几小时的数据"贴"到当前时刻。比如 `power_factor_lag_3` = 3小时前的功率因数。**比喻**：你不是只看锅里现在什么状态，还要回忆"3分钟前是什么味道"——趋势比单点更重要。 |
| **滚动统计**（rolling） | 计算过去一段时间的平均值/波动幅度。比如 `current_roll_6_mean` = 过去6小时电流的均值。**比喻**：你不光看当前火候，还看"过去10分钟火候稳不稳"——忽大忽小的火比稳定的小火更糟糕。 |
| **仿真试算** | 在真正执行之前，先用模型"假装执行"各种候选方案，看哪个效果最好。**比喻**：你先在脑子里想"加一勺盐会怎样？加两勺呢？加半勺呢？"——想好了再动手。 |

---

## 核心流程详解

### 一、测试数据生成流程

基于物理因果逻辑的三层数据模拟，从驱动因素逐步推导到结果指标。

```mermaid
flowchart TD
    subgraph init["初始化"]
        A1["设置随机种子 seed=42"] --> A2["生成时间索引<br/>2024-01-01 起，每小时一条"]
        A2 --> A3["创建空 DataFrame"]
    end

    subgraph layer1["第一层：驱动因素（可直接控制）"]
        B1["电极三相深度<br/>1.2m 基线 + sin 波动 + N(0,0.1)噪声<br/>clip 到 [0.8, 1.8]m"]
        B2["电极三相升降速度<br/>U(-0.5, 0.5) cm/s"]
        B3["C/CaO 摩尔比<br/>3.05 + N(0,0.05)，clip [2.9,3.2]"]
        B4["石灰流量<br/>80 + N(0,5)，clip [60,100] kg/h"]
        B5["焦炭固定碳<br/>85 + N(0,1)，clip [80,90]%"]
    end

    subgraph layer2["第二层：中间状态（由驱动层 + 噪声决定）"]
        C1["三相电流 = f(平均深度)<br/>深度↑ → 电流↑，100~140kA<br/>+ N(0,3)噪声"]
        C2["短网阻抗 = f(平均深度)<br/>深度↑ → 阻抗↓<br/>+ N(0,0.1)噪声"]
        C3["三相不平衡度 = f(深度标准差)<br/>std(depth)×10 + Exp(0.5)<br/>clip [0, 8]%"]
        C4["反应区温度 = f(电流)<br/>1800 + (I-100)×2 + N(0,30)<br/>clip [1600,2000]°C"]
        C5["炉内气相压力<br/>120 + N(0,15)<br/>clip [80,200] Pa"]
    end

    subgraph layer3["第三层：结果指标（由中间状态 + 噪声决定）"]
        D1["功率因数 = f(电流, 不平衡度)<br/>倒U型: 最优电流120kA<br/>pf = 0.93 - ((I-120)/20)²×0.1<br/>- imbalance×0.005 + N(0,0.01)<br/>clip [0.80, 0.98]"]
        D2["吨电耗 = f(功率因数, 焦炭品质)<br/>3200 - (pf-0.80)/0.18×400<br/>- (固定碳-80)×10 + N(0,30)<br/>clip [2800, 3300] kWh/t"]
    end

    init --> layer1
    layer1 --> layer2
    layer2 --> layer3
    layer3 --> E["输出完整 DataFrame<br/>列: timestamp + 9驱动 + 7中间 + 2结果<br/>行: n_hours 条时序记录"]
```

### 二、模型训练流程

两段 LightGBM 模型分别拟合因果链的两段映射关系，使用时序切分保证不泄露未来信息。

```mermaid
flowchart TD
    subgraph prep["数据准备"]
        P1["加载完整 DataFrame<br/>驱动层 + 中间状态 + 结果指标"] --> P2["按时间排序<br/>sort_values('timestamp')"]
    end

    subgraph model1["模型一训练：驱动层 → 中间状态"]
        M1A["构造特征 X1<br/>build_features_for_model1()"] --> M1B["原始驱动特征 + 滞后值(_lag_1,3,6,12)<br/>+ 滚动统计(_roll_mean, _roll_std)<br/>+ 时间特征(hour_sin/cos, 班次)"]
        M1B --> M1C["标签 Y1 = INTERMEDIATE_COLS<br/>current_a/b/c, impedance, imbalance,<br/>reaction_temp, furnace_pressure"]
        M1C --> M1D["时序切分（不随机shuffle!）<br/>前80%训练集 / 后20%测试集"]
        M1D --> M1E["对 Y1 的每个目标列<br/>独立训练 LightGBM"]
        M1E --> M1F["LGBMRegressor<br/>num_leaves=31, lr=0.05<br/>n_estimators=500<br/>early_stopping_rounds=50"]
        M1F --> M1G["评估: R² · MAE · MAPE<br/>保存 model1[col] = trained_model"]
    end

    subgraph model2["模型二训练：中间状态 → 结果指标"]
        M2A["构造特征 X2<br/>build_features_for_model2()"] --> M2B["中间状态特征 + 滞后/滚动<br/>结果指标历史滞后(_lag_1,3,6,12)<br/>+ 时间特征 + 驱动层原始值"]
        M2B --> M2C["标签 Y2 = RESULT_COLS<br/>power_factor, energy_per_ton"]
        M2C --> M2D["时序切分（不随机shuffle!）<br/>前80%训练集 / 后20%测试集"]
        M2D --> M2E["对 Y2 的每个目标列<br/>独立训练 LightGBM"]
        M2E --> M2F["LGBMRegressor<br/>num_leaves=31, lr=0.05<br/>n_estimators=500<br/>early_stopping_rounds=50"]
        M2F --> M2G["评估: R² · MAE · MAPE<br/>保存 model2[col] = trained_model"]
    end

    subgraph output["输出"]
        O1["model1: Dict[str, LGBM] — 7个模型"]
        O2["model2: Dict[str, LGBM] — 2个模型"]
        O3["eval_results: 全部评估指标"]
        O4["model1/2_feature_cols: 特征列名"]
    end

    prep --> model1
    prep --> model2
    model1 --> output
    model2 --> output
```

### 三、模型使用（推理 + 闭环决策）流程

加载训练好的模型后，系统以 30 秒为周期运行五步闭环决策，每次循环都经过安全硬约束检查。

```mermaid
flowchart TD
    subgraph load["模型加载"]
        L1["TwoStageModel.load()"] --> L2["恢复 model1(7个LightGBM)<br/>恢复 model2(2个LightGBM)<br/>恢复 feature_cols / eval_results"]
    end

    subgraph inference["两段推理 predict_pipeline()"]
        I1["输入：驱动层特征<br/>electrode_depth, c_cao_ratio, ..."] --> I2["模型一 predict_intermediate()<br/>对每个中间状态列独立预测"]
        I2 --> I3["输出：中间状态预测<br/>current_a/b/c, impedance,<br/>imbalance, temp, pressure"]
        I3 --> I4["拼接驱动特征 + 中间状态"]
        I4 --> I5["模型二 predict_result()<br/>对 power_factor, energy_per_ton 预测"]
        I5 --> I6["输出：结果指标预测<br/>power_factor, energy_per_ton"]
    end

    subgraph loop["闭环决策循环 engine.step() — 每30秒"]
        S0["输入：current_state 实时数据<br/>+ history_df 历史上下文"] --> S1
        S1{"步骤① 紧急检查<br/>safety_guard.check_emergency()"} -->|"PF < 0.80<br/>或炉压严重超限"| EMG["🚨 紧急状态<br/>status='emergency'<br/>切回人工控制"]
        S1 -->|"正常"| S2{"步骤① 实时比对<br/>当前 PF vs [0.92, 0.93]"}
        S2 -->|"已在目标区间"| HOLD["✓ 保持不动<br/>status='hold'"]
        S2 -->|"偏离目标"| S3["步骤② 仿真试算<br/>_simulate_candidates()"]
        S3 --> S3A["对每相电极尝试 ±1cm/±2cm/±4cm<br/>（超出[0.8,1.8]m直接跳过）<br/>共 ≤18 个候选动作"]
        S3A --> S3B["对每个候选动作<br/>用两段模型 predict_pipeline()<br/>预测执行后的 power_factor + energy_per_ton"]
        S3B --> S4["步骤③ 智能择优<br/>_score_and_rank()"]
        S4 --> S4A["综合评分：<br/>功率因数接近0.925 → 权重50%<br/>吨电耗越低越好 → 权重30%<br/>调节幅度越小越好 → 权重20%"]
        S4A --> S4B["按 score 降序排列"]
        S4B --> S5["步骤④ 安全拦截<br/>safety_guard.check() — 逐个检查"]
        S5 --> S5A["硬约束检查项：<br/>• 电极深度 ∈ [0.8, 1.8]m<br/>• 单次调节 ≤ 5cm<br/>• 电流 < 额定90%<br/>• 三相不平衡度 < 5%<br/>• 炉压 ∈ [80, 200]Pa"]
        S5A --> S5B{"通过?"}
        S5B -->|"全部被拒"| ESC["⚠ 转人工<br/>status='escalate'"]
        S5B -->|"有通过者"| S6["步骤⑤ 输出最优推荐<br/>best = safe_candidates[0]"]
        S6 --> S6A["生成人类可读建议<br/>记录 DecisionRecord"]
        S6A --> EXEC{"auto_execute?"}
        EXEC -->|"True"| DCS["下发 PLC/DCS 执行"]
        EXEC -->|"False"| HUMAN["展示建议，等待人工确认"]
    end

    subgraph feedback["步骤⑤ 回验（下一个采样周期）"]
        F1["实测到新的 power_factor"] --> F2["record_actual_result()"]
        F2 --> F3["计算预测误差<br/>error = |predicted - actual|"]
        F3 --> F4["累积误差样本<br/>用于在线学习 / 周期性重训"]
    end

    load --> inference
    inference --> loop
    DCS --> feedback
    HUMAN --> feedback
    feedback -.->|"周期性重训"| load
```

## 模块说明

```mermaid
flowchart TB
    main.py["main.py<br/>主程序入口"]
    subgraph data["data/"]
        simulator.py["simulator.py<br/>模拟数据生成器"]
    end
    subgraph models["models/"]
        two_stage_model.py["two_stage_model.py<br/>两段LightGBM模型"]
        saved["saved/<br/>模型持久化"]
    end
    subgraph engine["engine/"]
        decision_engine.py["decision_engine.py<br/>闭环决策引擎"]
        safety_guard.py["safety_guard.py<br/>安全约束检查"]
    end
    subgraph utils["utils/"]
        feature_engineering.py["feature_engineering.py<br/>特征工程"]
    end
    main.py --> data
    main.py --> models
    main.py --> engine
    main.py --> utils
```

### `simulator.py` — 数据模拟器

基于物理逻辑生成训练数据：
- 电极深度 → 电流 → 功率因数（倒U型曲线，最优电流 120kA）
- 三相深度差异 → 不平衡度 → 功率因数衰减
- 焦炭固定碳 → 反应效率 → 吨电耗

### `two_stage_model.py` — 两段模型

对每个目标列独立训练一个 LightGBM 回归器：

- **模型一**：驱动层特征 → 7个中间状态指标
- **模型二**：中间状态特征（含历史滞后/滚动） → 功率因数 + 吨电耗

关键设计决策：
- 时序数据按时间顺序切分（不随机shuffle），前80%训练，后20%测试
- 超参数针对工控机优化：`num_leaves=31`, `n_estimators=500`, early stopping
- SHAP TreeExplainer 用于可解释性分析

### `feature_engineering.py` — 特征工程

为原始数据构造三类特征：
- **滞后特征**：`_lag_1`, `_lag_3`, `_lag_6`, `_lag_12` — t-1, t-3, t-6, t-12 时刻的历史值
- **滚动统计**：`_roll_3_mean`, `_roll_6_mean`, `_roll_6_std` 等 — 历史窗口的均值/标准差
- **时间特征**：`hour`, `day_of_week`, `is_night_shift`, `hour_sin`, `hour_cos` — 捕捉班次和周期规律

### `decision_engine.py` — 决策引擎

每30秒一次的五步循环：

1. **紧急检查** — 功率因数 < 0.80 直接切回人工
2. **实时比对** — 当前PF若在 [0.92, 0.93] 区间则无需调整
3. **仿真试算** — 对每相电极尝试 ±1cm / ±2cm / ±4cm 共18个候选动作，用两段模型预测结果
4. **智能择优** — 综合评分：功率因数接近0.925(50%) + 吨电耗低(30%) + 调节幅度小(20%)
5. **安全拦截** — 硬约束检查（详见下方），不通过直接丢弃

### `safety_guard.py` — 安全守卫

**完全独立于AI模型**。每个候选动作必须通过以下检查才能被执行：

| 约束项 | 限制值 | 说明 |
|--------|--------|------|
| 电流上限 | 额定值的 90% (126 kA) | 防止过载 |
| 电极深度 | 0.8 ~ 1.8 m | 避免电弧闪烁 / 防烧损顶炉 |
| 三相不平衡度 | < 5% | 超过会损坏设备 |
| 单次调节幅度 | < 5 cm | 防止突变 |
| 炉压 | 80 ~ 200 Pa | 空气倒灌爆炸 / 密封损坏 |
| 功率因数底线 | ≥ 0.80 | 低于此值紧急停机 |

## 运行结果

### 训练精度

**模型一（驱动层 → 中间状态）** — 3000条数据，2400训练 / 600测试：

| 目标 | R² | MAE | MAPE |
|------|----|-----|------|
| current_a | 0.854 | 2.64 kA | 2.29% |
| current_b | 0.852 | 2.69 kA | 2.32% |
| current_c | 0.847 | 2.72 kA | 2.37% |
| short_net_impedance | 0.244 | 0.081 mΩ | 3.42% |
| imbalance | 0.440 | 0.43% | 34.54% |
| reaction_temp | 0.201 | 24.9 °C | 1.36% |
| furnace_pressure | 0.005 | 11.5 Pa | 9.84% |

三相电流预测精度高（R² > 0.84），炉压基本不可预测（由噪声主导）。

**模型二（中间状态 → 结果指标）：**

| 目标 | R² | MAE | MAPE |
|------|----|-----|------|
| power_factor | 0.799 | 0.0088 | 0.98% |
| energy_per_ton | 0.615 | 30.5 kWh/t | 1.04% |

功率因数预测 MAPE 仅 0.98%，实用精度良好。

### SHAP 特征重要性（功率因数）

Top 10 影响最大的特征，从强到弱：

1. **current_b** — B相电流（最强）
2. **current_c** — C相电流
3. **current_a** — A相电流
4. **imbalance** — 三相不平衡度
5. **electrode_depth_b** — B相电极深度
6. **reaction_temp_lag_3** — 3小时前反应温度
7. **current_a_roll_3_std** — A相电流3小时波动
8. **electrode_depth_c** — C相电极深度
9. **current_c_lag_12** — 12小时前C相电流
10. **reaction_temp_roll_3_std** — 反应温度3小时波动

结论：三相电流是最核心的驱动因素，验证了因果链设计的合理性。

### 闭环决策场景验证

| 场景 | 输入PF | 系统行为 | 结果 |
|------|--------|----------|------|
| A：功率因数偏低 | 0.885 | 推荐调整方案 | 预测PF 0.9106，实际 0.9124，误差 0.0018 |
| B：已在目标区间 | 0.925 | 保持不动 | 跳过调整 |
| C：三相不平衡 | 0.880, 不平衡度4.5% | 推荐方案 + 警告 | 预测PF 0.9099，实际 0.9097，误差 0.0001 |
| D：紧急状态 | 0.780 | 拒绝AI建议，切回人工 | 紧急状态触发 |
| E：过度深插 | 0.890, 电流超标 | 所有候选均被安全拦截 | 转人工处理 |

## 使用方法

### 环境要求

```bash
pip install numpy pandas lightgbm scikit-learn shap
```

### 运行

```bash
python main.py
```

### 训练新模型

```python
from models.two_stage_model import TwoStageModel

model = TwoStageModel()
model.train(df, test_ratio=0.2)
model.save("models/saved/")
```

### 加载模型做推理

```python
model = TwoStageModel()
model.load("models/saved/")
result = model.predict_pipeline(drive_features)
print(result["result"])  # power_factor, energy_per_ton
```

### 使用决策引擎

```python
from engine.decision_engine import DecisionEngine
from engine.safety_guard import SafetyGuard, SafetyLimits

guard = SafetyGuard(SafetyLimits(
    current_rated_ka=140.0,
    current_max_ratio=0.90,
    electrode_depth_min=0.8,
    electrode_depth_max=1.8,
    imbalance_max=5.0,
))
engine = DecisionEngine(model, guard)

result = engine.step(current_state, history_df)
# result["status"]: "recommend" | "hold" | "emergency" | "escalate"
```

## 部署替换清单

真实部署时需替换以下部分：

1. **`data/simulator.py`** → 替换为 DCS / 罗茨线圈实时数据读取
2. **`engine/decision_engine.py`** → 接入 PLC/DCS 下发接口，实现 `auto_execute=True`
3. 增加**定时任务**：每30秒调用 `engine.step()`
4. 增加**Web 界面**：展示建议、历史趋势、预测 vs 实测对比
5. 增加**在线学习**：累积预测误差样本，周期性重训模型
