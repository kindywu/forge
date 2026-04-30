"""
电石炉智能决策系统 - 主程序

运行这个文件会：
1. 生成模拟数据（真实场景替换为读取DCS/罗茨线圈数据）
2. 训练两段模型
3. 进行SHAP特征重要性分析
4. 模拟5轮闭环决策
5. 保存模型
"""
import sys
import os
import io
sys.path.insert(0, os.path.dirname(__file__))

# Fix GBK encoding on Windows terminals
if hasattr(sys.stdout, 'buffer'):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

from data.simulator import generate_furnace_data
from models.two_stage_model import TwoStageModel
from engine.safety_guard import SafetyGuard, SafetyLimits
from engine.decision_engine import DecisionEngine


def main():
    print("=" * 60)
    print("  电石炉电极操作智能决策系统")
    print("  一条因果链 · 两段模型 · 一层安全防护")
    print("=" * 60)

    # ── Step 1: 准备数据 ──────────────────────────────────────────
    print("\n【Step 1】生成模拟数据...")
    print("  （真实部署时替换为：df = pd.read_csv('dcs_data.csv')）")
    df = generate_furnace_data(n_hours=3000)
    print(f"  数据规模：{len(df)} 条记录，时间跨度约 {len(df)//24} 天")
    print(f"  字段数：{len(df.columns)} 个（驱动层 + 中间层 + 结果层）")

    # 快速查看数据分布
    print(f"\n  功率因数统计：")
    print(f"    均值={df['power_factor'].mean():.3f}  "
          f"最小={df['power_factor'].min():.3f}  "
          f"最大={df['power_factor'].max():.3f}")
    print(f"  吨电耗统计：")
    print(f"    均值={df['energy_per_ton'].mean():.0f}  "
          f"最小={df['energy_per_ton'].min():.0f}  "
          f"最大={df['energy_per_ton'].max():.0f} kWh/t")

    # 打印训练数据样例（参考 demo.py 的结构）
    pd.set_option('display.max_columns', 25)
    pd.set_option('display.width', 300)
    pd.set_option('display.float_format', '{:.4f}'.format)
    pd.set_option('display.max_colwidth', 12)

    print("\n===== 原始数据前 5 行（三层结构：驱动层 + 中间状态 + 结果指标）=====")
    print(df.head(5))

    from utils.feature_engineering import build_features_for_model1, build_features_for_model2
    X1, Y1 = build_features_for_model1(df)
    X2, Y2 = build_features_for_model2(df)

    print(f"\n===== 模型一训练输入 X1: {X1.shape}（特征）=====")
    print(X1.head(5))
    print(f"\n===== 模型一训练标签 Y1: {Y1.shape}（中间状态）=====")
    print(Y1.head(5))

    print(f"\n===== 模型二训练输入 X2: {X2.shape}（特征）=====")
    print(X2.head(5))
    print(f"\n===== 模型二训练标签 Y2: {Y2.shape}（结果指标）=====")
    print(Y2.head(5))

    # ── Step 2: 训练两段模型 ──────────────────────────────────────
    print("\n【Step 2】训练两段机器学习模型...")
    model = TwoStageModel()
    eval_results = model.train(df, test_ratio=0.2)

    # ── Step 3: SHAP 可解释性分析 ─────────────────────────────────
    print("\n【Step 3】SHAP 特征重要性分析（功率因数预测）")
    print("  （这告诉你：哪些操作/状态对功率因数影响最大）")
    try:
        importance = model.get_shap_importance("power_factor", top_n=10)
        if not importance.empty:
            print("\n  Top 10 最重要特征：")
            for i, row in importance.iterrows():
                bar = "█" * int(row["shap_importance"] / importance["shap_importance"].max() * 20)
                print(f"  {bar:<20} {row['feature']}")
        else:
            print("  （SHAP分析跳过，使用内置特征重要性）")
            # 用LightGBM自带的feature importance作为备选
            if "power_factor" in model.model2:
                m = model.model2["power_factor"]
                imp = pd.Series(
                    m.feature_importances_,
                    index=model.model2_feature_cols
                ).sort_values(ascending=False).head(10)
                print("\n  Top 10 特征（LightGBM内置重要性）：")
                for feat, score in imp.items():
                    bar = "█" * int(score / imp.max() * 20)
                    print(f"  {bar:<20} {feat}")
    except Exception as e:
        print(f"  SHAP分析跳过：{e}")

    # ── Step 4: 保存模型 ──────────────────────────────────────────
    print("\n【Step 4】保存模型...")
    model.save("models/saved/")

    # ── Step 5: 模拟闭环决策 ──────────────────────────────────────
    print("\n【Step 5】模拟闭环决策（5轮）")
    print("-" * 60)

    guard = SafetyGuard(SafetyLimits(
        current_rated_ka=140.0,
        current_max_ratio=0.95,         # 演示用：允许到 133kA
        electrode_depth_min=0.8,
        electrode_depth_max=1.8,
        imbalance_max=5.0,
        electrode_max_step_cm=12.0,     # 演示用：允许单次最多 12cm
    ))
    engine = DecisionEngine(model, guard)

    # 用数据集的最后100条作为历史上下文
    history_df = df.iloc[-200:-100].copy()

    # 模拟几个典型炉况场景
    test_scenarios = [
        {
            "name": "场景A：功率因数偏低，需要调整",
            "state": {
                "power_factor": 0.885,          # 低于目标0.92
                "energy_per_ton": 3050,
                "electrode_depth_a": 1.1,
                "electrode_depth_b": 1.2,
                "electrode_depth_c": 1.15,
                "current_a": 118, "current_b": 120, "current_c": 119,
                "short_net_impedance": 2.3,
                "imbalance": 1.2,
                "reaction_temp": 1820,
                "furnace_pressure": 125,
                "c_cao_ratio": 3.05,
                "lime_flow": 80,
                "coke_fixed_carbon": 84,
            }
        },
        {
            "name": "场景B：功率因数已在目标区间",
            "state": {
                "power_factor": 0.925,          # 在目标区间内，无需调整
                "energy_per_ton": 2950,
                "electrode_depth_a": 1.3,
                "electrode_depth_b": 1.3,
                "electrode_depth_c": 1.3,
                "current_a": 122, "current_b": 121, "current_c": 122,
                "short_net_impedance": 2.1,
                "imbalance": 0.8,
                "reaction_temp": 1860,
                "furnace_pressure": 118,
                "c_cao_ratio": 3.05,
                "lime_flow": 80,
                "coke_fixed_carbon": 85,
            }
        },
        {
            "name": "场景C：三相不平衡，触发安全限制",
            "state": {
                "power_factor": 0.880,
                "energy_per_ton": 3100,
                "electrode_depth_a": 0.85,      # A相很浅
                "electrode_depth_b": 1.5,        # B相很深
                "electrode_depth_c": 1.2,
                "current_a": 105, "current_b": 138, "current_c": 120,
                "short_net_impedance": 2.0,
                "imbalance": 4.5,               # 不平衡度已经很高
                "reaction_temp": 1780,
                "furnace_pressure": 130,
                "c_cao_ratio": 3.0,
                "lime_flow": 78,
                "coke_fixed_carbon": 83,
            }
        },
        {
            "name": "场景D：🚨 紧急状态（功率因数低于底线）",
            "state": {
                "power_factor": 0.78,           # 低于紧急底线0.80
                "energy_per_ton": 3200,
                "electrode_depth_a": 0.9,
                "electrode_depth_b": 0.9,
                "electrode_depth_c": 0.9,
                "current_a": 105, "current_b": 104, "current_c": 106,
                "short_net_impedance": 2.7,
                "imbalance": 2.0,
                "reaction_temp": 1700,
                "furnace_pressure": 115,
                "c_cao_ratio": 2.95,
                "lime_flow": 75,
                "coke_fixed_carbon": 82,
            }
        },
        {
            "name": "场景E：功率因数偏高（过度深插）",
            "state": {
                "power_factor": 0.89,
                "energy_per_ton": 3000,
                "electrode_depth_a": 1.7,       # 太深了
                "electrode_depth_b": 1.65,
                "electrode_depth_c": 1.6,
                "current_a": 138, "current_b": 135, "current_c": 133,
                "short_net_impedance": 1.7,
                "imbalance": 2.5,
                "reaction_temp": 1950,
                "furnace_pressure": 140,
                "c_cao_ratio": 3.1,
                "lime_flow": 82,
                "coke_fixed_carbon": 86,
            }
        },
    ]

    for i, scenario in enumerate(test_scenarios):
        print(f"\n{'─'*50}")
        print(f"  {scenario['name']}")
        print(f"{'─'*50}")
        result = engine.step(
            current_state=scenario["state"],
            history_df=history_df,
            auto_execute=False
        )

        # 模拟回验（真实场景中等待下一个采样周期的实测值）
        if result["status"] == "recommend" and result.get("best"):
            simulated_actual_pf = result["best"].predicted_pf + np.random.normal(0, 0.005)
            print(f"\n  [回验] 执行后实测功率因数: {simulated_actual_pf:.4f}")
            engine.record_actual_result(result["log"], simulated_actual_pf)

    # ── Step 6: 输出决策日志 ──────────────────────────────────────
    print("\n\n【Step 6】决策日志汇总")
    print("=" * 60)
    log_df = engine.get_log_df()
    print(log_df.to_string(index=False))

    print("\n\n✅ 演示完成！")
    print("\n真实部署时的替换点：")
    print("  1. data/simulator.py → 替换为实际DCS/罗茨线圈数据读取")
    print("  2. engine/decision_engine.py → 接入真实的PLC/DCS下发接口")
    print("  3. 增加一个定时任务（每30秒调用 engine.step()）")
    print("  4. 增加Web界面展示建议和历史趋势")


if __name__ == "__main__":
    main()
