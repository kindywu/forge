"""
电石炉数据模拟器
用途：在真实数据接入之前，生成用于开发和测试的模拟数据
三层数据：驱动层 → 中间层 → 结果层
"""
import numpy as np
import pandas as pd
from datetime import datetime, timedelta


def generate_furnace_data(n_hours: int = 2000, seed: int = 42) -> pd.DataFrame:
    """
    生成模拟的电石炉三层数据。

    现实逻辑：
    - 电极插入越深 → 电流越大 → 功率因数先升后降（有最优点）
    - 三相不平衡度越大 → 功率因数越低
    - 焦炭固定碳越高 → 反应效率越好 → 吨电耗越低
    - 以上关系都叠加了随机噪声，模拟真实炉况的不确定性

    参数：
        n_hours: 生成多少小时的数据（默认2000小时≈83天）
        seed: 随机种子，保证可复现

    返回：
        DataFrame，每行=1小时，包含三层所有字段
    """
    np.random.seed(seed)

    # 时间索引（每小时一条记录）
    start = datetime(2024, 1, 1)
    timestamps = [start + timedelta(hours=i) for i in range(n_hours)]
    df = pd.DataFrame({"timestamp": timestamps})

    # ── 第一层：驱动因素 ──────────────────────────────────────────
    # 电极插入深度（米），正常范围 0.8~1.8m，三相独立
    df["electrode_depth_a"] = 1.2 + 0.3 * np.sin(np.arange(n_hours) * 0.05) + np.random.normal(0, 0.1, n_hours)
    df["electrode_depth_b"] = 1.2 + 0.3 * np.sin(np.arange(n_hours) * 0.05 + 0.5) + np.random.normal(0, 0.1, n_hours)
    df["electrode_depth_c"] = 1.2 + 0.3 * np.sin(np.arange(n_hours) * 0.05 + 1.0) + np.random.normal(0, 0.1, n_hours)
    df[["electrode_depth_a", "electrode_depth_b", "electrode_depth_c"]] = df[
        ["electrode_depth_a", "electrode_depth_b", "electrode_depth_c"]
    ].clip(0.8, 1.8)

    # 电极升降速度（cm/s），正值=下降，负值=上升
    df["electrode_speed_a"] = np.random.uniform(-0.5, 0.5, n_hours)
    df["electrode_speed_b"] = np.random.uniform(-0.5, 0.5, n_hours)
    df["electrode_speed_c"] = np.random.uniform(-0.5, 0.5, n_hours)

    # 原料配比（C/CaO摩尔比），正常范围 2.9~3.2
    df["c_cao_ratio"] = 3.05 + np.random.normal(0, 0.05, n_hours)
    df["c_cao_ratio"] = df["c_cao_ratio"].clip(2.9, 3.2)

    # 石灰流量（kg/h）
    df["lime_flow"] = 80 + np.random.normal(0, 5, n_hours)
    df["lime_flow"] = df["lime_flow"].clip(60, 100)

    # 焦炭固定碳含量（%）
    df["coke_fixed_carbon"] = 85 + np.random.normal(0, 1, n_hours)
    df["coke_fixed_carbon"] = df["coke_fixed_carbon"].clip(80, 90)

    # ── 第二层：中间状态（由驱动因素决定 + 噪声）──────────────────

    # 平均电极深度（用于整体炉况特征）
    avg_depth = (df["electrode_depth_a"] + df["electrode_depth_b"] + df["electrode_depth_c"]) / 3

    # 三相电流（kA），每相电流主要由该相电极深度决定，叠加耦合效应
    for phase in ["a", "b", "c"]:
        depth_col = f"electrode_depth_{phase}"
        current_col = f"current_{phase}"
        other_phases = [p for p in ["a", "b", "c"] if p != phase]
        other_avg = df[[f"electrode_depth_{p}" for p in other_phases]].mean(axis=1)
        effective_depth = 0.7 * df[depth_col] + 0.3 * other_avg
        df[current_col] = 100 + (effective_depth - 0.8) / (1.8 - 0.8) * 40 + np.random.normal(0, 2, n_hours)
    for col in ["current_a", "current_b", "current_c"]:
        df[col] = df[col].clip(90, 150)

    # 短网阻抗（mΩ），与深度负相关
    df["short_net_impedance"] = 2.5 - 0.3 * (avg_depth - 0.8) / (1.8 - 0.8) + np.random.normal(0, 0.1, n_hours)
    df["short_net_impedance"] = df["short_net_impedance"].clip(1.5, 3.0)

    # 三相不平衡度（%），深度差异越大→不平衡越大
    depth_std = df[["electrode_depth_a", "electrode_depth_b", "electrode_depth_c"]].std(axis=1)
    df["imbalance"] = depth_std * 10 + np.random.exponential(0.5, n_hours)
    df["imbalance"] = df["imbalance"].clip(0, 8)

    # 反应区温度（°C），使用平均电流
    avg_current = (df["current_a"] + df["current_b"] + df["current_c"]) / 3
    df["reaction_temp"] = 1800 + (avg_current - 100) * 2 + np.random.normal(0, 30, n_hours)
    df["reaction_temp"] = df["reaction_temp"].clip(1600, 2000)

    # 炉内气相压力（Pa）
    df["furnace_pressure"] = 120 + np.random.normal(0, 15, n_hours)
    df["furnace_pressure"] = df["furnace_pressure"].clip(80, 200)

    # ── 第三层：结果指标（由中间状态决定 + 噪声）────────────────

    # 功率因数（0~1），最优电流区间约 115~125kA，过高过低都会下降
    avg_current = (df["current_a"] + df["current_b"] + df["current_c"]) / 3
    # 倒U型曲线：在optimal_current附近最高
    optimal_current = 120
    pf_base = 0.93 - ((avg_current - optimal_current) / 20) ** 2 * 0.1
    # 不平衡度越大→功率因数越低
    pf_base -= df["imbalance"] * 0.005
    # 加噪声
    df["power_factor"] = pf_base + np.random.normal(0, 0.01, n_hours)
    df["power_factor"] = df["power_factor"].clip(0.80, 0.98)

    # 吨电耗（kWh/t），功率因数越高→电耗越低
    # 理论下限2017，实际在2800~3200
    df["energy_per_ton"] = 3200 - (df["power_factor"] - 0.80) / 0.18 * 400
    # 焦炭品质越好→电耗越低
    df["energy_per_ton"] -= (df["coke_fixed_carbon"] - 80) * 10
    df["energy_per_ton"] += np.random.normal(0, 30, n_hours)
    df["energy_per_ton"] = df["energy_per_ton"].clip(2800, 3300)

    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


if __name__ == "__main__":
    df = generate_furnace_data(n_hours=2000)
    df.to_csv("furnace_data.csv", index=False)
    print(f"生成数据：{len(df)} 条记录，{len(df.columns)} 个字段")
    print(f"功率因数范围：{df['power_factor'].min():.3f} ~ {df['power_factor'].max():.3f}")
    print(f"吨电耗范围：{df['energy_per_ton'].min():.0f} ~ {df['energy_per_ton'].max():.0f} kWh/t")
    print(df.describe().round(2))
