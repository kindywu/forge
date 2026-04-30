"""
特征工程模块

核心思路（来自电价预测文章的启发）：
  - 最重要的特征是"近期历史信号"：滚动均值、滞后值
  - 时间特征捕捉班次/周期规律
  - 所有特征分为四类：驱动类、电气类、时间类、滞后/滚动类
"""
import pandas as pd
import numpy as np
from typing import List, Tuple


# ── 配置：哪些列属于哪一层 ─────────────────────────────────────────

DRIVE_COLS = [
    "electrode_depth_a", "electrode_depth_b", "electrode_depth_c",
    "electrode_speed_a", "electrode_speed_b", "electrode_speed_c",
    "c_cao_ratio", "lime_flow", "coke_fixed_carbon",
]

INTERMEDIATE_COLS = [
    "current_a", "current_b", "current_c",
    "short_net_impedance", "imbalance",
    "reaction_temp", "furnace_pressure",
]

RESULT_COLS = ["power_factor", "energy_per_ton"]

# 目标值区间
PF_TARGET_LOW = 0.92
PF_TARGET_HIGH = 0.93


def build_features_for_model1(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    模型一的特征和标签：
      输入X = 驱动层特征（电极操作 + 原料）—— 仅原始值，不加滞后/滚动
      输出Y = 中间状态（电气参数 + 炉况）

    注意：模型一代表瞬时因果链（驱动 → 中间状态），不应加入驱动列
    的滞后/滚动特征，否则模型会依赖历史模式而忽略当前驱动值的因果效应，
    导致反事实仿真（如"改变电极深度"）时预测结果几乎不变。

    返回 (X, Y)
    """
    df = df.copy().sort_values("timestamp").reset_index(drop=True)
    df = _add_time_features(df)

    # X：驱动层原始值 + 时间特征（不加滞后/滚动，保持因果性）
    feature_cols = [c for c in df.columns if c not in INTERMEDIATE_COLS + RESULT_COLS + ["timestamp"]]
    X = df[feature_cols].dropna()
    Y = df.loc[X.index, INTERMEDIATE_COLS]

    return X, Y


def build_features_for_model2(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    模型二的特征和标签：
      输入X = 中间状态（含近期历史）+ 驱动层原始值
      输出Y = 结果指标（功率因数 + 吨电耗）

    重点：这里加入了"近期功率因数的滞后/滚动"，
         因为文章SHAP分析显示近期价格信号是最重要的特征，
         功率因数同理。
    """
    df = df.copy().sort_values("timestamp").reset_index(drop=True)

    # 对中间状态列加滞后/滚动（窗口从3开始，避免window=1时std全NaN）
    df = _add_lag_rolling_features(df, cols=INTERMEDIATE_COLS, windows=[3, 6, 12])

    # 对结果列加历史滞后（用历史功率因数预测未来功率因数，窗口从3开始）
    df = _add_lag_rolling_features(df, cols=RESULT_COLS, windows=[3, 6, 12])
    # 单独加lag_1（只有滞后值，没有窗口std问题）
    for col in RESULT_COLS:
        df[f"{col}_lag_1"] = df[col].shift(1)

    df = _add_time_features(df)

    # X：排除当前时刻的结果值（只保留历史滞后/滚动版本）
    raw_result_cols = [c for c in df.columns if c in RESULT_COLS]
    feature_cols = [c for c in df.columns if c not in raw_result_cols + ["timestamp"]]
    X = df[feature_cols].dropna()
    Y = df.loc[X.index, RESULT_COLS]

    return X, Y


def build_inference_features(
    current_row: pd.Series,
    history_df: pd.DataFrame,
    mode: str = "model1"
) -> pd.DataFrame:
    """
    用于推理（预测）时构造单行特征。

    参数：
        current_row: 当前时刻的原始数据（一行）
        history_df: 过去若干小时的历史数据（用于计算滞后/滚动）
        mode: "model1" 或 "model2"

    返回：
        构造好特征的单行 DataFrame
    """
    # 把当前行拼到历史尾部，然后取最后一行的特征
    combined = pd.concat([history_df, current_row.to_frame().T], ignore_index=True)
    # 确保数值列保持数值类型（concat 可能把 float 列变成 object）
    for col in combined.columns:
        if col != "timestamp":
            combined[col] = pd.to_numeric(combined[col], errors="coerce")
    combined["timestamp"] = pd.to_datetime(
        combined["timestamp"] if "timestamp" in combined.columns
        else pd.date_range(end=pd.Timestamp.now(), periods=len(combined), freq="h")
    )

    if mode == "model1":
        X, _ = build_features_for_model1(combined)
    else:
        X, _ = build_features_for_model2(combined)

    # 返回最后一行
    return X.iloc[[-1]]


# ── 内部工具函数 ──────────────────────────────────────────────────

def _add_lag_rolling_features(
    df: pd.DataFrame,
    cols: List[str],
    windows: List[int]
) -> pd.DataFrame:
    """
    为指定列添加滞后值和滚动统计特征。

    例子：对 power_factor 列，windows=[3,6,12]，会生成：
      power_factor_lag_1    → 1小时前的值
      power_factor_lag_3    → 3小时前的值
      power_factor_roll_3_mean  → 过去3小时均值
      power_factor_roll_6_mean  → 过去6小时均值
      power_factor_roll_6_std   → 过去6小时标准差
    """
    for col in cols:
        if col not in df.columns:
            continue
        # 滞后1期（最重要）
        df[f"{col}_lag_1"] = df[col].shift(1)
        # 滞后更多期
        for w in windows:
            df[f"{col}_lag_{w}"] = df[col].shift(w)
            df[f"{col}_roll_{w}_mean"] = df[col].shift(1).rolling(w).mean()
            df[f"{col}_roll_{w}_std"] = df[col].shift(1).rolling(w).std()
    return df


def _add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    添加时间周期特征（捕捉班次、日夜规律）。

    捕捉的规律：
      - 白班/夜班：不同班次操作习惯不同
      - 小时：一天内的周期（和电价文章里"hour"是重要特征一样）
      - 星期几：周末/工作日
    """
    if "timestamp" not in df.columns:
        return df
    ts = pd.to_datetime(df["timestamp"])
    df["hour"] = ts.dt.hour
    df["day_of_week"] = ts.dt.dayofweek
    df["is_night_shift"] = ((ts.dt.hour >= 20) | (ts.dt.hour < 8)).astype(int)
    # 用sin/cos编码小时，让0点和23点"在数学上相邻"
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    return df


def get_feature_names(df: pd.DataFrame, mode: str = "model1") -> List[str]:
    """返回指定模式下的特征列名列表（用于SHAP分析等）"""
    if mode == "model1":
        X, _ = build_features_for_model1(df)
    else:
        X, _ = build_features_for_model2(df)
    return list(X.columns)
