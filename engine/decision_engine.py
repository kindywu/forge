"""
闭环决策引擎

每30秒执行一次五步循环：
  ① 实时比对    → 当前功率因数 vs 目标区间
  ② 仿真试算    → 枚举候选动作，用两段模型预测结果
  ③ 智能择优    → 综合评分选最优方案
  ④ 安全拦截    → 硬约束检查，不通过直接丢弃
  ⑤ 执行+回验   → 下发建议，记录预测vs实测偏差
"""
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
import time
from datetime import datetime

from engine.safety_guard import SafetyGuard, SafetyCheckResult
from utils.feature_engineering import INTERMEDIATE_COLS


# 目标功率因数区间
PF_TARGET_LOW = 0.92
PF_TARGET_HIGH = 0.93


@dataclass
class ActionCandidate:
    """一个候选动作方案"""
    action: Dict                  # 建议的操作（电极深度调整量）
    predicted_intermediate: Dict  # 预测的中间状态
    predicted_pf: float           # 预测功率因数
    predicted_energy: float       # 预测吨电耗
    score: float = 0.0            # 综合评分（越高越好）
    safety_result: Optional[SafetyCheckResult] = None

    def describe(self) -> str:
        """生成人类可读的建议描述"""
        lines = ["建议动作："]
        for k, v in self.action.items():
            if "depth" in k:
                phase = k.split("_")[-1].upper()
                lines.append(f"  电极{phase}调整至 {v:.2f}m")
        lines.append(f"预测功率因数：{self.predicted_pf:.4f}")
        lines.append(f"预测吨电耗：{self.predicted_energy:.0f} kWh/t")
        lines.append(f"综合评分：{self.score:.3f}")
        if self.safety_result and self.safety_result.warnings:
            lines.append(f"⚠ 警告：{'; '.join(self.safety_result.warnings)}")
        return "\n".join(lines)


@dataclass
class DecisionRecord:
    """每次决策的完整记录（用于在线学习反馈）"""
    timestamp: str
    current_state: Dict
    recommended_action: Optional[Dict]
    predicted_pf: float
    actual_pf: Optional[float] = None
    prediction_error: Optional[float] = None
    was_executed: bool = False
    skip_reason: str = ""


class DecisionEngine:
    """
    闭环决策引擎。

    用法：
        engine = DecisionEngine(model, safety_guard)
        recommendation = engine.step(current_state, history_df)
    """

    def __init__(self, model, safety_guard: Optional[SafetyGuard] = None):
        """
        参数：
            model: TwoStageModel 实例
            safety_guard: SafetyGuard 实例（不传则用默认配置）
        """
        self.model = model
        self.safety_guard = safety_guard or SafetyGuard()
        self.decision_log: List[DecisionRecord] = []

    def step(
        self,
        current_state: dict,
        history_df: pd.DataFrame,
        auto_execute: bool = False
    ) -> Dict:
        """
        执行一次完整的五步决策循环。

        参数：
            current_state: 当前时刻的实测数据（字典）
            history_df: 过去若干小时的历史数据（用于构造特征）
            auto_execute: True=自动执行，False=等待人工确认

        返回：
            {
                "status": "recommend" / "hold" / "emergency",
                "candidates": 排序后的候选动作列表,
                "best": 最优候选（或None）,
                "log": DecisionRecord
            }
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # ── 步骤①：紧急状态检查 ──────────────────────────────────
        is_emergency, reason = self.safety_guard.check_emergency(current_state)
        if is_emergency:
            print(f"[{timestamp}] 🚨 紧急状态：{reason}，切回人工！")
            record = DecisionRecord(
                timestamp=timestamp, current_state=current_state,
                recommended_action=None, predicted_pf=current_state.get("power_factor", 0),
                skip_reason=f"紧急状态：{reason}"
            )
            self.decision_log.append(record)
            return {"status": "emergency", "reason": reason, "log": record}

        # ── 步骤①：比对当前功率因数 vs 目标 ─────────────────────
        current_pf = current_state.get("power_factor", 0)
        if PF_TARGET_LOW <= current_pf <= PF_TARGET_HIGH:
            print(f"[{timestamp}] ✓ 功率因数 {current_pf:.4f} 在目标区间，无需调整")
            record = DecisionRecord(
                timestamp=timestamp, current_state=current_state,
                recommended_action=None, predicted_pf=current_pf,
                skip_reason="已在目标区间"
            )
            self.decision_log.append(record)
            return {"status": "hold", "log": record}

        deviation = current_pf - (PF_TARGET_LOW + PF_TARGET_HIGH) / 2
        print(f"[{timestamp}] 当前功率因数 {current_pf:.4f}，目标 {PF_TARGET_LOW}~{PF_TARGET_HIGH}，偏差 {deviation:+.4f}")

        # ── 步骤②：仿真试算候选动作 ─────────────────────────────
        candidates = self._simulate_candidates(current_state, history_df)
        print(f"  仿真了 {len(candidates)} 个候选方案")

        # ── 步骤③：智能择优 ─────────────────────────────────────
        candidates = self._score_and_rank(candidates)

        # ── 步骤④：安全拦截 ─────────────────────────────────────
        safe_candidates = []
        rejection_reasons: Dict[str, int] = {}
        for c in candidates:
            c.safety_result = self.safety_guard.check(
                action=c.action,
                current_state=current_state,
                predicted_state=c.predicted_intermediate
            )
            if c.safety_result.passed:
                safe_candidates.append(c)
            else:
                reason = c.safety_result.violations[0]
                rejection_reasons[reason] = rejection_reasons.get(reason, 0) + 1

        if rejection_reasons:
            for reason, count in rejection_reasons.items():
                print(f"  [安全拦截] {count} 个候选方案被拒绝：{reason}")

        if not safe_candidates:
            print(f"  ⚠ 所有候选方案均未通过安全检查，转人工")
            record = DecisionRecord(
                timestamp=timestamp, current_state=current_state,
                recommended_action=None, predicted_pf=current_pf,
                skip_reason="所有候选均未通过安全检查"
            )
            self.decision_log.append(record)
            return {"status": "escalate", "candidates": candidates, "log": record}

        best = safe_candidates[0]

        # ── 步骤⑤：输出建议 ─────────────────────────────────────
        print(f"\n  【推荐方案】\n{best.describe()}")

        record = DecisionRecord(
            timestamp=timestamp, current_state=current_state,
            recommended_action=best.action,
            predicted_pf=best.predicted_pf,
            was_executed=auto_execute
        )
        self.decision_log.append(record)

        return {
            "status": "recommend",
            "candidates": safe_candidates[:3],  # 返回前3个
            "best": best,
            "log": record,
        }

    def record_actual_result(self, record: DecisionRecord, actual_pf: float):
        """
        步骤⑤的回验：记录实际执行后的真实功率因数，计算预测误差。
        这个误差数据会作为在线学习的反馈样本。

        参数：
            record: 之前的决策记录
            actual_pf: 执行建议后实测到的功率因数
        """
        record.actual_pf = actual_pf
        record.prediction_error = abs(record.predicted_pf - actual_pf)

        status = "✓" if record.prediction_error < 0.03 else "⚠"
        print(f"  {status} 预测 {record.predicted_pf:.4f} vs 实测 {actual_pf:.4f}，误差 {record.prediction_error:.4f}")

    def get_log_df(self) -> pd.DataFrame:
        """返回所有决策记录，可用于分析和在线学习"""
        return pd.DataFrame([{
            "timestamp": r.timestamp,
            "current_pf": r.current_state.get("power_factor"),
            "recommended": r.recommended_action is not None,
            "predicted_pf": r.predicted_pf,
            "actual_pf": r.actual_pf,
            "prediction_error": r.prediction_error,
            "was_executed": r.was_executed,
            "skip_reason": r.skip_reason,
        } for r in self.decision_log])

    # ── 内部方法 ───────────────────────────────────────────────────

    def _simulate_candidates(
        self, current_state: dict, history_df: pd.DataFrame
    ) -> List[ActionCandidate]:
        """
        枚举候选动作并用模型预测结果。

        候选动作的生成策略：
          - 对每个电极，在当前深度基础上尝试 ±1cm, ±2cm, ±4cm
          - 三相组合太多，先只调最有影响力的那相（电流最高的那相）
        """
        candidates = []

        # 找出当前电流最高的相（优先调它）
        currents = {
            "a": current_state.get("current_a", 120),
            "b": current_state.get("current_b", 120),
            "c": current_state.get("current_c", 120),
        }
        # 使用完整列名作为键
        current_depths = {
            "electrode_depth_a": current_state.get("electrode_depth_a", 1.2),
            "electrode_depth_b": current_state.get("electrode_depth_b", 1.2),
            "electrode_depth_c": current_state.get("electrode_depth_c", 1.2),
        }

        # 尝试调节幅度（cm → m）
        steps_cm = [-4, -2, -1, 0, 1, 2, 4]
        steps_m = [s / 100 for s in steps_cm]

        for phase in ["electrode_depth_a", "electrode_depth_b", "electrode_depth_c"]:
            for step in steps_m:
                if step == 0:
                    continue  # 不动就不试了

                new_depth = current_depths[phase] + step
                if new_depth < 0.8 or new_depth > 1.8:
                    continue  # 超出物理范围直接跳过

                action = dict(current_depths)  # 从当前深度出发
                action[phase] = new_depth

                # 用模型预测这个动作的结果
                pred = self._predict_action(action, current_state, history_df)
                if pred is None:
                    continue

                candidates.append(ActionCandidate(
                    action=action,
                    predicted_intermediate=pred["intermediate"],
                    predicted_pf=pred["power_factor"],
                    predicted_energy=pred["energy_per_ton"],
                ))

        return candidates

    def _predict_action(
        self, action: dict, current_state: dict, history_df: pd.DataFrame
    ) -> Optional[dict]:
        """
        给定一个动作（电极深度配置），用两段模型预测结果。
        使用完整的 build_inference_features() 构造特征，确保滞后/滚动特征正确。
        """
        try:
            from utils.feature_engineering import build_inference_features, DRIVE_COLS
            from datetime import timedelta

            # 构造候选行：用当前状态做基础，覆盖电极深度，历史值补齐缺失列
            candidate = dict(current_state)
            for k, v in action.items():
                candidate[k] = v
            # 补全 current_state 中缺失的驱动列（从历史最后一行取）
            for col in DRIVE_COLS:
                if col not in candidate or candidate[col] is None:
                    if col in history_df.columns:
                        candidate[col] = history_df[col].iloc[-1]
                    else:
                        candidate[col] = 0.0
            candidate_row = pd.Series(candidate)

            # 给候选行一个合理的时间戳（历史最后一行 + 1 小时）
            last_ts = pd.to_datetime(history_df["timestamp"].iloc[-1])
            candidate_row["timestamp"] = last_ts + timedelta(hours=1)

            # 模型一：驱动特征 → 中间状态
            X1 = build_inference_features(candidate_row, history_df, mode="model1")
            intermediate_pred = self.model.predict_intermediate(X1)

            # 用模型一的预测结果更新候选行，再送入模型二
            for col in intermediate_pred.columns:
                candidate_row[col] = float(intermediate_pred[col].iloc[0])

            # 模型二：驱动 + 中间状态 → 结果指标
            X2 = build_inference_features(candidate_row, history_df, mode="model2")
            result_pred = self.model.predict_result(X2)

            return {
                "power_factor": float(result_pred["power_factor"].iloc[0]),
                "energy_per_ton": float(result_pred["energy_per_ton"].iloc[0]),
                "intermediate": intermediate_pred.iloc[0].to_dict(),
            }
        except Exception as e:
            import traceback
            traceback.print_exc()
            return None

    def _build_simple_drive_features(self, state: dict, history_df: pd.DataFrame) -> pd.DataFrame:
        """
        简化版特征构造（用于推理循环中）。
        只取模型需要的特征列，用0填充缺失的滞后/滚动特征。
        真实部署时替换为完整的 build_inference_features()。
        """
        row = {}
        for col in self.model.model1_feature_cols:
            if col in state:
                row[col] = state[col]
            else:
                # 从历史数据中提取滞后/滚动值
                base_col = col.split("_lag_")[0].split("_roll_")[0]
                if base_col in history_df.columns and len(history_df) > 0:
                    row[col] = history_df[base_col].iloc[-1]
                else:
                    row[col] = 0.0
        return pd.DataFrame([row])

    def _score_and_rank(self, candidates: List[ActionCandidate]) -> List[ActionCandidate]:
        """
        对候选方案综合评分：
          - 功率因数越接近目标中心（0.925）越好（权重50%）
          - 吨电耗越低越好（权重30%）
          - 调节幅度越小越好，减少频繁操作（权重20%）
        """
        if not candidates:
            return candidates

        pf_target = (PF_TARGET_LOW + PF_TARGET_HIGH) / 2  # 0.925

        pfs = np.array([c.predicted_pf for c in candidates])
        energies = np.array([c.predicted_energy for c in candidates])

        # 归一化到0~1
        pf_score = 1 - np.abs(pfs - pf_target) / 0.1  # 偏差0.1以内都给分
        pf_score = np.clip(pf_score, 0, 1)

        energy_score = 1 - (energies - energies.min()) / (energies.max() - energies.min() + 1e-6)

        for i, c in enumerate(candidates):
            c.score = 0.5 * pf_score[i] + 0.3 * energy_score[i] + 0.2 * 0.5  # 简化幅度评分

        return sorted(candidates, key=lambda x: x.score, reverse=True)
