"""
安全约束检查模块

核心原则：这个模块完全独立于AI模型。
无论AI给出什么建议，都必须先过这里的"安检门"。
任何一项不过，建议直接拒绝，交给人工。
"""
from dataclasses import dataclass, field
from typing import List, Tuple, Optional


@dataclass
class SafetyLimits:
    """
    硬约束配置。

    现场调试时可根据实际设备参数修改这里的数值。
    所有数值都有注释说明含义和建议范围。
    """
    # 电流（kA）
    current_max_ratio: float = 0.90   # 不能超过额定电流的90%
    current_rated_ka: float = 140.0   # 额定电流（kA）

    # 电极插入深度（米）
    electrode_depth_min: float = 0.8  # 最小插入深度，防止电弧太短引起闪烁
    electrode_depth_max: float = 1.8  # 最大插入深度，防止烧损/顶炉

    # 三相不平衡度（%）
    imbalance_max: float = 5.0        # 超过5%会损坏设备

    # 炉内气相压力（Pa）
    pressure_min: float = 80.0        # 太低→空气倒灌，有爆炸风险
    pressure_max: float = 200.0       # 太高→密封损坏

    # 功率因数安全底线（低于此值强制降负荷）
    power_factor_emergency: float = 0.80

    # 单次操作最大幅度（防止突变）
    electrode_max_step_cm: float = 5.0  # 单次最多调5cm


@dataclass
class SafetyCheckResult:
    """安全检查结果"""
    passed: bool
    violations: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def __str__(self):
        if self.passed:
            status = "✓ 通过"
            if self.warnings:
                status += f"（{len(self.warnings)}条警告）"
        else:
            status = f"✗ 拒绝（{len(self.violations)}项违规）"
        return status


class SafetyGuard:
    """
    安全守卫。

    用法：
        guard = SafetyGuard()
        result = guard.check(candidate_action, current_state)
        if result.passed:
            execute(candidate_action)
        else:
            escalate_to_human(result.violations)
    """

    def __init__(self, limits: Optional[SafetyLimits] = None):
        self.limits = limits or SafetyLimits()

    def check(
        self,
        action: dict,
        current_state: dict,
        predicted_state: Optional[dict] = None
    ) -> SafetyCheckResult:
        """
        对一个候选动作做完整安全检查。

        参数：
            action: 候选动作，例如：
                {
                    "electrode_depth_a": 1.4,  # 调整后的深度
                    "electrode_depth_b": 1.2,
                    "electrode_depth_c": 1.2,
                }
            current_state: 当前实测状态（炉况数据）
            predicted_state: 模型预测的执行后状态（可选）

        返回：
            SafetyCheckResult
        """
        violations = []
        warnings = []

        # 1. 检查电极深度范围
        for phase in ["a", "b", "c"]:
            key = f"electrode_depth_{phase}"
            if key in action:
                depth = action[key]
                if depth < self.limits.electrode_depth_min:
                    violations.append(
                        f"电极{phase.upper()}深度 {depth:.2f}m < 安全下限 {self.limits.electrode_depth_min}m"
                    )
                if depth > self.limits.electrode_depth_max:
                    violations.append(
                        f"电极{phase.upper()}深度 {depth:.2f}m > 安全上限 {self.limits.electrode_depth_max}m"
                    )

        # 2. 检查单次调节幅度（防止突变），加小epsilon避免浮点精度问题
        for phase in ["a", "b", "c"]:
            depth_key = f"electrode_depth_{phase}"
            if depth_key in action and depth_key in current_state:
                step = abs(action[depth_key] - current_state[depth_key]) * 100  # 转cm
                if step > self.limits.electrode_max_step_cm + 1e-9:
                    violations.append(
                        f"电极{phase.upper()}单次调节幅度 {step:.1f}cm > 限制 {self.limits.electrode_max_step_cm}cm"
                    )

        # 3. 检查预测后的电流（如果有预测结果）
        if predicted_state:
            current_limit = self.limits.current_rated_ka * self.limits.current_max_ratio
            for phase in ["a", "b", "c"]:
                current_key = f"current_{phase}"
                if current_key in predicted_state:
                    c = predicted_state[current_key]
                    if c > current_limit:
                        violations.append(
                            f"预测电流{phase.upper()}: {c:.1f}kA 超过限制 {current_limit:.1f}kA"
                        )
                    elif c > current_limit * 0.95:
                        warnings.append(f"预测电流{phase.upper()}: {c:.1f}kA 接近上限，注意观察")

            # 4. 检查三相不平衡度
            if "imbalance" in predicted_state:
                imb = predicted_state["imbalance"]
                if imb > self.limits.imbalance_max:
                    violations.append(
                        f"预测三相不平衡度 {imb:.1f}% > 限制 {self.limits.imbalance_max}%"
                    )
                elif imb > self.limits.imbalance_max * 0.8:
                    warnings.append(f"预测三相不平衡度 {imb:.1f}% 偏高，建议关注")

        # 5. 检查当前炉压是否在安全区间
        if "furnace_pressure" in current_state:
            p = current_state["furnace_pressure"]
            if p < self.limits.pressure_min or p > self.limits.pressure_max:
                violations.append(
                    f"当前炉压 {p:.0f}Pa 超出安全区间 [{self.limits.pressure_min}, {self.limits.pressure_max}]Pa"
                )

        return SafetyCheckResult(
            passed=len(violations) == 0,
            violations=violations,
            warnings=warnings,
        )

    def check_emergency(self, current_state: dict) -> Tuple[bool, str]:
        """
        紧急状态检查（独立于候选动作，每次循环都跑）。

        返回：(is_emergency, reason)
        """
        if "power_factor" in current_state:
            pf = current_state["power_factor"]
            if pf < self.limits.power_factor_emergency:
                return True, f"功率因数 {pf:.3f} 低于紧急底线 {self.limits.power_factor_emergency}"

        if "furnace_pressure" in current_state:
            p = current_state["furnace_pressure"]
            if p > self.limits.pressure_max * 1.2:
                return True, f"炉压 {p:.0f}Pa 严重超限！"

        return False, ""
