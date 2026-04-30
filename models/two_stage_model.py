"""
两段机器学习模型

模型一：驱动层 → 中间状态（LightGBM多输出回归）
模型二：中间状态 → 结果指标（LightGBM多输出回归）

每个"多输出"模型实际上是对每个目标列分别训练一个LightGBM，
这样可以针对每个输出调参，也方便单独评估精度。
"""

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import (
    mean_absolute_percentage_error,
    mean_absolute_error,
    r2_score,
)
import pickle
from pathlib import Path
from typing import Dict, List, Optional
import warnings

warnings.filterwarnings("ignore")

from utils.feature_engineering import (
    build_features_for_model1,
    build_features_for_model2,
)

# LightGBM 超参数（工控机友好：树不太大，训练快，内存小）
LGBM_PARAMS = {
    "objective": "regression",
    "metric": "mae",
    "num_leaves": 31,
    "learning_rate": 0.05,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "verbose": -1,
    "n_estimators": 500,
    "early_stopping_rounds": 50,
}


class TwoStageModel:
    """
    两段模型的封装类。

    使用方法：
        model = TwoStageModel()
        model.train(df)          # 训练
        result = model.predict_pipeline(drive_features)  # 推理
        model.save("models/")    # 保存
        model.load("models/")    # 加载
    """

    def __init__(self):
        # 模型一：对每个中间状态列分别训练一个LightGBM
        self.model1: Dict[str, lgb.LGBMRegressor] = {}
        # 模型二：对每个结果列分别训练一个LightGBM
        self.model2: Dict[str, lgb.LGBMRegressor] = {}
        # 保存特征列名（推理时用）
        self.model1_feature_cols: List[str] = []
        self.model2_feature_cols: List[str] = []
        # 训练评估结果
        self.eval_results: Dict = {}

    def train(self, df: pd.DataFrame, test_ratio: float = 0.2) -> Dict:
        """
        训练两段模型。

        注意：时序数据不能随机切分！用时间顺序切分：
          前80%数据 → 训练集
          后20%数据 → 测试集

        参数：
            df: 包含三层数据的完整DataFrame
            test_ratio: 测试集比例

        返回：
            各模型在测试集上的评估指标
        """
        print("=" * 50)
        print("开始训练两段模型")
        print("=" * 50)

        # ── 训练模型一 ─────────────────────────────────────────
        print("\n【模型一】驱动层 → 中间状态")
        X1, Y1 = build_features_for_model1(df)
        self.model1_feature_cols = list(X1.columns)
        results1 = self._train_multi_output(X1, Y1, test_ratio, tag="模型一")
        self.model1 = results1["models"]
        self.eval_results["model1"] = results1["metrics"]

        # ── 训练模型二 ─────────────────────────────────────────
        print("\n【模型二】中间状态 → 结果指标")
        X2, Y2 = build_features_for_model2(df)
        self.model2_feature_cols = list(X2.columns)
        results2 = self._train_multi_output(X2, Y2, test_ratio, tag="模型二")
        self.model2 = results2["models"]
        self.eval_results["model2"] = results2["metrics"]

        self._print_summary()
        return self.eval_results

    def predict_intermediate(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        模型一推理：给定驱动特征，预测中间状态。

        参数：
            X: 驱动层特征（列名必须和训练时一致）

        返回：
            预测的中间状态 DataFrame（列名 = INTERMEDIATE_COLS）
        """
        preds = {}
        for col, model in self.model1.items():
            X_aligned = self._align_features(X, self.model1_feature_cols)
            preds[col] = model.predict(X_aligned)
        return pd.DataFrame(preds)

    def predict_result(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        模型二推理：给定中间状态特征，预测结果指标。

        参数：
            X: 中间状态特征

        返回：
            预测的结果指标 DataFrame（列名 = RESULT_COLS）
        """
        preds = {}
        for col, model in self.model2.items():
            X_aligned = self._align_features(X, self.model2_feature_cols)
            preds[col] = model.predict(X_aligned)
        return pd.DataFrame(preds)

    def predict_pipeline(
        self,
        drive_features: pd.DataFrame,
        intermediate_override: Optional[pd.DataFrame] = None,
    ) -> Dict:
        """
        完整两段推理：驱动特征 → 中间状态 → 结果指标。

        参数：
            drive_features: 驱动层特征
            intermediate_override: 如果提供，跳过模型一，直接用真实中间状态送入模型二

        返回：
            {
                "intermediate": 预测的中间状态,
                "result": 预测的结果指标（power_factor, energy_per_ton）
            }
        """
        if intermediate_override is not None:
            intermediate_pred = intermediate_override
        else:
            intermediate_pred = self.predict_intermediate(drive_features)

        result_pred = self.predict_result(
            pd.concat(
                [
                    drive_features.reset_index(drop=True),
                    intermediate_pred.reset_index(drop=True),
                ],
                axis=1,
            )
        )

        return {
            "intermediate": intermediate_pred,
            "result": result_pred,
        }

    def get_shap_importance(
        self, target_col: str = "power_factor", top_n: int = 15
    ) -> pd.DataFrame:
        """
        对指定目标列做SHAP特征重要性分析。

        这告诉你：哪些操作/状态对功率因数影响最大？
        （对应电价文章里"滚动均价"是最重要特征的发现）

        参数：
            target_col: 要分析的目标列（"power_factor" 或 "energy_per_ton"）
            top_n: 返回前N个重要特征

        返回：
            按重要性排序的 DataFrame
        """
        try:
            import shap
        except ImportError:
            print("请先安装shap: pip install shap --break-system-packages")
            return pd.DataFrame()

        if target_col in self.model2:
            model = self.model2[target_col]
            feature_cols = self.model2_feature_cols
        elif target_col in self.model1:
            model = self.model1[target_col]
            feature_cols = self.model1_feature_cols
        else:
            raise ValueError(f"找不到目标列 '{target_col}' 对应的模型")

        # SHAP基于树模型，直接用TreeExplainer
        explainer = shap.TreeExplainer(model)
        # 用训练数据的一个小样本（节省时间）
        dummy_X = pd.DataFrame(
            np.random.randn(100, len(feature_cols)), columns=feature_cols
        )
        shap_values = explainer.shap_values(dummy_X)

        importance = (
            pd.DataFrame(
                {
                    "feature": feature_cols,
                    "shap_importance": np.abs(shap_values).mean(axis=0),
                }
            )
            .sort_values("shap_importance", ascending=False)
            .head(top_n)
        )

        return importance

    def save(self, save_dir: str = "models/"):
        """保存两段模型到本地"""
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        with open(f"{save_dir}/two_stage_model.pkl", "wb") as f:
            pickle.dump(
                {
                    "model1": self.model1,
                    "model2": self.model2,
                    "model1_feature_cols": self.model1_feature_cols,
                    "model2_feature_cols": self.model2_feature_cols,
                    "eval_results": self.eval_results,
                },
                f,
            )
        print(f"模型已保存到 {save_dir}/two_stage_model.pkl")

    def load(self, save_dir: str = "models/"):
        """从本地加载两段模型"""
        with open(f"{save_dir}/two_stage_model.pkl", "rb") as f:
            data = pickle.load(f)
        self.model1 = data["model1"]
        self.model2 = data["model2"]
        self.model1_feature_cols = data["model1_feature_cols"]
        self.model2_feature_cols = data["model2_feature_cols"]
        self.eval_results = data["eval_results"]
        print(
            f"模型已加载：model1({len(self.model1)}个目标), model2({len(self.model2)}个目标)"
        )

    # ── 内部方法 ───────────────────────────────────────────────────

    def _train_multi_output(
        self, X: pd.DataFrame, Y: pd.DataFrame, test_ratio: float, tag: str
    ) -> Dict:
        """对Y的每一列分别训练一个LightGBM"""
        # 时序切分（不能随机打乱！）
        split_idx = int(len(X) * (1 - test_ratio))
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        Y_train, Y_test = Y.iloc[:split_idx], Y.iloc[split_idx:]

        print(f"  训练集：{len(X_train)} 条 | 测试集：{len(X_test)} 条")

        models = {}
        metrics = {}

        for col in Y.columns:
            model = lgb.LGBMRegressor(**LGBM_PARAMS)
            model.fit(
                X_train,
                Y_train[col],
                eval_set=[(X_test, Y_test[col])],
                callbacks=[
                    lgb.early_stopping(50, verbose=False),
                    lgb.log_evaluation(period=-1),
                ],
            )
            y_pred = model.predict(X_test)
            metrics[col] = {
                "r2": round(r2_score(Y_test[col], y_pred), 4),
                "mae": round(mean_absolute_error(Y_test[col], y_pred), 4),
                "mape": round(
                    mean_absolute_percentage_error(Y_test[col], y_pred) * 100, 2
                ),
            }
            models[col] = model
            print(
                f"  [{tag}] {col}: R²={metrics[col]['r2']}, MAE={metrics[col]['mae']}, MAPE={metrics[col]['mape']}%"
            )

        return {"models": models, "metrics": metrics}

    def _align_features(
        self, X: pd.DataFrame, expected_cols: List[str]
    ) -> pd.DataFrame:
        """确保推理时的特征列和训练时一致，缺失列填0"""
        for col in expected_cols:
            if col not in X.columns:
                X = X.copy()
                X[col] = 0.0
        return X[expected_cols]

    def _print_summary(self):
        """打印训练总结"""
        print("\n" + "=" * 50)
        print("训练完成，精度汇总：")
        print("=" * 50)
        for model_tag, metrics in self.eval_results.items():
            print(f"\n{model_tag}:")
            for col, m in metrics.items():
                status = "✓" if m["mape"] < 5 else "△"
                print(f"  {status} {col}: R²={m['r2']}, MAPE={m['mape']}%")
