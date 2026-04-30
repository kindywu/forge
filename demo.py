import lightgbm as lgb
import pandas as pd
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# 1. 加载加州房价数据
data = fetch_california_housing()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target  # 房价（单位：10万美元）

# ===================== 新增：打印前10行特征数据 + 目标值 =====================
print("===== 加州房价数据集 前10行特征数据 =====")
print(X.head(10))  # 打印特征前10行

print("\n===== 前10行房价目标值（单位：10万美元）=====")
print(y[:10])
# ==========================================================================

# 2. 划分训练集、测试集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 3. 构造LightGBM专用数据集格式
lgb_train = lgb.Dataset(X_train, label=y_train)
lgb_eval = lgb.Dataset(X_test, label=y_test, reference=lgb_train)

params = {
    "boosting_type": "gbdt",
    "objective": "regression",
    "metric": "mse",
    "learning_rate": 0.05,
    "num_leaves": 31,
    "verbose": -1,
    "early_stopping_round": 20,
}

# 5. 训练模型
model = lgb.train(params, lgb_train, num_boost_round=200, valid_sets=[lgb_eval])

# 6. 预测 + 模型评估
y_pred = model.predict(X_test, num_iteration=model.best_iteration)

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f"\n均方误差 MSE: {mse:.4f}")
print(f"均方根误差 RMSE: {rmse:.4f}")
print(f"决定系数 R²: {r2:.4f}")

# 7. 特征重要性
print("\n===== 特征重要性（影响房价最大的因素）=====")
feature_importance = pd.DataFrame(
    {"feature": X.columns, "importance": model.feature_importance()}
).sort_values("importance", ascending=False)

print(feature_importance)


new_house = [
    8.3252,  # MedInc 收入中位数
    41.0,  # HouseAge 房龄
    6.9841,  # AveRooms 平均房间数
    1.0238,  # AveBedrms 平均卧室数
    322.0,  # Population 人口
    2.5556,  # AveOccup 平均居住人数
    37.88,  # Latitude 纬度
    -122.23,  # Longitude 经度
]

# 转换成模型能识别的格式
new_df = pd.DataFrame([new_house], columns=data.feature_names)

# 预测！
pred_price = model.predict(new_df)[0]

# 输出结果（单位：10万美元 → 乘以10万就是真实价格）
print(f"\n预测房价：${pred_price * 100000:.2f}")
