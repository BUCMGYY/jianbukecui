import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt

# 加载保存的随机森林模型
model = joblib.load('RF.pkl')

# 特征范围定义（根据提供的特征范围和数据类型）
feature_ranges = {
    # 数值型特征（基于数据统计）
    "年龄": {"type": "numerical", "min": 22.0, "max": 75.0, "default": 45.0},
    "病程": {"type": "numerical", "min": 1.0, "max": 9.0, "default": 4.0},
    "VAS评分": {"type": "numerical", "min": 2.0, "max": 9.0, "default": 6.0},

    # 关节活动角度（单位：度）
    "前屈角度": {"type": "numerical", "min": 65.0, "max": 150.0, "default": 100.0},
    "后伸角度": {"type": "numerical", "min": 18.0, "max": 55.0, "default": 35.0},
    "外展角度": {"type": "numerical", "min": 58.0, "max": 130.0, "default": 90.0},
    "内收角度": {"type": "numerical", "min": 33.0, "max": 66.0, "default": 45.0},
    "内旋角度": {"type": "numerical", "min": 25.0, "max": 82.0, "default": 45.0},
    "外旋角度": {"type": "numerical", "min": 8.0, "max": 46.0, "default": 20.0},

    # 分类特征（基于数据观察）
    "疾病": {"type": "categorical", "options": [1, 2]},  # 1和2代表两种疾病类型
}

# Streamlit 界面
st.title("肩不可摧")

# 动态生成输入项
st.header("Enter the following feature values:")
feature_values = []
for feature, properties in feature_ranges.items():
    if properties["type"] == "numerical":
        value = st.number_input(
            label=f"{feature} ({properties['min']} - {properties['max']})",
            min_value=float(properties["min"]),
            max_value=float(properties["max"]),
            value=float(properties["default"]),
        )
    elif properties["type"] == "categorical":
        value = st.selectbox(
            label=f"{feature} (Select a value)",
            options=properties["options"],
        )
    feature_values.append(value)

# 转换为模型输入格式
features = np.array([feature_values])

if st.button("Predict"):
    # 获取预测结果和概率
    predicted_class = model.predict(features)[0]
    predicted_proba = model.predict_proba(features)[0]

    # 构建包含所有证型概率的文本
    proba_text = "中医证型预测概率：\n"
    for class_idx, prob in enumerate(predicted_proba):
        proba_text += f"• 证型 {model.classes_[class_idx]}: {prob * 100:.2f}%\n"

    # 添加最可能证型的特别提示
    proba_text += f"\n最可能证型：{predicted_class}（{predicted_proba.max() * 100:.2f}%）"

    # 可视化配置
    fig, ax = plt.subplots(figsize=(10, 4))  # 增加高度以适应多行文本
    ax.text(0.5, 0.5,
            proba_text,
            fontsize=14,
            ha='left',  # 改为左对齐
            va='center',
            fontname='Microsoft YaHei',  # 建议使用中文字体
            linespacing=1.5,
            transform=ax.transAxes)

    ax.axis('off')
    plt.tight_layout()
    plt.savefig("proba_plot.png", bbox_inches='tight', dpi=300)
    st.image("proba_plot.png")

    # 计算 SHAP 值
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(pd.DataFrame([feature_values], columns=feature_ranges.keys()))

    # 生成 SHAP 力图
    class_index = predicted_class  # 当前预测类别
    shap_fig = shap.force_plot(
        explainer.expected_value[class_index],
        shap_values[:,:,class_index],
        pd.DataFrame([feature_values], columns=feature_ranges.keys()),
        matplotlib=True,
    )
    # 保存并显示 SHAP 图
    plt.savefig("shap_force_plot.png", bbox_inches='tight', dpi=1200)
    st.image("shap_force_plot.png")
