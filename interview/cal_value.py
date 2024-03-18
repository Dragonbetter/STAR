# 计算每组数据相对于F的斜率（假设y = ax，且b = 0）
F_values = (0.243, 0.433)  # F组的两个数值
groups_values = {
    "A": (0.321, 0.579),
    "B": (0.306, 0.547),
    "C": (0.282, 0.493),
    "D": (0.271, 0.451),
    "E": (0.260, 0.454)
}

# 计算斜率a，y = ax，a = y/x
slope_ratios = {}
for group, values in groups_values.items():
    slope_ratios[group] = (values[0] / F_values[0], values[1] / F_values[1])
# 新的F值
new_F_values = (0.188, 0.372)

# 使用上述计算的斜率来预测新的A、B、C、D、E值
predicted_values = {}
for group, slopes in slope_ratios.items():
    predicted_values[group] = (new_F_values[0] * slopes[0], new_F_values[1] * slopes[1])

# 由于执行环境被重置，重新定义变量并计算斜率
group_values = [
    (0.273, 0.536),
    (0.264, 0.457),
    (0.248, 0.437),
    (0.246, 0.434),
    (0.245, 0.436),
    (0.250, 0.440),
    (0.260, 0.452),
    (0.248, 0.439),
    (0.253, 0.446)
]
last_group_values = (0.243, 0.431)

# 计算每组数相对于最后一组数的斜率（对应两个数值）
slope_ratios_to_last_group = [(value[0] / last_group_values[0], value[1] / last_group_values[1]) for value in group_values]

