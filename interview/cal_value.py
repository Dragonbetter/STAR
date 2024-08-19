import numpy as np
from scipy.stats import linregress

# 计算每组数据相对于F的斜率（假设y = ax，且b = 0）
F_values = (0.243, 0.433)  # F组的两个数值
groups_values = {
    "A": (0.321, 0.579),
    "B": (0.306, 0.547),
    "C": (0.282, 0.493),
    "D": (0.271, 0.451),
    "E": (0.260, 0.454)
}
groups_values = {
    "1": (14.667, 33.431),
    "2": (11.29, 19.369),
    "3": (10.82, 18.97),
    "4": (11.47, 20.184),
    "5": (10.13, 17.05)
}
F_values = (0.182,0.351)
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

# Given data
A = np.array([1.270, 1.228, 1.153, 1.144, 1.139, 1.163, 1.209, 1.153, 1.177, 1.132])
B = np.array([2.155, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 1.381])

# Perform linear regression to find the slope (a) and intercept (b)
slope, intercept, r_value, p_value, std_err = linregress(A[~np.isnan(B)], B[~np.isnan(B)])

# Use the equation y = ax + b to predict the missing values in B
B_filled = A * slope + intercept
B_filled
groups_values = {
    "1":(16.453,35.25),
    "2":(11.68,22.24),
    "3": (10.62,18.76),
    "4": (9.858,16.24),
    }

if __name__ == '__main__':
    groups_values = {

    }
    list_1 = [18.50, 21.36, 7.96, 7.49, 9.67, 5.29, 10.41, 20.80]
    list_2 = [32.69, 35.95, 12.97, 15.24, 16.52, 8.42, 17.31, 34.57]
    groups_values = {
    "1": (list_1[0], list_2[0]),
    "2": (list_1[1], list_2[1]),
    "3": (list_1[2], list_2[2]),
    "4": (list_1[3], list_2[3]),
    "5": (list_1[4], list_2[4]),
    "6": (list_1[5], list_2[5]),
    "7": (list_1[6], list_2[6]),
    "8": (list_1[7], list_2[7]),
    # Continue for the rest of the elements...
    }

    F_values =  (14.18, 24.47)
    # 计算斜率a，y = ax，a = y/x
    slope_ratios = {}
    for group, values in groups_values.items():
        slope_ratios[group] = (values[0] / F_values[0], values[1] / F_values[1])
    # 新的F值
    new_F_values = (8.943,16.204)

    # 使用上述计算的斜率来预测新的A、B、C、D、E值
    predicted_values = {}
    for group, slopes in slope_ratios.items():
        predicted_values[group] = (new_F_values[0] * slopes[0], new_F_values[1] * slopes[1])
    print(predicted_values)