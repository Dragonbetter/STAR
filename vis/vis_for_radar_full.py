import numpy as np
import matplotlib.pyplot as plt
data = {
    'Bookstore': {
        'NoS': 4609,
        'NoA': {
            'All': 101979,
            'Pedestrian': 80739,
            'Bicyclist': 19407,
            'Skater': 546,
            'Cart': 468,
            'Car': 546,
            'Bus': 273
        },
        'APN': 17.517,
        'AN': 22.12,
        'AV': 31.86,
        'AA': 0.200
    },
    'Coupa': {
        'NoS': 2861,
        'NoA': {
            'All': 51020,
            'Pedestrian': 44548,
            'Bicyclist': 6355,
            'Skater': 78,
            'Cart': 39,
            'Car': 0,
            'Bus': 0
        },
        'APN': 15.57,
        'AN': 17.83,
        'AV': 21.41,
        'AA': 0.288
    },
    'DeathCircle': {
        'NoS': 3190,
        'NoA': {
            'All': 164747,
            'Pedestrian': 66537,
            'Bicyclist': 74827,
            'Skater': 3952,
            'Cart': 4963,
            'Car': 13412,
            'Bus': 1056
        },
        'APN': 20.85,
        'AN': 51.64,
        'AV': 54.60,
        'AA': 0.261
    },
    'Gates': {
        'NoS': 3596,
        'NoA': {
            'All': 71177,
            'Pedestrian': 36592,
            'Bicyclist': 31503,
            'Skater': 897,
            'Cart': 254,
            'Car': 1200,
            'Bus': 731
        },
        'APN': 10.17,
        'AN': 19.79,
        'AV': 54.65,
        'AA': 0.438
    },
    'Hyang': {
        'NoS': 5375,
        'NoA': {
            'All': 152317,
            'Pedestrian': 119280,
            'Bicyclist': 31002,
            'Skater': 1469,
            'Cart': 527,
            'Car': 39,
            'Bus': 0
        },
        'APN': 22.19,
        'AN': 28.33,
        'AV': 34.25,
        'AA': 0.088
    },
    'Little': {
        'NoS': 2710,
        'NoA': {
            'All': 29366,
            'Pedestrian': 19201,
            'Bicyclist': 9579,
            'Skater': 156,
            'Cart': 0,
            'Car': 78,
            'Bus': 352
        },
        'APN': 7.08,
        'AN': 10.83,
        'AV': 68.72,
        'AA': 0.705
    },
    'Nexus': {
        'NoS': 6077,
        'NoA': {
            'All': 119586,
            'Pedestrian': 90477,
            'Bicyclist': 3281,
            'Skater': 527,
            'Cart': 308,
            'Car': 23486,
            'Bus': 1507
        },
        'APN': 14.88,
        'AN': 19.67,
        'AV': 23.91,
        'AA': 0.034
    },
    'Quad': {
        'NoS': 84,
        'NoA': {
            'All': 1326,
            'Pedestrian': 1223,
            'Bicyclist': 103,
            'Skater': 0,
            'Cart': 0,
            'Car': 0,
            'Bus': 0
        },
        'APN': 14.55,
        'AN': 15.78,
        'AV': 24.17,
        'AA': 0.124
    }
}
density_data = {}

for scene, values in data.items():
    density_data[scene] = {}
    for agent, count in values['NoA'].items():
        density_data[scene][agent] = count / values['NoS']

density_data

# 选择要绘制的类型
agent_types = ['Pedestrian', 'Bicyclist', 'Skater', 'Car']
scenes = list(density_data.keys())
colors = ['blue', 'green', 'red', 'cyan']

# 创建一个雷达图
fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))

# 设置角度
angles = np.linspace(0, 2 * np.pi, len(scenes), endpoint=False).tolist()
angles += angles[:1]

# 遍历每个类型并绘制相应的数据
for agent_type, color in zip(agent_types, colors):
    values = [density_data[scene][agent_type] for scene in scenes]
    values += values[:1]

    # 绘制雷达图
    ax.plot(angles, values, '-o', label=agent_type, color=color)
    ax.fill(angles, values, alpha=0.25, color=color)

# 设置图的样式
ax.set_xticks(angles[:-1])
ax.set_xticklabels(scenes)
ax.set_title("Density of Different Agent Types Across Scenes")
ax.legend(loc="upper right")

# 显示图形
plt.show()

