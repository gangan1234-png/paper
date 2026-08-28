# -*- coding: utf-8 -*-
"""
潮宗街主题空间分析与功能分区关联 (论文复刻版)
功能：
1. 局部主题分析：计算不同功能分区(Zone)下的主题分布热力图
2. 空间分布分析：基于经纬度生成带有主题颜色的地理散点图
3. 自动生成用于论文插图的图表
"""

import pandas as pd
import numpy as np
import matplotlib

# 设置后端，防止PyCharm报错
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import os
import folium  # 需要安装: pip install folium
from folium.plugins import HeatMap

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False


class SpatialTopicAnalyzer:
    def __init__(self, data_path, output_dir="Spatial_Analysis_Result"):
        self.df = pd.read_csv(data_path, encoding='utf-8-sig')
        self.output_dir = output_dir
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # 简单清洗列名
        self.df.columns = [c.strip() for c in self.df.columns]
        print(f"数据加载成功，列名: {list(self.df.columns)}")

        # 检查必要的列是否存在
        required_cols = ['Topic', '分区', 'Longitude', 'Latitude']  # 根据你的实际列名调整
        # 模糊匹配列名（防止列名是 Latitude__ 或 lat 等）
        self.col_map = {}
        for req in required_cols:
            match = [c for c in self.df.columns if
                     req in c or (req == 'Longitude' and 'Lng' in c) or (req == 'Latitude' and 'Lat' in c)]
            if match:
                self.col_map[req] = match[0]
            else:
                print(f"⚠️ 警告: 未找到包含 {req} 的列，部分功能可能无法运行")

    def analyze_zonal_distribution(self):
        """核心功能：生成【功能分区-主题】热力图 (论文中的局部主题分析)"""
        print("1. 正在分析功能分区与主题的关系...")

        topic_col = 'Topic'
        zone_col = self.col_map.get('分区')

        if not zone_col:
            return

        # 过滤掉噪声数据 (Topic -1)
        clean_df = self.df[self.df[topic_col] != -1]

        # 1. 计算交叉表 (每个分区下，各主题的数量)
        cross_tab = pd.crosstab(clean_df[zone_col], clean_df[topic_col])

        # 2. 计算百分比 (归一化：看每个分区内部的主题构成)
        # axis=1 表示按行求和为1，即“在这个分区里，各个主题占多少比例”
        cross_tab_norm = cross_tab.div(cross_tab.sum(axis=1), axis=0)

        # 3. 绘制热力图
        plt.figure(figsize=(12, 8))
        sns.heatmap(cross_tab_norm, annot=True, fmt='.2f', cmap='YlGnBu', cbar_kws={'label': '主题占比'})
        plt.title('功能分区与感知主题关联热力图 (局部主题分析)')
        plt.ylabel('功能分区')
        plt.xlabel('主题ID')

        save_path = f"{self.output_dir}/1_zonal_topic_heatmap.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ 热力图已保存: {save_path}")

        # 保存数据表供论文引用
        cross_tab_norm.to_csv(f"{self.output_dir}/zonal_distribution_data.csv", encoding='utf-8-sig')

    def visualize_spatial_map(self):
        """核心功能：生成交互式地图 (论文中的空间分布图)"""
        print("2. 正在生成空间分布地图...")

        lat_col = self.col_map.get('Latitude')
        lng_col = self.col_map.get('Longitude')

        if not lat_col or not lng_col:
            return

        # 创建地图中心
        center_lat = self.df[lat_col].mean()
        center_lng = self.df[lng_col].mean()
        m = folium.Map(location=[center_lat, center_lng], zoom_start=16, tiles='CartoDB positron')

        # 为不同主题设置颜色
        # 颜色列表
        colors = ['red', 'blue', 'green', 'purple', 'orange', 'darkred',
                  'lightred', 'beige', 'darkblue', 'darkgreen', 'cadetblue',
                  'darkpurple', 'white', 'pink', 'lightblue', 'lightgreen',
                  'gray', 'black', 'lightgray']

        topic_colors = {}
        unique_topics = sorted(self.df['Topic'].unique())
        for i, topic in enumerate(unique_topics):
            if topic == -1:
                topic_colors[topic] = 'gray'
            else:
                topic_colors[topic] = colors[i % len(colors)]

        # 添加点
        for idx, row in self.df.iterrows():
            topic = row['Topic']
            # 获取关键词作为弹窗信息
            popup_text = f"Topic: {topic}<br>Zone: {row.get(self.col_map.get('分区'), 'N/A')}"

            folium.CircleMarker(
                location=[row[lat_col], row[lng_col]],
                radius=5,
                popup=popup_text,
                color=topic_colors.get(topic, 'black'),
                fill=True,
                fill_color=topic_colors.get(topic, 'black'),
                fill_opacity=0.7
            ).add_to(m)

        # 添加图例说明 (保存为HTML文件)
        map_path = f"{self.output_dir}/2_spatial_topic_map.html"
        m.save(map_path)
        print(f"✅ 交互式地图已保存: {map_path} (请用浏览器打开截图)")

    def generate_stacked_bar(self):
        """补充图表：堆叠柱状图 (看不同分区的构成)"""
        print("3. 生成堆叠柱状图...")
        zone_col = self.col_map.get('分区')
        if not zone_col: return

        clean_df = self.df[self.df['Topic'] != -1]

        # 统计
        counts = clean_df.groupby([zone_col, 'Topic']).size().unstack(fill_value=0)
        # 归一化
        props = counts.div(counts.sum(axis=1), axis=0)

        props.plot(kind='bar', stacked=True, figsize=(12, 6), colormap='tab20')
        plt.title('不同功能分区的主题构成比例')
        plt.xlabel('功能分区')
        plt.ylabel('比例')
        plt.legend(title='Topic', bbox_to_anchor=(1.05, 1), loc='upper left')

        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/3_zonal_stacked_bar.png", dpi=300)
        print("✅ 堆叠柱状图已保存")


if __name__ == "__main__":
    # 【注意】这里换成你刚刚跑出来的那个 csv 文件路径
    csv_path = r"文件路径"

    # 运行分析
    analyzer = SpatialTopicAnalyzer(csv_path)
    analyzer.analyze_zonal_distribution()
    analyzer.visualize_spatial_map()
    analyzer.generate_stacked_bar()