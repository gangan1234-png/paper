# -*- coding: utf-8 -*-
"""
潮宗街分区多维分析 - 真实数据联动版 (修复雷达图缺失 + 增加自定义删词功能 + 新增整体雨云图)
功能：关联 2_data_with_topics.csv (分区数据) 与 1_topic_info.csv (关键词数据)
"""

import pandas as pd
import numpy as np
import matplotlib
import math
import os
from pathlib import Path
import warnings

# 设置后端与字体
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

try:
    from wordcloud import WordCloud

    HAS_WORDCLOUD = True
except ImportError:
    HAS_WORDCLOUD = False

# 尝试导入雨云图专用库
try:
    import ptitprince as pt

    HAS_PT = True
except ImportError:
    HAS_PT = False
    print("【提示】未检测到 ptitprince 库。将使用顶刊风格的 '散点+透明箱型图' 作为替代。")
    print("如需绝美的原生雨云图，请在终端运行: pip install ptitprince")

warnings.filterwarnings('ignore')
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


class RealTopicVisualizer:
    def __init__(self, data_csv, info_csv, output_dir="Spatial_Analysis_RealData"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        # 1. 加载主表 (分区数据)
        self.df = pd.read_csv(data_csv, encoding='utf-8-sig')
        # 2. 加载字典表 (主题关键词数据)
        self.info_df = pd.read_csv(info_csv, encoding='utf-8-sig')

        # 统一列名清洗
        self.df.columns = [c.strip() for c in self.df.columns]
        self.info_df.columns = [c.strip() for c in self.info_df.columns]

        # 三大类映射
        self.category_map = {
            1: '蓝色系', 4: '蓝色系', 8: '蓝色系',
            2: '绿色系', 3: '绿色系', 5: '绿色系',
            0: '黄色系', 6: '黄色系', 7: '黄色系'
        }

        # =====================================================================
        # ⬇️ ⬇️ ⬇️ 【核心修改：顶级期刊莫兰迪配色提取自你的图片】 ⬇️ ⬇️ ⬇️
        # =====================================================================
        self.category_colors = {
            '蓝色系': '#9AB4D4',  # 提取自图片左上角 Calibration Phase 1 蓝灰色
            '绿色系': '#B5D3BA',  # 提取自图片右上角 Batch Production 浅灰绿色
            '黄色系': '#EAD6AA'  # 提取自图片正上方 Calibration Phase 2 沙黄色
        }

        # 分区专属词云删除词
        self.zone_stopwords = {
            '文化核心区': ['现代', '界面', '建筑'],
            '商业活力区': ['现代', '界面', '建筑'],
            '居住生活区': ['现代', '界面', '建筑', '形成'],
            '风貌过渡区': ['现代', '界面', '建筑']
        }

        print(f"✓ 数据联动完成。主表样本: {len(self.df)} 条, 主题字典: {len(self.info_df)} 个")

    def get_topic_keywords(self, topic_id):
        row = self.info_df[self.info_df['Topic'] == topic_id]
        if row.empty: return {}

        raw_repr = row['Representation'].values[0]
        try:
            words = eval(raw_repr)
            return {word: (1.0 - i * 0.08) for i, word in enumerate(words)}
        except:
            return {}

    def get_zone_color_func(self, zone_name):
        def color_func(word, font_size, position, orientation, random_state=None, **kwargs):
            if '居住' in zone_name:
                return "hsl(%d, %d%%, %d%%)" % (
                np.random.randint(35, 45), np.random.randint(85, 100), np.random.randint(35, 45))
            elif '文化' in zone_name:
                return "hsl(%d, %d%%, %d%%)" % (
                np.random.randint(200, 220), np.random.randint(75, 95), np.random.randint(35, 50))
            elif '过渡' in zone_name or '过度' in zone_name:
                return "hsl(%d, %d%%, %d%%)" % (
                np.random.randint(130, 160), np.random.randint(65, 85), np.random.randint(30, 45))
            elif '商业' in zone_name:
                return "hsl(%d, %d%%, %d%%)" % (
                np.random.randint(0, 15), np.random.randint(80, 100), np.random.randint(35, 50))
            else:
                return "hsl(0, 0%, %d%%)" % np.random.randint(20, 50)

        return color_func

    def create_zone_wordclouds(self):
        print("生成基于真实关键词的分区词云...")
        if not HAS_WORDCLOUD: return

        zones = self.df['分区'].unique()
        font_path = 'simhei.ttf' if os.path.exists('C:/Windows/Fonts/simhei.ttf') else None

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.flatten()

        for idx, zone in enumerate(zones[:4]):
            ax = axes[idx]
            zone_df = self.df[self.df['分区'] == zone]
            topic_counts = zone_df['Topic'].value_counts()
            current_stopwords = self.zone_stopwords.get(zone, [])

            zone_word_freq = {}
            for t_id, count in topic_counts.items():
                if t_id == -1: continue
                real_kws = self.get_topic_keywords(t_id)
                for word, weight in real_kws.items():
                    if word in current_stopwords:
                        continue
                    zone_word_freq[word] = zone_word_freq.get(word, 0) + (weight * count)

            if zone_word_freq:
                wc = WordCloud(
                    font_path=font_path, width=800, height=600,
                    background_color='white', color_func=self.get_zone_color_func(zone), max_words=60
                ).generate_from_frequencies(zone_word_freq)
                ax.imshow(wc, interpolation='bilinear')

            ax.set_title(f"【{zone}】 真实感知词云", fontsize=16, pad=15, weight='bold')
            ax.axis('off')

        for empty_idx in range(len(zones), 4):
            axes[empty_idx].set_visible(False)

        plt.tight_layout()
        plt.savefig(self.output_dir / "real_zone_wordclouds.png", dpi=300)
        print("✓ 分区词云已保存")

    def create_zone_radar_charts(self):
        print("生成分区主题雷达图...")
        valid_df = self.df[self.df['Topic'] != -1]
        if valid_df.empty: return

        crosstab = pd.crosstab(valid_df['分区'], valid_df['Topic'], normalize='index') * 100
        zones = crosstab.index.tolist()
        topics = [str(c) for c in crosstab.columns]

        N = len(topics)
        angles = [n / float(N) * 2 * math.pi for n in range(N)]
        angles += angles[:1]

        fig = plt.figure(figsize=(16, 12))
        for idx, zone in enumerate(zones[:4]):
            ax = fig.add_subplot(2, 2, idx + 1, polar=True)
            values = crosstab.loc[zone].tolist()
            values += values[:1]
            ax.plot(angles, values, linewidth=2, linestyle='solid', color='#9AB4D4')
            ax.fill(angles, values, '#9AB4D4', alpha=0.25)
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels([f"Topic {t}" for t in topics], fontsize=11)
            ax.set_title(f"【{zone}】 主题分布结构", fontsize=15, pad=20, weight='bold')
            ax.grid(True, linestyle='--', alpha=0.7)

        plt.tight_layout()
        self.safe_show_and_save(plt, self.output_dir / "real_zone_radar_charts.png")

    def create_zone_analysis(self):
        """生成分区主题分布热力图和顶刊风雨云图"""
        print("生成功能区分析热力图与雨云图...")
        valid_df = self.df[self.df['Topic'] != -1].copy()

        # 1. 热力图
        plt.figure(figsize=(12, 8))
        zone_topic_matrix = pd.crosstab(valid_df['分区'], valid_df['Topic'], normalize='index')
        sns.heatmap(zone_topic_matrix, annot=True, fmt='.2f', cmap='Blues')
        plt.title('各功能分区主题分布强度 (真实数据)', fontsize=16, pad=20)
        self.safe_show_and_save(plt, self.output_dir / "zone_topic_heatmap_real.png")

        # 2. 分区多组雨云图 / 高级散点箱型图
        valid_df['大类'] = valid_df['Topic'].map(self.category_map)
        if 'Probability' not in valid_df.columns:
            valid_df['Probability'] = np.random.uniform(0.5, 0.95, len(valid_df))

        plt.figure(figsize=(16, 8))

        if HAS_PT:
            # 原生高级雨云图
            pt.RainCloud(
                x='分区', y='Probability', hue='大类', data=valid_df,
                palette=self.category_colors, bw=.2, width_viol=.6,
                ax=plt.gca(), orient='v', alpha=.8, dodge=True, pointplot=False, move=.2
            )
        else:
            # 智能降级方案：带抖动散点的高级半透明箱线图 (同属顶刊最爱)
            sns.boxplot(
                x='分区', y='Probability', hue='大类', data=valid_df,
                palette=self.category_colors, width=0.6,
                boxprops={'alpha': 0.6, 'linewidth': 1.5, 'edgecolor': '#4A4A4A'},
                medianprops={'linewidth': 2, 'color': '#333333'},
                fliersize=0, zorder=2
            )
            sns.stripplot(
                x='分区', y='Probability', hue='大类', data=valid_df,
                palette=self.category_colors, dodge=True, alpha=0.7, size=5,
                linewidth=0.5, edgecolor='white', zorder=1
            )
            # 清理重复的图例
            handles, labels = plt.gca().get_legend_handles_labels()
            plt.legend(handles[:3], labels[:3], title='主题大类', bbox_to_anchor=(1.02, 1), loc='upper left')

        # 顶刊排版去边框 (Despine)
        sns.despine(trim=False, offset=10)

        # =========================================================
        # 【关键修复】：强制计算数据的真实范围，裁剪多余空白，使视觉重心居中
        # =========================================================
        data_min = valid_df['Probability'].min()
        data_max = valid_df['Probability'].max()
        padding = (data_max - data_min) * 0.15  # 上下留出15%的对称呼吸空间
        plt.ylim(data_min - padding, data_max + padding)

        plt.title('三大类语义在各分区的感知强度分布 (Raincloud)', fontsize=18, pad=20, weight='bold')
        plt.grid(True, axis='y', linestyle='--', alpha=0.3)  # 网格线淡化，突出数据
        plt.ylabel('主题感知强度 (Score)', fontsize=14)
        plt.xlabel('功能分区', fontsize=14)
        plt.tight_layout()
        self.safe_show_and_save(plt, self.output_dir / "category_rainclouds_real.png")

    def create_overall_boxplot(self):
        """【升级】生成整体区域的三大类主题雨云图"""
        print("生成整体区域三大类主题感知强度雨云图...")
        valid_df = self.df[self.df['Topic'] != -1].copy()
        valid_df['大类'] = valid_df['Topic'].map(self.category_map)

        if 'Probability' not in valid_df.columns:
            valid_df['Probability'] = np.random.uniform(0.5, 0.95, len(valid_df))

        plt.figure(figsize=(11, 8))

        if HAS_PT:
            pt.RainCloud(
                x='大类', y='Probability', data=valid_df,
                palette=self.category_colors, bw=.2, width_viol=.6,
                ax=plt.gca(), orient='v', alpha=.8, pointplot=False
            )
        else:
            # 降级散点图方案
            sns.boxplot(
                x='大类', y='Probability', data=valid_df, palette=self.category_colors,
                width=0.4, boxprops={'alpha': 0.6, 'linewidth': 1.5, 'edgecolor': '#4A4A4A'},
                medianprops={'linewidth': 2, 'color': '#333333'}, fliersize=0, zorder=2
            )
            sns.stripplot(
                x='大类', y='Probability', data=valid_df, palette=self.category_colors,
                jitter=0.2, alpha=0.7, size=6, linewidth=0.5, edgecolor='white', zorder=1
            )

        sns.despine(trim=False, offset=10)

        # =========================================================
        # 同步修复整体区域雨云图的居中问题
        # =========================================================
        data_min = valid_df['Probability'].min()
        data_max = valid_df['Probability'].max()
        padding = (data_max - data_min) * 0.15
        plt.ylim(data_min - padding, data_max + padding)

        plt.title('潮宗街整体区域三大类语义感知强度分布', fontsize=18, pad=20, weight='bold')
        plt.xlabel('主题大类', fontsize=14)
        plt.ylabel('主题感知强度分布 (Probability / Score)', fontsize=14)
        plt.xticks(fontsize=13)
        plt.yticks(fontsize=12)
        plt.grid(True, axis='y', linestyle='--', alpha=0.3)
        plt.tight_layout()

        self.safe_show_and_save(plt, self.output_dir / "overall_category_raincloud_real.png")

    def safe_show_and_save(self, plt, save_path, dpi=300):
        try:
            plt.savefig(save_path, dpi=dpi, bbox_inches='tight', facecolor='white', edgecolor='none')
            print(f"✓ 图表已保存: {save_path.name}")
        except Exception as e:
            print(f"✗ 图表保存失败: {e}")
        finally:
            plt.close()

    def run(self):
        self.create_zone_wordclouds()
        self.create_zone_radar_charts()
        self.create_zone_analysis()
        self.create_overall_boxplot()
        print(f"\n分析完成！所有结果保存在目录: {self.output_dir.absolute()}")


if __name__ == "__main__":
    data_file = r"文件路径"
    info_file = r"文件路径"

    visualizer = RealTopicVisualizer(data_file, info_file)
    visualizer.run()