# -*- coding: utf-8 -*-
"""
潮宗街古今对比分析 - BERTopic 最终章
功能：
1. 全局训练：基于合并数据集提取公共主题
2. 动态分析：计算每个主题在“现代”与“历史”中的分布差异 (c-TF-IDF)
3. 论文出图：生成“古今演变对比图” (顶刊截断式哑铃图)
"""

import os

# ========================================================
# 设置 HuggingFace 国内镜像站，解决连接超时问题
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
# ========================================================

import pandas as pd
import numpy as np
from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer
from umap import UMAP
from hdbscan import HDBSCAN
import jieba
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import matplotlib.ticker as ticker
from matplotlib.lines import Line2D

# 设置绘图风格和字体
matplotlib.use('Agg')
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False
sns.set(font='SimHei')

warnings.filterwarnings('ignore')


class ComparativeTopicAnalyzer:
    def __init__(self, file_path, output_dir="Comparative_Analysis_Result"):
        try:
            self.df = pd.read_excel(file_path, engine='openpyxl')
        except Exception as e:
            print(f"读取文件失败，请检查路径: {e}")
            return

        self.df = self.df.dropna(subset=['Text', 'Source_Type'])
        self.output_dir = output_dir
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        print(f"合并数据加载成功，总样本量: {len(self.df)}")

        jieba.add_word("潮宗街")
        jieba.add_word("长沙市")
        jieba.add_word("大西门")
        jieba.add_word("麻石路")
        jieba.add_word("时务学堂")
        jieba.add_word("九如里")

    def clean_text(self, text):
        stopwords = {
            '的', '了', '在', '是', '我', '有', '和', '就', '都', '而', '及', '与', '着', '或', '一个', '没有',
            '我们', '你们', '他们', '它们', '因为', '所以', '如果', '但是', '虽然', '并', '且', '去', '来',
            '长沙', '长沙市', '潮宗街', '街区', '片区', '地方', '区域', '街道', '空间'
        }
        visual_noise = {
            '拍摄', '图片', '照片', '画面', '图中', '显示', '镜头', '视角', '全景', '特写', '俯瞰', '正视', '仰视',
            '位于', '属于', '地处', '可见', '看到', '观察', '发现', '存在', '出现', '展示', '呈现', '表现',
            '主要', '部分', '整体', '具有', '非常', '十分', '比较', '这种', '那种', '这里', '那里', '一些',
            '进行', '采用', '使用', '可以', '可能', '不仅', '而且', '以及', '甚至', '方面', '情况', '状态',
            '天空', '路面', '地面', '道路', '马路', '两侧', '两旁', '中间', '背景', '前景', '远处', '近处',
            '颜色', '色彩', '风格', '造型', '设计', '装饰', '结构', '样式', '布局', '氛围'
        }
        stopwords.update(visual_noise)

        words = jieba.lcut(str(text))
        return " ".join([w for w in words if w not in stopwords and len(w) > 1 and not w.isdigit()])

    def run_comparison(self):
        print("1. 正在进行全局文本清洗...")
        self.df['processed_text'] = self.df['Text'].apply(self.clean_text)
        docs = self.df['processed_text'].tolist()
        classes = self.df['Source_Type'].tolist()

        print("2. 初始化全局模型...")
        umap_model = UMAP(n_neighbors=8, n_components=5, min_dist=0.0, metric='cosine', random_state=42)
        hdbscan_model = HDBSCAN(min_cluster_size=5, metric='euclidean', cluster_selection_method='eom',
                                prediction_data=True)
        vectorizer_model = CountVectorizer(stop_words=None, min_df=2)

        try:
            topic_model = BERTopic(
                language="multilingual",
                umap_model=umap_model,
                hdbscan_model=hdbscan_model,
                vectorizer_model=vectorizer_model,
                nr_topics="auto",
                top_n_words=10,
                verbose=True
            )
            print("3. 开始混合训练...")
            topics, probs = topic_model.fit_transform(docs)
            self.df['Topic'] = topics
        except Exception as e:
            print(f"模型训练失败: {e}")
            return

        freq = topic_model.get_topic_info()
        print("4. 计算古今差异...")
        topics_per_class = topic_model.topics_per_class(docs, classes=classes)

        self.save_and_plot(topic_model, freq, topics_per_class)

    def save_and_plot(self, topic_model, freq, topics_per_class):
        print("5. 正在生成顶刊排版对比图...")
        topics_per_class.to_excel(f"{self.output_dir}/1_topics_per_class_raw.xlsx", index=False)

        # 核心绘制截断式哑铃图
        self.plot_static_comparison(topic_model, topics_per_class)

        topic_dict = {row['Topic']: row['Name'] for _, row in freq.iterrows()}
        keywords_dict = {}
        for topic in set(self.df['Topic']):
            if topic != -1:
                try:
                    words = [x[0] for x in topic_model.get_topic(topic)[:5]]
                    keywords_dict[topic] = "_".join(words)
                except:
                    pass
            else:
                keywords_dict[topic] = "噪声"

        self.df['Topic_Name'] = self.df['Topic'].map(topic_dict)
        self.df['Topic_Keywords'] = self.df['Topic'].map(keywords_dict)
        self.df.to_excel(f"{self.output_dir}/3_final_comparison_data.xlsx", index=False)

    def plot_static_comparison(self, topic_model, topics_per_class):
        """
        [极致排版升级] 绘制顶刊截断式哑铃图 + 斑马纹背景
        """
        data = topics_per_class[topics_per_class['Topic'] != -1].copy()
        total_freq_by_class = data.groupby('Class')['Frequency'].sum()

        data['Normalized_Freq'] = data.apply(
            lambda x: (x['Frequency'] / total_freq_by_class[x['Class']]) * 100, axis=1
        )

        topic_names = {}
        for topic in data['Topic'].unique():
            words = [x[0] for x in topic_model.get_topic(topic)[:4]]
            topic_names[topic] = f"T{topic}: {' - '.join(words)}"

        classes = data['Class'].unique()
        history_col, modern_col = None, None
        for c in classes:
            if '历' in str(c) or 'Hist' in str(c): history_col = c
            if '现' in str(c) or 'Mod' in str(c): modern_col = c
        if not history_col: history_col = classes[0]
        if not modern_col and len(classes) > 1: modern_col = classes[1]

        pivot_df = data.pivot(index='Topic', columns='Class', values='Normalized_Freq').fillna(0)
        for col in [history_col, modern_col]:
            if col not in pivot_df.columns:
                pivot_df[col] = 0

        # 按现代热度升序排
        pivot_df = pivot_df.sort_values(by=modern_col, ascending=True)

        # 【核心逻辑】：设定 X 轴的画幅极限。根据你的数据，除了T0，其余最大约22%。
        # 所以我们将极限设为 24%，让 95% 的数据撑满屏幕！
        X_LIMIT = 24.0

        y_pos = np.arange(len(pivot_df))
        fig_height = max(9, len(pivot_df) * 0.55)
        fig, ax = plt.subplots(figsize=(14, fig_height))

        color_up = '#FA7F6F'
        color_down = '#2878B5'
        color_hist = '#B0B0B0'

        # 【视觉增强】添加顶刊级斑马纹背景 (Zebra Striping)，取代普通的白底
        for i in range(len(pivot_df)):
            if i % 2 == 0:
                ax.axhspan(i - 0.5, i + 0.5, color='#F6F8FA', zorder=0)

        for i, (idx, row) in enumerate(pivot_df.iterrows()):
            h_val = row[history_col]
            m_val = row[modern_col]
            trend_color = color_up if m_val > h_val else color_down

            # 如果数据超出极限，截断它
            plot_m = min(m_val, X_LIMIT)
            plot_h = min(h_val, X_LIMIT)

            # 画底部的浅色引导线
            ax.plot([0, X_LIMIT], [i, i], color='#EAEAEA', linewidth=1, zorder=1, linestyle='--')

            # 画演变跨度实线
            ax.plot([plot_h, plot_m], [i, i], color=trend_color, linewidth=3.5, zorder=2, alpha=0.85)

            # ---------------- 历史基准点 ----------------
            if h_val < X_LIMIT:
                ax.scatter(h_val, i, color=color_hist, s=120, zorder=3, edgecolors='white', linewidth=1.5)

            # ---------------- 现代落点 (包含超限处理) ----------------
            offset = 0.6  # 文字间距
            if m_val <= X_LIMIT:
                # 正常范围内的数据点
                ax.scatter(m_val, i, color=trend_color, s=200, zorder=4, edgecolors='white', linewidth=1.5)

                # 文字防重叠逻辑
                if m_val > h_val:
                    ax.text(m_val + offset, i, f"{m_val:.1f}%", va='center', ha='left', fontsize=11, color=trend_color,
                            weight='bold')
                    if h_val > 0.5:  # 避免左侧文字被Y轴切掉
                        ax.text(h_val - offset, i, f"{h_val:.1f}%", va='center', ha='right', fontsize=10,
                                color='#888888')
                else:
                    ax.text(m_val - offset, i, f"{m_val:.1f}%", va='center', ha='right', fontsize=11, color=trend_color,
                            weight='bold')
                    ax.text(h_val + offset, i, f"{h_val:.1f}%", va='center', ha='left', fontsize=10, color='#888888')
            else:
                # 【极端值爆发】：超出画幅的点变成右向箭头，视觉上“冲破图表”
                ax.scatter(X_LIMIT, i, marker='>', color=trend_color, s=250, zorder=5)
                # 将真实的 42.5% 标注在箭头后面，加上醒目的指示符
                ax.text(X_LIMIT + 0.3, i, f"{m_val:.1f}%", va='center', ha='left', fontsize=12, color=trend_color,
                        weight='bold', fontstyle='italic')
                ax.text(h_val - offset, i, f"{h_val:.1f}%", va='center', ha='right', fontsize=10, color='#888888')

        # 轴和刻度设置
        ax.set_xlim(-1, X_LIMIT + 3)  # 右侧多留出3个单位给冲顶的文字
        ax.set_ylim(-0.5, len(pivot_df) - 0.5)
        ax.set_yticks(y_pos)
        ax.set_yticklabels([topic_names[idx] for idx in pivot_df.index], fontsize=13)
        ax.tick_params(axis='y', length=0)
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f"{x:.0f}%"))
        ax.tick_params(axis='x', labelsize=12, color='#999999')

        sns.despine(left=True, bottom=False, trim=False)

        custom_lines = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor=color_up, markersize=11, label='现代热度跃升'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor=color_down, markersize=11, label='现代热度衰减'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor=color_hist, markersize=9, label='历史基准热度'),
            Line2D([0], [0], marker='>', color='w', markerfacecolor=color_up, markersize=11, label='极值溢出指示')
        ]
        ax.legend(handles=custom_lines, loc='lower right', frameon=True, fontsize=12, title='古今演变趋势',
                  title_fontsize=13, shadow=True, fancybox=True, edgecolor='#EEEEEE')

        plt.title('潮宗街古今主题演变结构图 (Dumbbell Plot)', fontsize=22, pad=30, weight='bold', color='#222222')
        plt.xlabel('主题相对热度占比', fontsize=14, labelpad=15, color='#555555')

        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/2_static_comparison_dumbbell.jpg", dpi=300, format='jpg', bbox_inches='tight')
        print("✅ 顶刊级截断式哑铃图已生成: 2_static_comparison_dumbbell.jpg")


if __name__ == "__main__":
    file_path = r"文件路径"
    if os.path.exists(file_path):
        analyzer = ComparativeTopicAnalyzer(file_path)
        analyzer.run_comparison()
    else:
        print(f"找不到文件: {file_path}")