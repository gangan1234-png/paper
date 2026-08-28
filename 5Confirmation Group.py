# -*- coding: utf-8 -*-
"""
潮宗街街景感知 - BERTopic 完整运行版：确认组
优化：沉稳高级学术色板、纯白无边框画布、全面支持SVG高精度输出
"""

import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import pandas as pd
import numpy as np
from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer
from umap import UMAP
from hdbscan import HDBSCAN
import jieba
import warnings
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe

import gensim.corpora as corpora
from gensim.models.coherencemodel import CoherenceModel

warnings.filterwarnings('ignore')
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False


class OptimizedBERTopicAnalyzer:
    def __init__(self, csv_path, output_dir="BERTopic_Analysis_Result_Final"):
        self.df = pd.read_csv(csv_path, encoding='utf-8-sig')
        self.df.columns = [c.strip() for c in self.df.columns]
        self.output_dir = output_dir
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        print(f"数据加载成功，共 {len(self.df)} 条数据")

        self.color_palette = {
            0: '#1F4E79',  # 深海蓝
            1: '#A6761D',  # 赭石黄
            2: '#8B3A3A',  # 砖红
            3: '#668B8B',  # 灰绿
            4: '#4A5D23',  # 橄榄绿
            5: '#685369',  # 莫兰迪暗紫
            6: '#D97A3E',  # 陶土橘
            7: '#3E5C76',  # 灰蓝
            8: '#8B658B',  # 梅子色
            -1: '#E5E5E5'  # 噪声底色
        }

    def chinese_process(self, text):
        stopwords = {'的', '了', '在', '是', '我', '有', '和', '就', '都', '而', '及', '与', '着', '或', '一个', '没有',
                     '我们', '你们', '他们', '它们', '因为', '所以', '如果', '但是', '虽然', '并', '且', '去', '来',
                     '长沙', '长沙市', '潮宗街', '街区', '片区', '地方', '区域', '街道', '空间'}
        visual_noise = {'拍摄', '图片', '照片', '画面', '图中', '显示', '镜头', '视角', '全景', '特写', '俯瞰', '正视',
                        '仰视', '位于', '属于', '地处', '可见', '看到', '观察', '发现', '存在', '出现', '展示', '呈现',
                        '表现', '主要', '部分', '整体', '具有', '非常', '十分', '比较', '这种', '那种', '这里', '那里',
                        '一些', '进行', '采用', '使用', '可以', '可能', '不仅', '而且', '以及', '甚至', '方面', '情况',
                        '状态', '天空', '路面', '地面', '道路', '马路', '两侧', '两旁', '中间', '背景', '前景', '远处',
                        '近处', '颜色', '色彩', '风格', '造型', '设计', '装饰', '结构', '样式', '布局', '氛围'}
        stopwords.update(visual_noise)
        words = jieba.lcut(str(text))
        return " ".join([w for w in words if w not in stopwords and len(w) > 1 and not w.isdigit()])

    def run_analysis(self):
        print("1. 正在进行中文分词预处理...")
        text_cols = [c for c in self.df.columns if '文本' in c or 'Text' in c or 'text' in c]
        self.df['processed_text'] = self.df[text_cols[0]].apply(self.chinese_process)
        docs = self.df['processed_text'].tolist()

        print("2. 初始化 BERTopic 模型...")
        umap_model = UMAP(n_neighbors=5, n_components=5, min_dist=0.0, metric='cosine', random_state=42)
        hdbscan_model = HDBSCAN(min_cluster_size=4, metric='euclidean', cluster_selection_method='eom',
                                prediction_data=True)
        vectorizer_model = CountVectorizer(stop_words=None, min_df=2)

        topic_model = BERTopic(
            language="multilingual", umap_model=umap_model, hdbscan_model=hdbscan_model,
            vectorizer_model=vectorizer_model, nr_topics=10, top_n_words=10, calculate_probabilities=True
        )

        print("3. 开始训练模型...")
        topics, probs = topic_model.fit_transform(docs)
        self.df['Topic'] = topics
        freq = topic_model.get_topic_info()

        self.save_results(topic_model, freq)
        self.plot_static_clusters(topic_model)
        self.visualize_results(topic_model)
        self.evaluate_model(topic_model, docs)
        return topic_model

    def plot_static_clusters(self, topic_model):
        print(">> 正在绘制静态聚类气泡图 (学术色板、中心编号、纯白净空)...")
        try:
            topics_info = topic_model.get_topic_info()
            topics_info = topics_info[topics_info['Topic'] != -1]
            if len(topics_info) == 0: return

            freq = topics_info['Count'].values
            topics = topics_info['Topic'].values
            embeddings = topic_model.topic_embeddings_[topics + 1]

            umap_2d = UMAP(n_neighbors=5, n_components=2, min_dist=0.05, metric='cosine',
                           random_state=42).fit_transform(embeddings)
            bubble_colors = [self.color_palette.get(t, '#CCCCCC') for t in topics]

            plt.figure(figsize=(10, 8), dpi=300)
            bubble_sizes = freq * 35 + 600

            plt.scatter(umap_2d[:, 0], umap_2d[:, 1], s=bubble_sizes, c=bubble_colors, alpha=0.9, edgecolors='white',
                        linewidth=1.5)

            for i, topic_id in enumerate(topics):
                plt.annotate(str(topic_id),
                             xy=(umap_2d[i, 0], umap_2d[i, 1]),
                             ha='center', va='center',
                             fontsize=16, fontweight='bold', color='white',
                             path_effects=[pe.withStroke(linewidth=1, foreground="black")])

            plt.title('确认组空间感知聚类 (n_neighbors=5)', fontsize=16, pad=20)
            plt.xticks([])
            plt.yticks([])

            ax = plt.gca()
            for spine in ax.spines.values():
                spine.set_visible(False)

            # 【修改】保存为矢量图 SVG 格式
            save_path = f"{self.output_dir}/5_1_1_static_cluster_map.svg"
            plt.savefig(save_path, bbox_inches='tight', facecolor='white', transparent=False, format='svg')
            print(f"✅ 静态 UMAP 气泡图已完美保存为SVG: {save_path}")

        except Exception as e:
            print(f"⚠️ 静态绘图失败: {e}")

    def visualize_results(self, topic_model):
        try:
            fig_hierarchy = topic_model.visualize_hierarchy()
            fig_hierarchy.update_traces(marker=dict(color='#1F4E79'))
            # 【修改】彻底去除Plotly的灰色底色，替换为纯白
            fig_hierarchy.update_layout(
                title="主题语义层级树状图 (确认组)",
                plot_bgcolor='white',
                paper_bgcolor='white',
                font=dict(color='black')
            )

            # 【修改】尝试输出 SVG 格式
            try:
                fig_hierarchy.write_image(f"{self.output_dir}/viz_hierarchy_dendrogram.svg", format='svg')
                print("✅ 纯白底色层级树状图已保存为SVG")
            except Exception as e:
                print("⚠️ SVG导出失败，请确保终端已运行：pip install -U kaleido")

            fig_hierarchy.write_html(f"{self.output_dir}/viz_hierarchy_dendrogram.html")
        except:
            pass

    def evaluate_model(self, topic_model, docs_list):
        try:
            topic_words = [[word for word, _ in topic_model.get_topics()[t]][:10] for t in
                           topic_model.get_topics().keys() if t != -1]
            if not topic_words: return
            texts = [doc.split() for doc in docs_list]
            dictionary = corpora.Dictionary(texts)
            cm = CoherenceModel(topics=topic_words, texts=texts, dictionary=dictionary, coherence='c_npmi', processes=1)
            all_words = [word for topic in topic_words for word in topic]
            print(f"✅ NPMI Score: {cm.get_coherence():.4f}")
            print(f"✅ Diversity Score: {len(set(all_words)) / len(all_words):.4f}")
        except:
            pass

    def save_results(self, topic_model, freq):
        freq.to_csv(f"{self.output_dir}/1_topic_info.csv", index=False, encoding='utf-8-sig')


if __name__ == "__main__":
    csv_path = r"文件路径：Street view text.csv"
    analyzer = OptimizedBERTopicAnalyzer(csv_path)
    analyzer.run_analysis()