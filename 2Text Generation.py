"""
潮宗街街景分析系统 - 使用推荐的 gemini-flash-latest 模型
基于新API密钥和可用模型
"""
import os
import google.generativeai as genai
from PIL import Image
import pandas as pd
import time
import json
from datetime import datetime
import traceback

# ===================== 代理配置 =====================
# 根据您的Clash设置
CLASH_PROXY = "http://127.0.0.1:7890"
os.environ['HTTP_PROXY'] = CLASH_PROXY
os.environ['HTTPS_PROXY'] = CLASH_PROXY
os.environ['ALL_PROXY'] = CLASH_PROXY

# ===================== 提示词模板 =====================
PROMPT_TEMPLATES = {
    "default": """你是一位具有城市规划、建筑学和历史文化保护专业背景的研究者，正在对长沙潮宗街历史文化街区进行深入研究。

请对这张街景图片进行专业分析：

## 第一部分：研究区域背景概况
1. 空间定位：描述此点在潮宗街街区中的可能位置（主街/支巷/节点空间等）
2. 街区肌理：分析可见的街区肌理特征（路网结构、地块划分、空间尺度）
3. 功能混合：识别当前功能（商业/居住/文化/过渡/公共服务等）及其混合模式
4. 更新状态：评估街区的保护更新状况（原真性/干预程度/新旧协调）

## 第二部分：城市规划专业知识视角
### 2.1 空间形态特征
- 界面分析：街道界面连续性、透明度、退线关系
- 尺度比例：街巷高宽比(D/H)、空间围合感
- 空间序列：前景-中景-背景的层次组织
- 视线通廊：视觉通透性与空间引导

### 2.2 建筑形态与风貌
- 建筑类型：识别主要建筑类型（传统民居/历史建筑/现代建筑/临时建筑）
- 风貌特征：材料（青砖/木材/石材/现代材料）、色彩、细部装饰
- 历史分层：不同时期建筑叠加的痕迹与协调性
- 保护状况：历史建筑保存完整性、干预措施的适宜性

### 2.3 环境与设施
- 街道家具：路灯、座椅、标识系统、垃圾桶
- 绿化景观：行道树、花池、垂直绿化
- 地面铺装：材料、图案、历史特征保留
- 公共艺术：壁画、雕塑、文化展示元素

### 2.4 活动与活力
- 静态活动：停留、休憩、社交的行为特征
- 动态活动：步行、骑行、商业活动的频率与分布
- 时间痕迹：临时性使用痕迹、更新施工状态
- 场所氛围：历史文化氛围感知、空间温度评价

### 2.5 城市设计评估
- 可读性：空间导向的清晰度
- 可达性：进入和使用空间的便利性
- 舒适性：物理环境与心理感受
- 包容性：对不同人群的友好程度
- 可持续性：环境友好特征体现

## 第三部分：专业见解
1. 核心价值识别：最有价值的空间要素或历史特征
2. 潜在问题诊断：保护与发展中的矛盾与挑战
3. 优化建议：基于"微更新"理念的具体改善措施
4. 研究线索：值得深入研究的学术问题

请用中文、分点清晰回答，使用城市规划专业术语，描述具体、客观。""",

    "风貌过渡区": """[针对风貌过渡区特化提示]
你正在分析潮宗街风貌过渡区的街景，这是传统风貌与现代城市之间的缓冲区域。

重点关注：
1. 过渡特征：新旧建筑如何衔接？空间如何渐变？
2. 风貌协调：传统元素与现代元素的整合方式
3. 功能转型：功能混合与转换的痕迹
4. 空间韧性：过渡空间的适应性与弹性
""",

    "商业活力区": """[针对商业活力区特化提示]
你正在分析潮宗街商业活力区的街景，这是街区经济活动的核心。

重点关注：
1. 业态特征：商业类型、店面形态、招牌广告
2. 人流特征：人流密度、行为模式、停留意愿
3. 商业界面：店铺透明度、外摆空间、商业氛围
4. 经济活力：营业时间、商品陈列、消费场景
""",

    "生活区": """[针对生活区特化提示]
你正在分析潮宗街生活区的街景，这是居民日常生活的场所。

重点关注：
1. 居住环境：建筑宜居性、私密性保障
2. 社区设施：便民服务、公共活动空间
3. 邻里关系：共享空间、社交节点
4. 生活痕迹：晾晒、绿植、儿童活动痕迹
""",

    "文化核心区": """[针对文化核心区特化提示]
你正在分析潮宗街文化核心区的街景，这是历史文化遗产最集中的区域。

重点关注：
1. 历史原真性：历史建筑保存状况、材料工艺
2. 文化表达：文化符号、历史叙事呈现
3. 保护措施：保护工程的技术适宜性
4. 展示利用：文化遗产的活化利用方式
"""
}


# ===================== Gemini分析器 =====================
class GeminiStreetViewAnalyzer:
    def __init__(self, api_key, model_name="gemini-flash-latest"):
        """初始化Gemini分析器，使用推荐的模型"""
        try:
            print(f"初始化Gemini分析器，使用模型: {model_name}")

            # 配置Gemini API
            genai.configure(api_key=api_key)

            # 使用推荐的模型
            self.model = genai.GenerativeModel(model_name)

            # 简单测试模型是否能工作
            print("测试模型连接...")
            test_response = self.model.generate_content("用一句话介绍长沙")

            if test_response.text:
                print(f"✅ 模型 {model_name} 初始化成功")
                print(f"   测试响应: {test_response.text[:100]}...")
            else:
                print("⚠ 模型测试返回空响应")

            self.results = []
            self.api_key = api_key
            self.model_name = model_name

        except Exception as e:
            print(f"❌ Gemini初始化失败: {e}")
            print("请检查:")
            print("1. API密钥是否正确")
            print("2. 网络代理是否正常工作")
            print("3. 模型名称是否正确")
            raise

    def analyze_single_image(self, image_path, zone="default", max_retries=3):
        """分析单张图片，带重试机制"""
        for attempt in range(max_retries):
            try:
                print(f"正在分析: {os.path.basename(image_path)} (尝试 {attempt + 1}/{max_retries})")

                # 1. 加载图片
                img = Image.open(image_path)

                # 2. 获取提示词
                prompt = PROMPT_TEMPLATES.get(zone, PROMPT_TEMPLATES["default"])

                # 3. 调用API
                start_time = time.time()
                response = self.model.generate_content([prompt, img])
                elapsed = time.time() - start_time

                # 4. 构建结果
                result = {
                    "图片路径": image_path,
                    "文件名": os.path.basename(image_path),
                    "所属分区": zone,
                    "分析时间": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "Gemini响应": response.text,
                    "状态": "成功",
                    "分析耗时": f"{elapsed:.2f}秒",
                    "尝试次数": attempt + 1
                }

                print(f"✅ 分析完成: {os.path.basename(image_path)} ({elapsed:.2f}秒)")
                return result

            except Exception as e:
                error_msg = str(e)
                print(f"❌ 分析失败 (尝试 {attempt + 1}/{max_retries}): {error_msg[:100]}")

                if attempt < max_retries - 1:
                    # 如果是配额错误，等待更长时间
                    if "429" in error_msg or "quota" in error_msg.lower():
                        wait_time = 10 * (attempt + 1)  # 10秒, 20秒, 30秒
                        print(f"   配额限制，等待 {wait_time} 秒后重试...")
                        time.sleep(wait_time)
                    else:
                        wait_time = 2 * (attempt + 1)  # 2秒, 4秒, 6秒
                        print(f"   等待 {wait_time} 秒后重试...")
                        time.sleep(wait_time)

        # 所有重试都失败
        error_result = {
            "图片路径": image_path,
            "文件名": os.path.basename(image_path),
            "所属分区": zone,
            "分析时间": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "Gemini响应": f"分析失败: 达到最大重试次数 {max_retries}",
            "状态": "失败",
            "分析耗时": "N/A"
        }
        return error_result

    def batch_analyze(self, csv_path, output_dir="分析结果"):
        """批量分析图片"""
        try:
            # 读取元数据
            df = pd.read_csv(csv_path, encoding='utf-8-sig')
            print(f"读取到 {len(df)} 条记录")

        except Exception as e:
            print(f"❌ 读取CSV文件失败: {e}")
            return

        # 创建输出目录
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"{output_dir}_{timestamp}"
        os.makedirs(output_dir, exist_ok=True)

        # 统计信息
        stats = {
            "total": len(df),
            "success": 0,
            "failed": 0,
            "start_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "model": self.model_name
        }

        all_results = []

        print(f"\n开始批量分析，保存到: {output_dir}")
        print("=" * 60)

        for idx, row in df.iterrows():
            try:
                # 获取图片路径和分区
                img_path = row.get("完整路径", "")
                zone = row.get("分区", "default")
                point_id = row.get("图片ID", f"IMG_{idx}")

                if not img_path or not os.path.exists(img_path):
                    print(f"[{idx + 1}/{len(df)}] ❌ 图片不存在: {img_path}")
                    stats["failed"] += 1
                    continue

                print(f"[{idx + 1}/{len(df)}] 分析: {point_id} ({zone})")

                # 分析单张图片
                result = self.analyze_single_image(img_path, zone)

                # 合并元数据
                combined_result = {**row.to_dict(), **result}
                all_results.append(combined_result)

                if result["状态"] == "成功":
                    stats["success"] += 1

                    # 保存单个结果
                    result_file = os.path.join(output_dir, f"{point_id}.json")
                    with open(result_file, 'w', encoding='utf-8') as f:
                        json.dump(combined_result, f, ensure_ascii=False, indent=2)
                else:
                    stats["failed"] += 1

                # 每处理5张保存一次进度
                if len(all_results) % 5 == 0:
                    self._save_progress(all_results, output_dir, stats)
                    print(f"  进度: 已处理 {len(all_results)}/{len(df)}")

                # 延迟避免API限制
                time.sleep(1.5)

            except Exception as e:
                print(f"[{idx + 1}/{len(df)}] ❌ 处理异常: {e}")
                stats["failed"] += 1

        # 最终保存
        self._save_final_results(all_results, output_dir, stats)

        return stats

    def _save_progress(self, results, output_dir, stats):
        """保存进度"""
        progress_file = os.path.join(output_dir, "progress.json")
        with open(progress_file, 'w', encoding='utf-8') as f:
            json.dump({
                "processed": len(results),
                "stats": stats,
                "last_update": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }, f, ensure_ascii=False, indent=2)

    def _save_final_results(self, results, output_dir, stats):
        """保存最终结果"""
        # 更新统计信息
        stats["end_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        stats["duration"] = (datetime.now() - datetime.strptime(stats["start_time"], "%Y-%m-%d %H:%M:%S")).seconds

        # 保存统计
        stats_file = os.path.join(output_dir, "分析统计.json")
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)

        # 保存汇总CSV
        if results:
            df_results = pd.DataFrame(results)
            summary_file = os.path.join(output_dir, "分析结果汇总.csv")
            df_results.to_csv(summary_file, index=False, encoding='utf-8-sig')

            # 按分区保存
            for zone in df_results["分区"].unique():
                zone_df = df_results[df_results["分区"] == zone]
                zone_file = os.path.join(output_dir, f"{zone}_分析结果.csv")
                zone_df.to_csv(zone_file, index=False, encoding='utf-8-sig')

        # 输出结果
        print("\n" + "=" * 60)
        print("🎉 批量分析完成！")
        print("=" * 60)
        print(f"总处理数: {stats['total']}")
        print(f"成功: {stats['success']}")
        print(f"失败: {stats['failed']}")
        print(f"成功率: {stats['success'] / stats['total'] * 100:.1f}%" if stats['total'] > 0 else "0%")
        print(f"总耗时: {stats['duration']}秒")
        print(f"结果目录: {output_dir}")


# ===================== 主程序 =====================
def main():
    print("=" * 60)
    print("潮宗街街景分析系统 v4.0")
    print("使用模型: gemini-flash-latest")
    print("=" * 60)

    # 1. 输入新申请的API密钥
    API_KEY = input("请输入您新申请的Google AI Studio API密钥: ").strip()

    if not API_KEY:
        print("❌ 未提供API密钥")
        return

    # 2. 初始化分析器
    try:
        analyzer = GeminiStreetViewAnalyzer(API_KEY, "gemini-flash-latest")
    except Exception as e:
        print(f"分析器初始化失败，请检查以上错误信息")
        return

    # 3. 测试单张图片
    test_option = input("\n是否先测试单张图片？(y/n, 默认y): ").strip().lower()

    if test_option != 'n':
        # 查找测试图片
        search_paths = [
            r"街景图存储路径",
            r"文件路径",
            os.getcwd()
        ]

        test_images = []
        for folder in search_paths:
            if os.path.exists(folder):
                files = os.listdir(folder)
                for file in files:
                    if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                        test_images.append(os.path.join(folder, file))
                        if len(test_images) >= 3:
                            break
                if test_images:
                    break

        if test_images:
            print(f"\n找到 {len(test_images)} 张测试图片:")
            for i, img in enumerate(test_images[:3]):
                print(f"  {i + 1}. {os.path.basename(img)}")

            choice = input(f"\n选择测试图片 (1-{len(test_images)}，默认1): ").strip()
            try:
                choice_idx = int(choice) - 1 if choice else 0
                if 0 <= choice_idx < len(test_images):
                    test_image = test_images[choice_idx]
                else:
                    test_image = test_images[0]
            except:
                test_image = test_images[0]

            zone = input("请输入分区 (风貌过渡区/商业活力区/生活区/文化核心区，默认default): ").strip()
            if not zone:
                zone = "default"

            print(f"\n测试图片: {os.path.basename(test_image)}")
            print(f"分区: {zone}")

            result = analyzer.analyze_single_image(test_image, zone)

            if result["状态"] == "成功":
                print(f"\n✅ 测试成功！")
                print(f"   耗时: {result.get('分析耗时', 'N/A')}")
                print(f"   响应长度: {len(result['Gemini响应'])} 字符")
                print("\n响应预览:")
                print("-" * 40)
                preview = result["Gemini响应"][:500] + "..." if len(result["Gemini响应"]) > 500 else result[
                    "Gemini响应"]
                print(preview)

                # 保存测试结果
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                test_file = f"测试结果_{timestamp}.txt"
                with open(test_file, 'w', encoding='utf-8') as f:
                    f.write(result["Gemini响应"])
                print(f"\n✅ 测试结果已保存: {test_file}")
            else:
                print(f"❌ 测试失败")
        else:
            print("❌ 未找到测试图片，跳过测试")

    # 4. 批量处理
    csv_option = input("\n是否进行批量处理？(y/n, 默认y): ").strip().lower()

    if csv_option != 'n':
        csv_file = "潮宗街街景图片元数据表.csv"

        if os.path.exists(csv_file):
            print(f"\n找到元数据文件: {csv_file}")

            # 显示文件信息
            df = pd.read_csv(csv_file, encoding='utf-8-sig')
            print(f"文件包含 {len(df)} 条记录")
            print("前3条记录:")
            print(df.head(3))

            confirm = input("\n是否开始批量分析？(y/n, 默认y): ").strip().lower()

            if confirm != 'n':
                stats = analyzer.batch_analyze(csv_file)

                if stats["success"] > 0:
                    print(f"\n✅ 批量分析完成！")
                    print(f"   成功分析 {stats['success']} 张图片")
                else:
                    print(f"\n❌ 批量分析失败，没有成功分析的图片")
            else:
                print("取消批量分析")
        else:
            print(f"❌ 元数据文件不存在: {csv_file}")
            print("请确保 '潮宗街街景图片元数据表.csv' 文件在当前目录下")


if __name__ == "__main__":
    main()