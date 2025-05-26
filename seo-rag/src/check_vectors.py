import os
from dotenv import load_dotenv
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
import logging
from collections import defaultdict
from pinecone import Pinecone
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import io
from contextlib import redirect_stdout
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文字体
matplotlib.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def analyze_vectors(vectors: List[Dict[str, Any]]) -> None:
    """分析向量数据

    Args:
        vectors: 向量数据列表
    """
    print("\n=== 向量数据结构分析 ===")
    print("=" * 80)

    # 打印第一个向量的完整结构
    if vectors:
        first_vector = vectors[0]
        print("\n向量基本结构:")
        print("-" * 40)
        print(f"向量ID: {first_vector['id']}")
        print(f"向量分数: {first_vector.get('score', 0):.4f}")

        print("\n元数据字段:")
        print("-" * 40)
        for key, value in first_vector['metadata'].items():
            print(f"{key}: {value} ({type(value).__name__})")

        print("\n所有可用字段列表:")
        print("-" * 40)
        all_fields = set()
        for vector in vectors:
            all_fields.update(vector['metadata'].keys())
        print("元数据字段:", sorted(list(all_fields)))

        print("\n数据类型统计:")
        print("-" * 40)
        type_stats = defaultdict(int)
        for vector in vectors:
            for value in vector['metadata'].values():
                type_stats[type(value).__name__] += 1
        for type_name, count in type_stats.items():
            print(f"{type_name}: {count}个值")

    print("\n=== 数据结构分析结束 ===")
    print("=" * 80)

    # 转换为DataFrame
    df = pd.DataFrame([v['metadata'] for v in vectors])

    # 检查必需字段是否存在
    required_fields = ['avg_rankning', 'total_clicks',
                       'total_impressions', 'month_start', 'query']
    missing_fields = [
        field for field in required_fields if field not in df.columns]

    if missing_fields:
        print(f"\n错误: 缺少必需字段: {', '.join(missing_fields)}")
        print("可用的字段有:", sorted(list(df.columns)))
        return

    # 处理日期字段
    try:
        df['month_start'] = pd.to_datetime(df['month_start'], errors='coerce')
        df['month'] = df['month_start'].dt.to_period('M')
    except Exception as e:
        print(f"处理日期时出错: {str(e)}")
        return

    # 排名分布分析
    try:
        df['ranking'] = pd.cut(df['avg_rankning'],
                               bins=[0, 10, 20, 50, 100, float('inf')],
                               labels=['1-10', '11-20', '21-50', '51-100', '>100'])
        ranking_dist = df['ranking'].value_counts().sort_index()

        print("\n1. 排名分布分析")
        print("-" * 40)
        print(ranking_dist)
    except Exception as e:
        print(f"排名分析时出错: {str(e)}")

    # 月度趋势分析
    try:
        monthly_stats = df.groupby('month').agg({
            'total_clicks': 'sum',
            'total_impressions': 'sum',
            'avg_rankning': 'mean'
        }).round(2)

        print("\n2. 月度趋势分析")
        print("-" * 40)
        print(monthly_stats)
    except Exception as e:
        print(f"月度趋势分析时出错: {str(e)}")

    # 表现最好的关键词
    try:
        top_keywords = df.nlargest(5, 'total_clicks')[
            ['query', 'total_clicks', 'total_impressions', 'avg_rankning', 'month_start']]

        print("\n3. 表现最好的关键词 (点击量前5)")
        print("-" * 40)
        print(top_keywords)
    except Exception as e:
        print(f"分析表现最好的关键词时出错: {str(e)}")

    # 表现最差的关键词
    try:
        worst_keywords = df.nlargest(5, 'avg_rankning')[
            ['query', 'total_clicks', 'total_impressions', 'avg_rankning', 'month_start']]

        print("\n4. 表现最差的关键词 (排名后5)")
        print("-" * 40)
        print(worst_keywords)
    except Exception as e:
        print(f"分析表现最差的关键词时出错: {str(e)}")

    print("\n=== 分析报告结束 ===")
    print("=" * 80)


def generate_client_report(vectors: List[Dict[str, Any]],
                           output_dir: str = "reports",
                           report_title: str = "SEO Keyword Analysis",
                           client_name: str = "Genesis",
                           report_date: Optional[str] = None) -> str:
    """生成客户端SEO分析报告

    Args:
        vectors: 向量数据列表
        output_dir: 报告输出目录
        report_title: 报告标题
        client_name: 客户名称
        report_date: 报告日期（可选，默认使用当前日期）

    Returns:
        str: 报告文件路径
    """
    try:
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)

        # 设置报告日期
        if report_date is None:
            report_date = datetime.now().strftime("%Y-%m-%d")

        # 转换为DataFrame
        df = pd.DataFrame([v['metadata'] for v in vectors])

        # 创建报告文件
        report_path = os.path.join(
            output_dir, f"SEO_Report_{report_date}.html")

        # 生成报告内容
        report_content = []
        report_content.append(f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <title>{report_title}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                h1, h2, h3 {{ color: #2c3e50; }}
                .section {{ margin: 20px 0; padding: 20px; background: #f8f9fa; border-radius: 5px; }}
                .highlight {{ color: #e74c3c; }}
                .chart {{ margin: 20px 0; text-align: center; }}
                table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
                th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }}
                th {{ background-color: #f8f9fa; }}
                .summary {{ background: #e8f4f8; padding: 15px; border-radius: 5px; }}
            </style>
        </head>
        <body>
            <h1>{report_title}</h1>
            <div class="summary">
                <p><strong>客户名称：</strong>{client_name}</p>
                <p><strong>报告日期：</strong>{report_date}</p>
                <p><strong>数据范围：</strong>{df['month_start'].min()} 至 {df['month_start'].max()}</p>
                <p><strong>关键词总数：</strong>{len(df['query'].unique())}</p>
            </div>
        """)

        # 1. 关键指标概览
        report_content.append("""
            <div class="section">
                <h2>1. 关键指标概览</h2>
        """)

        # 计算关键指标
        total_clicks = df['total_clicks'].sum()
        total_impressions = df['total_impressions'].sum()
        avg_ranking = df['avg_rankning'].mean()
        ctr = (total_clicks / total_impressions *
               100) if total_impressions > 0 else 0

        report_content.append(f"""
                <table>
                    <tr>
                        <th>指标</th>
                        <th>数值</th>
                        <th>环比变化</th>
                    </tr>
                    <tr>
                        <td>总点击量</td>
                        <td>{total_clicks:,.0f}</td>
                        <td class="highlight">{df['clicks_mom'].mean():.1f}%</td>
                    </tr>
                    <tr>
                        <td>总展示量</td>
                        <td>{total_impressions:,.0f}</td>
                        <td class="highlight">{df['impressions_mom'].mean():.1f}%</td>
                    </tr>
                    <tr>
                        <td>平均排名</td>
                        <td>{avg_ranking:.1f}</td>
                        <td class="highlight">{df['rank_mom'].mean():.1f}</td>
                    </tr>
                    <tr>
                        <td>点击率(CTR)</td>
                        <td>{ctr:.2f}%</td>
                        <td>-</td>
                    </tr>
                </table>
            </div>
        """)

        # 3. 排名分布分析
        report_content.append("""
            <div class="section">
                <h2>2. 排名分布分析</h2>
        """)

        # 生成排名分布图
        plt.figure(figsize=(10, 6))
        df['ranking_group'] = pd.cut(df['avg_rankning'],
                                     bins=[0, 10, 20, 50, 100, float('inf')],
                                     labels=['1-10', '11-20', '21-50', '51-100', '>100'])
        ranking_dist = df['ranking_group'].value_counts().sort_index()
        # # 2. 意图分布分析
        # report_content.append("""
        #     <div class="section">
        #         <h2>2. 搜索意图分布</h2>
        # """)

        # # 生成意图分布图
        # plt.figure(figsize=(10, 6))
        # intent_dist = df['intent'].value_counts()
        # intent_dist.plot(kind='pie', autopct='%1.1f%%')
        # plt.title('搜索意图分布')
        # plt.ylabel('')

        # # 保存图表
        # intent_chart_path = os.path.join(output_dir, 'intent_distribution.png')
        # plt.savefig(intent_chart_path)
        # plt.close()

        # report_content.append(f"""
        #         <div class="chart">
        #             <img src="intent_distribution.png" alt="搜索意图分布" style="max-width: 100%;">
        #         </div>
        #         <table>
        #             <tr>
        #                 <th>意图类型</th>
        #                 <th>关键词数量</th>
        #                 <th>占比</th>
        #             </tr>
        # """)

        # for intent, count in intent_dist.items():
        #     percentage = (count / len(df)) * 100
        #     report_content.append(f"""
        #             <tr>
        #                 <td>{intent}</td>
        #                 <td>{count}</td>
        #                 <td>{percentage:.1f}%</td>
        #             </tr>
        #     """)

        # report_content.append("</table></div>")

        # 3. 排名分布分析
        report_content.append("""
            <div class="section">
                <h2>3. 排名分布分析</h2>
        """)

        # 生成排名分布图
        plt.figure(figsize=(10, 6))
        df['ranking_group'] = pd.cut(df['avg_rankning'],
                                     bins=[0, 10, 20, 50, 100, float('inf')],
                                     labels=['1-10', '11-20', '21-50', '51-100', '>100'])
        ranking_dist = df['ranking_group'].value_counts().sort_index()
        ranking_dist.plot(kind='bar')
        plt.title('关键词排名分布')
        plt.xlabel('排名区间')
        plt.ylabel('关键词数量')

        # 保存图表
        ranking_chart_path = os.path.join(
            output_dir, 'ranking_distribution.png')
        plt.savefig(ranking_chart_path)
        plt.close()

        report_content.append(f"""
                <div class="chart">
                    <img src="ranking_distribution.png" alt="排名分布" style="max-width: 100%;">
                </div>
                <table>
                    <tr>
                        <th>排名区间</th>
                        <th>关键词数量</th>
                        <th>占比</th>
                    </tr>
        """)

        for rank, count in ranking_dist.items():
            percentage = (count / len(df)) * 100
            report_content.append(f"""
                    <tr>
                        <td>{rank}</td>
                        <td>{count}</td>
                        <td>{percentage:.1f}%</td>
                    </tr>
            """)

        report_content.append("</table></div>")

        # 4. 表现最佳/最差的关键词
        report_content.append("""
            <div class="section">
                <h2>4. 关键词表现分析</h2>
                <h3>4.1 表现最佳的关键词（点击量前5）</h3>
        """)

        top_keywords = df.nlargest(5, 'total_clicks')
        report_content.append("""
                <table>
                    <tr>
                        <th>关键词</th>
                        <th>点击量</th>
                        <th>展示量</th>
                        <th>平均排名</th>
                        <th>意图</th>
                    </tr>
        """)

        for _, row in top_keywords.iterrows():
            report_content.append(f"""
                    <tr>
                        <td>{row['query']}</td>
                        <td>{row['total_clicks']:,.0f}</td>
                        <td>{row['total_impressions']:,.0f}</td>
                        <td>{row['avg_rankning']:.1f}</td>
                   
                    </tr>
            """)

        report_content.append("""
                </table>
                <h3>4.2 需要优化的关键词（排名后5）</h3>
                <table>
                    <tr>
                        <th>关键词</th>
                        <th>点击量</th>
                        <th>展示量</th>
                        <th>平均排名</th>
                  
                    </tr>
        """)

        worst_keywords = df.nlargest(5, 'avg_rankning')
        for _, row in worst_keywords.iterrows():
            report_content.append(f"""
                    <tr>
                        <td>{row['query']}</td>
                        <td>{row['total_clicks']:,.0f}</td>
                        <td>{row['total_impressions']:,.0f}</td>
                 
                    </tr>
            """)

        report_content.append("</table></div>")

        # 5. 环比变化分析
        report_content.append("""
            <div class="section">
                <h2>5. 环比变化分析</h2>
        """)

        # 生成环比变化图
        plt.figure(figsize=(12, 6))
        df['month'] = pd.to_datetime(df['month_start']).dt.to_period('M')
        monthly_stats = df.groupby('month').agg({
            'total_clicks': 'sum',
            'total_impressions': 'sum',
            'avg_rankning': 'mean'
        })

        # 绘制点击量和展示量的环比变化
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

        # 点击量环比
        clicks_mom = monthly_stats['total_clicks'].pct_change() * 100
        clicks_mom.plot(kind='bar', ax=ax1)
        ax1.set_title('点击量环比变化')
        ax1.set_xlabel('月份')
        ax1.set_ylabel('环比变化率(%)')

        # 排名环比
        rank_mom = monthly_stats['avg_rankning'].diff() * -1
        rank_mom.plot(kind='bar', ax=ax2)
        ax2.set_title('排名环比变化')
        ax2.set_xlabel('月份')
        ax2.set_ylabel('排名变化')

        plt.tight_layout()

        # 保存图表
        mom_chart_path = os.path.join(output_dir, 'monthly_changes.png')
        plt.savefig(mom_chart_path)
        plt.close()

        report_content.append(f"""
                <div class="chart">
                    <img src="monthly_changes.png" alt="月度环比变化" style="max-width: 100%;">
                </div>
            </div>
        """)

        # 结束HTML
        report_content.append("""
            </body>
        </html>
        """)

        # 写入报告文件
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report_content))

        print(f"报告已生成：{report_path}")
        return report_path

    except Exception as e:
        print(f"生成报告时出错: {str(e)}")
        raise


def main():
    # 加载环境变量
    load_dotenv()

    # 获取 API key
    api_key = os.getenv(
        "PINECONE_API_KEY", "YOUR_API")

    # 初始化Pinecone
    pc = Pinecone(api_key=api_key)

    # 获取索引
    index_name = "quickstart-py"
    index = pc.Index(index_name)

    # 获取所有向量
    print("正在获取所有向量...")
    all_vectors = []
    namespace = "seo-full"
    # 获取所有向量ID
    vector_ids = index.describe_index_stats(
    )['namespaces'][namespace]['vector_count']
    print(f"命名空间 '{namespace}' 中共有 {vector_ids} 条向量")

    # 分批获取所有向量
    batch_size = 1000
    for i in range(0, vector_ids, batch_size):
        batch_ids = [f"doc_{j}" for j in range(
            i, min(i + batch_size, vector_ids))]
        results = index.fetch(ids=batch_ids, namespace=namespace)

        # 更新响应处理方式
        for id, vector in results.vectors.items():
            all_vectors.append({
                'id': id,
                'metadata': vector.metadata,
                'score': 0  # 因为没有进行相似度搜索，所以分数为0
            })
    analyze_vectors(all_vectors)
    # 生成客户端报告
    # try:
    #     report_path = generate_client_report(
    #         vectors=all_vectors,
    #         output_dir="reports",
    #         report_title="SEO关键词分析报告",
    #         client_name="医疗美容诊所",
    #         report_date=datetime.now().strftime("%Y-%m-%d")
    #     )
    #     print(f"报告生成成功：{report_path}")
    # except Exception as e:
    #     print(f"生成报告失败：{str(e)}")


if __name__ == "__main__":
    main()
