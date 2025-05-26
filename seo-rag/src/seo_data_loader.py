from typing import List, Dict, Optional
import pandas as pd
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain.schema import Document
from datetime import datetime
import logging
import os
# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SEOCSVLoader:
    """SEO 数据加载器，专门用于处理 SEO 相关的 CSV 数据"""

    def __init__(self, file_path: str):
        """初始化 SEO 数据加载器

        Args:
            file_path: CSV 文件路径
        """
        self.file_path = file_path
        self.data = None
        self.required_columns = [
            'site_url',
            'month_start',
            'query',
            'total_clicks',
            'total_impressions',
            'avg_rankning'
        ]

    def load_and_process(self) -> List[Document]:
        """加载并处理 CSV 数据

        Returns:
            List[Document]: 处理后的文档列表
        """
        try:
            # 使用 pandas 读取 CSV 文件
            logger.info(f"正在读取 CSV 文件: {self.file_path}")
            df = pd.read_csv(self.file_path)

            # 验证必要的列是否存在
            missing_columns = [
                col for col in self.required_columns if col not in df.columns]
            if missing_columns:
                raise ValueError(f"CSV 文件缺少必要的列: {missing_columns}")

            # 数据预处理
            df = self._preprocess_data(df)

            # 转换为文档列表
            documents = self._convert_to_documents(df)
            logger.info(f"成功加载 {len(documents)} 条数据")

            return documents

        except Exception as e:
            logger.error(f"加载数据时出错: {str(e)}")
            raise

    def _preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """预处理数据

        Args:
            df: 原始数据 DataFrame

        Returns:
            pd.DataFrame: 处理后的数据
        """
        try:
            # 复制数据
            df = df.copy()

            # 1. 数据类型转换
            df['month_start'] = pd.to_datetime(df['month_start'])
            numeric_columns = ['total_clicks',
                               'total_impressions', 'avg_rankning']
            for col in numeric_columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

            # 2. 计算环比变化
            df = self._calculate_mom_changes(df)

            # 3. 添加默认意图分类
            if 'intent' not in df.columns:
                df['intent'] = 'informational'  # 直接设置默认意图

            return df

        except Exception as e:
            logger.error(f"数据预处理失败: {str(e)}")
            raise

    def _calculate_mom_changes(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算环比变化

        Args:
            df: 原始数据 DataFrame

        Returns:
            pd.DataFrame: 添加环比变化后的数据
        """
        try:
            # 按关键词分组
            grouped = df.groupby('query')

            # 计算点击量环比
            df['clicks_mom'] = grouped['total_clicks'].transform(
                lambda x: ((x - x.shift(1)) / x.shift(1).replace(0, 1)) * 100
            ).fillna(0).replace([float('inf'), float('-inf')], 0)

            # 计算展示量环比
            df['impressions_mom'] = grouped['total_impressions'].transform(
                lambda x: ((x - x.shift(1)) / x.shift(1).replace(0, 1)) * 100
            ).fillna(0).replace([float('inf'), float('-inf')], 0)

            # 计算排名环比（不需要处理除以0的情况，因为只是差值）
            df['rank_mom'] = grouped['avg_rankning'].transform(
                lambda x: x.diff() * -1  # 排名上升为正值
            ).fillna(0)

            return df

        except Exception as e:
            logger.error(f"计算环比变化失败: {str(e)}")
            raise

    def _convert_to_documents(self, df: pd.DataFrame) -> List[Document]:
        """将 DataFrame 转换为文档列表

        Args:
            df: 处理后的数据 DataFrame

        Returns:
            List[Document]: 文档列表
        """
        documents = []

        for _, row in df.iterrows():
            # 构建文档内容
            content_parts = [
                f"关键词: {row['query']}",
                f"时间: {row['month_start'].strftime('%Y-%m-%d')}",
                f"搜索意图: {row['intent']}",
                f"数据表现:",
                f"- 点击量: {row['total_clicks']:.0f} (环比: {row['clicks_mom']:.1f}%)",
                f"- 展示量: {row['total_impressions']:.0f} (环比: {row['impressions_mom']:.1f}%)",
                f"- 平均排名: {row['avg_rankning']:.1f} (环比: {row['rank_mom']:.1f}%)"
            ]

            # 合并内容
            content = "\n".join(content_parts)

            # 构建元数据
            metadata = {
                'query': row['query'],
                'month_start': row['month_start'].strftime('%Y-%m-%d'),
                'source': row.get('site_url', 'unknown'),
                'total_clicks': float(row['total_clicks']),
                'total_impressions': float(row['total_impressions']),
                'avg_rankning': float(row['avg_rankning']),

            }

            # 创建文档
            doc = Document(
                page_content=content,
                metadata=metadata
            )
            documents.append(doc)

        return documents


def load_seo_data(file_path: str) -> List[Document]:
    """加载 SEO 数据的便捷函数

    Args:
        file_path: CSV 文件路径

    Returns:
        List[Document]: 处理后的文档列表
    """
    loader = SEOCSVLoader(file_path)
    return loader.load_and_process()


if __name__ == "__main__":
    # 测试代码
    try:
        # 替换为实际的 CSV 文件路径
        file_path = os.path.join('data', 'keywords_processed.csv') 
        documents = load_seo_data(file_path)

        # 打印前两个文档作为示例
        for doc in documents[:2]:
            print("\n文档内容:")
            print(doc.page_content)
            print("\n元数据:")
            for key, value in doc.metadata.items():
                print(f"{key}: {value}")
            print("-" * 80)

    except Exception as e:
        logger.error(f"测试过程中出错: {str(e)}")
