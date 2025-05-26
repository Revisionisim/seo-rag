import os
from typing import List, Dict, Optional, Any, Tuple
import pandas as pd
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.base import Embeddings
from langchain_community.vectorstores import FAISS
from langchain_pinecone import PineconeVectorStore
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from langchain.chat_models.base import BaseChatModel
from langchain.schema import Document, HumanMessage, SystemMessage, AIMessage
import numpy as np
import requests
import pinecone
from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.schema import ChatResult, ChatGeneration, BaseMessage
from pydantic import BaseModel, Field
import time  # 添加到文件顶部的导入语句中
import functools
import hashlib
import json
from datetime import datetime, timedelta
from seo_data_loader import SEOCSVLoader, load_seo_data  # 导入 SEO 数据加载器
from check_vectors import analyze_vectors

from contextlib import redirect_stdout
from keyword_time_series import get_keyword_time_series, load_and_prepare_data
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import BaseOutputParser
from langchain_openai import ChatOpenAI
import logging
import re

import argparse
PROCESSED_DATA_PATH = os.path.join('data', 'keywords_processed.csv') 
# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from config import (
    SILICONFLOW_API_KEY,
    EMBEDDING_MODEL,
    PINECONE_API_KEY,
    PINECONE_ENVIRONMENT,
    PINECONE_INDEX_NAME,
    PINECONE_NAMESPACE
)


class SiliconFlowEmbeddings(Embeddings):
    """使用 SiliconFlow API 的嵌入模型实现"""

    def __init__(self, api_key: str, model_name: str = "BAAI/bge-m3"):
        self.api_key = api_key
        self.model_name = model_name
        self.api_url = "https://api.siliconflow.cn/v1/embeddings"
        self.max_retries = 3
        self.retry_delay = 2  # 重试延迟秒数
        self.batch_size = 5   # 减小批处理大小

    def _make_request(self, payload: dict, retry_count: int = 0) -> dict:
        """发送API请求，包含重试逻辑"""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        try:
            response = requests.post(
                self.api_url,
                headers=headers,
                json=payload,
                timeout=30  # 设置超时时间
            )
            response.raise_for_status()
            return response.json()
        except (requests.exceptions.RequestException, ConnectionError) as e:
            if retry_count < self.max_retries:
                print(f"请求失败，正在进行第 {retry_count + 1} 次重试...")
                time.sleep(self.retry_delay * (retry_count + 1))  # 指数退避
                return self._make_request(payload, retry_count + 1)
            raise Exception(f"API请求失败，已重试{self.max_retries}次: {str(e)}")

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """为文档列表生成嵌入向量"""
        embeddings = []
        total_batches = (len(texts) + self.batch_size - 1) // self.batch_size

        for i in range(0, len(texts), self.batch_size):
            batch_texts = texts[i:i + self.batch_size]
            current_batch = (i // self.batch_size) + 1
            print(f"正在处理第 {current_batch}/{total_batches} 批数据...")

            payload = {
                "model": self.model_name,
                "input": batch_texts,
                "encoding_format": "float"
            }

            try:
                result = self._make_request(payload)
                batch_embeddings = [item['embedding']
                                    for item in result['data']]
                embeddings.extend(batch_embeddings)
                print(f"第 {current_batch} 批数据处理完成")
            except Exception as e:
                print(f"处理第 {current_batch} 批数据时出错: {str(e)}")
                raise

        return embeddings

    def embed_query(self, text: str) -> List[float]:
        """为单个查询生成嵌入向量"""
        payload = {
            "model": self.model_name,
            "input": text,
            "encoding_format": "float"
        }

        try:
            result = self._make_request(payload)
            return result['data'][0]['embedding']
        except Exception as e:
            raise Exception(f"生成查询嵌入向量时出错: {str(e)}")


class KeywordDataProcessor:
    def __init__(self):
        self.data = None

    def load_data(self, file_path: str =os.path.join('data', 'keywords_processed.csv') ) -> List[Document]:
        """使用 SEOCSVLoader 加载并处理数据

        Args:
            file_path: GSC 数据 CSV 文件路径

        Returns:
            List[Document]: 处理后的文档列表
        """
        try:
            print(f"正在从 {file_path} 加载数据...")
            # 使用 SEOCSVLoader 加载数据
            documents = load_seo_data(file_path)
            print(f"成功加载 {len(documents)} 条记录")
            return documents

        except Exception as e:
            raise Exception(f"加载数据时出错: {str(e)}")

    def create_documents(self) -> List[Document]:
        """获取已加载的文档

        Returns:
            List[Document]: 文档列表
        """
        if not hasattr(self, 'documents'):
            raise ValueError("未加载数据，请先调用 load_data()")
        return self.documents

    def _generate_trend_description(self, row: pd.Series) -> str:
        """Generate trend description based on performance metrics

        Args:
            row: 包含性能指标的 DataFrame 行

        Returns:
            str: 趋势描述
        """
        trends = []

        # 只保留基于绝对值的趋势判断
        if row['total_clicks'] > 100:
            trends.append(f"点击量较高({row['total_clicks']:.0f})")
        if row['total_impressions'] > 1000:
            trends.append(f"展示量较高({row['total_impressions']:.0f})")
        if row['avg_rankning'] <= 3:
            trends.append(f"排名靠前({row['avg_rankning']:.1f})")

        if not trends:
            return "表现稳定，无明显变化"

        return "；".join(trends)

    def classify_keyword_intent(self, keyword: str) -> Tuple[str, Dict[str, float]]:
        """返回默认的意图分类结果

        Args:
            keyword: 搜索关键词

        Returns:
            Tuple[str, Dict[str, float]]: (默认意图, 默认意图得分)
        """
        # 返回默认值
        default_intent = "informational"
        default_scores = {
            'transactional': 0.0,
            'informational': 1.0,  # 设置默认意图的得分为1.0
            'navigational': 0.0,
            'commercial': 0.0,
            'local': 0.0,
            'other': 0.0
        }
        return default_intent, default_scores

    def process_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """处理数据，包括意图分类和计算环比变化

        Args:
            df: 原始数据 DataFrame

        Returns:
            pd.DataFrame: 处理后的数据
        """
        try:
            # 复制数据
            df = df.copy()

            # 确保必要的列存在
            required_columns = ['query', 'total_clicks',
                                'total_impressions', 'avg_rankning', 'month_start']
            for col in required_columns:
                if col not in df.columns:
                    raise ValueError(f"缺少必要的列: {col}")

            # 重命名列
            df = df.rename(columns={
                'query': 'keyword',
                'total_clicks': 'clicks',
                'total_impressions': 'impressions',
                'avg_rankning': 'avg_ranking'
            })

            # 转换日期格式并添加时间戳
            df['month_start'] = pd.to_datetime(df['month_start'])
            df['month_start_timestamp'] = df['month_start'].apply(
                lambda x: int(x.timestamp() * 1000)
            )

            # 计算 CTR
            df['ctr'] = (df['clicks'] / df['impressions'] * 100).fillna(0)

            # 按关键词分组计算环比变化
            df = self._calculate_mom_changes(df)

            # 修改意图分类部分
            print("使用默认意图分类...")
            intent_results = []
            for keyword in df['keyword'].unique():
                primary_intent, scores = self.classify_keyword_intent(keyword)
                intent_results.append({
                    'keyword': keyword,
                    'intent': primary_intent,
                    'intent_scores': scores,
                    'intent_method': 'default',  # 使用默认方法
                    'intent_explanation': '使用默认意图分类'  # 使用默认解释
                })
                print(f"已分类: {keyword} -> {primary_intent} (默认分类)")

            # 将意图分类结果合并到数据中
            intent_df = pd.DataFrame(intent_results)
            df = df.merge(intent_df, on='keyword', how='left')

            # 确保所有必要的列都存在
            for col in ['intent', 'intent_scores', 'intent_method', 'intent_explanation']:
                if col not in df.columns:
                    df[col] = None

            return df

        except Exception as e:
            print(f"数据处理失败: {str(e)}")
            raise


class ChatSiliconFlowConfig(BaseModel):
    """SiliconFlow 聊天模型配置"""
    api_key: str = Field(..., description="SiliconFlow API密钥")
    model_name: str = Field(
        default="deepseek-ai/DeepSeek-V3", description="模型名称")
    temperature: float = Field(default=0.2, description="温度参数")
    api_url: str = Field(
        default="https://api.siliconflow.cn/v1/chat/completions", description="API URL")


class ChatSiliconFlow(BaseChatModel):
    """自定义的 SiliconFlow 聊天模型实现"""

    config: ChatSiliconFlowConfig = Field(..., description="模型配置")

    def __init__(self, api_key: str, model_name: str = "deepseek-ai/DeepSeek-V3", temperature: float = 0.5):
        """初始化聊天模型

        Args:
            api_key: SiliconFlow API密钥
            model_name: 模型名称
            temperature: 温度参数
        """
        config = ChatSiliconFlowConfig(
            api_key=api_key,
            model_name=model_name,
            temperature=temperature
        )
        super().__init__(config=config)

    @property
    def _llm_type(self) -> str:
        """返回LLM类型"""
        return "siliconflow"

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """实现生成方法"""
        response = self._call(messages, stop=stop,
                              run_manager=run_manager, **kwargs)
        return ChatResult(generations=[ChatGeneration(message=response)])

    def _call(self, messages, stop=None, run_manager=None, **kwargs):
        """实现聊天模型的调用"""
        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json"
        }

        # 转换消息格式
        formatted_messages = []
        for msg in messages:
            if isinstance(msg, SystemMessage):
                formatted_messages.append(
                    {"role": "system", "content": msg.content})
            elif isinstance(msg, HumanMessage):
                formatted_messages.append(
                    {"role": "user", "content": msg.content})
            elif isinstance(msg, AIMessage):
                formatted_messages.append(
                    {"role": "assistant", "content": msg.content})

        payload = {
            "model": self.config.model_name,
            "messages": formatted_messages,
            "temperature": self.config.temperature,
            "max_tokens": 1000
        }

        try:
            response = requests.post(
                self.config.api_url, headers=headers, json=payload)
            response.raise_for_status()
            result = response.json()
            return AIMessage(content=result['choices'][0]['message']['content'])
        except Exception as e:
            raise Exception(f"调用 SiliconFlow API 时出错: {str(e)}")


class CacheManager:
    """缓存管理器"""

    def __init__(self, ttl_seconds: int = 3600):
        self.cache: Dict[str, Tuple[Any, datetime]] = {}
        self.ttl_seconds = ttl_seconds

    def get(self, key: str) -> Optional[Any]:
        """获取缓存值"""
        if key in self.cache:
            value, timestamp = self.cache[key]
            if datetime.now() - timestamp < timedelta(seconds=self.ttl_seconds):
                return value
            else:
                del self.cache[key]
        return None

    def set(self, key: str, value: Any):
        """设置缓存值"""
        self.cache[key] = (value, datetime.now())

    def clear(self):
        """清除缓存"""
        self.cache.clear()


class RAGSystem:
    def __init__(self, embedding_model_name: str = EMBEDDING_MODEL, use_pinecone: bool = True):
        """初始化RAG系统

        Args:
            embedding_model_name: 嵌入模型名称
            use_pinecone: 是否使用 Pinecone 向量存储        """
        self.use_pinecone = use_pinecone
        self.embedding_model_name = embedding_model_name

        # 初始化嵌入模型（使用 SiliconFlow API）
        self.embeddings = SiliconFlowEmbeddings(
            api_key=SILICONFLOW_API_KEY,
            model_name=embedding_model_name
        )

        if use_pinecone:
            # 初始化 Pinecone 向量存储
            try:
                # 先初始化 Pinecone 客户端
                pc = pinecone.Pinecone(
                    api_key=PINECONE_API_KEY,
                    environment=PINECONE_ENVIRONMENT
                )

                # 检查索引是否存在，如果不存在则创建
                if PINECONE_INDEX_NAME not in pc.list_indexes().names():
                    print(f"创建新的 Pinecone 索引: {PINECONE_INDEX_NAME}")
                    pc.create_index(
                        name=PINECONE_INDEX_NAME,
                        dimension=1024,  # BGE-M3 模型的维度
                        metric="cosine",
                        spec={
                            "serverless": {
                                "cloud": "aws",
                                "region": "us-west-2"
                            }
                        }
                    )

                index = pc.Index(PINECONE_INDEX_NAME)

                # 然后创建向量存储
                self.vectorstore = PineconeVectorStore(
                    index=index,
                    embedding=self.embeddings,
                    namespace=PINECONE_NAMESPACE,
                    text_key="query"  # 改用 query 字段作为文本内容
                )
                print(f"成功连接到 Pinecone 索引: {PINECONE_INDEX_NAME}")
            except Exception as e:
                raise Exception(f"初始化 Pinecone 时出错: {str(e)}")
        else:
            # 使用 FAISS 本地向量存储
            self.text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                length_function=len,
            )
            self.vectorstore = None

        # 初始化LLM
        self.llm = ChatSiliconFlow(
            api_key=SILICONFLOW_API_KEY,
            model_name="Qwen/Qwen3-32B",
            temperature=0.1
        )

        # 初始化提示模板
        self.prompt = ChatPromptTemplate.from_template("""
         You are an SEO expert analyzing keyword performance for a medical aesthetics clinic.
         The following keywords have shown declining performance (worsening rankings or decreasing clicks) over the past 4 months:
        
        Context Information:
        {context}
        
        data analysis report:
        {analysis_report}
        
        Please provide detailed analysis and insights based on the context information, focusing on:

        1. Directly answer the user's question
        2. Support your points with data
        3. Provide specific, actionable recommendations
        4. Use professional yet accessible language
    
        Focus on actionable insights that will have the biggest impact on their SEO performance.
        
        Answer:
        """)

        # 初始化缓存管理器
        self.cache_manager = CacheManager()

        # 初始化元数据过滤器为空字典
        self.metadata_filters = {}

        # 加载时序数据
        try:
            self.time_series_df = load_and_prepare_data()
            print("成功加载时序数据")
        except Exception as e:
            print(f"加载时序数据失败: {str(e)}")
            self.time_series_df = None

        self.use_exact_match = False  # 添加精确匹配开关

    # def create_vectorstore(self, documents: List[Document] = None):
    #     """创建或连接向量存储"""
    #     if not self.use_pinecone:
    #         # 使用 FAISS 本地向量存储
    #         if documents is None:
    #             raise ValueError("使用 FAISS 时需要提供文档")
    #         splits = self.text_splitter.split_documents(documents)
    #         self.vectorstore = FAISS.from_documents(splits, self.embeddings)
    #         print("已创建 FAISS 向量存储")

    def setup_rag_chain(self):
        """设置RAG链"""
        if self.vectorstore is None:
            raise ValueError("向量存储尚未初始化，请先调用 create_vectorstore()")

        retriever = self.vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={"k": 5,
                           "score_threshold": 0.5}
        )

        # 构建RAG链
        self.rag_chain = (
            {"context": retriever, "question": RunnablePassthrough()}
            | self.prompt
            | self.llm
            | StrOutputParser()
        )

    def format_embedding_text(self, doc: Document) -> str:
        """优化 embedding 文本格式"""
        metadata = doc.metadata
        return f"""
        Keyword: {metadata['query']}
        Search Intent: {metadata['intent']}
        Performance Metrics:
        - Clicks: {metadata['total_clicks']} 
        - Impressions: {metadata['total_impressions']} 
        - Average Ranking: {metadata['avg_rankning']} 
     
       
        """

    def _generate_cache_key(self, query: str, filters: Optional[Dict] = None) -> str:
        """生成缓存键"""
        cache_data = {
            'query': query,
            'filters': filters or {},
            'timestamp': datetime.now().strftime('%Y%m%d%H')  # 按小时缓存
        }
        return hashlib.md5(json.dumps(cache_data, sort_keys=True).encode()).hexdigest()

    def set_metadata_filters(self, filters: Dict[str, Any]):
        """设置元数据过滤条件

        Args:
            filters: 过滤条件字典，例如：
                {
                    'intent': 'informational',
                    'clicks_mom': {'$gt': 10},
                    'month_start': '2024-01'
                }
        """
        self.metadata_filters = filters

    def _build_filter_query(self) -> Dict:
        """构建 Pinecone 过滤查询"""
        filter_query = {}

        for field, condition in self.metadata_filters.items():
            if isinstance(condition, dict):
                # 处理比较操作符
                for op, value in condition.items():
                    if op == '$gt':
                        filter_query[field] = {'$gt': value}
                    elif op == '$lt':
                        filter_query[field] = {'$lt': value}
                    elif op == '$gte':
                        filter_query[field] = {'$gte': value}
                    elif op == '$lte':
                        filter_query[field] = {'$lte': value}
            else:
                # 处理精确匹配
                filter_query[field] = condition

        return filter_query

    @functools.lru_cache(maxsize=100)
    def _get_cached_embedding(self, text: str) -> List[float]:
        """获取缓存的嵌入向量"""
        embedding = self.embeddings.embed_query(text)
        # 确保返回的向量值都是 float 类型
        return [float(x) for x in embedding]

    def search_with_score(self, query: str, min_score: Optional[float] = None) -> List[Tuple[Document, float]]:
        """带评分的向量检索

        Args:
            query: 查询文本
            min_score: 最小相似度分数阈值

        Returns:
            包含文档和分数的元组列表
        """
        # 检查缓存
        cache_key = self._generate_cache_key(query, self.metadata_filters)
        cached_results = self.cache_manager.get(cache_key)
        if cached_results:
            return cached_results

        # 生成查询向量
        query_vector = self._get_cached_embedding(query)

        # 构建过滤条件
        filter_query = self._build_filter_query()

        try:
            # 执行向量搜索，增加检索数量并降低相似度阈值
            search_results = self.vectorstore.similarity_search_by_vector_with_score(
                embedding=query_vector,
                k=5,  # 增加检索数量
                filter=filter_query if filter_query else None
            )

            # 处理结果
            results = []
            min_threshold = min_score or 0.4  # 降低默认阈值到0.4

            for doc, score in search_results:
                # 检查相似度分数是否达到阈值，并且文档的 'query' 字段包含原始查询关键词
                doc_query = doc.metadata.get('query', '').lower()
                if score >= min_threshold and query.lower() in doc_query:
                    # 确保从文档的 page_content 中获取关键词（如果元数据中没有）
                    metadata = doc.metadata.copy() if doc.metadata else {}
                    if not metadata.get('query') and doc.page_content:
                        metadata['query'] = doc.page_content.strip()

                    # 创建新的文档对象，确保包含所有必要的元数据
                    new_doc = Document(
                        page_content=doc.page_content,
                        metadata=metadata
                    )
                    results.append((new_doc, score))

            # 按相似度分数排序
            results.sort(key=lambda x: x[1], reverse=True)

            # 缓存结果
            self.cache_manager.set(cache_key, results)


            return results

        except Exception as e:
            print(f"搜索过程中出错: {str(e)}")
            return []

    def _get_all_vectors(self, keyword: Optional[str] = None) -> List[Dict[str, Any]]:
        """获取所有向量数据，如果提供了关键词则只返回相关数据

        Args:
            keyword: 可选的关键词过滤条件

        Returns:
            List[Dict[str, Any]]: 向量数据列表
        """
        try:
            all_vectors = []
            namespace = PINECONE_NAMESPACE

            # 获取所有向量ID
            vector_ids = self.vectorstore.index.describe_index_stats(
            )['namespaces'][namespace]['vector_count']

            # 分批获取所有向量
            batch_size = 1000
            for i in range(0, vector_ids, batch_size):
                batch_ids = [f"doc_{j}" for j in range(
                    i, min(i + batch_size, vector_ids))]
                results = self.vectorstore.index.fetch(
                    ids=batch_ids, namespace=namespace)

                for id, vector in results.vectors.items():
                    # 如果提供了关键词，只添加包含该关键词的向量
                    if keyword:
                        if keyword.lower() in vector.metadata.get('query', '').lower():
                            all_vectors.append({
                                'id': id,
                                'metadata': vector.metadata,
                            })
                    else:
                        all_vectors.append({
                            'id': id,
                            'metadata': vector.metadata,
                        })

            return all_vectors

        except Exception as e:
            print(f"获取向量数据时出错: {str(e)}")
            return []

    def get_analysis_report(self, keyword: Optional[str] = None) -> str:
        """Get vector analysis report, if keyword is provided only return analysis for related keywords

        Args:
            keyword: Optional keyword filter

        Returns:
            str: Analysis report content
        """
        try:
            # Read data directly from CSV file
            try:
                df = pd.read_csv(PROCESSED_DATA_PATH)
                if df.empty:
                    return "Unable to read data file"
            except Exception as e:
                print(f"Error reading CSV file: {str(e)}")
                return "Unable to read data file"

            # Exclude branded keywords
            branded_keywords = ['genesis med spa', 'genesis medspa', 'medspa']
            df_non_branded = df[~df['query'].str.lower().str.contains(
                '|'.join(branded_keywords), na=False)]

            # If keyword provided, only keep data containing that keyword
            if keyword:
                df_non_branded = df_non_branded[df_non_branded['query'].str.lower().str.contains(
                    keyword.lower(), na=False)]
                if df_non_branded.empty:
                    return f"No data found containing keyword '{keyword}'"

            # Take average for same keywords
            df_non_branded = df_non_branded.groupby('query').agg({
                'total_clicks': 'mean',
                'total_impressions': 'mean',
                'avg_rankning': 'mean'
            }).reset_index()

            # Initialize report content list
            report_parts = []
            title = f"=== {'Keyword ' + keyword + ' Analysis' if keyword else 'Non-branded Keywords'} Report ===\n"
            report_parts.append(title)
            report_parts.append(f"Total Keywords: {len(df_non_branded)} (branded keywords excluded)\n")

            # 1. Ranking Distribution Analysis
            report_parts.append("\n1. Ranking Distribution:")
            rank_ranges = {
                "1-3": (1, 3),
                "4-10": (4, 10),
                "11-20": (11, 20),
                "21-50": (21, 50),
                "50+": (51, float('inf'))
            }

            for range_name, (min_rank, max_rank) in rank_ranges.items():
                count = len(df_non_branded[(df_non_branded['avg_rankning'] >= min_rank)
                            & (df_non_branded['avg_rankning'] <= max_rank)])
                percentage = (count / len(df_non_branded)) * \
                    100 if len(df_non_branded) > 0 else 0
                report_parts.append(
                    f"  {range_name}: {count} keywords ({percentage:.1f}%)")

            # 2. Top 10 Keywords by Click Rate
            report_parts.append("\n2. Top 10 Keywords by Click Rate:")
            top_clicks = df_non_branded.nlargest(10, 'total_clicks')[
                ['query', 'total_clicks', 'avg_rankning']]
            for _, row in top_clicks.iterrows():
                report_parts.append(
                    f"  {row['query']}: {row['total_clicks']:.1f} clicks (rank: {row['avg_rankning']:.1f})")

            # 3. Top 10 Keywords by Impression Rate
            report_parts.append("\n3. Top 10 Keywords by Impression Rate:")
            top_impressions = df_non_branded.nlargest(10, 'total_impressions')[
                ['query', 'total_impressions', 'avg_rankning']]
            for _, row in top_impressions.iterrows():
                report_parts.append(
                    f"  {row['query']}: {row['total_impressions']:.1f} impressions (rank: {row['avg_rankning']:.1f})")

            # 4. Calculate Average Metrics
            report_parts.append("\n4. Overall Metrics:")
            report_parts.append(
                f"  Average Clicks: {df_non_branded['total_clicks'].mean():.1f}")
            report_parts.append(
                f"  Average Impressions: {df_non_branded['total_impressions'].mean():.1f}")
            report_parts.append(
                f"  Average Rank: {df_non_branded['avg_rankning'].mean():.1f}")

            return "\n".join(report_parts)

        except Exception as e:
            print(f"Error generating analysis report: {str(e)}")
            return "Unable to generate analysis report"

    def get_keyword_time_context(self, keyword: str) -> str:
        """
        获取关键词的时序信息作为上下文

        Args:
            keyword: 要分析的关键词

        Returns:
            str: 格式化的时序信息
        """
        if self.time_series_df is None:
            return "无法获取时序数据"

        try:
            # 获取关键词的时序数据
            keyword_data = get_keyword_time_series(
                self.time_series_df, keyword)

            if keyword_data.empty:
                return f"未找到关键词 '{keyword}' 的时序数据"

            # 构建时序信息字符串
            time_context = []

            # 按URL分组处理数据
            for url in keyword_data['site_url'].unique():
                url_data = keyword_data[keyword_data['site_url'] == url]

                # 添加URL信息
                time_context.append(f"\nURL: {url}")
                time_context.append(
                    f"数据范围: {url_data['month_start'].min().strftime('%Y-%m')} 至 {url_data['month_start'].max().strftime('%Y-%m')}")
                time_context.append(f"数据点数量: {len(url_data)}")

                # Add monthly data
                time_context.append("\nMonthly Data:")
                for _, row in url_data.iterrows():
                    time_context.append(
                        f"  {row['month_start'].strftime('%Y-%m')}:")
                    time_context.append(f"    Keyword: {row['query']}")
                    time_context.append(f"    Clicks: {row['total_clicks']}")
                    time_context.append(
                        f"    Impressions: {row['total_impressions']}")
                    time_context.append(
                        f"    Average Rank: {row['avg_rankning']:.2f}")

                # 计算首尾月份的变化
                if len(url_data) > 1:
                    first_month = url_data.iloc[0]
                    last_month = url_data.iloc[-1]

                    time_context.append("\n  变化情况 (末月 vs 首月):")
                    clicks_diff = last_month['total_clicks'] - \
                        first_month['total_clicks']
                    impressions_diff = last_month['total_impressions'] - \
                        first_month['total_impressions']
                    ranking_diff = first_month['avg_rankning'] - \
                        last_month['avg_rankning']

                    time_context.append(f"    点击量变化: {clicks_diff:+d}")
                    time_context.append(f"    展示量变化: {impressions_diff:+d}")
                    time_context.append(f"    排名变化: {ranking_diff:+.2f}")

            return "\n".join(time_context)

        except Exception as e:
            print(f"获取时序上下文时出错: {str(e)}")
            return f"获取关键词 '{keyword}' 的时序数据时出错"

    def extract_keywords_from_question(self, question: str) -> List[str]:
        """
        从问题中提取关键词，使用多种策略匹配数据集中的关键词

        Args:
            question: 用户问题

        Returns:
            List[str]: 提取的关键词列表
        """
        try:
            # 预处理问题文本
            question = question.lower().strip()
            keywords = set()  # 使用集合去重
            
            # 1. 从数据集中加载所有关键词
            if not hasattr(self, '_all_keywords'):
                df = pd.read_csv(PROCESSED_DATA_PATH)
                # 将所有数据集关键词转换为小写并去重
                self._all_keywords = set(df['query'].str.lower().unique())
            
            # 2. 使用正则表达式提取 [] 中的内容作为精确匹配
            # 查找所有 [] 包裹的内容，使用非贪婪匹配 (.*?)
            matches = re.findall(r'\[(.*?)\]', question)
            
            for kw in matches:
                extracted_kw = kw.strip()
                if extracted_kw and len(extracted_kw) > 1:
                    # 检查提取的关键词是否在数据集中存在 (不区分大小写)
                    if extracted_kw.lower() in self._all_keywords:
                        keywords.add(extracted_kw.lower()) # 转换为小写再添加
            
            # 如果从 [] 中找到了关键词，则立即返回
            if keywords:
                logger.info(f"Extracted keywords from []: {keywords}") # 添加日志确认提取结果
                return list(keywords)
            
            results=list(keywords)
            print("extracted keywords:",keywords)
            return results  # 转换回列表

        except Exception as e:
            logger.error(f"提取关键词时出错: {str(e)}")
            return []  # 出错时返回空列表

    def get_keyword_analysis_context(self, keyword: str) -> str:
        """
        获取特定关键词的分析上下文

        Args:
            keyword: 要分析的关键词

        Returns:
            str: 格式化的分析报告
        """
        try:
            # 获取关键词的时序数据
            keyword_data = self.get_keyword_time_context(keyword)
            
            # 过滤掉平均排名大于30的数据
            if keyword_data and isinstance(keyword_data, str):
                try:
                    # 将字符串数据转换为字典列表
                    data_list = json.loads(keyword_data)
                    # 过滤掉平均排名大于30的数据
                    filtered_data = []
                    for item in data_list:
                        avg_ranking = item.get('avg_rankning')
                        if avg_ranking is not None and float(avg_ranking) <= 30:
                            filtered_data.append(item)
                    
                    if not filtered_data:
                        logger.info(f"关键词 '{keyword}' 没有排名在30以内的数据")
                        return "没有找到排名在30以内的数据"
                        
                    keyword_data = json.dumps(filtered_data, ensure_ascii=False)
                    logger.info(f"过滤后的数据条数: {len(filtered_data)}")
                except json.JSONDecodeError:
                    logger.warning(f"无法解析关键词数据为JSON格式: {keyword}")
                    return keyword_data

            return keyword_data

        except Exception as e:
            logger.error(f"Error generating keyword analysis report: {str(e)}")
            return f"Error analyzing keyword '{keyword}'"

    def query(self, question: str) -> str:
        """执行混合查询，支持关键词搜索和性能分析查询"""
        if not self.rag_chain:
            raise ValueError("RAG链尚未设置，请先调用setup_rag_chain()")

        try:
            # 构建上下文
            context_parts = []
            context = []
            keywords = []  # 初始化 keywords 变量

            # 1. 分析查询类型
            query_type = self._analyze_query_type(question)
            print(query_type)
            # 2. 根据查询类型执行不同的搜索策略
            if query_type == "ranking_analysis":
                try:
                    # 排名分析查询 - 移除排名过滤，获取所有相关关键词
                    search_results = self.search_with_score(
                        question,
                        min_score=0.4  # 降低相似度阈值
                    )

                    if search_results:
                        context_parts.append("\n相关关键词分析:")
                        # 按关键词分组
                        keyword_groups = {}
                        for doc, score in search_results:
                            keyword = doc.metadata.get('query', '')
                            if keyword not in keyword_groups:
                                keyword_groups[keyword] = []
                            keyword_groups[keyword].append((doc, score))

                        # 对每个关键词组进行分析
                        for keyword, results in keyword_groups.items():
                            # 获取该关键词的最新数据
                            latest_result = max(
                                results, key=lambda x: x[0].metadata.get('month_start', ''))
                            doc, score = latest_result

                            context_part = self._format_search_result(
                                doc, score)
                            context_parts.append(context_part)
                            keywords.append(keyword)

                            # 获取详细分析
                            keyword_analysis = self.get_keyword_analysis_context(
                                keyword
                            )
                            if keyword_analysis:
                                context.append(keyword_analysis)
                except Exception as e:
                    logger.error(f"执行排名分析查询时出错: {str(e)}")
                 
            else:
                # 关键词特定查询或通用查询
                keywords = self.extract_keywords_from_question(question)
                print(keywords)
                if keywords:
                    # 关键词特定查询
                    for keyword in keywords:
                        # try:
                        #     search_results = self.search_with_score(
                        #         keyword,
                        #         min_score=0.4
                        #     )
                        #     if search_results:
                                # context_parts.append(
                                #     f"\n相似度搜索关键词 '{keyword}' 的结果:")
                                # for doc, score in search_results:
                                #     context_part = self._format_search_result(
                                #         doc, score)
                                #     context_parts.append(context_part)

                                    # 获取时序分析
                                    keyword_analysis = self.get_keyword_analysis_context(
                                        keyword
                                    )
                                    if keyword_analysis:
                                        context_parts.append(keyword_analysis)
                            # else:
                            #     context_parts.append(
                            #         f"\n未找到相似的关键词 '{keyword}'")
                        # except Exception as e:
                        #     logger.error(f"搜索关键词 '{keyword}' 时出错: {str(e)}")
                       
                else:
                    # 通用查询，使用问题本身进行搜索
                    try:
                        search_results = self.search_with_score(
                            question,
                            min_score=0.4
                        )

                        if search_results:
                            context_parts.append("\n相关关键词:")
                            for doc, score in search_results:
                                context_part = self._format_search_result(
                                    doc, score)
                                context_parts.append(context_part)
                    except Exception as e:
                        logger.error(f"执行通用查询时出错: {str(e)}")
                   

            context = "\n---\n".join(context_parts)
            if context:
                print("context:", context)
         
            # 获取整体分析报告
            analysis_report = self.get_analysis_report(
                keyword=keywords[0] if keywords else None)
            print(analysis_report)

            # 使用优化后的提示模板
            prompt = ChatPromptTemplate.from_template("""
            You are a professional SEO data analyst specializing in analyzing keyword performance for medical aesthetic clinics.
            Please analyze the following data based on the user's specific question always in English:

            Retrieved Keyword Data:
            {context}
            Important Keywords to Focus On:
            - Highlight keywords with average ranking in top 10 positions
            - Emphasize keywords with impressions > 50
            - always report the import keywords in your final response
       
              
            Overall Analysis Report:
            {analysis_report}
            
            Please structure your response to the user's question "{question}" as follows:
            
            1. Data Overview
               - Summarize the retrieved keyword data
               - Highlight the most important findings
               - Specify the data time range
            
            2. Detailed Analysis
               - Analyze based on the user's specific question
               - Support your points with data
               - Identify key trends and changes
            
            3. Optimization Recommendations
               - Provide specific, actionable recommendations
               - Prioritize by importance
               - Explain expected outcomes
            
            4. Data Limitations
               - Explain the limitations of the analysis
               - Suggest additional data that could be valuable
            
            Please use professional yet accessible language, ensuring recommendations are actionable.
            
            Response:
            """).format(
                context=context,
                analysis_report=analysis_report,
                question=question
            )

            # 生成回答
            response = self.llm.invoke(prompt)
            return response

        except Exception as e:
            logger.error(f"查询过程中出错: {str(e)}")
            return "抱歉，查询过程中出现错误。请稍后重试。"

    def clear_cache(self):
        """清除缓存"""
        self.cache_manager.clear()
        self._get_cached_embedding.cache_clear()

    def add_documents_in_batches(self, documents: List[Document], batch_size: int = 50, max_retries: int = 3, retry_delay: int = 5):
        """分批将文档添加到 Pinecone 向量存储中，支持断点续传和重试机制

        Args:
            documents: 要添加的文档列表
            batch_size: 每批处理的文档数量（默认减小到50）
            max_retries: 最大重试次数
            retry_delay: 重试延迟（秒）
        """
        if not self.use_pinecone:
            raise ValueError("此方法仅支持 Pinecone 向量存储")

        total_docs = len(documents)
        total_batches = (total_docs + batch_size - 1) // batch_size

        # 创建进度文件路径
        progress_file = "pinecone_upload_progress.json"

        # 尝试加载上次的进度
        start_batch = 0
        if os.path.exists(progress_file):
            try:
                with open(progress_file, 'r') as f:
                    progress = json.load(f)
                    start_batch = progress.get('last_batch', 0)
                    print(f"从第 {start_batch + 1} 批继续上传...")
            except Exception as e:
                print(f"加载进度文件失败: {str(e)}")

        print(f"开始分批导入文档，共 {total_docs} 条数据，分 {total_batches} 批处理")

        for i in range(start_batch * batch_size, total_docs, batch_size):
            batch = documents[i:i + batch_size]
            current_batch = (i // batch_size) + 1
            print(f"正在处理第 {current_batch}/{total_batches} 批数据...")

            retry_count = 0
            while retry_count < max_retries:
                try:
                    # 获取当前批次的文本和元数据
                    texts = [doc.page_content for doc in batch]
                    metadatas = [doc.metadata for doc in batch]

                    # 生成嵌入向量
                    embeddings = self.embeddings.embed_documents(texts)

                    # 准备向量数据
                    vectors = []
                    for j, (text, metadata, embedding) in enumerate(zip(texts, metadatas, embeddings)):
                        # 确保所有向量值都是 float 类型
                        embedding_values = [float(x) for x in embedding]

                        # 确保元数据中的数值也是 float 类型
                        processed_metadata = {}
                        for key, value in metadata.items():
                            if isinstance(value, (int, float)):
                                processed_metadata[key] = float(value)
                            else:
                                processed_metadata[key] = value

                        vector_id = f"doc_{i + j}"  # 生成唯一的向量ID
                        vectors.append({
                            "id": vector_id,
                            "values": embedding_values,
                            "metadata": processed_metadata
                        })

                    # 使用 Pinecone 的 upsert 方法直接上传向量，设置超时时间
                    self.vectorstore.index.upsert(
                        vectors=vectors,
                        namespace=PINECONE_NAMESPACE,
                        timeout=30  # 设置30秒超时
                    )

                    # 保存进度
                    with open(progress_file, 'w') as f:
                        json.dump({'last_batch': current_batch}, f)

                    print(f"第 {current_batch} 批数据处理完成")
                    break  # 成功处理，跳出重试循环

                except Exception as e:
                    retry_count += 1
                    if retry_count < max_retries:
                        wait_time = retry_delay * \
                            (2 ** (retry_count - 1))  # 指数退避
                        print(f"处理第 {current_batch} 批数据时出错: {str(e)}")
                        print(f"等待 {wait_time} 秒后进行第 {retry_count + 1} 次重试...")
                        time.sleep(wait_time)
                    else:
                        print(f"处理第 {current_batch} 批数据失败，已达到最大重试次数")
                        # 保存当前进度，以便下次继续
                        with open(progress_file, 'w') as f:
                            json.dump({'last_batch': current_batch - 1}, f)
                        raise Exception(f"上传数据失败: {str(e)}")

        # 所有批次处理完成后，删除进度文件
        if os.path.exists(progress_file):
            try:
                os.remove(progress_file)
                print("进度文件已清理")
            except Exception as e:
                print(f"清理进度文件失败: {str(e)}")

        print("所有文档导入完成")

    def _analyze_query_type(self, question: str) -> str:
        """分析查询类型

        Args:
            question: 用户问题

        Returns:
            str: 查询类型，包括：
                - keyword_specific: 关键词特定查询
                - performance_trend: 性能趋势查询
                - ranking_analysis: 排名分析查询
                - general: 通用查询
        """
        # 性能趋势相关关键词
        trend_keywords = [
         
            "increase", "growth", "improve", "trend",
            "decrease", "decline", "drop"
        ]

        # 排名相关关键词
        ranking_keywords = [
                    "ranking", "position", "top", "first"
        ]

        # 检查是否包含性能趋势关键词
        if any(keyword in question.lower() for keyword in trend_keywords):
            return "performance_trend"

        # 检查是否包含排名相关关键词
        if any(keyword in question.lower() for keyword in ranking_keywords):
            return "ranking_analysis"

        # 检查是否包含特定关键词查询
        if self.extract_keywords_from_question(question):
            return "keyword_specific"

        # 默认为通用查询
        return "general"

    def _format_search_result(self, doc: Document, score: float) -> str:
        """格式化搜索结果

        Args:
            doc: 文档对象
            score: 相似度分数

        Returns:
            str: 格式化后的搜索结果
        """
        return f"""
Similarity: {score:.4f}
Keyword: {doc.metadata.get('query', 'N/A')}
Month: {doc.metadata.get('month_start', 'N/A')}
Clicks: {doc.metadata.get('total_clicks', 'N/A')}
Impressions: {doc.metadata.get('total_impressions', 'N/A')}
Average Rank: {doc.metadata.get('avg_rankning', 'N/A')}
"""


def main():
    try:
        # 初始化数据处理器
        print("正在加载数据...")
        processor = KeywordDataProcessor()

        # 使用 SEOCSVLoader 加载数据
        documents = processor.load_data(
            "dataset.csv")
        processor.documents = documents  # 保存文档列表
        print(f"成功加载 {len(documents)} 条数据")

        # 初始化RAG系统（使用 Pinecone）
        print("正在初始化 RAG 系统...")
        rag_system = RAGSystem(
            embedding_model_name="BAAI/bge-m3",
            use_pinecone=True
        )

        # 检查向量存储状态
        print("\n检查向量存储状态...")
        # rag_system.check_vector_structure()

        # 分批将文档导入到 Pinecone
        print("\n正在将文档导入到 Pinecone...")
        rag_system.add_documents_in_batches(
            documents, batch_size=100)  # 每批处理100条数据
        print("文档导入完成")

        # 再次检查向量存储状态
        print("\n检查导入后的向量存储状态...")
        rag_system.check_vector_structure()

        rag_system.setup_rag_chain()
        print("RAG 系统初始化完成")

        # 测试增强的查询功能
        test_questions = [
            "Analyze keywords with recent ranking decline",
            "Find keywords with click growth exceeding 20%",
            "Analyze performance of informational keywords"
        ]

        # 设置元数据过滤

        for question in test_questions:
            print(f"\n问题: {question}")
            print("-" * 50)
            answer = rag_system.query(question)
            print(f"回答: {answer}")
            print("-" * 50)

    except Exception as e:
        print(f"运行出错: {str(e)}")
        import traceback
        print(traceback.format_exc())


def query_existing_index(question: str, use_filters: bool = True, max_keywords: int = 5):
    """直接查询已存在的 Pinecone 索引，优化输出格式并确保数据准确性

    Args:
        question: 用户问题
        use_filters: 是否使用元数据过滤
        max_keywords: 最多显示的关键词数量
    """
    try:
        print("正在连接到 Pinecone 索引...")
        rag_system = RAGSystem(use_pinecone=True)
        rag_system.setup_rag_chain()
        print("RAG 系统初始化完成")

        print(f"\n问题: {question}")
        print("-" * 50)

        # 执行查询并获取结果
        max_retries = 3
        retry_count = 0
        while retry_count < max_retries:
            try:
                # 获取更多结果以便查看完整历史
                search_results = rag_system.vectorstore.similarity_search(
                    query=question,
                    k=500,  # 增加检索数量，确保获取所有历史数据
                    filter=rag_system._build_filter_query() if rag_system.metadata_filters else None
                )

                if search_results:
                    # 按关键词和月份排序
                    sorted_results = sorted(
                        search_results,
                        key=lambda x: (x.metadata.get('query', ''),
                                       x.metadata.get('month_start', ''))
                    )

                    # 按关键词分组并验证数据
                    keyword_groups = {}
                    for doc in sorted_results:
                        keyword = doc.metadata.get('query', 'N/A')
                        if keyword not in keyword_groups:
                            keyword_groups[keyword] = []
                        keyword_groups[keyword].append(doc)

                    # 筛选关键词：只保留有显著变化的关键词
                    significant_keywords = {}
                    for keyword, docs in keyword_groups.items():
                        # 按月份排序
                        sorted_docs = sorted(docs, key=lambda x: x.metadata.get(
                            'month_start', ''), reverse=True)

                        # 检查是否有显著变化
                        has_significant_change = False
                        for doc in sorted_docs[:2]:  # 只检查最近两个月
                            clicks_mom = float(
                                doc.metadata.get('clicks_mom', 0))
                            impressions_mom = float(
                                doc.metadata.get('impressions_mom', 0))
                            rank_mom = float(doc.metadata.get('rank_mom', 0))

                            if (abs(clicks_mom) > 10 or
                                abs(impressions_mom) > 10 or
                                    abs(rank_mom) > 5):
                                has_significant_change = True
                                break

                        if has_significant_change:
                            # 保存该关键词的所有数据
                            significant_keywords[keyword] = sorted_docs

                    # 只显示前 max_keywords 个关键词
                    display_keywords = dict(
                        list(significant_keywords.items())[:max_keywords])

                # 生成完整回答
                answer = rag_system.query(question)
                print(f"\n综合分析回答:\n{answer}")
                print("-" * 50)
                break  # 成功执行，跳出重试循环

            except Exception as e:
                retry_count += 1
                if retry_count < max_retries:
                    wait_time = 2 ** retry_count  # 指数退避
                    print(f"查询出错，正在进行第 {retry_count} 次重试，等待 {wait_time} 秒...")
                    time.sleep(wait_time)
                else:
                    print(f"查询失败，已达到最大重试次数: {str(e)}")
                    raise

    except Exception as e:
        print(f"查询出错: {str(e)}")
        import traceback
        print(traceback.format_exc())


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='SEO关键词分析系统')
    parser.add_argument('question', nargs='?', help='要分析的问题')
    parser.add_argument('--exact', action='store_true', help='使用精确匹配模式')
    parser.add_argument('--debug', action='store_true', help='启用调试模式')
    args = parser.parse_args()

    try:
        # 初始化RAG系统
        rag_system = RAGSystem(
            embedding_model_name="BAAI/bge-m3",
            use_pinecone=True
        )
        rag_system.setup_rag_chain()

        # 设置搜索模式
        # rag_system.set_search_mode(exact_match=args.exact)

        if args.question:
            # 清理问题文本，移除可能的命令行参数
            question = args.question.split('--')[0].strip()
            print(f"\n问题: {question}")
            print(f"搜索模式: {'精确匹配' if args.exact else '向量相似度搜索'}")
            print("-" * 50)

            # 执行查询
            answer = rag_system.query(question)
            print(f"\n分析结果:\n{answer}")

    except Exception as e:
        print(f"运行出错: {str(e)}")
        if args.debug:
            import traceback
            print(traceback.format_exc())
