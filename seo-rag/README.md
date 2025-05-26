# SEO Data Analysis Agent

An intelligent SEO data analysis agent based on RAG (Retrieval-Augmented Generation) technology, capable of analyzing keyword performance and providing data-driven recommendations. The system integrates large language models, vector databases, and data analysis capabilities to provide intelligent decision support for SEO optimization.

## Features

- Intelligent Keyword Performance Analysis
  - Multi-dimensional data analysis (clicks, impressions, rankings)
  - Month-over-Month (MoM) change analysis
  - Trend prediction
  - Intent classification analysis
- Multi-turn Conversation Support
  - Context understanding
  - Intelligent follow-up questions
  - Analysis summarization
- Data Quality Assessment
  - Data completeness check
  - Outlier detection
  - Trend analysis
  - Quality improvement suggestions
- Automated Analysis Reports
  - Intent distribution analysis
  - Ranking distribution analysis
  - MoM change analysis
  - Monthly trend analysis
  - Best/worst performing keywords analysis
- Multiple Query Strategies
  - Semantic search
  - Metadata filtering
  - Similarity scoring
  - Hybrid retrieval
- Comprehensive Error Handling
  - Data loading error handling
  - API call error handling
  - Invalid query handling
  - Outlier handling

## Project Structure

```
GSC-RAG/
├── src/
│   ├── langchain_rag.py          # RAG系统核心实现
│   ├── feature_engineering.py    # 特征工程和数据处理
│   ├── embedding.py             # 向量生成和存储
│   ├── keyword_time_series.py   # 时间序列特征处理
│   └── web_ui.py               # Web界面实现
├── data/
│   ├── keywords_processed.csv   # 原始关键词数据
│   ├── keywords_enriched.parquet # 特征工程后的数据
│   ├── embeddings_for_pinecone.parquet # 向量数据
│   └── dictionaries/           # 词典文件
│       ├── intent_patterns.json # 意图分类词典
│       └── keyword_groups.json  # 关键词分组词典
├── tests/
│   └── test_seo_analyst.py     # 测试脚本
├── requirements.txt            # 项目依赖
└── README.md                  # 项目文档
```

## Requirements

- Python 3.8+
- Dependencies:
  - langchain>=0.1.0
  - pandas>=2.0.0
  - pinecone-client>=2.2.0
  - requests>=2.31.0
  - numpy>=1.24.0
  - streamlit>=1.30.0
  - python-dotenv>=1.0.0
  - scikit-learn>=1.3.0

## Installation

1. Clone the repository

```bash
git clone https://github.com/yourusername/GSC-RAG.git
cd GSC-RAG
```

2. Create and activate virtual environment (recommended)

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

3. Install dependencies

```bash
pip install -r requirements.txt
```

4. Configure environment variables
Create a `.env` file and add necessary configurations:

```env
PINECONE_API_KEY=your_api_key
PINECONE_ENVIRONMENT=your_environment
OPENAI_API_KEY=your_api_key
```

## Usage

### 1. Data Preparation

Save your SEO data as a CSV file with the following required fields:

- query: Search keyword
- total_clicks: Total click count
- total_impressions: Total impression count
- avg_rankning: Average ranking
- month_start: Month (YYYY-MM-DD format)
- intent: Search intent (informational/transactional/navigational/other)
- trend_desc: Trend description (auto-generated)

### 2. Data Import

```python
from src.seo_data_loader import SEOCSVLoader

# Initialize data loader
loader = SEOCSVLoader()

# Load and process data
df = loader.load_data("path/to/your/seo_data.csv")

# Generate vectors and upload to Pinecone
from src.generate_embedding import generate_and_upload_vectors
generate_and_upload_vectors(df)
```

### 3. Launch Web Interface

```bash
streamlit run src/web_ui.py
```

### 4. Use RAG System

```python
from src.langchain_rag import RAGSystem

# Initialize RAG system
rag_system = RAGSystem(use_pinecone=True)
rag_system.setup_rag_chain()

# Execute query
question = "Analyze the best performing keywords recently"
answer = rag_system.query(question)
print(answer)
```

### 5. Data Validation

```python
from src.check_vectors import analyze_vectors

# Get and analyze vector data
vectors = rag_system._get_all_vectors()
analyze_vectors(vectors)
```

## Core Features

### 1. Data Analysis Capabilities

- Keyword Performance Analysis
  - Click trend analysis
  - Impression change analysis
  - Ranking fluctuation analysis
  - CTR (Click-Through Rate) analysis
- MoM Change Analysis
  - Click MoM
  - Impression MoM
  - Ranking MoM
- Intent Analysis
  - Intent distribution statistics
  - Intent-performance correlation analysis
- Trend Analysis
  - Monthly trends
  - Quarterly trends
  - Annual trends

### 2. RAG System Features

- Intelligent Retrieval
  - Semantic similarity search
  - Metadata filtering
  - Hybrid ranking
- Context Understanding
  - Multi-turn conversation support
  - History tracking
  - Intent understanding
- Analysis Report Generation
  - Automated analysis summary
  - Key metrics extraction
  - Trend visualization

### 3. Data Quality Control

- Data Preprocessing
  - Missing value handling
  - Outlier detection
  - Data normalization
- Data Validation
  - Field completeness check
  - Value range validation
  - Time series continuity check
- Quality Reporting
  - Data quality scoring
  - Issue itemization
  - Improvement suggestions
