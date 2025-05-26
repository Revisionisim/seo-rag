```markdown
# SEO Data Analysis Agent

An intelligent SEO data analysis agent based on RAG (Retrieval-Augmented Generation) technology, capable of analyzing keyword performance and providing data-driven recommendations. The system integrates RAG, vector databases, and LLM to provide intelligent decision support for SEO optimization.

## Requirements

- Python 3.8+

## Dependencies

- `langchain`
- `pandas`
- `numpy`
- `pyarrow`
- `matplotlib`
- `seaborn`
- `streamlit`
- `faiss-cpu`
- `sentence-transformers`
- `scikit-learn`
- `tqdm`
- `llama-cpp-python`
- `pinecone-client`
- `jupyter`
- `openpyxl`

## Installation

Download and unzip the package

Create and activate virtual environment (recommended)

```bash
python -m venv venv
```

```bash
source venv/bin/activate # Linux/Mac
venv\Scripts\activate # Windows
```

Install dependencies

```bash
pip install -r requirements.txt
```

## Configure Environment Variables

Create a `.env` file and add necessary configurations:

```ini
PINECONE_API_KEY=your_api_key
PINECONE_ENVIRONMENT=your_environment
PINECONE_INDEX_NAME = your_index_name
PINECONE_NAMESPACE = your_namespace
SILICONFLOW_API_KEY=your_api_key
```

## Usage

### Data Preparation

**What this step does:**
- transform datatype
- group the keywords
- intent classification

Save your SEO data as a CSV file under the `data` folder and rename it to `dataset.csv`

```python
file_path = os.path.join('data', 'dataset.csv')
```

### Vector Upsert into Pinecone

**What this step does:**
- get unique embedding for each row
- build metadata
- connect to Pinecone vector space and upsert embedding

**How to run:**
```bash
python -u ".\src\generate_embedding.py"
```

### RAG System

**How to run:**
```bash
python -u ".\src\langchain_rag.py" "YOUR INSTRUCTION"
```

### Example Usage

To retrieve info about a specific keyword, make sure the keyword is covered under `[]`:

```bash
python -u .\src\langchain_rag.py "describe the performance trending of keyword [filler]"
```

Or retrieve global info about the whole data, like:

```bash
python -u .\src\langchain_rag.py "what are the top-5 keywords with the biggest rank increasement?"
```
### 	4. Visualization(unfinished)

**How to run:**
```bash
python src/visualization.py
```
### 5. WebUI(unfinished)
```bash
streamlit run ui/app.py
```
