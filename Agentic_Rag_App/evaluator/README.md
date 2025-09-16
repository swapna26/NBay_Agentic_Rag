# RAGAs Evaluator

A comprehensive RAG evaluation system using RAGAs (Retrieval Augmented Generation Assessment) framework to evaluate the quality of your Agentic RAG implementation.

## Features

- **Automatic Test Dataset Generation**: Generate test questions from indexed documents
- **Comprehensive RAG Evaluation**: Evaluate using multiple RAGAs metrics
- **Custom Question Support**: Evaluate with your own custom questions
- **Detailed Reporting**: Generate comprehensive evaluation reports with recommendations
- **CLI Interface**: Easy-to-use command-line interface
- **Rich Console Output**: Beautiful formatted output with progress tracking

## RAGAs Metrics Supported

1. **Faithfulness**: Measures factual consistency of generated answers
2. **Answer Relevancy**: Evaluates relevance of answers to questions
3. **Context Precision**: Measures precision of retrieved context
4. **Context Recall**: Evaluates recall of relevant context
5. **Answer Similarity**: Compares semantic similarity of answers
6. **Answer Correctness**: Evaluates factual and semantic correctness

## Installation

1. Install dependencies:
```bash
cd evaluator
pip install -r requirements.txt
```

## Configuration

The evaluator automatically connects to your existing infrastructure:
- **PostgreSQL Database**: Same database used by the indexer
- **Ollama Models**: Uses the same embedding and LLM models
- **Backend API**: Connects to your RAG backend API

Configuration is handled in `config.py` with these defaults:
- Database: `postgresql://raguser:ragpassword@localhost:5432/agentic_rag`
- Ollama: `http://localhost:11434`
- Backend API: `http://localhost:8001`
- Embedding Model: `nomic-embed-text:v1.5`
- LLM Model: `llama3.2:1b`

## Usage

### 1. Check System Status
```bash
python main.py status
```

### 2. Generate Test Dataset
```bash
# Generate 20 test questions from indexed documents
python main.py generate-dataset

# Generate custom number of questions
python main.py generate-dataset --num-questions 50

# Save to custom location
python main.py generate-dataset --output /path/to/dataset.json
```

### 3. Run Evaluation
```bash
# Run complete evaluation with all metrics
python main.py evaluate

# Evaluate specific metrics only
python main.py evaluate --metrics faithfulness --metrics answer_relevancy

# Evaluate with custom questions
python main.py evaluate --questions "What is the procurement process?" --questions "How to handle HR issues?"

# Use custom dataset
python main.py evaluate --dataset /path/to/custom_dataset.json

# Save report to custom location
python main.py evaluate --output /path/to/report.json
```

### 4. View Reports
```bash
# List all evaluation reports
python main.py list-reports

# View specific report
python main.py show-report --report /path/to/report.json
```

## CLI Commands

### `generate-dataset`
Generate test questions from indexed documents.

**Options:**
- `--num-questions, -n`: Number of questions to generate (default: 20)
- `--output, -o`: Output path for the dataset file

### `evaluate`
Run RAGAs evaluation on the RAG system.

**Options:**
- `--dataset, -d`: Path to test dataset JSON file
- `--metrics, -m`: Specific metrics to evaluate (can be repeated)
- `--output, -o`: Output path for evaluation report
- `--questions, -q`: Custom questions to evaluate (can be repeated)

### `status`
Check the status of the evaluation system.

### `show-report`
Display evaluation report in formatted output.

**Options:**
- `--report, -r`: Path to evaluation report JSON file (required)

### `list-reports`
List available evaluation reports.

**Options:**
- `--path, -p`: Path to reports directory

## Example Workflow

1. **Check if your system is ready:**
```bash
python main.py status
```

2. **Generate test dataset:**
```bash
python main.py generate-dataset --num-questions 30
```

3. **Run evaluation:**
```bash
python main.py evaluate
```

4. **View the results:**
```bash
python main.py list-reports
python main.py show-report --report reports/rag_evaluation_report_20240914_120000.json
```

## Output Structure

### Test Dataset
Generated test datasets contain:
- Questions with different types (factual, conceptual, analytical, procedural)
- Ground truth answers
- Source document references
- Context information

### Evaluation Reports
Comprehensive JSON reports include:
- Metric scores with statistics (mean, std, min, max)
- Individual question results
- System configuration and metadata
- Performance summary and recommendations
- Database statistics

## Advanced Usage

### Custom Metrics
Modify `config.py` to select specific metrics:
```python
config.metrics = ["faithfulness", "answer_relevancy", "context_precision"]
```

### Custom Configuration
Override default settings in `config.py`:
```python
config.backend_api_url = "http://your-backend:8000"
config.database_url = "postgresql://user:pass@host:port/db"
config.num_test_questions = 50
```

## Troubleshooting

### Common Issues

1. **Database Connection Error**
   - Ensure PostgreSQL is running
   - Check database credentials in `config.py`
   - Verify the database contains indexed documents

2. **Backend API Unavailable**
   - Start your RAG backend server
   - Check the API URL in configuration
   - Verify the backend is healthy

3. **Ollama Models Not Available**
   - Ensure Ollama is running
   - Pull required models: `ollama pull nomic-embed-text:v1.5` and `ollama pull llama3.2:1b`

4. **No Documents Found**
   - Run the indexer first to populate the database
   - Check that documents are properly indexed

### Performance Tips

- Start with a smaller number of test questions (10-20) for faster evaluation
- Use specific metrics instead of all metrics for quicker results
- Generate test datasets once and reuse them for multiple evaluations

## Integration

The evaluator is designed to work seamlessly with your existing Agentic RAG setup:
- Reads from the same PostgreSQL vector database
- Uses the same Ollama models and configuration
- Connects to your backend API for RAG queries
- Generates reports in the `reports/` directory

No changes to other components are required!