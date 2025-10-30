# PDF-based RAG System with Multi-Model Evaluation

A comprehensive Retrieval-Augmented Generation (RAG) system that processes PDF documents and evaluates multiple LLM models with extensive metrics and visualizations.

## 🚀 Features

### Supported Models
- **Mistral AI** - Fast and efficient language model
- **Qwen 2.5** - High-quality multilingual model
- **Llama 3** - Meta's powerful language model

### PDF Processing
- Automatic PDF loading and parsing
- Intelligent document chunking
- FAISS vector store for fast retrieval
- HuggingFace embeddings (all-MiniLM-L6-v2)

### Comprehensive Evaluation Metrics

#### Similarity Metrics
- **Cosine Similarity** - Semantic similarity between generated and reference answers
- **BLEU Score** - N-gram overlap metric
- **METEOR Score** - Alignment-based metric with synonyms
- **BERTScore** - Contextual embeddings-based evaluation (F1, Precision, Recall)

#### Quality Metrics
- **Completeness** - Coverage of reference information
- **Hallucination Detection** - Information not present in source documents
- **Irrelevance Detection** - Deviation from the question

#### Performance Metrics
- **Response Time / Latency** - Time taken to generate responses
- **Average Response Time** - Mean across trials

### Visualization & Reporting
- **Trial Scores** - Individual performance across 3 trials
- **Bar Charts** - Comparative analysis of all metrics
- **Distribution Plots** - Response time distributions
- **Summary Tables** - Comprehensive metric overview

## 📋 Prerequisites

### 1. Install Ollama
```bash
# macOS/Linux
curl -fsSL https://ollama.com/install.sh | sh

# Or visit: https://ollama.com/download
```

### 2. Install Required Models
```bash
ollama pull mistral
ollama pull qwen2.5
ollama pull llama3
```

### 3. Verify Ollama is Running
```bash
ollama serve  # Start Ollama server if not already running
```

## 🛠️ Installation

### 1. Clone the Repository
```bash
git clone <repository-url>
cd <repository-directory>
```

### 2. Create Virtual Environment
```bash
python -m venv venv

# Activate virtual environment
# On Linux/Mac:
source venv/bin/activate

# On Windows:
venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Download NLTK Data
```python
python -c "import nltk; nltk.download('punkt'); nltk.download('wordnet'); nltk.download('omw-1.4')"
```

## 📁 Project Structure

```
.
├── pdf_rag_system.py          # Main RAG system implementation
├── evaluation_metrics.py       # Comprehensive evaluation metrics
├── visualization.py            # Visualization and plotting tools
├── run_evaluation.py          # Main evaluation script
├── requirements.txt           # Python dependencies
├── README.md                  # This file
├── pdfs/                      # Place your PDF files here
└── plots/                     # Generated visualization charts
```

## 🎯 Usage

### Quick Start

1. **Add PDF Files**
   ```bash
   # Create pdfs directory and add your PDF files
   mkdir -p pdfs
   # Copy your PDF files to pdfs/ directory
   ```

2. **Run Complete Evaluation**
   ```bash
   python run_evaluation.py
   ```

### Advanced Usage

#### Using Individual Components

**1. Initialize RAG System**
```python
from pdf_rag_system import PDFRAGSystem

# Initialize with specific model
rag = PDFRAGSystem(pdf_directory="pdfs", model_name="llama3")
rag.initialize()

# Query the system
result = rag.query("What is this document about?")
print(f"Answer: {result['answer']}")
print(f"Response Time: {result['response_time']:.2f}s")
```

**2. Run Custom Evaluation**
```python
from evaluation_metrics import RAGEvaluator

# Initialize evaluator
evaluator = RAGEvaluator()

# Evaluate single response
metrics = evaluator.evaluate_response(
    question="What is AI?",
    generated="AI is artificial intelligence...",
    reference="Artificial intelligence is...",
    source_docs=result['source_documents'],
    response_time=result['response_time']
)

print(f"Cosine Similarity: {metrics['cosine_similarity']:.4f}")
print(f"BLEU Score: {metrics['bleu']:.4f}")
```

**3. Compare Multiple Models**
```python
from pdf_rag_system import PDFRAGSystem
from evaluation_metrics import RAGEvaluator

# Initialize models
models = {
    'mistral': PDFRAGSystem("pdfs", "mistral"),
    'qwen3': PDFRAGSystem("pdfs", "qwen2.5"),
    'llama3': PDFRAGSystem("pdfs", "llama3")
}

for rag in models.values():
    rag.initialize()

# Run comparison
evaluator = RAGEvaluator()
questions = ["What is the main topic?", "Summarize the key points."]

results = evaluator.compare_models(
    rag_systems=models,
    questions=questions,
    num_trials=3
)
```

**4. Generate Visualizations**
```python
from visualization import RAGVisualizer

visualizer = RAGVisualizer()

# Generate all plots
visualizer.generate_all_plots(results, output_dir="plots")

# Or generate specific plots
visualizer.plot_latency_comparison(results)
visualizer.plot_metric_comparison(results, 'cosine_similarity')
visualizer.plot_trial_scores(results)
```

## 📊 Evaluation Metrics Explained

### Cosine Similarity (0-1, higher is better)
Measures semantic similarity using sentence embeddings. Scores > 0.7 indicate high similarity.

### BLEU Score (0-1, higher is better)
Measures n-gram overlap. Common in machine translation. Scores > 0.5 are good.

### METEOR Score (0-1, higher is better)
Alignment-based metric considering synonyms and stemming. More sophisticated than BLEU.

### BERTScore F1 (0-1, higher is better)
Uses contextual embeddings from BERT. Captures semantic similarity better than n-gram metrics.

### Completeness (0-1, higher is better)
Measures how much of the reference information is covered in the generated answer.

### Hallucination (0-1, lower is better)
Detects information in the answer that's not present in source documents. Scores < 0.3 are good.

### Irrelevance (0-1, lower is better)
Measures how much the answer deviates from the question. Scores < 0.3 indicate relevance.

### Latency (seconds, lower is better)
Time taken to generate response. Varies by model and hardware.

## 📈 Generated Visualizations

After running evaluation, you'll find these charts in the `plots/` directory:

1. **latency_comparison.png** - Response time across models
2. **cosine_similarity_comparison.png** - Semantic similarity scores
3. **bleu_comparison.png** - BLEU scores
4. **meteor_comparison.png** - METEOR scores
5. **bertscore_f1_comparison.png** - BERTScore F1 scores
6. **completeness_comparison.png** - Completeness scores
7. **hallucination_comparison.png** - Hallucination detection
8. **irrelevance_comparison.png** - Irrelevance detection
9. **trial_scores.png** - Performance across 3 trials
10. **all_metrics_comparison.png** - Comprehensive view of all metrics
11. **response_time_distribution.png** - Distribution of response times
12. **summary_table.png** - Tabular summary of all metrics

## 🔧 Configuration

### Customize Number of Trials
Edit `run_evaluation.py`:
```python
num_trials = 5  # Change from default 3
```

### Add Custom Questions
Edit `run_evaluation.py`:
```python
questions = [
    "Your custom question 1?",
    "Your custom question 2?",
    # Add more questions
]
```

### Add Reference Answers (Ground Truth)
Edit `run_evaluation.py`:
```python
references = [
    "Reference answer for question 1",
    "Reference answer for question 2",
    # Add more references (must match questions)
]
```

### Adjust Chunk Size
Edit `pdf_rag_system.py`:
```python
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,      # Change from 1000
    chunk_overlap=100,   # Change from 200
)
```

### Change Embedding Model
Edit `pdf_rag_system.py`:
```python
self.embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)
```

## 🐛 Troubleshooting

### Ollama Connection Error
```bash
# Ensure Ollama is running
ollama serve

# Check if models are installed
ollama list
```

### CUDA/GPU Issues
```bash
# The system uses CPU by default. For GPU support:
pip install faiss-gpu
```

### Memory Issues
- Reduce chunk_size in text splitter
- Process fewer PDFs at once
- Use smaller embedding model

### BERTScore Installation Issues
```bash
# If BERTScore fails, it will fall back to cosine similarity
pip install bert-score --no-deps
pip install transformers torch
```

## 📝 Example Output

```
================================================================================
PDF-based RAG System - Comprehensive Evaluation
Models: Mistral AI, Qwen3, Llama3
================================================================================

✓ Found 3 PDF file(s) in 'pdfs/'
  - document1.pdf
  - document2.pdf
  - document3.pdf

================================================================================
EVALUATION RESULTS
================================================================================

────────────────────────────────────────────────────────────────────────────────
Model: LLAMA3
────────────────────────────────────────────────────────────────────────────────

📊 Performance Metrics:
  Latency (avg):              2.3456s ± 0.1234s

🎯 Similarity Metrics:
  Cosine Similarity:          0.8456 ± 0.0234
  BLEU Score:                 0.6789 ± 0.0456
  METEOR Score:               0.7234 ± 0.0345
  BERTScore F1:               0.8123 ± 0.0234
  Completeness:               0.7890 ± 0.0345

🔍 Quality Metrics:
  Hallucination (lower=better): 0.2345 ± 0.0456
  Irrelevance (lower=better):   0.1234 ± 0.0234

================================================================================
EVALUATION COMPLETE!
================================================================================
```

## 🤝 Contributing

Contributions are welcome! Feel free to:
- Add new evaluation metrics
- Support additional LLM models
- Improve visualizations
- Add new features

## 📄 License

This project is open-source and available under the MIT License.

## 🙏 Acknowledgments

- LangChain for RAG framework
- Ollama for local LLM inference
- HuggingFace for embeddings and evaluation metrics
- FAISS for efficient vector search

## 📧 Support

For issues and questions:
- Create an issue on GitHub
- Check existing documentation
- Review troubleshooting section

---

**Happy Evaluating! 🎉**
