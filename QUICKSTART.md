# Quick Start Guide

Get started with the PDF RAG System in 5 minutes!

## 🚀 Quick Setup (Automated)

### Option 1: Using Setup Script (Recommended)

```bash
# Run the automated setup script
bash setup.sh
```

The script will:
- ✅ Check Python installation
- ✅ Check Ollama installation
- ✅ Create virtual environment
- ✅ Install all dependencies
- ✅ Download NLTK data
- ✅ Pull required Ollama models
- ✅ Create necessary directories

### Option 2: Manual Setup

```bash
# 1. Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download NLTK data
python -c "import nltk; nltk.download('punkt'); nltk.download('wordnet'); nltk.download('omw-1.4')"

# 4. Install Ollama models
ollama pull mistral
ollama pull qwen2.5
ollama pull llama3
```

## 📝 Usage in 3 Steps

### Step 1: Add Your PDFs
```bash
# Copy your PDF files to the pdfs directory
cp /path/to/your/document.pdf pdfs/
```

### Step 2: Run Evaluation
```bash
# Activate virtual environment if not already active
source venv/bin/activate

# Run the complete evaluation
python run_evaluation.py
```

### Step 3: View Results
```bash
# Check the generated plots
ls plots/

# View detailed metrics
cat evaluation_results.json
```

## 📊 What You'll Get

After running the evaluation, you'll have:

### 📁 Files
- `evaluation_results.json` - Detailed metrics in JSON format
- `plots/` directory with 12 visualization charts

### 📈 Charts Generated
1. ✅ **latency_comparison.png** - Response time comparison
2. ✅ **cosine_similarity_comparison.png** - Semantic similarity
3. ✅ **bleu_comparison.png** - BLEU scores
4. ✅ **meteor_comparison.png** - METEOR scores
5. ✅ **bertscore_f1_comparison.png** - BERTScore F1
6. ✅ **completeness_comparison.png** - Answer completeness
7. ✅ **hallucination_comparison.png** - Hallucination detection
8. ✅ **irrelevance_comparison.png** - Irrelevance detection
9. ✅ **trial_scores.png** - Performance across 3 trials
10. ✅ **all_metrics_comparison.png** - All metrics overview
11. ✅ **response_time_distribution.png** - Response time distribution
12. ✅ **summary_table.png** - Summary table

## 🎯 Example: Custom Evaluation

```python
from pdf_rag_system import PDFRAGSystem
from evaluation_metrics import RAGEvaluator
from visualization import RAGVisualizer

# 1. Initialize RAG system
rag = PDFRAGSystem(pdf_directory="pdfs", model_name="llama3")
rag.initialize()

# 2. Ask a question
result = rag.query("What is the main topic of this document?")
print(f"Answer: {result['answer']}")
print(f"Time: {result['response_time']:.2f}s")

# 3. Evaluate with reference answer (optional)
evaluator = RAGEvaluator()
metrics = evaluator.evaluate_response(
    question="What is the main topic?",
    generated=result['answer'],
    reference="The main topic is...",  # Your ground truth
    source_docs=result['source_documents'],
    response_time=result['response_time']
)

print(f"Cosine Similarity: {metrics['cosine_similarity']:.4f}")
print(f"BLEU Score: {metrics['bleu']:.4f}")
```

## 🔧 Common Customizations

### Change Number of Trials
Edit `run_evaluation.py`:
```python
num_trials = 5  # Default is 3
```

### Add Custom Questions
Edit `run_evaluation.py`:
```python
questions = [
    "Your question 1?",
    "Your question 2?",
    "Your question 3?",
]
```

### Use Only Specific Models
Edit `run_evaluation.py`:
```python
models = ['llama3']  # Only use Llama3
# models = ['mistral', 'llama3']  # Use Mistral and Llama3
```

## 🐛 Troubleshooting

### Problem: "No PDF files found"
**Solution:** Add PDF files to the `pdfs/` directory
```bash
cp your_document.pdf pdfs/
```

### Problem: "Ollama connection refused"
**Solution:** Start Ollama server
```bash
ollama serve
```

### Problem: "Model not found"
**Solution:** Pull the required model
```bash
ollama pull mistral
ollama pull qwen2.5
ollama pull llama3
```

### Problem: Import errors
**Solution:** Reinstall dependencies
```bash
pip install -r requirements.txt --force-reinstall
```

## 💡 Tips for Best Results

1. **Use Quality PDFs**: Clear, text-based PDFs work best (not scanned images)
2. **Multiple Trials**: Run 3-5 trials for reliable metrics
3. **Ground Truth**: Provide reference answers for more metrics
4. **Hardware**: Models run faster with GPU support
5. **Chunk Size**: Adjust based on document structure (see README.md)

## 📚 Next Steps

- 📖 Read the full [README.md](README.md) for detailed documentation
- 💻 Check [example_usage.py](example_usage.py) for code examples
- 🎨 Customize visualizations in [visualization.py](visualization.py)
- 📊 Add more metrics in [evaluation_metrics.py](evaluation_metrics.py)

## 🎉 Ready to Go!

```bash
# Add your PDFs
cp my_document.pdf pdfs/

# Run evaluation
python run_evaluation.py

# View results
ls plots/
```

That's it! You now have a comprehensive RAG evaluation system running! 🚀
