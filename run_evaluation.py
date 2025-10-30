"""
Main script to run comprehensive RAG system evaluation
Evaluates Mistral, Qwen3, and Llama3 models with all metrics
"""

import os
import sys
from pdf_rag_system import PDFRAGSystem
from evaluation_metrics import RAGEvaluator
from visualization import RAGVisualizer


def main():
    """Run complete evaluation pipeline"""
    
    print("=" * 80)
    print("PDF-based RAG System - Comprehensive Evaluation")
    print("Models: Mistral AI, Qwen3, Llama3")
    print("=" * 80)
    
    # Configuration
    pdf_directory = "pdfs"
    num_trials = 3
    
    # Create PDF directory if it doesn't exist
    os.makedirs(pdf_directory, exist_ok=True)
    
    # Check if PDFs exist
    pdf_files = [f for f in os.listdir(pdf_directory) if f.endswith('.pdf')]
    if not pdf_files:
        print(f"\n⚠️  No PDF files found in '{pdf_directory}/' directory!")
        print(f"Please add PDF files to '{pdf_directory}/' and run again.")
        print("=" * 80)
        return
    
    print(f"\n✓ Found {len(pdf_files)} PDF file(s) in '{pdf_directory}/'")
    for pdf in pdf_files:
        print(f"  - {pdf}")
    
    # Define test questions
    print("\n" + "=" * 80)
    print("Test Questions:")
    print("=" * 80)
    
    questions = [
        "What is the main topic of this document?",
        "Can you summarize the key points?",
        "What are the most important details mentioned?",
    ]
    
    for i, q in enumerate(questions, 1):
        print(f"{i}. {q}")
    
    # Optional: Define reference answers if you have them
    # If you don't have reference answers, set to None
    references = None  # You can add ground truth answers here
    
    # Initialize models
    print("\n" + "=" * 80)
    print("Initializing RAG Systems...")
    print("=" * 80)
    
    models = ['mistral', 'qwen2.5', 'llama3']
    rag_systems = {}
    
    for model_name in models:
        print(f"\nInitializing {model_name}...")
        try:
            rag = PDFRAGSystem(pdf_directory=pdf_directory, model_name=model_name)
            rag.initialize()
            rag_systems[model_name] = rag
            print(f"✓ {model_name} initialized successfully")
        except Exception as e:
            print(f"✗ Error initializing {model_name}: {str(e)}")
            print(f"  Make sure Ollama is running and model is installed:")
            print(f"  ollama pull {model_name}")
    
    if not rag_systems:
        print("\n⚠️  No models were successfully initialized!")
        print("Please ensure Ollama is running and models are installed.")
        return
    
    # Initialize evaluator
    print("\n" + "=" * 80)
    print("Initializing Evaluator...")
    print("=" * 80)
    evaluator = RAGEvaluator()
    
    # Run evaluation
    print("\n" + "=" * 80)
    print(f"Running Evaluation ({num_trials} trials per model)...")
    print("=" * 80)
    
    comparison_results = evaluator.compare_models(
        rag_systems=rag_systems,
        questions=questions,
        references=references,
        num_trials=num_trials
    )
    
    # Display results
    print("\n" + "=" * 80)
    print("EVALUATION RESULTS")
    print("=" * 80)
    
    for model_name, results in comparison_results.items():
        print(f"\n{'─' * 80}")
        print(f"Model: {model_name.upper()}")
        print(f"{'─' * 80}")
        
        agg = results['aggregated']
        
        print(f"\n📊 Performance Metrics:")
        print(f"  Latency (avg):              {agg.get('latency_mean', 0):.4f}s ± {agg.get('latency_std', 0):.4f}s")
        
        if agg.get('cosine_similarity_mean') is not None:
            print(f"\n🎯 Similarity Metrics:")
            print(f"  Cosine Similarity:          {agg.get('cosine_similarity_mean', 0):.4f} ± {agg.get('cosine_similarity_std', 0):.4f}")
            print(f"  BLEU Score:                 {agg.get('bleu_mean', 0):.4f} ± {agg.get('bleu_std', 0):.4f}")
            print(f"  METEOR Score:               {agg.get('meteor_mean', 0):.4f} ± {agg.get('meteor_std', 0):.4f}")
            print(f"  BERTScore F1:               {agg.get('bertscore_f1_mean', 0):.4f} ± {agg.get('bertscore_f1_std', 0):.4f}")
            print(f"  Completeness:               {agg.get('completeness_mean', 0):.4f} ± {agg.get('completeness_std', 0):.4f}")
        
        print(f"\n🔍 Quality Metrics:")
        print(f"  Hallucination (lower=better): {agg.get('hallucination_mean', 0):.4f} ± {agg.get('hallucination_std', 0):.4f}")
        print(f"  Irrelevance (lower=better):   {agg.get('irrelevance_mean', 0):.4f} ± {agg.get('irrelevance_std', 0):.4f}")
    
    # Generate visualizations
    print("\n" + "=" * 80)
    print("Generating Visualizations...")
    print("=" * 80)
    
    visualizer = RAGVisualizer()
    visualizer.generate_all_plots(comparison_results, output_dir="plots")
    
    # Save detailed results
    import json
    
    # Prepare results for JSON (remove non-serializable objects)
    json_results = {}
    for model_name, results in comparison_results.items():
        json_results[model_name] = {
            'aggregated': results['aggregated'],
            'num_trials': results['num_trials'],
            'num_questions': results['num_questions']
        }
    
    with open('evaluation_results.json', 'w') as f:
        json.dump(json_results, f, indent=2)
    
    print("\n✓ Detailed results saved to 'evaluation_results.json'")
    
    print("\n" + "=" * 80)
    print("EVALUATION COMPLETE!")
    print("=" * 80)
    print("\n📁 Generated Files:")
    print("  - plots/ directory with all visualization charts")
    print("  - evaluation_results.json with detailed metrics")
    print("\n📈 Available Charts:")
    print("  - latency_comparison.png")
    print("  - cosine_similarity_comparison.png")
    print("  - bleu_comparison.png")
    print("  - meteor_comparison.png")
    print("  - bertscore_f1_comparison.png")
    print("  - completeness_comparison.png")
    print("  - hallucination_comparison.png")
    print("  - irrelevance_comparison.png")
    print("  - trial_scores.png (Trials 1, 2, 3)")
    print("  - all_metrics_comparison.png")
    print("  - response_time_distribution.png")
    print("  - summary_table.png")
    print("=" * 80)


if __name__ == "__main__":
    main()
