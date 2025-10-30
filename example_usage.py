"""
Example usage demonstrating all features of the PDF RAG System
"""

import os
from pdf_rag_system import PDFRAGSystem
from evaluation_metrics import RAGEvaluator
from visualization import RAGVisualizer


def example_1_basic_usage():
    """Example 1: Basic RAG system usage with a single model"""
    print("\n" + "=" * 80)
    print("EXAMPLE 1: Basic RAG Usage")
    print("=" * 80)
    
    # Initialize RAG system with Llama3
    rag = PDFRAGSystem(pdf_directory="pdfs", model_name="llama3")
    rag.initialize()
    
    # Ask questions
    questions = [
        "What is the main topic of this document?",
        "Can you summarize the key points?",
    ]
    
    for question in questions:
        result = rag.query(question)
        print(f"\nQ: {question}")
        print(f"A: {result['answer']}")
        print(f"Time: {result['response_time']:.2f}s")
        print(f"Sources: {len(result['source_documents'])} documents")


def example_2_compare_models():
    """Example 2: Compare multiple models"""
    print("\n" + "=" * 80)
    print("EXAMPLE 2: Compare Multiple Models")
    print("=" * 80)
    
    # Initialize all models
    models = ['mistral', 'qwen2.5', 'llama3']
    rag_systems = {}
    
    for model_name in models:
        print(f"\nInitializing {model_name}...")
        rag = PDFRAGSystem(pdf_directory="pdfs", model_name=model_name)
        rag.initialize()
        rag_systems[model_name] = rag
    
    # Ask same question to all models
    question = "What is this document about?"
    
    print(f"\nQuestion: {question}\n")
    print("-" * 80)
    
    for model_name, rag in rag_systems.items():
        result = rag.query(question)
        print(f"\n{model_name.upper()}:")
        print(f"Answer: {result['answer'][:200]}...")
        print(f"Time: {result['response_time']:.2f}s")


def example_3_evaluation_metrics():
    """Example 3: Calculate evaluation metrics"""
    print("\n" + "=" * 80)
    print("EXAMPLE 3: Evaluation Metrics")
    print("=" * 80)
    
    # Initialize RAG and evaluator
    rag = PDFRAGSystem(pdf_directory="pdfs", model_name="llama3")
    rag.initialize()
    evaluator = RAGEvaluator()
    
    # Query
    question = "What is artificial intelligence?"
    result = rag.query(question)
    
    # Reference answer (ground truth)
    reference = "Artificial intelligence is the simulation of human intelligence by machines."
    
    # Evaluate
    metrics = evaluator.evaluate_response(
        question=question,
        generated=result['answer'],
        reference=reference,
        source_docs=result['source_documents'],
        response_time=result['response_time']
    )
    
    # Display metrics
    print(f"\nQuestion: {question}")
    print(f"\nGenerated: {result['answer'][:200]}...")
    print(f"\nReference: {reference}")
    print("\nMetrics:")
    print(f"  Cosine Similarity: {metrics['cosine_similarity']:.4f}")
    print(f"  BLEU Score:        {metrics['bleu']:.4f}")
    print(f"  METEOR Score:      {metrics['meteor']:.4f}")
    print(f"  BERTScore F1:      {metrics['bertscore_f1']:.4f}")
    print(f"  Completeness:      {metrics['completeness']:.4f}")
    print(f"  Hallucination:     {metrics['hallucination']:.4f}")
    print(f"  Irrelevance:       {metrics['irrelevance']:.4f}")
    print(f"  Response Time:     {metrics['response_time']:.4f}s")


def example_4_multiple_trials():
    """Example 4: Run multiple trials and aggregate results"""
    print("\n" + "=" * 80)
    print("EXAMPLE 4: Multiple Trials Evaluation")
    print("=" * 80)
    
    # Initialize
    rag = PDFRAGSystem(pdf_directory="pdfs", model_name="llama3")
    rag.initialize()
    evaluator = RAGEvaluator()
    
    # Questions
    questions = [
        "What is the main topic?",
        "Summarize the key points.",
    ]
    
    # Run 3 trials
    results = evaluator.evaluate_multiple_trials(
        rag_system=rag,
        questions=questions,
        references=None,  # No ground truth
        num_trials=3
    )
    
    # Display aggregated results
    print("\nAggregated Results (3 trials):")
    agg = results['aggregated']
    print(f"  Avg Latency:       {agg['latency_mean']:.4f}s ± {agg['latency_std']:.4f}s")
    print(f"  Avg Hallucination: {agg['hallucination_mean']:.4f} ± {agg['hallucination_std']:.4f}")
    print(f"  Avg Irrelevance:   {agg['irrelevance_mean']:.4f} ± {agg['irrelevance_std']:.4f}")


def example_5_full_comparison_with_viz():
    """Example 5: Full model comparison with visualizations"""
    print("\n" + "=" * 80)
    print("EXAMPLE 5: Full Comparison with Visualizations")
    print("=" * 80)
    
    # Initialize all models
    models = {
        'mistral': PDFRAGSystem("pdfs", "mistral"),
        'qwen3': PDFRAGSystem("pdfs", "qwen2.5"),
        'llama3': PDFRAGSystem("pdfs", "llama3")
    }
    
    for name, rag in models.items():
        print(f"Initializing {name}...")
        rag.initialize()
    
    # Questions
    questions = [
        "What is the main topic?",
        "Summarize the document.",
    ]
    
    # Evaluate
    evaluator = RAGEvaluator()
    comparison_results = evaluator.compare_models(
        rag_systems=models,
        questions=questions,
        num_trials=3
    )
    
    # Visualize
    visualizer = RAGVisualizer()
    
    # Generate specific plots
    print("\nGenerating visualizations...")
    visualizer.plot_latency_comparison(comparison_results, "example_latency.png")
    visualizer.plot_trial_scores(comparison_results, "example_trials.png")
    
    print("✓ Visualizations saved!")
    
    # Display summary
    print("\nModel Performance Summary:")
    for model_name, results in comparison_results.items():
        agg = results['aggregated']
        print(f"\n{model_name.upper()}:")
        print(f"  Avg Latency: {agg['latency_mean']:.4f}s")
        print(f"  Hallucination: {agg['hallucination_mean']:.4f}")


def example_6_save_load_vectorstore():
    """Example 6: Save and load vectorstore for faster initialization"""
    print("\n" + "=" * 80)
    print("EXAMPLE 6: Save and Load Vectorstore")
    print("=" * 80)
    
    # First time: create and save
    print("\nCreating vectorstore...")
    rag = PDFRAGSystem(pdf_directory="pdfs", model_name="llama3")
    
    # Load and process PDFs
    documents = rag.load_pdfs()
    chunks = rag.split_documents(documents)
    rag.create_vectorstore(chunks)
    
    # Save vectorstore
    rag.save_vectorstore("my_vectorstore")
    print("✓ Vectorstore saved!")
    
    # Later: load from disk (much faster!)
    print("\nLoading vectorstore from disk...")
    rag2 = PDFRAGSystem(pdf_directory="pdfs", model_name="llama3")
    rag2.load_vectorstore("my_vectorstore")
    rag2.setup_qa_chain()
    print("✓ Vectorstore loaded!")
    
    # Query
    result = rag2.query("What is this about?")
    print(f"\nAnswer: {result['answer'][:200]}...")


def main():
    """Run all examples"""
    
    # Check if PDFs exist
    if not os.path.exists("pdfs") or not any(f.endswith('.pdf') for f in os.listdir("pdfs")):
        print("\n⚠️  No PDF files found!")
        print("Please create 'pdfs/' directory and add PDF files before running examples.")
        return
    
    print("=" * 80)
    print("PDF RAG System - Example Usage")
    print("=" * 80)
    
    examples = {
        '1': ("Basic Usage", example_1_basic_usage),
        '2': ("Compare Models", example_2_compare_models),
        '3': ("Evaluation Metrics", example_3_evaluation_metrics),
        '4': ("Multiple Trials", example_4_multiple_trials),
        '5': ("Full Comparison with Viz", example_5_full_comparison_with_viz),
        '6': ("Save/Load Vectorstore", example_6_save_load_vectorstore),
    }
    
    print("\nAvailable Examples:")
    for key, (name, _) in examples.items():
        print(f"  {key}. {name}")
    
    print("\nNote: Make sure Ollama is running with required models installed!")
    print("Run all examples by uncommenting the function calls below.")
    
    # Uncomment to run specific examples:
    # example_1_basic_usage()
    # example_2_compare_models()
    # example_3_evaluation_metrics()
    # example_4_multiple_trials()
    # example_5_full_comparison_with_viz()
    # example_6_save_load_vectorstore()


if __name__ == "__main__":
    main()
