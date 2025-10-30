"""
Comprehensive Evaluation Metrics for RAG Systems
Includes: Cosine Similarity, F1, BERTScore, METEOR, BLEU, Completeness, Hallucination, Irrelevance
"""

import time
import numpy as np
from typing import List, Dict, Any, Optional
import warnings
warnings.filterwarnings('ignore')

# NLP and evaluation metrics
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from nltk.tokenize import word_tokenize
import evaluate
from sentence_transformers import SentenceTransformer

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('omw-1.4', quiet=True)


class RAGEvaluator:
    """Comprehensive evaluation for RAG systems"""
    
    def __init__(self):
        """Initialize evaluation metrics"""
        print("Initializing evaluation metrics...")
        
        # Load embedding model for semantic similarity
        self.embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        
        # Load BERTScore
        try:
            self.bertscore = evaluate.load("bertscore")
        except:
            print("Warning: BERTScore not available, will use alternative metrics")
            self.bertscore = None
        
        print("Evaluator initialized successfully")
    
    def calculate_cosine_similarity(self, reference: str, generated: str) -> float:
        """
        Calculate cosine similarity between reference and generated text
        
        Args:
            reference: Ground truth text
            generated: Generated text from model
            
        Returns:
            Cosine similarity score (0-1)
        """
        # Get embeddings
        ref_embedding = self.embedding_model.encode([reference])
        gen_embedding = self.embedding_model.encode([generated])
        
        # Calculate cosine similarity
        similarity = cosine_similarity(ref_embedding, gen_embedding)[0][0]
        
        return float(similarity)
    
    def calculate_bleu(self, reference: str, generated: str) -> float:
        """
        Calculate BLEU score
        
        Args:
            reference: Ground truth text
            generated: Generated text from model
            
        Returns:
            BLEU score (0-1)
        """
        # Tokenize
        reference_tokens = word_tokenize(reference.lower())
        generated_tokens = word_tokenize(generated.lower())
        
        # Calculate BLEU with smoothing
        smoothing = SmoothingFunction().method1
        score = sentence_bleu(
            [reference_tokens], 
            generated_tokens,
            smoothing_function=smoothing
        )
        
        return float(score)
    
    def calculate_meteor(self, reference: str, generated: str) -> float:
        """
        Calculate METEOR score
        
        Args:
            reference: Ground truth text
            generated: Generated text from model
            
        Returns:
            METEOR score (0-1)
        """
        # Tokenize
        reference_tokens = word_tokenize(reference.lower())
        generated_tokens = word_tokenize(generated.lower())
        
        # Calculate METEOR
        score = meteor_score([reference_tokens], generated_tokens)
        
        return float(score)
    
    def calculate_bertscore(self, reference: str, generated: str) -> Dict[str, float]:
        """
        Calculate BERTScore (Precision, Recall, F1)
        
        Args:
            reference: Ground truth text
            generated: Generated text from model
            
        Returns:
            Dictionary with precision, recall, and F1 scores
        """
        if self.bertscore is None:
            # Fallback to cosine similarity
            sim = self.calculate_cosine_similarity(reference, generated)
            return {
                'precision': sim,
                'recall': sim,
                'f1': sim
            }
        
        results = self.bertscore.compute(
            predictions=[generated],
            references=[reference],
            lang="en"
        )
        
        return {
            'precision': float(results['precision'][0]),
            'recall': float(results['recall'][0]),
            'f1': float(results['f1'][0])
        }
    
    def calculate_completeness(self, reference: str, generated: str, source_docs: List[Any]) -> float:
        """
        Measure completeness - how much of the reference information is covered
        
        Args:
            reference: Ground truth text
            generated: Generated text from model
            source_docs: Source documents used for generation
            
        Returns:
            Completeness score (0-1)
        """
        # Tokenize reference and generated
        ref_tokens = set(word_tokenize(reference.lower()))
        gen_tokens = set(word_tokenize(generated.lower()))
        
        # Calculate overlap
        if len(ref_tokens) == 0:
            return 0.0
        
        overlap = len(ref_tokens.intersection(gen_tokens))
        completeness = overlap / len(ref_tokens)
        
        return float(completeness)
    
    def calculate_hallucination(self, generated: str, source_docs: List[Any]) -> float:
        """
        Measure hallucination - information in generated text not in source
        
        Args:
            generated: Generated text from model
            source_docs: Source documents used for generation
            
        Returns:
            Hallucination score (0-1, lower is better)
        """
        # Combine source documents
        source_text = " ".join([doc.page_content for doc in source_docs])
        
        # Get embeddings
        gen_embedding = self.embedding_model.encode([generated])
        source_embedding = self.embedding_model.encode([source_text])
        
        # Calculate similarity
        similarity = cosine_similarity(gen_embedding, source_embedding)[0][0]
        
        # Hallucination is inverse of similarity
        hallucination = 1.0 - similarity
        
        return float(max(0.0, hallucination))
    
    def calculate_irrelevance(self, question: str, generated: str) -> float:
        """
        Measure irrelevance - how much the answer deviates from the question
        
        Args:
            question: The input question
            generated: Generated answer
            
        Returns:
            Irrelevance score (0-1, lower is better)
        """
        # Get embeddings
        question_embedding = self.embedding_model.encode([question])
        answer_embedding = self.embedding_model.encode([generated])
        
        # Calculate similarity
        similarity = cosine_similarity(question_embedding, answer_embedding)[0][0]
        
        # Irrelevance is inverse of similarity
        irrelevance = 1.0 - similarity
        
        return float(max(0.0, irrelevance))
    
    def evaluate_response(
        self,
        question: str,
        generated: str,
        reference: Optional[str],
        source_docs: List[Any],
        response_time: float
    ) -> Dict[str, Any]:
        """
        Comprehensive evaluation of a single response
        
        Args:
            question: The input question
            generated: Generated answer
            reference: Ground truth answer (optional)
            source_docs: Source documents used
            response_time: Time taken to generate response
            
        Returns:
            Dictionary with all evaluation metrics
        """
        metrics = {
            'response_time': response_time,
            'latency': response_time,
        }
        
        # Metrics that require reference
        if reference:
            metrics['cosine_similarity'] = self.calculate_cosine_similarity(reference, generated)
            metrics['bleu'] = self.calculate_bleu(reference, generated)
            metrics['meteor'] = self.calculate_meteor(reference, generated)
            
            bertscore = self.calculate_bertscore(reference, generated)
            metrics['bertscore_precision'] = bertscore['precision']
            metrics['bertscore_recall'] = bertscore['recall']
            metrics['bertscore_f1'] = bertscore['f1']
            
            metrics['completeness'] = self.calculate_completeness(reference, generated, source_docs)
        else:
            # Set to None if no reference
            metrics['cosine_similarity'] = None
            metrics['bleu'] = None
            metrics['meteor'] = None
            metrics['bertscore_precision'] = None
            metrics['bertscore_recall'] = None
            metrics['bertscore_f1'] = None
            metrics['completeness'] = None
        
        # Metrics that don't require reference
        metrics['hallucination'] = self.calculate_hallucination(generated, source_docs)
        metrics['irrelevance'] = self.calculate_irrelevance(question, generated)
        
        return metrics
    
    def evaluate_multiple_trials(
        self,
        rag_system,
        questions: List[str],
        references: Optional[List[str]] = None,
        num_trials: int = 3
    ) -> Dict[str, Any]:
        """
        Evaluate RAG system across multiple trials
        
        Args:
            rag_system: Initialized RAG system
            questions: List of questions to ask
            references: Optional list of reference answers
            num_trials: Number of trials to run
            
        Returns:
            Dictionary with trial results and aggregated metrics
        """
        all_trials = []
        
        for trial_num in range(1, num_trials + 1):
            print(f"\nRunning Trial {trial_num}/{num_trials}...")
            trial_results = []
            
            for i, question in enumerate(questions):
                # Get response
                result = rag_system.query(question)
                
                # Get reference if available
                reference = references[i] if references and i < len(references) else None
                
                # Evaluate
                metrics = self.evaluate_response(
                    question=question,
                    generated=result['answer'],
                    reference=reference,
                    source_docs=result['source_documents'],
                    response_time=result['response_time']
                )
                
                metrics['question'] = question
                metrics['answer'] = result['answer']
                metrics['trial'] = trial_num
                
                trial_results.append(metrics)
            
            all_trials.append(trial_results)
            print(f"Trial {trial_num} completed")
        
        # Aggregate results
        aggregated = self._aggregate_trials(all_trials)
        
        return {
            'trials': all_trials,
            'aggregated': aggregated,
            'num_trials': num_trials,
            'num_questions': len(questions)
        }
    
    def _aggregate_trials(self, all_trials: List[List[Dict]]) -> Dict[str, Any]:
        """Aggregate metrics across trials"""
        
        metric_names = [
            'response_time', 'latency', 'cosine_similarity', 'bleu', 'meteor',
            'bertscore_f1', 'completeness', 'hallucination', 'irrelevance'
        ]
        
        aggregated = {}
        
        for metric in metric_names:
            values = []
            for trial in all_trials:
                for result in trial:
                    if result.get(metric) is not None:
                        values.append(result[metric])
            
            if values:
                aggregated[f'{metric}_mean'] = np.mean(values)
                aggregated[f'{metric}_std'] = np.std(values)
                aggregated[f'{metric}_min'] = np.min(values)
                aggregated[f'{metric}_max'] = np.max(values)
        
        return aggregated
    
    def compare_models(
        self,
        rag_systems: Dict[str, Any],
        questions: List[str],
        references: Optional[List[str]] = None,
        num_trials: int = 3
    ) -> Dict[str, Any]:
        """
        Compare multiple models
        
        Args:
            rag_systems: Dictionary of model_name -> RAG system
            questions: List of questions
            references: Optional reference answers
            num_trials: Number of trials
            
        Returns:
            Comparison results for all models
        """
        comparison = {}
        
        for model_name, rag_system in rag_systems.items():
            print(f"\n{'='*60}")
            print(f"Evaluating model: {model_name}")
            print(f"{'='*60}")
            
            results = self.evaluate_multiple_trials(
                rag_system=rag_system,
                questions=questions,
                references=references,
                num_trials=num_trials
            )
            
            comparison[model_name] = results
        
        return comparison


def main():
    """Example usage"""
    print("RAG Evaluation Metrics Module")
    print("=" * 60)
    print("\nAvailable metrics:")
    print("  - Cosine Similarity")
    print("  - BLEU Score")
    print("  - METEOR Score")
    print("  - BERTScore (F1, Precision, Recall)")
    print("  - Completeness")
    print("  - Hallucination Detection")
    print("  - Irrelevance Detection")
    print("  - Response Time / Latency")
    print("=" * 60)


if __name__ == "__main__":
    main()
