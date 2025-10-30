"""
Visualization Module for RAG System Evaluation
Creates comprehensive comparison charts and plots
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Dict, List, Any
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)


class RAGVisualizer:
    """Visualization tools for RAG evaluation results"""
    
    def __init__(self):
        """Initialize visualizer"""
        self.colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8']
        
    def plot_latency_comparison(
        self,
        comparison_results: Dict[str, Any],
        save_path: str = "latency_comparison.png"
    ):
        """
        Create bar chart comparing latency across models
        
        Args:
            comparison_results: Results from evaluator.compare_models()
            save_path: Path to save the plot
        """
        models = list(comparison_results.keys())
        latencies = []
        errors = []
        
        for model in models:
            agg = comparison_results[model]['aggregated']
            latencies.append(agg.get('latency_mean', 0))
            errors.append(agg.get('latency_std', 0))
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        x_pos = np.arange(len(models))
        bars = ax.bar(x_pos, latencies, yerr=errors, capsize=5,
                      color=self.colors[:len(models)], alpha=0.8, edgecolor='black')
        
        ax.set_xlabel('Model', fontsize=12, fontweight='bold')
        ax.set_ylabel('Average Response Time (seconds)', fontsize=12, fontweight='bold')
        ax.set_title('Latency Comparison Across Models', fontsize=14, fontweight='bold')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(models, rotation=45, ha='right')
        ax.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for i, bar in enumerate(bars):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{latencies[i]:.3f}s',
                   ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Latency comparison saved to {save_path}")
        plt.close()
    
    def plot_metric_comparison(
        self,
        comparison_results: Dict[str, Any],
        metric: str,
        save_path: str = None
    ):
        """
        Create bar chart for a specific metric
        
        Args:
            comparison_results: Results from evaluator.compare_models()
            metric: Metric name (e.g., 'cosine_similarity', 'bleu', etc.)
            save_path: Path to save the plot
        """
        if save_path is None:
            save_path = f"{metric}_comparison.png"
        
        models = list(comparison_results.keys())
        values = []
        errors = []
        
        for model in models:
            agg = comparison_results[model]['aggregated']
            values.append(agg.get(f'{metric}_mean', 0))
            errors.append(agg.get(f'{metric}_std', 0))
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        x_pos = np.arange(len(models))
        bars = ax.bar(x_pos, values, yerr=errors, capsize=5,
                      color=self.colors[:len(models)], alpha=0.8, edgecolor='black')
        
        ax.set_xlabel('Model', fontsize=12, fontweight='bold')
        ax.set_ylabel(f'{metric.replace("_", " ").title()}', fontsize=12, fontweight='bold')
        ax.set_title(f'{metric.replace("_", " ").title()} Comparison', fontsize=14, fontweight='bold')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(models, rotation=45, ha='right')
        ax.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for i, bar in enumerate(bars):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{values[i]:.3f}',
                   ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"{metric} comparison saved to {save_path}")
        plt.close()
    
    def plot_all_metrics_comparison(
        self,
        comparison_results: Dict[str, Any],
        save_path: str = "all_metrics_comparison.png"
    ):
        """
        Create comprehensive comparison of all metrics
        
        Args:
            comparison_results: Results from evaluator.compare_models()
            save_path: Path to save the plot
        """
        metrics = [
            'cosine_similarity', 'bleu', 'meteor', 'bertscore_f1',
            'completeness', 'hallucination', 'irrelevance', 'latency'
        ]
        
        models = list(comparison_results.keys())
        
        # Prepare data
        data = []
        for model in models:
            agg = comparison_results[model]['aggregated']
            row = {'Model': model}
            for metric in metrics:
                row[metric] = agg.get(f'{metric}_mean', 0)
            data.append(row)
        
        df = pd.DataFrame(data)
        
        # Create subplots
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        axes = axes.flatten()
        
        for idx, metric in enumerate(metrics):
            ax = axes[idx]
            
            # Skip if metric not available
            if metric not in df.columns:
                continue
            
            values = df[metric].values
            x_pos = np.arange(len(models))
            
            bars = ax.bar(x_pos, values, color=self.colors[:len(models)],
                         alpha=0.8, edgecolor='black')
            
            ax.set_xlabel('Model', fontsize=10, fontweight='bold')
            ax.set_ylabel('Score', fontsize=10, fontweight='bold')
            ax.set_title(metric.replace('_', ' ').title(), fontsize=11, fontweight='bold')
            ax.set_xticks(x_pos)
            ax.set_xticklabels(models, rotation=45, ha='right', fontsize=9)
            ax.grid(axis='y', alpha=0.3)
            
            # Add value labels
            for i, bar in enumerate(bars):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{values[i]:.3f}',
                       ha='center', va='bottom', fontsize=8, fontweight='bold')
        
        plt.suptitle('Comprehensive Metrics Comparison Across Models',
                    fontsize=16, fontweight='bold', y=0.995)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"All metrics comparison saved to {save_path}")
        plt.close()
    
    def plot_trial_scores(
        self,
        comparison_results: Dict[str, Any],
        save_path: str = "trial_scores.png"
    ):
        """
        Plot scores across different trials
        
        Args:
            comparison_results: Results from evaluator.compare_models()
            save_path: Path to save the plot
        """
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        models = list(comparison_results.keys())
        num_trials = comparison_results[models[0]]['num_trials']
        
        for trial_idx in range(num_trials):
            ax = axes[trial_idx]
            
            trial_data = []
            for model in models:
                trials = comparison_results[model]['trials']
                if trial_idx < len(trials):
                    trial_results = trials[trial_idx]
                    avg_score = np.mean([
                        r.get('cosine_similarity', 0) or 0
                        for r in trial_results
                    ])
                    trial_data.append(avg_score)
                else:
                    trial_data.append(0)
            
            x_pos = np.arange(len(models))
            bars = ax.bar(x_pos, trial_data, color=self.colors[:len(models)],
                         alpha=0.8, edgecolor='black')
            
            ax.set_xlabel('Model', fontsize=11, fontweight='bold')
            ax.set_ylabel('Average Cosine Similarity', fontsize=11, fontweight='bold')
            ax.set_title(f'Trial {trial_idx + 1} Scores', fontsize=12, fontweight='bold')
            ax.set_xticks(x_pos)
            ax.set_xticklabels(models, rotation=45, ha='right')
            ax.grid(axis='y', alpha=0.3)
            ax.set_ylim([0, 1])
            
            # Add value labels
            for i, bar in enumerate(bars):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{trial_data[i]:.3f}',
                       ha='center', va='bottom', fontweight='bold')
        
        plt.suptitle('Model Performance Across Trials', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Trial scores saved to {save_path}")
        plt.close()
    
    def plot_response_time_distribution(
        self,
        comparison_results: Dict[str, Any],
        save_path: str = "response_time_distribution.png"
    ):
        """
        Plot distribution of response times
        
        Args:
            comparison_results: Results from evaluator.compare_models()
            save_path: Path to save the plot
        """
        fig, ax = plt.subplots(figsize=(12, 6))
        
        for idx, (model_name, results) in enumerate(comparison_results.items()):
            response_times = []
            for trial in results['trials']:
                for result in trial:
                    response_times.append(result['response_time'])
            
            ax.hist(response_times, bins=20, alpha=0.6, label=model_name,
                   color=self.colors[idx % len(self.colors)], edgecolor='black')
        
        ax.set_xlabel('Response Time (seconds)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Frequency', fontsize=12, fontweight='bold')
        ax.set_title('Response Time Distribution', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Response time distribution saved to {save_path}")
        plt.close()
    
    def create_summary_table(
        self,
        comparison_results: Dict[str, Any],
        save_path: str = "summary_table.png"
    ):
        """
        Create a summary table of all metrics
        
        Args:
            comparison_results: Results from evaluator.compare_models()
            save_path: Path to save the plot
        """
        metrics = [
            'latency', 'cosine_similarity', 'bleu', 'meteor',
            'bertscore_f1', 'completeness', 'hallucination', 'irrelevance'
        ]
        
        models = list(comparison_results.keys())
        
        # Prepare data
        table_data = []
        for metric in metrics:
            row = [metric.replace('_', ' ').title()]
            for model in models:
                agg = comparison_results[model]['aggregated']
                value = agg.get(f'{metric}_mean', 0)
                row.append(f'{value:.4f}')
            table_data.append(row)
        
        # Create table
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.axis('tight')
        ax.axis('off')
        
        columns = ['Metric'] + models
        table = ax.table(cellText=table_data, colLabels=columns,
                        cellLoc='center', loc='center',
                        colColours=['lightgray'] * len(columns))
        
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        
        # Style header
        for i in range(len(columns)):
            table[(0, i)].set_facecolor('#4ECDC4')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        plt.title('Summary Table of All Metrics', fontsize=14, fontweight='bold', pad=20)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Summary table saved to {save_path}")
        plt.close()
    
    def generate_all_plots(self, comparison_results: Dict[str, Any], output_dir: str = "plots"):
        """
        Generate all visualization plots
        
        Args:
            comparison_results: Results from evaluator.compare_models()
            output_dir: Directory to save plots
        """
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"\nGenerating all visualization plots in '{output_dir}/'...")
        print("=" * 60)
        
        # Individual metric plots
        metrics = ['cosine_similarity', 'bleu', 'meteor', 'bertscore_f1',
                  'completeness', 'hallucination', 'irrelevance']
        
        for metric in metrics:
            self.plot_metric_comparison(
                comparison_results,
                metric,
                save_path=f"{output_dir}/{metric}_comparison.png"
            )
        
        # Latency comparison
        self.plot_latency_comparison(
            comparison_results,
            save_path=f"{output_dir}/latency_comparison.png"
        )
        
        # All metrics comparison
        self.plot_all_metrics_comparison(
            comparison_results,
            save_path=f"{output_dir}/all_metrics_comparison.png"
        )
        
        # Trial scores
        self.plot_trial_scores(
            comparison_results,
            save_path=f"{output_dir}/trial_scores.png"
        )
        
        # Response time distribution
        self.plot_response_time_distribution(
            comparison_results,
            save_path=f"{output_dir}/response_time_distribution.png"
        )
        
        # Summary table
        self.create_summary_table(
            comparison_results,
            save_path=f"{output_dir}/summary_table.png"
        )
        
        print("=" * 60)
        print(f"All plots generated successfully in '{output_dir}/' directory!")


def main():
    """Example usage"""
    print("RAG Visualization Module")
    print("=" * 60)
    print("\nAvailable visualizations:")
    print("  - Latency Comparison")
    print("  - Cosine Similarity Comparison")
    print("  - BLEU Score Comparison")
    print("  - METEOR Score Comparison")
    print("  - BERTScore F1 Comparison")
    print("  - Completeness Comparison")
    print("  - Hallucination Comparison")
    print("  - Irrelevance Comparison")
    print("  - Trial Scores (1, 2, 3)")
    print("  - Response Time Distribution")
    print("  - Summary Table")
    print("=" * 60)


if __name__ == "__main__":
    main()
