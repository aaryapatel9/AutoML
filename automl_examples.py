"""
AutoML Framework - Complete Examples
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris, load_breast_cancer, load_diabetes, make_classification, make_regression

from automl_optuna_framework import ClassificationAutoML, RegressionAutoML

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 6)


def example_1_iris_classification():
    print("\n" + "="*80)
    print("EXAMPLE 1: IRIS CLASSIFICATION")
    print("="*80)
    
    X, y = load_iris(return_X_y=True)
    
    automl = ClassificationAutoML(task_type='classification', verbose=True)
    automl.fit(X, y, n_trials=25, cv_folds=5)
    
    print("\nTop 10 Models:")
    summary = automl.get_summary()
    print(summary.head(10).to_string())
    
    fig1 = automl.plot_optimization_history()
    fig2 = automl.plot_model_comparison(top_n=8)
    
    return automl, summary


def example_2_breast_cancer_classification():
    print("\n" + "="*80)
    print("EXAMPLE 2: BREAST CANCER CLASSIFICATION")
    print("="*80)
    
    X, y = load_breast_cancer(return_X_y=True)
    
    automl = ClassificationAutoML(task_type='classification', verbose=True)
    automl.fit(X, y, test_size=0.2, n_trials=40, cv_folds=5)
    
    print("\nTop 10 Models:")
    summary = automl.get_summary()
    print(summary.head(10).to_string())
    
    print(f"\nBest Model Details:")
    print(f"  Model: {automl.best_result.model_name}")
    print(f"  CV Score: {automl.best_result.cv_score:.4f} ± {automl.best_result.cv_std:.4f}")
    print(f"  Test Score: {automl.best_result.test_score:.4f}")
    print(f"  Train Score: {automl.best_result.train_score:.4f}")
    print(f"  Hyperparameters: {automl.best_result.hyperparameters}")
    
    fig1 = automl.plot_optimization_history()
    fig2 = automl.plot_model_comparison(top_n=10)
    fig3 = automl.plot_hyperparameter_importance(top_n=8)
    
    return automl, summary


def example_3_diabetes_regression():
    print("\n" + "="*80)
    print("EXAMPLE 3: DIABETES REGRESSION")
    print("="*80)
    
    X, y = load_diabetes(return_X_y=True)
    
    automl = RegressionAutoML(task_type='regression', verbose=True)
    automl.fit(X, y, test_size=0.2, n_trials=40, cv_folds=5)
    
    print("\nTop 10 Models:")
    summary = automl.get_summary()
    print(summary.head(10).to_string())
    
    print(f"\nBest Model Details:")
    print(f"  Model: {automl.best_result.model_name}")
    print(f"  CV Score (R²): {automl.best_result.cv_score:.4f} ± {automl.best_result.cv_std:.4f}")
    print(f"  Test Score (R²): {automl.best_result.test_score:.4f}")
    print(f"  Train Score (R²): {automl.best_result.train_score:.4f}")
    print(f"  Hyperparameters: {automl.best_result.hyperparameters}")
    
    fig1 = automl.plot_optimization_history()
    fig2 = automl.plot_model_comparison(top_n=10)
    fig3 = automl.plot_hyperparameter_importance(top_n=8)
    
    return automl, summary


def example_4_synthetic_large_classification():
    print("\n" + "="*80)
    print("EXAMPLE 4: LARGE SYNTHETIC CLASSIFICATION")
    print("="*80)
    
    X, y = make_classification(
        n_samples=5000, n_features=50, n_informative=30,
        n_redundant=10, random_state=42
    )
    
    automl = ClassificationAutoML(task_type='classification', verbose=True)
    automl.fit(X, y, test_size=0.2, n_trials=50, cv_folds=5)
    
    print("\nTop 10 Models:")
    summary = automl.get_summary()
    print(summary.head(10).to_string())
    
    fig1 = automl.plot_optimization_history()
    fig2 = automl.plot_model_comparison(top_n=10)
    
    return automl, summary


def example_5_synthetic_large_regression():
    print("\n" + "="*80)
    print("EXAMPLE 5: LARGE SYNTHETIC REGRESSION")
    print("="*80)
    
    X, y = make_regression(
        n_samples=2000, n_features=30, n_informative=20,
        random_state=42
    )
    
    automl = RegressionAutoML(task_type='regression', verbose=True)
    automl.fit(X, y, test_size=0.2, n_trials=50, cv_folds=5)
    
    print("\nTop 10 Models:")
    summary = automl.get_summary()
    print(summary.head(10).to_string())
    
    print(f"\nBest Model Details:")
    print(f"  Model: {automl.best_result.model_name}")
    print(f"  CV Score (R²): {automl.best_result.cv_score:.4f} ± {automl.best_result.cv_std:.4f}")
    print(f"  Test Score (R²): {automl.best_result.test_score:.4f}")
    print(f"  Train Score (R²): {automl.best_result.train_score:.4f}")
    
    fig1 = automl.plot_optimization_history()
    fig2 = automl.plot_model_comparison(top_n=10)
    fig3 = automl.plot_hyperparameter_importance(top_n=8)
    
    return automl, summary


def compare_results(results_dict):
    print("\n" + "="*80)
    print("COMPARISON OF ALL EXPERIMENTS")
    print("="*80)
    
    comparison_data = []
    for name, (automl, summary) in results_dict.items():
        best = automl.best_result
        comparison_data.append({
            'Experiment': name,
            'Best Model': best.model_name,
            'CV Score': best.cv_score,
            'Test Score': best.test_score,
            'Train Score': best.train_score,
            'Time (s)': best.train_time,
            'Num Trials': len(automl.results)
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    print("\n" + comparison_df.to_string(index=False))
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(comparison_df))
    width = 0.25
    
    ax.bar(x - width, comparison_df['CV Score'], width, label='CV Score', alpha=0.8)
    ax.bar(x, comparison_df['Test Score'], width, label='Test Score', alpha=0.8)
    ax.bar(x + width, comparison_df['Train Score'], width, label='Train Score', alpha=0.8)
    
    ax.set_xlabel('Experiment')
    ax.set_ylabel('Score')
    ax.set_title('Comparison of Best Models Across Experiments')
    ax.set_xticks(x)
    ax.set_xticklabels(comparison_df['Experiment'], rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('C:/Users/AARYA PATEL/OneDrive/Desktop/AIML/Projects/automl/outputs/automl_comparison.png', dpi=300, bbox_inches='tight')
    
    return comparison_df


def save_results_to_csv(results_dict):
    print("\n" + "="*80)
    print("SAVING RESULTS")
    print("="*80)
    
    for name, (automl, summary) in results_dict.items():
        filename = f'C:/Users/AARYA PATEL/OneDrive/Desktop/AIML/Projects/automl/outputs/automl_results_{name.replace(" ", "_").lower()}.csv'
        summary.to_csv(filename, index=False)
        print(f"Saved: {filename}")


if __name__ == "__main__":
    results = {}
    
    automl1, summary1 = example_1_iris_classification()
    results['Iris Classification'] = (automl1, summary1)
    
    automl2, summary2 = example_2_breast_cancer_classification()
    results['Breast Cancer Classification'] = (automl2, summary2)
    
    automl3, summary3 = example_3_diabetes_regression()
    results['Diabetes Regression'] = (automl3, summary3)
    
    automl4, summary4 = example_4_synthetic_large_classification()
    results['Synthetic Large Classification'] = (automl4, summary4)
    
    automl5, summary5 = example_5_synthetic_large_regression()
    results['Synthetic Large Regression'] = (automl5, summary5)
    
    comparison = compare_results(results)
    save_results_to_csv(results)
    
    plt.show()