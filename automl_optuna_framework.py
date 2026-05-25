"""
Advanced AutoML Mini-Framework with Optuna
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Any, Union, Callable
from dataclasses import dataclass, field
import warnings
import time
from abc import ABC, abstractmethod
import json

import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler

from sklearn.model_selection import cross_val_score, train_test_split, KFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import (
    RandomForestClassifier, RandomForestRegressor,
    GradientBoostingClassifier, GradientBoostingRegressor,
    AdaBoostClassifier, AdaBoostRegressor
)

from sklearn.svm import SVC, SVR
from sklearn.linear_model import (
    LogisticRegression, LinearRegression, Ridge, Lasso, ElasticNet
)

from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    mean_squared_error, r2_score, mean_absolute_error, mean_absolute_percentage_error
)

warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

@dataclass
class ModelResult:
    model_name: str
    hyperparameters: Dict[str, Any]
    cv_score: float
    test_score: float
    train_score: float
    cv_std: float
    train_time: float
    cv_scores: List[float] = field(default_factory=list)
    model: Any = None
    predictions: np.ndarray = None
    trial_number: int = None
    
    def __repr__(self):
        return f"{self.model_name} | CV: {self.cv_score:.4f} | Test: {self.test_score:.4f} | Time: {self.train_time:.2f}s"
    
    def to_dict(self):
        return {
            'Model': self.model_name,
            'Hyperparameters': self.hyperparameters,
            'CV Score': self.cv_score,
            'Test Score': self.test_score,
            'Train Score': self.train_score,
            'CV Std': self.cv_std,
            'Train Time (s)': self.train_time,
            'Trial': self.trial_number
        }


class BaseAutoML(ABC):
    
    def __init__(self, task_type: str, verbose: bool = True, n_jobs: int = -1, random_state: int = 42):
        self.task_type = task_type
        self.verbose = verbose
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.results: List[ModelResult] = []
        self.best_model = None
        self.best_result = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scaler = None
        self.cv_results_history = []
        
    @abstractmethod
    def get_models(self) -> Dict[str, Any]:
        pass
    
    @abstractmethod
    def define_search_space(self, trial: optuna.Trial) -> Tuple[str, Dict[str, Any]]:
        pass
    
    @abstractmethod
    def get_scoring_metric(self) -> str:
        pass
    
    def _log(self, message: str):
        if self.verbose:
            print(message)
    
    def _objective(self, trial: optuna.Trial, cv_folds: int = 5) -> float:
        model_name, hyperparams = self.define_search_space(trial)
        models = self.get_models()
        
        if model_name not in models:
            return 0.0
        
        try:
            model_class = models[model_name]
            try:
                model = model_class(**hyperparams, random_state=self.random_state)
            except TypeError:
                model = model_class(**hyperparams)
            
            start_time = time.time()
            metric = self.get_scoring_metric()
            
            cv_scores = cross_val_score(
                model, self.X_train, self.y_train,
                cv=cv_folds, scoring=metric, n_jobs=self.n_jobs
            )
            
            train_time = time.time() - start_time
            score = cv_scores.mean()
            
            model.fit(self.X_train, self.y_train)
            test_score = self._evaluate(model, self.X_test, self.y_test)
            train_score = self._evaluate(model, self.X_train, self.y_train)
            
            result = ModelResult(
                model_name=model_name,
                hyperparameters=hyperparams,
                cv_score=score,
                test_score=test_score,
                train_score=train_score,
                cv_std=cv_scores.std(),
                train_time=train_time,
                cv_scores=cv_scores.tolist(),
                model=model,
                predictions=model.predict(self.X_test),
                trial_number=trial.number
            )
            
            self.results.append(result)
            self.cv_results_history.append(score)
            
            self._log(f"Trial {trial.number}: {result}")
            
            return score
        
        except Exception as e:
            self._log(f"Trial {trial.number} failed: {str(e)}")
            return 0.0
    
    @abstractmethod
    def _evaluate(self, model, X, y) -> float:
        pass
    
    def fit(self, X: Union[np.ndarray, pd.DataFrame], y: Union[np.ndarray, pd.Series],
            test_size: float = 0.2, random_state: int = 42, cv_folds: int = 5,
            n_trials: int = 50, timeout: float = None):

        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(y, pd.Series):
            y = y.values
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        self.scaler = StandardScaler()
        self.X_train = self.scaler.fit_transform(self.X_train)
        self.X_test = self.scaler.transform(self.X_test)
        
        self._log(f"\n{'='*80}")
        self._log(f"AutoML {self.task_type.upper()}")
        self._log(f"{'='*80}")
        self._log(f"Training set size: {self.X_train.shape}")
        self._log(f"Test set size: {self.X_test.shape}")
        self._log(f"Number of trials: {n_trials}")
        self._log(f"CV folds: {cv_folds}\n")
        
        sampler = TPESampler(seed=self.random_state)
        pruner = MedianPruner(n_startup_trials=10)
        
        study = optuna.create_study(
            direction='maximize',
            sampler=sampler,
            pruner=pruner
        )
        
        study.optimize(
            lambda trial: self._objective(trial, cv_folds=cv_folds),
            n_trials=n_trials,
            timeout=timeout,
            show_progress_bar=self.verbose
        )
        
        if self.results:
            self.results.sort(key=lambda x: x.cv_score, reverse=True)
            self.best_result = self.results[0]
            self.best_model = self.best_result.model
            
            self._log(f"\n{'='*80}")
            self._log(f"BEST MODEL: {self.best_result.model_name}")
            self._log(f"Hyperparameters:\n{json.dumps(self.best_result.hyperparameters, indent=2)}")
            self._log(f"CV Score: {self.best_result.cv_score:.4f} ± {self.best_result.cv_std:.4f}")
            self._log(f"Test Score: {self.best_result.test_score:.4f}")
            self._log(f"Train Score: {self.best_result.train_score:.4f}")
            self._log(f"{'='*80}\n")
        
        return self
    
    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        if self.best_model is None:
            raise ValueError("Model not fitted yet. Call fit() first.")
        
        if isinstance(X, pd.DataFrame):
            X = X.values
        
        X_scaled = self.scaler.transform(X)
        return self.best_model.predict(X_scaled)
    
    def predict_proba(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        if self.best_model is None:
            raise ValueError("Model not fitted yet. Call fit() first.")
        
        if not hasattr(self.best_model, 'predict_proba'):
            raise ValueError("Model doesn't support predict_proba")
        
        if isinstance(X, pd.DataFrame):
            X = X.values
        
        X_scaled = self.scaler.transform(X)
        return self.best_model.predict_proba(X_scaled)
    
    def get_top_models(self, n: int = 5) -> List[ModelResult]:
        return self.results[:n]
    
    def get_summary(self) -> pd.DataFrame:
        data = [result.to_dict() for result in self.results]
        df = pd.DataFrame(data)
        return df.sort_values('CV Score', ascending=False).reset_index(drop=True)
    
    def plot_optimization_history(self, figsize: Tuple[int, int] = (12, 5)):
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        axes[0].plot(range(len(self.cv_results_history)), self.cv_results_history, 'o-', alpha=0.6)
        axes[0].set_xlabel('Trial Number')
        axes[0].set_ylabel('CV Score')
        axes[0].set_title('Optimization History')
        axes[0].grid(True, alpha=0.3)
        
        best_scores = []
        for i, score in enumerate(self.cv_results_history):
            best_scores.append(max(self.cv_results_history[:i+1]))
        axes[1].plot(range(len(best_scores)), best_scores, 'g-', linewidth=2)
        axes[1].set_xlabel('Trial Number')
        axes[1].set_ylabel('Best CV Score')
        axes[1].set_title('Best Score Progress')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_model_comparison(self, top_n: int = 10, figsize: Tuple[int, int] = (12, 6)):
        top_results = self.get_top_models(top_n)
        
        models = [f"{r.model_name}" for r in top_results]
        cv_scores = [r.cv_score for r in top_results]
        test_scores = [r.test_score for r in top_results]
        
        x = np.arange(len(models))
        width = 0.35
        
        fig, ax = plt.subplots(figsize=figsize)
        bars1 = ax.bar(x - width/2, cv_scores, width, label='CV Score', alpha=0.8)
        bars2 = ax.bar(x + width/2, test_scores, width, label='Test Score', alpha=0.8)
        
        ax.set_xlabel('Model')
        ax.set_ylabel('Score')
        ax.set_title('Top Models Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.3f}', ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        return fig
    
    def plot_hyperparameter_importance(self, top_n: int = 10, figsize: Tuple[int, int] = (12, 6)):
        top_results = self.get_top_models(top_n)
        
        all_hps = {}
        for result in top_results:
            for hp_name, hp_value in result.hyperparameters.items():
                if hp_name not in all_hps:
                    all_hps[hp_name] = []
                all_hps[hp_name].append(hp_value)
        
        numeric_hps = {k: v for k, v in all_hps.items() if isinstance(v[0], (int, float))}
        
        if not numeric_hps:
            print("No numeric hyperparameters to plot")
            return None
        
        n_hps = len(numeric_hps)
        fig, axes = plt.subplots(1, min(n_hps, 3), figsize=figsize)
        
        if n_hps == 1:
            axes = [axes]
        
        for idx, (hp_name, hp_values) in enumerate(list(numeric_hps.items())[:3]):
            axes[idx].scatter(range(len(hp_values)), hp_values, alpha=0.6, s=100)
            axes[idx].set_xlabel('Top Models')
            axes[idx].set_ylabel(hp_name)
            axes[idx].set_title(f'{hp_name} Distribution')
            axes[idx].grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig


class ClassificationAutoML(BaseAutoML):
    
    def get_models(self) -> Dict[str, Any]:
        return {
            'Logistic Regression': LogisticRegression,
            'Decision Tree': DecisionTreeClassifier,
            'Random Forest': RandomForestClassifier,
            'Gradient Boosting': GradientBoostingClassifier,
            'AdaBoost': AdaBoostClassifier,
            'SVM': SVC,
            'KNN': KNeighborsClassifier,
        }
    
    def define_search_space(self, trial: optuna.Trial) -> Tuple[str, Dict[str, Any]]:
        model_name = trial.suggest_categorical(
            'model',
            list(self.get_models().keys())
        )
        
        if model_name == 'Logistic Regression':
            return model_name, {
                'C': trial.suggest_float('C', 1e-3, 1e3, log=True),
                'max_iter': trial.suggest_int('max_iter', 100, 1000),
            }
        
        elif model_name == 'Decision Tree':
            return model_name, {
                'max_depth': trial.suggest_int('max_depth', 2, 20),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
            }
        
        elif model_name == 'Random Forest':
            return model_name, {
                'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                'max_depth': trial.suggest_int('max_depth', 5, 30),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 15),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
            }
        
        elif model_name == 'Gradient Boosting':
            return model_name, {
                'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                'learning_rate': trial.suggest_float('gb_learning_rate', 0.001, 0.5, log=True),
                'max_depth': trial.suggest_int('max_depth', 2, 10),
                'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            }
        
        elif model_name == 'AdaBoost':
            return model_name, {
                'n_estimators': trial.suggest_int('n_estimators', 50, 200),
                'learning_rate': trial.suggest_float('adaboost_learning_rate', 0.5, 2.0),
            }
        
        elif model_name == 'SVM':
            return model_name, {
                'C': trial.suggest_float('C', 1e-2, 1e3, log=True),
                'kernel': trial.suggest_categorical('kernel', ['rbf', 'poly', 'linear']),
                'gamma': trial.suggest_categorical('gamma', ['scale', 'auto']),
            }
        
        elif model_name == 'KNN':
            return model_name, {
                'n_neighbors': trial.suggest_int('n_neighbors', 3, 30),
                'weights': trial.suggest_categorical('weights', ['uniform', 'distance']),
                'metric': trial.suggest_categorical('metric', ['euclidean', 'manhattan']),
            }
    
    def get_scoring_metric(self) -> str:
        return 'f1_weighted'
    
    def _evaluate(self, model, X, y) -> float:
        return f1_score(y, model.predict(X), average='weighted', zero_division=0)


class RegressionAutoML(BaseAutoML):
    
    def get_models(self) -> Dict[str, Any]:
        return {
            'Linear Regression': LinearRegression,
            'Ridge': Ridge,
            'Lasso': Lasso,
            'ElasticNet': ElasticNet,
            'Decision Tree': DecisionTreeRegressor,
            'Random Forest': RandomForestRegressor,
            'Gradient Boosting': GradientBoostingRegressor,
            'AdaBoost': AdaBoostRegressor,
            'SVR': SVR,
            'KNN': KNeighborsRegressor,
        }
    
    def define_search_space(self, trial: optuna.Trial) -> Tuple[str, Dict[str, Any]]:
        model_name = trial.suggest_categorical(
            'model',
            list(self.get_models().keys())
        )
        
        if model_name == 'Linear Regression':
            return model_name, {}
        
        elif model_name == 'Ridge':
            return model_name, {
                'alpha': trial.suggest_float('alpha', 1e-3, 1e3, log=True),
            }
        
        elif model_name == 'Lasso':
            return model_name, {
                'alpha': trial.suggest_float('alpha', 1e-4, 1, log=True),
                'max_iter': trial.suggest_int('max_iter', 1000, 5000),
            }
        
        elif model_name == 'ElasticNet':
            return model_name, {
                'alpha': trial.suggest_float('alpha', 1e-4, 1, log=True),
                'l1_ratio': trial.suggest_float('l1_ratio', 0.0, 1.0),
                'max_iter': trial.suggest_int('max_iter', 1000, 5000),
            }
        
        elif model_name == 'Decision Tree':
            return model_name, {
                'max_depth': trial.suggest_int('max_depth', 2, 20),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
            }
        
        elif model_name == 'Random Forest':
            return model_name, {
                'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                'max_depth': trial.suggest_int('max_depth', 5, 30),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 15),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
            }
        
        elif model_name == 'Gradient Boosting':
            return model_name, {
                'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                'learning_rate': trial.suggest_float('gb_learning_rate', 0.001, 0.5, log=True),
                'max_depth': trial.suggest_int('max_depth', 2, 10),
                'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            }
        
        elif model_name == 'AdaBoost':
            return model_name, {
                'n_estimators': trial.suggest_int('n_estimators', 50, 200),
                'learning_rate': trial.suggest_float('adaboost_learning_rate', 0.5, 2.0),
            }
        
        elif model_name == 'SVR':
            return model_name, {
                'C': trial.suggest_float('C', 1e-2, 1e3, log=True),
                'kernel': trial.suggest_categorical('kernel', ['rbf', 'poly', 'linear']),
                'gamma': trial.suggest_categorical('gamma', ['scale', 'auto']),
                'epsilon': trial.suggest_float('epsilon', 0.01, 1.0),
            }
        
        elif model_name == 'KNN':
            return model_name, {
                'n_neighbors': trial.suggest_int('n_neighbors', 3, 30),
                'weights': trial.suggest_categorical('weights', ['uniform', 'distance']),
                'metric': trial.suggest_categorical('metric', ['euclidean', 'manhattan']),
            }
    
    def get_scoring_metric(self) -> str:
        return 'r2'
    
    def _evaluate(self, model, X, y) -> float:
        return r2_score(y, model.predict(X))


if __name__ == "__main__":
    from sklearn.datasets import load_breast_cancer
    
    X, y = load_breast_cancer(return_X_y=True)
    
    automl_clf = ClassificationAutoML(task_type='classification', verbose=True)
    automl_clf.fit(X, y, n_trials=30, cv_folds=5)
    
    print("\nTop 5 Models:")
    print(automl_clf.get_summary().head())
    
    automl_clf.plot_optimization_history()
    automl_clf.plot_model_comparison()
    plt.show()
    
    from sklearn.datasets import load_diabetes
    
    X, y = load_diabetes(return_X_y=True)
    
    automl_reg = RegressionAutoML(task_type='regression', verbose=True)
    automl_reg.fit(X, y, n_trials=30, cv_folds=5)
    
    print("\nTop 5 Models:")
    print(automl_reg.get_summary().head())
    
    automl_reg.plot_optimization_history()
    automl_reg.plot_model_comparison()
    plt.show()