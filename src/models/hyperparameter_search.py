import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import roc_auc_score, make_scorer
from typing import Dict, List, Tuple, Union, Any, Optional
import logging
from datetime import datetime


class HyperparameterSearcher:
    """Handle hyperparameter optimization"""
    
    def __init__(self, search_method: str = 'grid'):
        """
        Initialize hyperparameter searcher
        
        Args:
            search_method: 'grid', 'random', or 'bayesian'
        """
        self.search_method = search_method
        self.search_results = {}
        self.best_params = {}
        self.best_score = None
        
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def grid_search(self, 
                   model, 
                   param_grid: Dict, 
                   X: Union[pd.DataFrame, np.ndarray], 
                   y: Union[pd.Series, np.ndarray], 
                   cv: int = 5,
                   scoring: str = 'roc_auc',
                   n_jobs: int = -1) -> Dict:
        """
        Perform grid search
        
        Args:
            model: Model to optimize
            param_grid: Parameter grid
            X: Training features
            y: Training labels
            cv: Cross-validation folds
            scoring: Scoring metric
            n_jobs: Number of parallel jobs
            
        Returns:
            Grid search results
        """
        start_time = datetime.now()
        
        # Setup cross-validation
        cv_splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
        
        # Perform grid search
        grid_search = GridSearchCV(
            estimator=model,
            param_grid=param_grid,
            cv=cv_splitter,
            scoring=scoring,
            n_jobs=n_jobs,
            verbose=1,
            return_train_score=True
        )
        
        grid_search.fit(X, y)
        
        search_time = (datetime.now() - start_time).total_seconds()
        
        # Store results
        self.best_params = grid_search.best_params_
        self.best_score = grid_search.best_score_
        
        results = {
            'method': 'grid_search',
            'best_params': self.best_params,
            'best_score': self.best_score,
            'best_estimator': grid_search.best_estimator_,
            'cv_results': grid_search.cv_results_,
            'search_time': search_time,
            'n_combinations': len(grid_search.cv_results_['params']),
            'scoring': scoring
        }
        
        self.search_results = results
        
        self.logger.info(f"Grid search completed in {search_time:.2f}s")
        self.logger.info(f"Best score: {self.best_score:.4f}")
        self.logger.info(f"Best params: {self.best_params}")
        
        return results
    
    def random_search(self, 
                     model, 
                     param_dist: Dict, 
                     X: Union[pd.DataFrame, np.ndarray], 
                     y: Union[pd.Series, np.ndarray], 
                     n_iter: int = 100,
                     cv: int = 5,
                     scoring: str = 'roc_auc',
                     n_jobs: int = -1,
                     random_state: int = 42) -> Dict:
        """
        Perform random search
        
        Args:
            model: Model to optimize
            param_dist: Parameter distributions
            X: Training features
            y: Training labels
            n_iter: Number of parameter settings sampled
            cv: Cross-validation folds
            scoring: Scoring metric
            n_jobs: Number of parallel jobs
            random_state: Random seed
            
        Returns:
            Random search results
        """
        start_time = datetime.now()
        
        # Setup cross-validation
        cv_splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
        
        # Perform random search
        random_search = RandomizedSearchCV(
            estimator=model,
            param_distributions=param_dist,
            n_iter=n_iter,
            cv=cv_splitter,
            scoring=scoring,
            n_jobs=n_jobs,
            verbose=1,
            random_state=random_state,
            return_train_score=True
        )
        
        random_search.fit(X, y)
        
        search_time = (datetime.now() - start_time).total_seconds()
        
        # Store results
        self.best_params = random_search.best_params_
        self.best_score = random_search.best_score_
        
        results = {
            'method': 'random_search',
            'best_params': self.best_params,
            'best_score': self.best_score,
            'best_estimator': random_search.best_estimator_,
            'cv_results': random_search.cv_results_,
            'search_time': search_time,
            'n_iter': n_iter,
            'scoring': scoring
        }
        
        self.search_results = results
        
        self.logger.info(f"Random search completed in {search_time:.2f}s")
        self.logger.info(f"Best score: {self.best_score:.4f}")
        self.logger.info(f"Best params: {self.best_params}")
        
        return results
    
    def bayesian_search(self, 
                       model, 
                       param_space: Dict, 
                       X: Union[pd.DataFrame, np.ndarray], 
                       y: Union[pd.Series, np.ndarray], 
                       n_calls: int = 50,
                       cv_folds: int = 5,
                       random_state: int = 42,
                       acq_func: str = 'EI') -> Dict:
        """
        Perform Bayesian optimization using scikit-optimize
        
        Args:
            model: Model to optimize
            param_space: Parameter space (should be converted to skopt dimensions)
            X: Training features
            y: Training labels
            n_calls: Number of optimization calls
            cv_folds: Cross-validation folds
            random_state: Random seed
            acq_func: Acquisition function ('EI', 'LCB', 'PI')
            
        Returns:
            Bayesian optimization results
        """
        try:
            from skopt import gp_minimize
            from skopt.space import Real, Integer, Categorical
            from skopt.utils import use_named_args
            
        except ImportError:
            self.logger.warning("scikit-optimize not available, falling back to random search")
            return self.random_search(model, param_space, X, y, n_iter=n_calls, cv=cv_folds)
        
        start_time = datetime.now()
        
        # Convert parameter space to skopt dimensions
        dimensions = []
        param_names = []
        
        for param_name, param_config in param_space.items():
            param_names.append(param_name)
            
            if isinstance(param_config, dict):
                if param_config['type'] == 'real':
                    dimensions.append(Real(param_config['low'], param_config['high'], name=param_name))
                elif param_config['type'] == 'integer':
                    dimensions.append(Integer(param_config['low'], param_config['high'], name=param_name))
                elif param_config['type'] == 'categorical':
                    dimensions.append(Categorical(param_config['categories'], name=param_name))
            else:
                # Assume it's a list of values (categorical)
                dimensions.append(Categorical(param_config, name=param_name))
        
        # Setup cross-validation
        cv_splitter = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
        
        # Define objective function
        @use_named_args(dimensions)
        def objective(**params):
            # Set model parameters
            model.set_params(**params)
            
            # Perform cross-validation
            scores = []
            for train_idx, val_idx in cv_splitter.split(X, y):
                if hasattr(X, 'iloc'):
                    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
                else:
                    X_train, X_val = X[train_idx], X[val_idx]
                    y_train, y_val = y[train_idx], y[val_idx]
                
                model.fit(X_train, y_train)
                y_pred = model.predict_proba(X_val)[:, 1]
                score = roc_auc_score(y_val, y_pred)
                scores.append(score)
            
            # Return negative score (since gp_minimize minimizes)
            return -np.mean(scores)
        
        # Perform Bayesian optimization
        result = gp_minimize(
            func=objective,
            dimensions=dimensions,
            n_calls=n_calls,
            random_state=random_state,
            acq_func=acq_func,
            n_initial_points=min(10, n_calls // 4)
        )
        
        search_time = (datetime.now() - start_time).total_seconds()
        
        # Extract best parameters
        best_params = dict(zip(param_names, result.x))
        best_score = -result.fun  # Convert back to positive
        
        # Store results
        self.best_params = best_params
        self.best_score = best_score
        
        results = {
            'method': 'bayesian_search',
            'best_params': best_params,
            'best_score': best_score,
            'optimization_result': result,
            'search_time': search_time,
            'n_calls': n_calls,
            'acq_func': acq_func,
            'cv_folds': cv_folds
        }
        
        self.search_results = results
        
        self.logger.info(f"Bayesian optimization completed in {search_time:.2f}s")
        self.logger.info(f"Best score: {best_score:.4f}")
        self.logger.info(f"Best params: {best_params}")
        
        return results
    
    def get_search_results_summary(self) -> Dict[str, Any]:
        """
        Get summary of search results
        
        Returns:
            Summary dictionary
        """
        if not self.search_results:
            return {}
        
        summary = {
            'method': self.search_results['method'],
            'best_score': self.best_score,
            'best_params': self.best_params,
            'search_time': self.search_results.get('search_time', 0),
            'n_evaluations': self.search_results.get('n_combinations', 
                                                    self.search_results.get('n_iter', 
                                                                           self.search_results.get('n_calls', 0)))
        }
        
        return summary
    
    def plot_search_results(self, figsize: Tuple[int, int] = (12, 8)) -> Optional[Any]:
        """
        Plot search results (convergence, parameter importance, etc.)
        
        Args:
            figsize: Figure size
            
        Returns:
            Matplotlib figure or None
        """
        if not self.search_results:
            self.logger.warning("No search results to plot")
            return None
        
        import matplotlib.pyplot as plt
        
        method = self.search_results['method']
        
        if method in ['grid_search', 'random_search'] and 'cv_results' in self.search_results:
            # Plot parameter vs score for grid/random search
            cv_results = self.search_results['cv_results']
            
            fig, axes = plt.subplots(2, 2, figsize=figsize)
            
            # Score distribution
            axes[0, 0].hist(cv_results['mean_test_score'], bins=20, alpha=0.7)
            axes[0, 0].axvline(self.best_score, color='red', linestyle='--', label='Best Score')
            axes[0, 0].set_xlabel('Cross-validation Score')
            axes[0, 0].set_ylabel('Frequency')
            axes[0, 0].set_title('Score Distribution')
            axes[0, 0].legend()
            
            # Training vs Validation scores
            axes[0, 1].scatter(cv_results['mean_train_score'], cv_results['mean_test_score'], alpha=0.6)
            axes[0, 1].plot([0, 1], [0, 1], 'r--', alpha=0.8)
            axes[0, 1].set_xlabel('Mean Training Score')
            axes[0, 1].set_ylabel('Mean Validation Score')
            axes[0, 1].set_title('Training vs Validation Scores')
            
            # Best parameters visualization (if reasonable number)
            if len(self.best_params) <= 6:
                param_names = list(self.best_params.keys())
                param_values = list(self.best_params.values())
                
                axes[1, 0].barh(param_names, [float(v) if isinstance(v, (int, float)) else 0 for v in param_values])
                axes[1, 0].set_xlabel('Parameter Value')
                axes[1, 0].set_title('Best Parameters')
            
            # Search convergence (for random search)
            if method == 'random_search':
                scores = cv_results['mean_test_score']
                best_scores = np.maximum.accumulate(scores)
                axes[1, 1].plot(best_scores, label='Best Score So Far')
                axes[1, 1].set_xlabel('Iteration')
                axes[1, 1].set_ylabel('Best Score')
                axes[1, 1].set_title('Search Convergence')
                axes[1, 1].legend()
            
            plt.tight_layout()
            return fig
        
        elif method == 'bayesian_search' and 'optimization_result' in self.search_results:
            # Plot Bayesian optimization results
            try:
                from skopt.plots import plot_convergence, plot_objective, plot_evaluations
                
                result = self.search_results['optimization_result']
                
                fig, axes = plt.subplots(2, 2, figsize=figsize)
                
                # Convergence plot
                plot_convergence(result, ax=axes[0, 0])
                axes[0, 0].set_title('Convergence Plot')
                
                # Objective function (if 1D or 2D)
                if len(result.space.dimensions) <= 2:
                    plot_objective(result, ax=axes[0, 1])
                    axes[0, 1].set_title('Objective Function')
                
                # Evaluations
                plot_evaluations(result, ax=axes[1, 0])
                axes[1, 0].set_title('Evaluations')
                
                plt.tight_layout()
                return fig
                
            except ImportError:
                self.logger.warning("skopt plotting functions not available")
                return None
        
        return None
    
    def create_param_distributions(self, param_grid: Dict[str, List]) -> Dict[str, Any]:
        """
        Convert parameter grid to distributions for random search
        
        Args:
            param_grid: Parameter grid with lists of values
            
        Returns:
            Parameter distributions
        """
        from scipy.stats import uniform, randint
        
        param_distributions = {}
        
        for param_name, param_values in param_grid.items():
            if all(isinstance(v, (int, float)) for v in param_values):
                # Numeric parameter
                if all(isinstance(v, int) for v in param_values):
                    # Integer parameter
                    param_distributions[param_name] = randint(min(param_values), max(param_values) + 1)
                else:
                    # Float parameter
                    param_distributions[param_name] = uniform(min(param_values), max(param_values) - min(param_values))
            else:
                # Categorical parameter
                param_distributions[param_name] = param_values
        
        return param_distributions
    
    def create_bayesian_param_space(self, param_grid: Dict[str, List]) -> Dict[str, Dict]:
        """
        Convert parameter grid to Bayesian optimization space
        
        Args:
            param_grid: Parameter grid with lists of values
            
        Returns:
            Bayesian parameter space
        """
        param_space = {}
        
        for param_name, param_values in param_grid.items():
            if all(isinstance(v, (int, float)) for v in param_values):
                # Numeric parameter
                if all(isinstance(v, int) for v in param_values):
                    # Integer parameter
                    param_space[param_name] = {
                        'type': 'integer',
                        'low': min(param_values),
                        'high': max(param_values)
                    }
                else:
                    # Float parameter
                    param_space[param_name] = {
                        'type': 'real',
                        'low': min(param_values),
                        'high': max(param_values)
                    }
            else:
                # Categorical parameter
                param_space[param_name] = {
                    'type': 'categorical',
                    'categories': param_values
                }
        
        return param_space
    
    def adaptive_search(self, 
                       model,
                       param_grid: Dict[str, List],
                       X: Union[pd.DataFrame, np.ndarray],
                       y: Union[pd.Series, np.ndarray],
                       initial_method: str = 'random',
                       refinement_method: str = 'bayesian',
                       initial_budget: int = 50,
                       refinement_budget: int = 30) -> Dict[str, Any]:
        """
        Perform adaptive hyperparameter search
        
        Args:
            model: Model to optimize
            param_grid: Parameter grid
            X: Training features
            y: Training labels
            initial_method: Initial search method ('random', 'grid')
            refinement_method: Refinement method ('bayesian', 'grid')
            initial_budget: Budget for initial search
            refinement_budget: Budget for refinement
            
        Returns:
            Combined search results
        """
        self.logger.info("Starting adaptive hyperparameter search")
        
        # Phase 1: Initial broad search
        if initial_method == 'random':
            param_dist = self.create_param_distributions(param_grid)
            initial_results = self.random_search(model, param_dist, X, y, n_iter=initial_budget)
        else:
            initial_results = self.grid_search(model, param_grid, X, y)
        
        # Phase 2: Refinement around best parameters
        if refinement_method == 'bayesian':
            # Create refined parameter space around best parameters
            param_space = self.create_bayesian_param_space(param_grid)
            refinement_results = self.bayesian_search(model, param_space, X, y, n_calls=refinement_budget)
        else:
            # Create refined grid around best parameters
            refined_grid = self._create_refined_grid(param_grid, self.best_params)
            refinement_results = self.grid_search(model, refined_grid, X, y)
        
        # Combine results
        combined_results = {
            'method': 'adaptive',
            'initial_search': initial_results,
            'refinement_search': refinement_results,
            'best_params': self.best_params,
            'best_score': self.best_score,
            'total_evaluations': (initial_results.get('n_combinations', initial_results.get('n_iter', 0)) + 
                                 refinement_results.get('n_combinations', refinement_results.get('n_calls', 0)))
        }
        
        self.logger.info(f"Adaptive search completed with final score: {self.best_score:.4f}")
        
        return combined_results
    
    def _create_refined_grid(self, original_grid: Dict[str, List], best_params: Dict[str, Any]) -> Dict[str, List]:
        """Create a refined parameter grid around best parameters"""
        refined_grid = {}
        
        for param_name, param_values in original_grid.items():
            if param_name in best_params:
                best_value = best_params[param_name]
                
                if isinstance(best_value, (int, float)) and len(param_values) > 2:
                    # Find position of best value
                    sorted_values = sorted(param_values)
                    try:
                        best_idx = sorted_values.index(best_value)
                    except ValueError:
                        # Best value not in original grid, find closest
                        best_idx = min(range(len(sorted_values)), 
                                     key=lambda i: abs(sorted_values[i] - best_value))
                    
                    # Create refined range around best value
                    start_idx = max(0, best_idx - 1)
                    end_idx = min(len(sorted_values), best_idx + 2)
                    refined_grid[param_name] = sorted_values[start_idx:end_idx]
                else:
                    # Keep original values for categorical or small grids
                    refined_grid[param_name] = param_values
            else:
                refined_grid[param_name] = param_values
        
        return refined_grid