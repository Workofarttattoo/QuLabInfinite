"""
Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light). All Rights Reserved. PATENT PENDING.

MACHINE LEARNING LAB
Production-ready machine learning algorithms implemented from scratch.
Free gift to the scientific community from QuLabInfinite.
"""

import numpy as np
from typing import List, Tuple, Optional, Dict, Any, Callable
from dataclasses import dataclass, field
import warnings
warnings.filterwarnings('ignore')

@dataclass
class MLConfig:
    """Configuration for machine learning models."""
    learning_rate: float = 0.01
    epochs: int = 100
    batch_size: int = 32
    regularization: str = 'l2'  # 'l1', 'l2', 'elastic'
    reg_lambda: float = 0.01
    reg_alpha: float = 0.5  # For elastic net
    tolerance: float = 1e-6
    random_state: int = 42
    verbose: bool = False

class MachineLearningLab:
    """Production-ready machine learning algorithms implemented from scratch."""

    def __init__(self, config: MLConfig = None):
        self.config = config or MLConfig()
        np.random.seed(self.config.random_state)
        self.models = {}
        self.metrics_history = []

    def gradient_descent(self, X: np.ndarray, y: np.ndarray,
                         theta_init: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Implement gradient descent optimization from scratch.

        Args:
            X: Feature matrix (n_samples, n_features)
            y: Target values (n_samples,)
            theta_init: Initial parameters

        Returns:
            Optimized parameters
        """
        n_samples, n_features = X.shape
        theta = theta_init if theta_init is not None else np.zeros(n_features)

        for epoch in range(self.config.epochs):
            # Compute predictions
            y_pred = X.dot(theta)

            # Compute gradient
            gradient = (2/n_samples) * X.T.dot(y_pred - y)

            # Apply regularization
            if self.config.regularization == 'l2':
                gradient += (self.config.reg_lambda / n_samples) * theta
            elif self.config.regularization == 'l1':
                gradient += (self.config.reg_lambda / n_samples) * np.sign(theta)
            elif self.config.regularization == 'elastic':
                l1_term = self.config.reg_alpha * np.sign(theta)
                l2_term = (1 - self.config.reg_alpha) * theta
                gradient += (self.config.reg_lambda / n_samples) * (l1_term + l2_term)

            # Update parameters
            theta -= self.config.learning_rate * gradient

            # Check convergence
            if np.linalg.norm(gradient) < self.config.tolerance:
                if self.config.verbose:
                    print(f"Converged at epoch {epoch}")
                break

        return theta

    def stochastic_gradient_descent(self, X: np.ndarray, y: np.ndarray,
                                   theta_init: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Implement stochastic gradient descent with mini-batches.

        Args:
            X: Feature matrix
            y: Target values
            theta_init: Initial parameters

        Returns:
            Optimized parameters
        """
        n_samples, n_features = X.shape
        theta = theta_init if theta_init is not None else np.zeros(n_features)

        for epoch in range(self.config.epochs):
            # Shuffle data
            indices = np.random.permutation(n_samples)
            X_shuffled = X[indices]
            y_shuffled = y[indices]

            # Mini-batch updates
            for i in range(0, n_samples, self.config.batch_size):
                X_batch = X_shuffled[i:i+self.config.batch_size]
                y_batch = y_shuffled[i:i+self.config.batch_size]

                # Compute gradient for batch
                y_pred = X_batch.dot(theta)
                gradient = (2/len(X_batch)) * X_batch.T.dot(y_pred - y_batch)

                # Apply regularization
                if self.config.regularization == 'l2':
                    gradient += (self.config.reg_lambda / len(X_batch)) * theta

                # Update with learning rate decay
                lr = self.config.learning_rate / (1 + epoch * 0.001)
                theta -= lr * gradient

        return theta

    def cross_validation(self, X: np.ndarray, y: np.ndarray,
                        k_folds: int = 5,
                        model_fn: Callable = None) -> Dict[str, float]:
        """
        Implement k-fold cross-validation from scratch.

        Args:
            X: Feature matrix
            y: Target values
            k_folds: Number of folds
            model_fn: Model training function

        Returns:
            Dictionary with validation metrics
        """
        n_samples = X.shape[0]
        fold_size = n_samples // k_folds
        indices = np.arange(n_samples)
        np.random.shuffle(indices)

        scores = []

        for fold in range(k_folds):
            # Create train/validation split
            val_start = fold * fold_size
            val_end = val_start + fold_size if fold < k_folds - 1 else n_samples

            val_indices = indices[val_start:val_end]
            train_indices = np.concatenate([indices[:val_start], indices[val_end:]])

            X_train, y_train = X[train_indices], y[train_indices]
            X_val, y_val = X[val_indices], y[val_indices]

            # Train model
            if model_fn is None:
                theta = self.gradient_descent(X_train, y_train)
                y_pred = X_val.dot(theta)
            else:
                y_pred = model_fn(X_train, y_train, X_val)

            # Calculate MSE
            mse = np.mean((y_val - y_pred) ** 2)
            scores.append(mse)

        return {
            'mean_score': np.mean(scores),
            'std_score': np.std(scores),
            'scores': scores
        }

    def feature_selection_rfe(self, X: np.ndarray, y: np.ndarray,
                             n_features_to_select: int) -> np.ndarray:
        """
        Recursive Feature Elimination for feature selection.

        Args:
            X: Feature matrix
            y: Target values
            n_features_to_select: Number of features to keep

        Returns:
            Boolean mask of selected features
        """
        n_features = X.shape[1]
        selected_features = np.ones(n_features, dtype=bool)

        while np.sum(selected_features) > n_features_to_select:
            # Train model with current features
            X_subset = X[:, selected_features]
            theta = self.gradient_descent(X_subset, y)

            # Find feature with smallest weight
            min_idx = np.argmin(np.abs(theta))

            # Remove feature
            active_indices = np.where(selected_features)[0]
            selected_features[active_indices[min_idx]] = False

        return selected_features

    def lasso_coordinate_descent(self, X: np.ndarray, y: np.ndarray,
                                alpha: float = 1.0) -> np.ndarray:
        """
        Implement LASSO regression using coordinate descent.

        Args:
            X: Feature matrix
            y: Target values
            alpha: L1 regularization strength

        Returns:
            Sparse coefficient vector
        """
        n_samples, n_features = X.shape
        theta = np.zeros(n_features)

        for _ in range(self.config.epochs):
            for j in range(n_features):
                # Compute residual without feature j
                theta_j = theta[j]
                theta[j] = 0
                residual = y - X.dot(theta)

                # Compute rho
                rho = X[:, j].dot(residual)

                # Soft thresholding
                if rho < -alpha/2:
                    theta[j] = (rho + alpha/2) / (X[:, j].dot(X[:, j]) / n_samples)
                elif rho > alpha/2:
                    theta[j] = (rho - alpha/2) / (X[:, j].dot(X[:, j]) / n_samples)
                else:
                    theta[j] = 0

        return theta

    def ridge_regression(self, X: np.ndarray, y: np.ndarray,
                        alpha: float = 1.0) -> np.ndarray:
        """
        Implement Ridge regression with closed-form solution.

        Args:
            X: Feature matrix
            y: Target values
            alpha: L2 regularization strength

        Returns:
            Coefficient vector
        """
        n_features = X.shape[1]
        # Closed-form solution: (X^T X + alpha*I)^-1 X^T y
        XtX = X.T.dot(X)
        theta = np.linalg.inv(XtX + alpha * np.eye(n_features)).dot(X.T).dot(y)
        return theta

    def elastic_net(self, X: np.ndarray, y: np.ndarray,
                   alpha: float = 1.0, l1_ratio: float = 0.5) -> np.ndarray:
        """
        Implement Elastic Net regression combining L1 and L2 penalties.

        Args:
            X: Feature matrix
            y: Target values
            alpha: Total regularization strength
            l1_ratio: Balance between L1 (1.0) and L2 (0.0)

        Returns:
            Coefficient vector
        """
        n_samples, n_features = X.shape
        theta = np.zeros(n_features)

        for epoch in range(self.config.epochs):
            for j in range(n_features):
                # Compute residual without feature j
                theta_j = theta[j]
                theta[j] = 0
                residual = y - X.dot(theta)

                # Compute gradient components
                rho = X[:, j].dot(residual)
                z = X[:, j].dot(X[:, j]) / n_samples + alpha * (1 - l1_ratio)

                # Soft thresholding with elastic net
                l1_penalty = alpha * l1_ratio * n_samples / 2

                if rho < -l1_penalty:
                    theta[j] = (rho + l1_penalty) / z
                elif rho > l1_penalty:
                    theta[j] = (rho - l1_penalty) / z
                else:
                    theta[j] = 0

        return theta

    def random_forest_regressor(self, X: np.ndarray, y: np.ndarray,
                              n_trees: int = 10, max_depth: int = 5) -> 'RandomForest':
        """
        Implement Random Forest regressor from scratch.

        Args:
            X: Feature matrix
            y: Target values
            n_trees: Number of trees
            max_depth: Maximum tree depth

        Returns:
            Trained random forest model
        """
        class DecisionNode:
            def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
                self.feature = feature
                self.threshold = threshold
                self.left = left
                self.right = right
                self.value = value

        class RandomForest:
            def __init__(self, trees):
                self.trees = trees

            def predict(self, X):
                predictions = np.array([self._predict_tree(tree, X) for tree in self.trees])
                return np.mean(predictions, axis=0)

            def _predict_tree(self, tree, X):
                if tree.value is not None:
                    return np.full(X.shape[0], tree.value)

                mask = X[:, tree.feature] <= tree.threshold
                predictions = np.zeros(X.shape[0])

                if tree.left and np.any(mask):
                    predictions[mask] = self._predict_tree(tree.left, X[mask])
                if tree.right and np.any(~mask):
                    predictions[~mask] = self._predict_tree(tree.right, X[~mask])

                return predictions

        def build_tree(X, y, depth=0):
            if depth >= max_depth or len(np.unique(y)) == 1 or len(y) < 2:
                return DecisionNode(value=np.mean(y))

            # Random feature selection
            n_features = X.shape[1]
            features = np.random.choice(n_features, int(np.sqrt(n_features)), replace=False)

            best_feature = None
            best_threshold = None
            best_mse = float('inf')

            for feature in features:
                thresholds = np.unique(X[:, feature])
                for threshold in thresholds:
                    mask = X[:, feature] <= threshold
                    if np.sum(mask) == 0 or np.sum(~mask) == 0:
                        continue

                    left_mse = np.var(y[mask]) * np.sum(mask)
                    right_mse = np.var(y[~mask]) * np.sum(~mask)
                    mse = (left_mse + right_mse) / len(y)

                    if mse < best_mse:
                        best_mse = mse
                        best_feature = feature
                        best_threshold = threshold

            if best_feature is None:
                return DecisionNode(value=np.mean(y))

            mask = X[:, best_feature] <= best_threshold
            left = build_tree(X[mask], y[mask], depth + 1)
            right = build_tree(X[~mask], y[~mask], depth + 1)

            return DecisionNode(feature=best_feature, threshold=best_threshold,
                              left=left, right=right)

        # Build forest
        trees = []
        n_samples = X.shape[0]

        for _ in range(n_trees):
            # Bootstrap sampling
            indices = np.random.choice(n_samples, n_samples, replace=True)
            X_boot = X[indices]
            y_boot = y[indices]

            tree = build_tree(X_boot, y_boot)
            trees.append(tree)

        return RandomForest(trees)

    def gradient_boosting_regressor(self, X: np.ndarray, y: np.ndarray,
                                  n_estimators: int = 10,
                                  learning_rate: float = 0.1) -> List[np.ndarray]:
        """
        Implement Gradient Boosting regressor from scratch.

        Args:
            X: Feature matrix
            y: Target values
            n_estimators: Number of boosting rounds
            learning_rate: Boosting learning rate

        Returns:
            List of weak learner predictions
        """
        n_samples = X.shape[0]
        predictions = np.zeros(n_samples)
        models = []

        for i in range(n_estimators):
            # Compute residuals
            residuals = y - predictions

            # Fit weak learner (simple linear model)
            theta = self.gradient_descent(X, residuals)
            weak_predictions = X.dot(theta)

            # Update predictions
            predictions += learning_rate * weak_predictions
            models.append(theta)

            # Store metrics
            mse = np.mean((y - predictions) ** 2)
            self.metrics_history.append({'iteration': i, 'mse': mse})

        return models

    def hyperparameter_optimization(self, X: np.ndarray, y: np.ndarray,
                                  param_grid: Dict[str, List[Any]]) -> Dict[str, Any]:
        """
        Grid search for hyperparameter optimization.

        Args:
            X: Feature matrix
            y: Target values
            param_grid: Dictionary of parameters to search

        Returns:
            Best parameters and score
        """
        best_params = None
        best_score = float('inf')
        results = []

        # Generate all parameter combinations
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())

        def generate_combinations(index=0, current={}):
            if index == len(param_names):
                return [current.copy()]

            combinations = []
            for value in param_values[index]:
                current[param_names[index]] = value
                combinations.extend(generate_combinations(index + 1, current))
            return combinations

        all_params = generate_combinations()

        for params in all_params:
            # Update config with current parameters
            for key, value in params.items():
                setattr(self.config, key, value)

            # Evaluate with cross-validation
            cv_results = self.cross_validation(X, y, k_folds=3)
            score = cv_results['mean_score']

            results.append({'params': params.copy(), 'score': score})

            if score < best_score:
                best_score = score
                best_params = params.copy()

        return {
            'best_params': best_params,
            'best_score': best_score,
            'all_results': results
        }

    def compute_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """
        Compute comprehensive evaluation metrics.

        Args:
            y_true: True values
            y_pred: Predicted values

        Returns:
            Dictionary of metrics
        """
        n = len(y_true)

        # MSE
        mse = np.mean((y_true - y_pred) ** 2)

        # RMSE
        rmse = np.sqrt(mse)

        # MAE
        mae = np.mean(np.abs(y_true - y_pred))

        # R-squared
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0

        # Adjusted R-squared (assuming p features)
        p = 1  # Default to 1 feature
        adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1) if n > p + 1 else r2

        # MAPE
        non_zero_mask = y_true != 0
        if np.any(non_zero_mask):
            mape = np.mean(np.abs((y_true[non_zero_mask] - y_pred[non_zero_mask]) / y_true[non_zero_mask])) * 100
        else:
            mape = 0

        return {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'adjusted_r2': adj_r2,
            'mape': mape
        }

    def polynomial_features(self, X: np.ndarray, degree: int = 2) -> np.ndarray:
        """
        Generate polynomial features for non-linear modeling.

        Args:
            X: Feature matrix
            degree: Polynomial degree

        Returns:
            Extended feature matrix with polynomial terms
        """
        n_samples, n_features = X.shape

        # Calculate number of output features
        from math import comb
        n_output_features = comb(n_features + degree, degree) - 1

        # Initialize output
        X_poly = np.ones((n_samples, n_output_features + 1))

        # Generate polynomial features
        idx = 1
        for d in range(1, degree + 1):
            # Generate all combinations of features for degree d
            def generate_terms(features, remaining_degree, start_idx=0):
                if remaining_degree == 0:
                    return [np.prod([X[:, f] for f in features], axis=0)]

                terms = []
                for i in range(start_idx, n_features):
                    new_features = features + [i]
                    terms.extend(generate_terms(new_features, remaining_degree - 1, i))
                return terms

            terms = generate_terms([], d)
            for term in terms:
                if idx < X_poly.shape[1]:
                    X_poly[:, idx] = term
                    idx += 1

        return X_poly

def run_demo():
    """Demonstrate the machine learning lab capabilities."""
    print("=" * 80)
    print("MACHINE LEARNING LAB - Production Demo")
    print("Copyright (c) 2025 Corporation of Light")
    print("=" * 80)

    # Initialize lab
    config = MLConfig(learning_rate=0.01, epochs=100, verbose=False)
    ml_lab = MachineLearningLab(config)

    # Generate synthetic dataset
    np.random.seed(42)
    n_samples, n_features = 100, 5
    X = np.random.randn(n_samples, n_features)
    true_weights = np.random.randn(n_features)
    y = X.dot(true_weights) + 0.1 * np.random.randn(n_samples)

    print("\n1. GRADIENT DESCENT OPTIMIZATION")
    print("-" * 40)
    theta_gd = ml_lab.gradient_descent(X, y)
    y_pred_gd = X.dot(theta_gd)
    metrics_gd = ml_lab.compute_metrics(y, y_pred_gd)
    print(f"   MSE: {metrics_gd['mse']:.4f}")
    print(f"   R²: {metrics_gd['r2']:.4f}")

    print("\n2. STOCHASTIC GRADIENT DESCENT")
    print("-" * 40)
    theta_sgd = ml_lab.stochastic_gradient_descent(X, y)
    y_pred_sgd = X.dot(theta_sgd)
    metrics_sgd = ml_lab.compute_metrics(y, y_pred_sgd)
    print(f"   MSE: {metrics_sgd['mse']:.4f}")
    print(f"   R²: {metrics_sgd['r2']:.4f}")

    print("\n3. CROSS-VALIDATION (5-FOLD)")
    print("-" * 40)
    cv_results = ml_lab.cross_validation(X, y, k_folds=5)
    print(f"   Mean CV Score: {cv_results['mean_score']:.4f}")
    print(f"   Std CV Score: {cv_results['std_score']:.4f}")

    print("\n4. FEATURE SELECTION (RFE)")
    print("-" * 40)
    selected = ml_lab.feature_selection_rfe(X, y, n_features_to_select=3)
    print(f"   Selected features: {np.where(selected)[0]}")

    print("\n5. REGULARIZATION METHODS")
    print("-" * 40)

    # LASSO
    theta_lasso = ml_lab.lasso_coordinate_descent(X, y, alpha=0.1)
    sparsity = np.sum(np.abs(theta_lasso) < 1e-6) / len(theta_lasso)
    print(f"   LASSO sparsity: {sparsity:.2%}")

    # Ridge
    theta_ridge = ml_lab.ridge_regression(X, y, alpha=0.1)
    y_pred_ridge = X.dot(theta_ridge)
    metrics_ridge = ml_lab.compute_metrics(y, y_pred_ridge)
    print(f"   Ridge R²: {metrics_ridge['r2']:.4f}")

    # Elastic Net
    theta_elastic = ml_lab.elastic_net(X, y, alpha=0.1, l1_ratio=0.5)
    y_pred_elastic = X.dot(theta_elastic)
    metrics_elastic = ml_lab.compute_metrics(y, y_pred_elastic)
    print(f"   Elastic Net R²: {metrics_elastic['r2']:.4f}")

    print("\n6. ENSEMBLE METHODS")
    print("-" * 40)

    # Random Forest
    rf_model = ml_lab.random_forest_regressor(X, y, n_trees=10, max_depth=3)
    y_pred_rf = rf_model.predict(X)
    metrics_rf = ml_lab.compute_metrics(y, y_pred_rf)
    print(f"   Random Forest R²: {metrics_rf['r2']:.4f}")

    # Gradient Boosting
    gb_models = ml_lab.gradient_boosting_regressor(X, y, n_estimators=10)
    print(f"   Gradient Boosting models: {len(gb_models)}")
    if ml_lab.metrics_history:
        final_mse = ml_lab.metrics_history[-1]['mse']
        print(f"   Final MSE: {final_mse:.4f}")

    print("\n7. HYPERPARAMETER OPTIMIZATION")
    print("-" * 40)
    param_grid = {
        'learning_rate': [0.001, 0.01, 0.1],
        'reg_lambda': [0.001, 0.01, 0.1]
    }
    best_results = ml_lab.hyperparameter_optimization(X[:50], y[:50], param_grid)
    print(f"   Best params: {best_results['best_params']}")
    print(f"   Best score: {best_results['best_score']:.4f}")

    print("\n8. POLYNOMIAL FEATURES")
    print("-" * 40)
    X_small = X[:5, :2]  # Use smaller subset for demo
    X_poly = ml_lab.polynomial_features(X_small, degree=2)
    print(f"   Original shape: {X_small.shape}")
    print(f"   Polynomial shape: {X_poly.shape}")

    print("\n" + "=" * 80)
    print("Demo completed successfully!")
    print("=" * 80)

if __name__ == '__main__':
    run_demo()