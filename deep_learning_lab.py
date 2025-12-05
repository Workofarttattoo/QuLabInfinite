"""
Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light). All Rights Reserved. PATENT PENDING.

DEEP LEARNING LAB
Production-ready deep learning algorithms implemented from scratch.
Free gift to the scientific community from QuLabInfinite.
"""

import numpy as np
from typing import List, Tuple, Dict, Optional, Callable, Any
from dataclasses import dataclass, field
import warnings
warnings.filterwarnings('ignore')

@dataclass
class DeepLearningConfig:
    """Configuration for deep learning models."""
    layers: List[int] = field(default_factory=lambda: [784, 128, 64, 10])
    learning_rate: float = 0.01
    epochs: int = 100
    batch_size: int = 32
    optimizer: str = 'adam'  # 'sgd', 'adam', 'rmsprop', 'adagrad'
    activation: str = 'relu'  # 'sigmoid', 'tanh', 'relu', 'leaky_relu', 'gelu'
    dropout_rate: float = 0.2
    batch_norm: bool = True
    weight_init: str = 'he'  # 'xavier', 'he', 'normal'
    lr_schedule: str = 'cosine'  # 'constant', 'step', 'exponential', 'cosine'
    momentum: float = 0.9
    beta1: float = 0.9  # Adam
    beta2: float = 0.999  # Adam
    epsilon: float = 1e-8
    clip_gradient: float = 1.0
    l2_reg: float = 0.0001
    random_state: int = 42

class DeepLearningLab:
    """Production-ready deep learning algorithms implemented from scratch."""

    def __init__(self, config: DeepLearningConfig = None):
        self.config = config or DeepLearningConfig()
        np.random.seed(self.config.random_state)

        # Initialize network architecture
        self.layers = self.config.layers
        self.weights = []
        self.biases = []

        # Initialize weights
        for i in range(len(self.layers) - 1):
            self._init_layer(self.layers[i], self.layers[i+1])

        # Optimizer state
        self.optimizer_state = self._init_optimizer_state()

        # Batch normalization parameters
        if self.config.batch_norm:
            self.bn_params = self._init_batch_norm()

        # Training history
        self.history = {'loss': [], 'accuracy': []}
        self.gradient_norms = []

    def _init_layer(self, input_dim: int, output_dim: int):
        """Initialize weights for a layer using specified initialization."""
        if self.config.weight_init == 'xavier':
            # Xavier/Glorot initialization
            scale = np.sqrt(2.0 / (input_dim + output_dim))
            W = np.random.randn(input_dim, output_dim) * scale
        elif self.config.weight_init == 'he':
            # He initialization (better for ReLU)
            scale = np.sqrt(2.0 / input_dim)
            W = np.random.randn(input_dim, output_dim) * scale
        else:
            # Standard normal initialization
            W = np.random.randn(input_dim, output_dim) * 0.01

        b = np.zeros((1, output_dim))

        self.weights.append(W.astype(np.float64))
        self.biases.append(b.astype(np.float64))

    def _init_optimizer_state(self) -> Dict:
        """Initialize optimizer-specific state variables."""
        state = {'step': 0}

        if self.config.optimizer == 'adam':
            state['m_weights'] = [np.zeros_like(W) for W in self.weights]
            state['v_weights'] = [np.zeros_like(W) for W in self.weights]
            state['m_biases'] = [np.zeros_like(b) for b in self.biases]
            state['v_biases'] = [np.zeros_like(b) for b in self.biases]

        elif self.config.optimizer == 'rmsprop':
            state['cache_weights'] = [np.zeros_like(W) for W in self.weights]
            state['cache_biases'] = [np.zeros_like(b) for b in self.biases]

        elif self.config.optimizer == 'adagrad':
            state['grad_weights'] = [np.zeros_like(W) for W in self.weights]
            state['grad_biases'] = [np.zeros_like(b) for b in self.biases]

        elif self.config.optimizer == 'sgd':
            if self.config.momentum > 0:
                state['v_weights'] = [np.zeros_like(W) for W in self.weights]
                state['v_biases'] = [np.zeros_like(b) for b in self.biases]

        return state

    def _init_batch_norm(self) -> Dict:
        """Initialize batch normalization parameters."""
        params = {
            'gamma': [np.ones((1, size)) for size in self.layers[1:]],
            'beta': [np.zeros((1, size)) for size in self.layers[1:]],
            'running_mean': [np.zeros((1, size)) for size in self.layers[1:]],
            'running_var': [np.ones((1, size)) for size in self.layers[1:]],
            'momentum': 0.9
        }
        return params

    # Activation functions
    def relu(self, x: np.ndarray) -> np.ndarray:
        """ReLU activation function."""
        return np.maximum(0, x)

    def relu_derivative(self, x: np.ndarray) -> np.ndarray:
        """Derivative of ReLU."""
        return (x > 0).astype(np.float64)

    def leaky_relu(self, x: np.ndarray, alpha: float = 0.01) -> np.ndarray:
        """Leaky ReLU activation function."""
        return np.where(x > 0, x, alpha * x)

    def leaky_relu_derivative(self, x: np.ndarray, alpha: float = 0.01) -> np.ndarray:
        """Derivative of Leaky ReLU."""
        return np.where(x > 0, 1, alpha)

    def sigmoid(self, x: np.ndarray) -> np.ndarray:
        """Sigmoid activation function."""
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

    def sigmoid_derivative(self, x: np.ndarray) -> np.ndarray:
        """Derivative of sigmoid."""
        s = self.sigmoid(x)
        return s * (1 - s)

    def tanh(self, x: np.ndarray) -> np.ndarray:
        """Tanh activation function."""
        return np.tanh(x)

    def tanh_derivative(self, x: np.ndarray) -> np.ndarray:
        """Derivative of tanh."""
        return 1 - np.tanh(x) ** 2

    def gelu(self, x: np.ndarray) -> np.ndarray:
        """Gaussian Error Linear Unit (GELU) activation."""
        return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))

    def gelu_derivative(self, x: np.ndarray) -> np.ndarray:
        """Derivative of GELU."""
        c = 0.044715
        a = np.sqrt(2 / np.pi)
        tanh_arg = a * (x + c * x**3)
        tanh_val = np.tanh(tanh_arg)
        sech2 = 1 - tanh_val**2

        return 0.5 * (1 + tanh_val) + 0.5 * x * sech2 * a * (1 + 3 * c * x**2)

    def softmax(self, x: np.ndarray) -> np.ndarray:
        """Softmax activation function."""
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

    def get_activation(self, name: str) -> Tuple[Callable, Callable]:
        """Get activation function and its derivative by name."""
        activations = {
            'sigmoid': (self.sigmoid, self.sigmoid_derivative),
            'tanh': (self.tanh, self.tanh_derivative),
            'relu': (self.relu, self.relu_derivative),
            'leaky_relu': (self.leaky_relu, self.leaky_relu_derivative),
            'gelu': (self.gelu, self.gelu_derivative)
        }
        return activations.get(name, (self.relu, self.relu_derivative))

    def dropout(self, x: np.ndarray, rate: float, training: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """Apply dropout regularization."""
        if not training or rate == 0:
            return x, np.ones_like(x)

        keep_prob = 1 - rate
        mask = np.random.binomial(1, keep_prob, size=x.shape) / keep_prob
        return x * mask, mask

    def batch_normalization(self, x: np.ndarray, layer_idx: int,
                          training: bool = True, eps: float = 1e-8) -> np.ndarray:
        """Apply batch normalization."""
        if not self.config.batch_norm:
            return x

        params = self.bn_params

        if training:
            # Calculate batch statistics
            batch_mean = np.mean(x, axis=0, keepdims=True)
            batch_var = np.var(x, axis=0, keepdims=True)

            # Normalize
            x_norm = (x - batch_mean) / np.sqrt(batch_var + eps)

            # Scale and shift
            out = params['gamma'][layer_idx] * x_norm + params['beta'][layer_idx]

            # Update running statistics
            momentum = params['momentum']
            params['running_mean'][layer_idx] = (momentum * params['running_mean'][layer_idx] +
                                                (1 - momentum) * batch_mean)
            params['running_var'][layer_idx] = (momentum * params['running_var'][layer_idx] +
                                               (1 - momentum) * batch_var)
        else:
            # Use running statistics
            x_norm = ((x - params['running_mean'][layer_idx]) /
                     np.sqrt(params['running_var'][layer_idx] + eps))
            out = params['gamma'][layer_idx] * x_norm + params['beta'][layer_idx]

        return out

    def forward_propagation(self, X: np.ndarray, training: bool = True) -> Tuple[np.ndarray, Dict]:
        """
        Perform forward propagation through the network.

        Args:
            X: Input data
            training: Whether in training mode

        Returns:
            Output predictions and cache for backpropagation
        """
        cache = {'activations': [X], 'z_values': [], 'dropout_masks': []}

        activation_fn, _ = self.get_activation(self.config.activation)
        current_input = X

        for i in range(len(self.weights)):
            # Linear transformation
            z = current_input @ self.weights[i] + self.biases[i]
            cache['z_values'].append(z)

            # Batch normalization (except last layer)
            if i < len(self.weights) - 1 and self.config.batch_norm:
                z = self.batch_normalization(z, i, training)

            # Activation (softmax for last layer, configured activation for others)
            if i == len(self.weights) - 1:
                a = self.softmax(z)
            else:
                a = activation_fn(z)

            # Dropout (except last layer)
            if i < len(self.weights) - 1 and training:
                a, mask = self.dropout(a, self.config.dropout_rate, training)
                cache['dropout_masks'].append(mask)
            else:
                cache['dropout_masks'].append(np.ones_like(a))

            cache['activations'].append(a)
            current_input = a

        return current_input, cache

    def backpropagation(self, y_true: np.ndarray, cache: Dict) -> Tuple[List, List]:
        """
        Perform backpropagation to compute gradients.

        Args:
            y_true: True labels (one-hot encoded)
            cache: Cache from forward propagation

        Returns:
            Gradients for weights and biases
        """
        m = y_true.shape[0]
        grad_weights = []
        grad_biases = []

        _, activation_derivative = self.get_activation(self.config.activation)

        # Start with output layer
        delta = cache['activations'][-1] - y_true

        for i in range(len(self.weights) - 1, -1, -1):
            # Apply dropout mask
            if i < len(self.weights) - 1:
                delta *= cache['dropout_masks'][i+1]

            # Compute gradients
            grad_w = (cache['activations'][i].T @ delta) / m
            grad_b = np.sum(delta, axis=0, keepdims=True) / m

            # Add L2 regularization
            if self.config.l2_reg > 0:
                grad_w += self.config.l2_reg * self.weights[i]

            grad_weights.insert(0, grad_w)
            grad_biases.insert(0, grad_b)

            # Propagate error backward (if not first layer)
            if i > 0:
                delta = delta @ self.weights[i].T

                # Apply activation derivative
                delta *= activation_derivative(cache['z_values'][i-1])

        return grad_weights, grad_biases

    def clip_gradients(self, gradients: List[np.ndarray]) -> List[np.ndarray]:
        """Clip gradients to prevent exploding gradients."""
        if self.config.clip_gradient <= 0:
            return gradients

        total_norm = np.sqrt(sum(np.sum(g**2) for g in gradients))

        if total_norm > self.config.clip_gradient:
            clip_factor = self.config.clip_gradient / total_norm
            gradients = [g * clip_factor for g in gradients]

        self.gradient_norms.append(total_norm)
        return gradients

    def sgd_update(self, grad_weights: List, grad_biases: List, lr: float):
        """SGD with momentum optimizer update."""
        if self.config.momentum > 0:
            # Update velocity
            for i in range(len(self.weights)):
                self.optimizer_state['v_weights'][i] = (
                    self.config.momentum * self.optimizer_state['v_weights'][i] -
                    lr * grad_weights[i]
                )
                self.optimizer_state['v_biases'][i] = (
                    self.config.momentum * self.optimizer_state['v_biases'][i] -
                    lr * grad_biases[i]
                )

                # Update parameters
                self.weights[i] += self.optimizer_state['v_weights'][i]
                self.biases[i] += self.optimizer_state['v_biases'][i]
        else:
            # Standard SGD
            for i in range(len(self.weights)):
                self.weights[i] -= lr * grad_weights[i]
                self.biases[i] -= lr * grad_biases[i]

    def adam_update(self, grad_weights: List, grad_biases: List, lr: float):
        """Adam optimizer update."""
        self.optimizer_state['step'] += 1
        t = self.optimizer_state['step']

        for i in range(len(self.weights)):
            # Update biased first moment estimate
            self.optimizer_state['m_weights'][i] = (
                self.config.beta1 * self.optimizer_state['m_weights'][i] +
                (1 - self.config.beta1) * grad_weights[i]
            )
            self.optimizer_state['m_biases'][i] = (
                self.config.beta1 * self.optimizer_state['m_biases'][i] +
                (1 - self.config.beta1) * grad_biases[i]
            )

            # Update biased second moment estimate
            self.optimizer_state['v_weights'][i] = (
                self.config.beta2 * self.optimizer_state['v_weights'][i] +
                (1 - self.config.beta2) * grad_weights[i]**2
            )
            self.optimizer_state['v_biases'][i] = (
                self.config.beta2 * self.optimizer_state['v_biases'][i] +
                (1 - self.config.beta2) * grad_biases[i]**2
            )

            # Compute bias-corrected moment estimates
            m_hat_w = self.optimizer_state['m_weights'][i] / (1 - self.config.beta1**t)
            v_hat_w = self.optimizer_state['v_weights'][i] / (1 - self.config.beta2**t)
            m_hat_b = self.optimizer_state['m_biases'][i] / (1 - self.config.beta1**t)
            v_hat_b = self.optimizer_state['v_biases'][i] / (1 - self.config.beta2**t)

            # Update parameters
            self.weights[i] -= lr * m_hat_w / (np.sqrt(v_hat_w) + self.config.epsilon)
            self.biases[i] -= lr * m_hat_b / (np.sqrt(v_hat_b) + self.config.epsilon)

    def rmsprop_update(self, grad_weights: List, grad_biases: List, lr: float):
        """RMSprop optimizer update."""
        decay_rate = 0.99

        for i in range(len(self.weights)):
            # Update cache
            self.optimizer_state['cache_weights'][i] = (
                decay_rate * self.optimizer_state['cache_weights'][i] +
                (1 - decay_rate) * grad_weights[i]**2
            )
            self.optimizer_state['cache_biases'][i] = (
                decay_rate * self.optimizer_state['cache_biases'][i] +
                (1 - decay_rate) * grad_biases[i]**2
            )

            # Update parameters
            self.weights[i] -= (lr * grad_weights[i] /
                               (np.sqrt(self.optimizer_state['cache_weights'][i]) + self.config.epsilon))
            self.biases[i] -= (lr * grad_biases[i] /
                              (np.sqrt(self.optimizer_state['cache_biases'][i]) + self.config.epsilon))

    def adagrad_update(self, grad_weights: List, grad_biases: List, lr: float):
        """Adagrad optimizer update."""
        for i in range(len(self.weights)):
            # Accumulate gradients
            self.optimizer_state['grad_weights'][i] += grad_weights[i]**2
            self.optimizer_state['grad_biases'][i] += grad_biases[i]**2

            # Update parameters
            self.weights[i] -= (lr * grad_weights[i] /
                               (np.sqrt(self.optimizer_state['grad_weights'][i]) + self.config.epsilon))
            self.biases[i] -= (lr * grad_biases[i] /
                              (np.sqrt(self.optimizer_state['grad_biases'][i]) + self.config.epsilon))

    def update_parameters(self, grad_weights: List, grad_biases: List, lr: float):
        """Update parameters using configured optimizer."""
        # Clip gradients
        grad_weights = self.clip_gradients(grad_weights)
        grad_biases = self.clip_gradients(grad_biases)

        # Apply optimizer
        if self.config.optimizer == 'adam':
            self.adam_update(grad_weights, grad_biases, lr)
        elif self.config.optimizer == 'rmsprop':
            self.rmsprop_update(grad_weights, grad_biases, lr)
        elif self.config.optimizer == 'adagrad':
            self.adagrad_update(grad_weights, grad_biases, lr)
        else:  # sgd
            self.sgd_update(grad_weights, grad_biases, lr)

    def get_learning_rate(self, epoch: int) -> float:
        """Get learning rate based on schedule."""
        base_lr = self.config.learning_rate

        if self.config.lr_schedule == 'constant':
            return base_lr

        elif self.config.lr_schedule == 'step':
            # Decay every 30 epochs
            decay_rate = 0.5
            decay_steps = 30
            return base_lr * (decay_rate ** (epoch // decay_steps))

        elif self.config.lr_schedule == 'exponential':
            # Exponential decay
            decay_rate = 0.95
            return base_lr * (decay_rate ** epoch)

        elif self.config.lr_schedule == 'cosine':
            # Cosine annealing
            return base_lr * 0.5 * (1 + np.cos(np.pi * epoch / self.config.epochs))

        else:
            return base_lr

    def cross_entropy_loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate cross-entropy loss."""
        eps = 1e-7
        y_pred = np.clip(y_pred, eps, 1 - eps)
        return -np.mean(np.sum(y_true * np.log(y_pred), axis=1))

    def mse_loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate mean squared error loss."""
        return np.mean((y_true - y_pred) ** 2)

    def train(self, X_train: np.ndarray, y_train: np.ndarray,
             X_val: Optional[np.ndarray] = None,
             y_val: Optional[np.ndarray] = None,
             verbose: bool = True) -> Dict:
        """
        Train the deep neural network.

        Args:
            X_train: Training features
            y_train: Training labels (one-hot encoded)
            X_val: Validation features
            y_val: Validation labels
            verbose: Whether to print progress

        Returns:
            Training history
        """
        n_samples = X_train.shape[0]
        n_batches = n_samples // self.config.batch_size

        for epoch in range(self.config.epochs):
            # Shuffle training data
            indices = np.random.permutation(n_samples)
            X_shuffled = X_train[indices]
            y_shuffled = y_train[indices]

            epoch_loss = 0
            correct = 0

            # Get learning rate for this epoch
            lr = self.get_learning_rate(epoch)

            # Mini-batch training
            for batch_idx in range(n_batches):
                start = batch_idx * self.config.batch_size
                end = start + self.config.batch_size

                X_batch = X_shuffled[start:end]
                y_batch = y_shuffled[start:end]

                # Forward pass
                y_pred, cache = self.forward_propagation(X_batch, training=True)

                # Calculate loss
                batch_loss = self.cross_entropy_loss(y_batch, y_pred)
                epoch_loss += batch_loss

                # Calculate accuracy
                correct += np.sum(np.argmax(y_pred, axis=1) == np.argmax(y_batch, axis=1))

                # Backward pass
                grad_weights, grad_biases = self.backpropagation(y_batch, cache)

                # Update parameters
                self.update_parameters(grad_weights, grad_biases, lr)

            # Calculate epoch metrics
            avg_loss = epoch_loss / n_batches
            accuracy = correct / n_samples

            self.history['loss'].append(avg_loss)
            self.history['accuracy'].append(accuracy)

            # Validation
            if X_val is not None and y_val is not None:
                val_loss, val_acc = self.evaluate(X_val, y_val)

                if verbose and epoch % 10 == 0:
                    print(f"Epoch {epoch:3d}: Loss={avg_loss:.4f}, Acc={accuracy:.4f}, "
                          f"Val_Loss={val_loss:.4f}, Val_Acc={val_acc:.4f}, LR={lr:.6f}")
            else:
                if verbose and epoch % 10 == 0:
                    print(f"Epoch {epoch:3d}: Loss={avg_loss:.4f}, Acc={accuracy:.4f}, LR={lr:.6f}")

        return self.history

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions on input data."""
        y_pred, _ = self.forward_propagation(X, training=False)
        return y_pred

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
        """Evaluate model on test data."""
        y_pred = self.predict(X)
        loss = self.cross_entropy_loss(y, y_pred)
        accuracy = np.mean(np.argmax(y_pred, axis=1) == np.argmax(y, axis=1))
        return loss, accuracy

    def get_parameter_count(self) -> int:
        """Get total number of trainable parameters."""
        total = 0
        for W, b in zip(self.weights, self.biases):
            total += W.size + b.size
        if self.config.batch_norm:
            for gamma, beta in zip(self.bn_params['gamma'], self.bn_params['beta']):
                total += gamma.size + beta.size
        return total

def run_demo():
    """Demonstrate the deep learning lab capabilities."""
    print("=" * 80)
    print("DEEP LEARNING LAB - Production Demo")
    print("Copyright (c) 2025 Corporation of Light")
    print("=" * 80)

    # Generate synthetic dataset (simplified MNIST-like)
    np.random.seed(42)
    n_samples = 1000
    n_features = 64  # 8x8 images
    n_classes = 10

    # Create synthetic data
    X = np.random.randn(n_samples, n_features)
    y_labels = np.random.randint(0, n_classes, n_samples)
    y = np.eye(n_classes)[y_labels]  # One-hot encode

    # Split data
    split = int(0.8 * n_samples)
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]

    print("\n1. NETWORK ARCHITECTURE")
    print("-" * 40)
    config = DeepLearningConfig(
        layers=[n_features, 128, 64, 32, n_classes],
        learning_rate=0.001,
        epochs=50,
        batch_size=32,
        optimizer='adam',
        activation='relu',
        dropout_rate=0.2,
        batch_norm=True,
        lr_schedule='cosine'
    )

    model = DeepLearningLab(config)
    print(f"   Layers: {config.layers}")
    print(f"   Total parameters: {model.get_parameter_count():,}")
    print(f"   Optimizer: {config.optimizer}")
    print(f"   Activation: {config.activation}")

    print("\n2. ACTIVATION FUNCTIONS DEMO")
    print("-" * 40)
    test_input = np.array([[-2, -1, 0, 1, 2]])

    activations = ['sigmoid', 'tanh', 'relu', 'leaky_relu', 'gelu']
    for act_name in activations:
        act_fn, _ = model.get_activation(act_name)
        output = act_fn(test_input)
        print(f"   {act_name:12s}: {output[0]}")

    print("\n3. TRAINING WITH ADAM OPTIMIZER")
    print("-" * 40)
    history = model.train(X_train, y_train, X_val, y_val, verbose=True)

    print(f"\n   Final Training Loss: {history['loss'][-1]:.4f}")
    print(f"   Final Training Accuracy: {history['accuracy'][-1]:.4f}")

    print("\n4. DIFFERENT OPTIMIZERS COMPARISON")
    print("-" * 40)
    optimizers = ['sgd', 'adam', 'rmsprop', 'adagrad']
    results = {}

    for opt in optimizers:
        config_opt = DeepLearningConfig(
            layers=[n_features, 32, n_classes],
            learning_rate=0.01 if opt == 'sgd' else 0.001,
            epochs=20,
            batch_size=32,
            optimizer=opt,
            activation='relu'
        )

        model_opt = DeepLearningLab(config_opt)
        model_opt.train(X_train[:200], y_train[:200], verbose=False)
        loss, acc = model_opt.evaluate(X_val, y_val)
        results[opt] = {'loss': loss, 'accuracy': acc}
        print(f"   {opt:8s}: Loss={loss:.4f}, Accuracy={acc:.4f}")

    print("\n5. LEARNING RATE SCHEDULES")
    print("-" * 40)
    schedules = ['constant', 'step', 'exponential', 'cosine']

    for schedule in schedules:
        config_lr = DeepLearningConfig(
            layers=[n_features, 32, n_classes],
            epochs=30,
            lr_schedule=schedule
        )
        model_lr = DeepLearningLab(config_lr)

        # Show learning rates over epochs
        lrs = [model_lr.get_learning_rate(e) for e in range(30)]
        print(f"   {schedule:12s}: LR range [{min(lrs):.6f}, {max(lrs):.6f}]")

    print("\n6. REGULARIZATION TECHNIQUES")
    print("-" * 40)

    # Without regularization
    config_no_reg = DeepLearningConfig(
        layers=[n_features, 128, n_classes],
        epochs=30,
        dropout_rate=0.0,
        batch_norm=False,
        l2_reg=0.0
    )
    model_no_reg = DeepLearningLab(config_no_reg)
    model_no_reg.train(X_train[:200], y_train[:200], verbose=False)
    _, acc_no_reg = model_no_reg.evaluate(X_val, y_val)

    # With regularization
    config_reg = DeepLearningConfig(
        layers=[n_features, 128, n_classes],
        epochs=30,
        dropout_rate=0.3,
        batch_norm=True,
        l2_reg=0.001
    )
    model_reg = DeepLearningLab(config_reg)
    model_reg.train(X_train[:200], y_train[:200], verbose=False)
    _, acc_reg = model_reg.evaluate(X_val, y_val)

    print(f"   Without regularization: {acc_no_reg:.4f}")
    print(f"   With regularization: {acc_reg:.4f}")

    print("\n7. GRADIENT CLIPPING ANALYSIS")
    print("-" * 40)
    if len(model.gradient_norms) > 0:
        print(f"   Max gradient norm: {max(model.gradient_norms):.4f}")
        print(f"   Mean gradient norm: {np.mean(model.gradient_norms):.4f}")
        print(f"   Gradient clipping threshold: {config.clip_gradient}")

    print("\n8. LOSS FUNCTIONS")
    print("-" * 40)
    y_true_sample = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    y_pred_sample = np.array([[0.7, 0.2, 0.1], [0.1, 0.8, 0.1], [0.2, 0.2, 0.6]])

    ce_loss = model.cross_entropy_loss(y_true_sample, y_pred_sample)
    mse_loss = model.mse_loss(y_true_sample, y_pred_sample)

    print(f"   Cross-entropy loss: {ce_loss:.4f}")
    print(f"   MSE loss: {mse_loss:.4f}")

    print("\n" + "=" * 80)
    print("Demo completed successfully!")
    print("=" * 80)

if __name__ == '__main__':
    run_demo()