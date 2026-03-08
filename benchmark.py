import numpy as np
import time

def original_encode(n_neurons, n_dimensions, preferred_directions, tuning_widths, stimulus, noise_level=0.1):
    responses = np.zeros(n_neurons)
    for i in range(n_neurons):
        projection = np.dot(stimulus, preferred_directions[i])
        responses[i] = np.exp(projection / tuning_widths[i])
    np.random.seed(42)
    responses = np.random.poisson(responses * 10 + noise_level) / 10
    return responses

def vector_encode(n_neurons, n_dimensions, preferred_directions, tuning_widths, stimulus, noise_level=0.1):
    projections = np.dot(preferred_directions, stimulus)
    responses = np.exp(projections / tuning_widths)
    np.random.seed(42)
    responses = np.random.poisson(responses * 10 + noise_level) / 10
    return responses

n_neurons = 1000
n_dimensions = 10
preferred_directions = np.random.randn(n_neurons, n_dimensions)
norms = np.linalg.norm(preferred_directions, axis=1, keepdims=True)
preferred_directions /= norms
tuning_widths = np.random.uniform(0.5, 2.0, n_neurons)
stimulus = np.random.randn(n_dimensions)

out_orig = original_encode(n_neurons, n_dimensions, preferred_directions, tuning_widths, stimulus)
out_vec = vector_encode(n_neurons, n_dimensions, preferred_directions, tuning_widths, stimulus)

print("Encode match:", np.allclose(out_orig, out_vec))

t0 = time.time()
for _ in range(100):
    original_encode(n_neurons, n_dimensions, preferred_directions, tuning_widths, stimulus)
t1 = time.time()
print("Orig encode:", t1-t0)

t0 = time.time()
for _ in range(100):
    vector_encode(n_neurons, n_dimensions, preferred_directions, tuning_widths, stimulus)
t1 = time.time()
print("Vec encode:", t1-t0)

def original_fisher(n_neurons, n_dimensions, preferred_directions, tuning_widths, rates):
    derivatives = np.zeros((n_neurons, n_dimensions))
    for i in range(n_neurons):
        derivatives[i] = (rates[i] / tuning_widths[i]) * preferred_directions[i]
    F = np.zeros((n_dimensions, n_dimensions))
    for i in range(n_neurons):
        F += np.outer(derivatives[i], derivatives[i]) / (rates[i] + 1e-8)
    return F

def vector_fisher(n_neurons, n_dimensions, preferred_directions, tuning_widths, rates):
    derivatives = (rates / tuning_widths)[:, np.newaxis] * preferred_directions
    F = derivatives.T @ (derivatives / (rates + 1e-8)[:, np.newaxis])
    return F

rates = vector_encode(n_neurons, n_dimensions, preferred_directions, tuning_widths, stimulus, noise_level=0)

out_f_orig = original_fisher(n_neurons, n_dimensions, preferred_directions, tuning_widths, rates)
out_f_vec = vector_fisher(n_neurons, n_dimensions, preferred_directions, tuning_widths, rates)

print("Fisher match:", np.allclose(out_f_orig, out_f_vec))

t0 = time.time()
for _ in range(100):
    original_fisher(n_neurons, n_dimensions, preferred_directions, tuning_widths, rates)
t1 = time.time()
print("Orig fisher:", t1-t0)

t0 = time.time()
for _ in range(100):
    vector_fisher(n_neurons, n_dimensions, preferred_directions, tuning_widths, rates)
t1 = time.time()
print("Vec fisher:", t1-t0)
