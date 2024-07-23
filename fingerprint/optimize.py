import numpy as np
from scipy.optimize import minimize

def bfgs_optimizer(obj_and_grad, init_x, callback=None, iterations=100):
    def epoch_generator():
        epoch = 0
        while True:
            yield epoch
            epoch += 1

    epoch_gen = epoch_generator()
    wrapped_callback = None

    if callback:
        def wrapped_callback(x):
            callback(x, next(epoch_gen))

    result = minimize(fun=obj_and_grad, x0=init_x, jac=True, callback=wrapped_callback,
                      options={'maxiter': iterations, 'disp': True})
    return result.x

def rmsprop(grad_func, init_x, callback=None, iterations=100, learning_rate=0.1, decay_rate=0.9, epsilon=1e-8):
    #Adagrad paper
    avg_sq_grad = np.ones(len(init_x))
    x = init_x
    for i in range(iterations):
        grad = grad_func(x, i)
        if callback:
            callback(x, i)
        avg_sq_grad = avg_sq_grad * decay_rate + grad**2 * (1 - decay_rate)
        x -= learning_rate * grad / (np.sqrt(avg_sq_grad) + epsilon)
    return x

def stochastic_gradient_descent(grad_func, init_x, callback=None, iterations=200, learning_rate=0.1, momentum=0.9):
    velocity = np.zeros(len(init_x))
    x = init_x
    for i in range(iterations):
        grad = grad_func(x, i)
        if callback:
            callback(x, i)
        velocity = momentum * velocity - (1.0 - momentum) * grad
        x += learning_rate * velocity
    return x

def adam_optimizer(grad_func, init_x, callback=None, iterations=100, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
    m = np.zeros(len(init_x))
    v = np.zeros(len(init_x))
    x = init_x
    for i in range(iterations):
        grad = grad_func(x, i)
        if callback:
            callback(x, i)
        m = (1 - beta1) * grad + beta1 * m  # First moment estimate.
        v = (1 - beta2) * (grad**2) + beta2 * v  # Second moment estimate.
        mhat = m / (1 - beta1**(i + 1))  # Bias correction.
        vhat = v / (1 - beta2**(i + 1))
        x -= learning_rate * mhat / (np.sqrt(vhat) + epsilon)
    return x
