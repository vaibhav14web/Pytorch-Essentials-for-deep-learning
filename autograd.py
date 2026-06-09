# Example 1 -> Normalfunction

import torch
x = torch.tensor(3.0, requires_grad=True) # requires_grad=True -> This tells PyTorch to track all operations on this tensor, so that it can compute gradients later.

y = x**2
print(y.backward()) # This computes the gradient of y with respect to x. Since y = x^2, the gradient will be 2*x.
print(x.grad) # This will print the gradient of y with respect to x, which should be 6.0 (since x is 3.0, and the gradient is 2*x).

# Example 2 -> More complex function

x = torch.tensor(2.0, requires_grad=True)
y = x**2
z = torch.sin(y)
print(z.backward())
print(x.grad) # This will print the gradient of z with respect to x. The gradient will be computed using the chain rule: dz/dx = dz/dy * dy/dx. Since z = sin(y) and y = x^2, we have dz/dy = cos(y) and dy/dx = 2*x. Therefore, dz/dx = cos(y) * 2*x.

# Example 3 -> Simple neural network

import torch

x = torch.tensor(6.7) # input value
y = torch.tensor(0.0)# target value

w = torch.tensor(1.0, requires_grad=True) # weight parameter
b = torch.tensor(0.0, requires_grad=True) # bias parameter

def binary_cross_entropy(prediction, target):
    epsilon = 1e-8 # small constant to prevent log(0)
    prediction.clamp(epsilon, 1-epsilon) # clamp the prediction to avoid log(0)
    return - (target * torch.log(prediction) + (1 - target) * torch.log(1 - prediction))

z = w*x + b # linear transformation
y_pred = torch.sigmoid(z) # apply sigmoid activation to get predicted probability  
loss = binary_cross_entropy(y_pred, y) # compute binary cross-entropy loss
print(loss.backward()) # compute gradients of loss with respect to w and b
print(w.grad)
print(b.grad)