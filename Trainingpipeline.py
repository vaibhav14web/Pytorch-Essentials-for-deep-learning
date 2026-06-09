import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder

# Load Dataset
df = pd.read_csv("breast_cancer.csv")

# Drop unnecessary columns
cols_to_drop = [col for col in ["id", "Unnamed: 32"] if col in df.columns]
df.drop(columns=cols_to_drop, inplace=True)

# Features and Target
X_train, X_test, y_train, y_test = train_test_split(
    df.iloc[:, 1:],
    df.iloc[:, 0],
    test_size=0.2,
    random_state=42
)

# Feature Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Label Encoding
encoder = LabelEncoder()
y_train = encoder.fit_transform(y_train)
y_test = encoder.transform(y_test)

# NumPy to Torch Tensor
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)

X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

print("X_train:", X_train_tensor.shape)
print("y_train:", y_train_tensor.shape)
print("X_test :", X_test_tensor.shape)
print("y_test :", y_test_tensor.shape)

# Neural Network
class MySimpleNN:

    def __init__(self, X):
        self.weights = torch.rand(
            X.shape[1],
            1,
            dtype=torch.float32,
            requires_grad=True
        )

        self.bias = torch.rand(
            1,
            dtype=torch.float32,
            requires_grad=True
        )

    def forward(self, X):
        z = torch.matmul(X, self.weights) + self.bias
        y_pred = torch.sigmoid(z)
        return y_pred


# Model
model = MySimpleNN(X_train_tensor)

# Built-in BCE Loss
criterion = nn.BCELoss()

learning_rate = 0.1
epochs = 250

print("Initial Weights:")
print(model.weights)

print("Initial Bias:")
print(model.bias)

# Training Loop
for epoch in range(epochs):

    # Forward Pass
    y_pred = model.forward(X_train_tensor)

    # Loss Calculation
    loss = criterion(
        y_pred,
        y_train_tensor
    )

    # Backward Pass
    loss.backward()

    # Gradient Descent
    with torch.no_grad():

        model.weights -= learning_rate * model.weights.grad
        model.bias -= learning_rate * model.bias.grad

        # Reset Gradients
        model.weights.grad.zero_()
        model.bias.grad.zero_()

    if (epoch + 1) % 50 == 0:
        print(
            f"Epoch [{epoch+1}/{epochs}] "
            f"Loss: {loss.item():.4f}"
        )

# Evaluation
with torch.no_grad():

    y_test_pred = model.forward(X_test_tensor)

    y_test_pred_labels = (
        y_test_pred >= 0.5
    ).float()

    accuracy = (
        y_test_pred_labels == y_test_tensor
    ).float().mean()

    print(f"\nTest Accuracy: {accuracy.item():.4f}")