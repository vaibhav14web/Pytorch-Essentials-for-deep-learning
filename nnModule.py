# Example 1 : Neural network with only one layer no hidden layer

import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self, num_features):
        super().__init__()
        self.linear = nn.Linear(num_features, 1)
        self.sigmoid = nn.Sigmoid()
        
    def  forward(self, features):
        out = self.linear(features)
        out = self.sigmoid(out)
        
        return out
    
features = torch.rand(10, 5)
print("Model features:" ,features)

model = Model(features.shape[1])  # model class object
model(features)                # calling model
print("Model Weights:")
print(model.linear.weight)    
    

# Example 2: neural network with one hidden layer

class Model2(nn.Module):
    def __init__(self, num_features):
        super().__init__()
       # self.linear1 = nn.Linear(num_features, 3)
        # self.relu = nn.ReLU()
        # self.linear2 = nn.Linear(3, 1)
        # self.sigmoid = nn.Sigmoid()
        
        """
        Instead of the above code, we can
        use a sequential container.
        
        """
        self.network = nn.Sequential(
            nn.Linear(num_features, 3),
            nn.ReLU(),
            nn.Linear(3,1),
            nn.Sigmoid()
        )
        
    def forward(self, features):
        """
        out = self.linear1(features)
        out = self.relu(out)
        out = self.linear2(out)
        out = self.sigmoid(out)
        """
        out = self.network(features)
        return out

features2 = torch.rand(10, 5)
print("Model Features 2:",features2)
model2 = Model2(features.shape[1])
model2(features)
print(model2.network[2].weight)        
        

