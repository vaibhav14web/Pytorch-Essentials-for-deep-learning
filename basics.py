import torch
print(torch.__version__)

if torch.cuda.is_available():
    print("Gpu is Available!")
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
else:
    print("Running on  CPU")


    # Creating a tensor
  #1. using Empty 
a = torch.empty(2, 3)  # -> this will create a tensor of size 2x3 with uninitialized values (random values)
print(a)

# using zeros 
b = torch.zeros(2, 3)  # -> this will create a tensor of size 2x3 filled with zeros
print(b)

# using ones 
c = torch.ones(2,3) # -> this will create a tensor of size 2x3 filled with ones
print(c)

#using random
d = torch.rand(2, 3) # -> this will create a tensor of size 2x3 filled with random values between 0 and 1   
print(d)
  
# use of seed
torch.manual_seed(42) # -> this will set the seed for generating random numbers to 42, ensuring reproducibility
e = torch.rand(2, 3) # -> this will create a tensor of size 2x3 filled with random values between 0 and 1, but the values will be the same every time you run the code due to the seed
print(e)  

# using tensor
f = torch.tensor([[1,2,3], [4,5,6]])  #-> this will create a tensor with specified values rows representing the dimensions of the tensor and columns representing the values in each dimension
print(f)

# using arange
print("Arange tensor:", torch.arange(0, 10, 2)) # -> this will create a tensor with values starting from 0 to 10 (exclusive) with a step of 2, resulting in a tensor containing the values [0, 2, 4, 6, 8]

#using Linspace
print("Linspace tensor:", torch.linspace(0, 1, steps=5)) # -> this will create a tensor with 5 equally spaced values between 0 and 1, resulting in a tensor containing the values [0.0, 0.25, 0.5, 0.75, 1.0]

# USING eye
print("Eye tensor:", torch.eye(3)) # -> this will create a 3x3 identity matrix, which is a square matrix with ones on the main diagonal and zeros elsewhere, resulting in a tensor containing the values [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]

# Tensor shapes
x = torch.tensor([[1, 2, 3], [4, 5, 6]])
print(x.shape) # -> this will print the shape of the tensor x, which is (2, 3) indicating that it has 2 rows and 3 columns

print(torch.empty_like(x)) # -> this will create a new tensor with the same shape as x but with uninitialized values (random values)

print(torch.zeros_like(x)) # -> this will create a new tensor with the same shape as x but filled with zeros

print(torch.ones_like(x)) # -> this will create a new tensor with the same shape as x but filled with ones

# print(torch.rand_like(x)) # -> this will create a new tensor with the same shape as x but filled with random values between 0 and 1, and the data type of the tensor will be float32, if we not specify the data type, it will default to the same data type as x, which is int64 in this case.

# tensor data types
x.dtype # -> this will print the data type of the tensor x, which is torch.int64, indicating that the elements of the tensor are 64-bit integers

# assign data types
torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32) # -> this will create a tensor with the specified values and data type of float32, resulting in a tensor containing the values [1.0, 2.0, 3.0] with a data type of float32

# using to() method to change data type
x.to(torch.int32) # -> this will create a new tensor with the same values as x but with a data type of int32, resulting in a tensor containing the values [[1, 2, 3], [4, 5, 6]] with a data type of int32

# Arithmetic operations
y = torch.tensor([[7, 8, 9], [10, 11, 12]])
print(x + y) # -> this will perform element-wise addition between tensors x and y, resulting in a new tensor containing the values [[8, 10, 12], [14, 16, 18]]

print(x - y) # -> this will perform element-wise subtraction between tensors x and y, resulting in a new tensor containing the values [[-6, -6, -6], [-6, -6, -6]]

print(x * y) # -> this will perform element-wise multiplication between tensors x and y, resulting in a new tensor containing the values [[7, 16, 27], [40, 55, 72]]    

print(x / y) # -> this will perform element-wise division between tensors x and y, resulting in a new tensor containing the values [[0.1429, 0.25, 0.3333], [0.4, 0.4545, 0.5]] (rounded to 4 decimal places)

c = torch.tensor([[1, 2], [3, 4]])

# absolute value
print(torch.abs(c)) # -> this will compute the absolute value of each element in the tensor
# exponentiation
print(torch.exp(c)) # -> this will compute the exponential of each element in the tensor

#clamp
torch.clamp(c, min=2) # -> this will clamp all values in the tensor to a minimum of 2, meaning that any value less than 2 will be set to 2, resulting in a new tensor containing the values [[2, 2], [3, 4]]

# Reduction operations
e = torch.randint(size=(2, 3), low = 0, high = 10)  # -> this will create a tensor of size 2x3 filled with random integers between 0 (inclusive) and 10 (exclusive)

torch.sum(e) # -> this will compute the sum of all elements in the tensor e, resulting in a single scalar value

torch.sum(e, dim = 0) # -> this will compute the sum of elements in the tensor e along the specified dimension (dim=0), which means it will sum the elements column-wise, resulting in a new tensor containing the sums of each column

torch.sum(e, dim = 1) # -> this will compute the sum of elements in the tensor e along the specified dimension (dim=1), which means it will sum the elements row-wise, resulting in a new tensor containing the sums of each row    

torch.argmax(e) # -> this will return the indices of the maximum values in the tensor e, resulting in a new tensor containing the indices of the maximum values along the specified dimension (default is dim=0, which means it will return the indices of the maximum values column-wise)

torch.argmin(e) # -> this will return the indices of the minimum values in the tensor e, resulting in a new tensor containing the indices of the minimum values along the specified dimension (default is dim=0, which means it will return the indices of the minimum values column-wise)

# Matrix operations
f = torch.randint(size=(2, 3), low = 0, high = 10) 
g = torch.randint(size=(3, 4), low = 0, high = 10)

torch.matmul(f, g) # -> this will perform matrix multiplication between tensors f and g, resulting in a new tensor containing the product of the two matrices   

#dot product
vector1 = torch.tensor([1, 2, 3])
vector2 = torch.tensor([4, 5, 6])
torch.dot(vector1, vector2) # -> this will compute the dot product of vectors vector1 and vector2, resulting in a single scalar value

# transpose
torch.transpose(f, 0, 1) # -> this will transpose the tensor f by swapping its dimensions, resulting in a new tensor where the rows and columns are interchanged

# comparision operations
i = torch.randint(size=(2, 3), low = 0, high = 10)
j = torch.randint(size=(2, 3), low = 0, high = 10)

print(i == j) # -> this will perform element-wise comparison between tensors i and j, resulting in a new tensor containing boolean values (True or False) indicating whether the corresponding elements in i and j are equal

print(i > j) # -> this will perform element-wise comparison between tensors i and j, resulting in a new tensor containing boolean values (True or False) indicating whether the corresponding elements in i are greater than those in j

print(i < j) # -> this will perform element-wise comparison between tensors i and j, resulting in a new tensor containing boolean values (True or False) indicating whether the corresponding elements in i are less than those in j

# special functions
k = torch.tensor([-1.0, 0.0, 1.0,2.0, 3.0]) 
torch.sigmoid(k) # -> this will apply the sigmoid function to each element in the tensor k, resulting in a new tensor containing the sigmoid values of each element

torch.softmax(k, dim=0) # -> this will apply the softmax function to the tensor k along the specified dimension (dim=0), resulting in a new tensor containing the softmax values of each element, which represent probabilities that sum up to 1 across the specified dimension

# Inplace operations -> to save memory, we can perform operations in-place, which means that the original tensor will be modified instead of creating a new tensor. In-place operations are denoted by an underscore (_) at the end of the function name.
l = torch.tensor([1, 2, 3])
l.add_(1) # -> this will add 1 to each element of the tensor l in-place, modifying the original tensor, resulting in a new tensor containing the values [2, 3, 4] and the original tensor l will also be updated to [2, 3, 4]
print(l) # -> this will print the modified tensor l, which now contains the values [2, 3, 4] after the in-place addition operation

# copying a tensor 
a = torch.rand(2, 3)
b = a # -> this will create a new reference b that points to the same tensor as a, meaning that any changes made to b will also affect a, and vice versa, since they both refer to the same underlying data in memory

b = a.clone() # -> this will create a new tensor b that is a copy of tensor a, meaning that b will have the same values as a but will be stored in a different location in memory, so changes made to b will not affect a, and vice versa, since they are now separate tensors with their own data in memory

# tensor operation on gpu

torch.cuda.is_available() # -> this will check if a GPU is available for use with PyTorch, returning True if a GPU is available and False otherwise

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # -> this will set the device variable to "cuda" if a GPU is available, or "cpu" if a GPU is not available, allowing you to specify the device on which to perform tensor operations  

# creating a tensor on 

torch.rand(2, 3, device=device) # -> this will create a tensor of size 2x3 filled with random values between 0 and 1 on the specified device (either "cuda" for GPU or "cpu" for CPU)   

# moving an existing tensor 
a = torch.rand(2, 3) # -> this will create a tensor of size 2x3 filled with random values between 0 and 1 on the default device (usually CPU)
a.to(device) # -> this will move the tensor a to the specified device (either "cuda" for GPU or "cpu" for CPU), allowing you to perform operations on the tensor using the capabilities of the chosen device

# Reshaping tensor
a = torch.ones(2, 3)
a.reshape(1, 6) # -> this will reshape the tensor a from its original shape to a new shape of 1x6, meaning that the tensor will now have 1 row and 6 columns, while still containing the same number of elements (6) as the original tensor.

a.flatten() # -> this will flatten the tensor a into a 1D tensor, meaning that all the elements of the original tensor will be arranged in a single dimension, resulting in a new tensor containing the values [1, 1, 1, 1, 1, 1] with a shape of (6,)

# unsqueeze 
c = torch.rand(226, 226, 3) # -> this will create a tensor of size 226x226x3 filled with random values between 0 and 1, which could represent an image with height and width of 226 pixels and 3 color channels (e.g., RGB)
c.unsqueeze(0) # -> this will add a new dimension to the tensor c at the specified position (dim=0), resulting in a new tensor with a shape of (1, 226, 226, 3), which could represent a batch of images with a batch size of 1 

# numpy and torch interoperability
import numpy as np
a = torch.tensor([1, 2, 3])
b = a.numpy() # -> this will convert the PyTorch tensor a into a NumPy array, resulting in a new NumPy array containing the values [1, 2, 3] with a data type of int64

c = np.array([1, 2, 3])
d = torch.from_numpy(c) # -> this will convert the NumPy array c into a PyTorch tensor, resulting in a new PyTorch tensor containing the values [1, 2, 3] with a data type of int64

print(b)
print(d)

#indexing and slicing

a = torch.tensor([1, 2, 3, 4, 5])
print(a[0]) # -> this will access the first element of the tensor a, resulting in a scalar value of 1
print(a[1:4]) # -> this will slice the tensor a from index 1 to index 4 (exclusive), resulting in a new tensor containing the values [2, 3, 4]
print(a[:3]) # -> this will slice the tensor a from the beginning to index 3 (exclusive), resulting in a new tensor containing the values [1, 2, 3]
print(a[3:]) # -> this will slice the tensor a from index 3 to the end, resulting in a new tensor containing the values [4, 5]
print(a[-1]) # -> this will access the last element of the tensor a, resulting  in a scalar value of 5