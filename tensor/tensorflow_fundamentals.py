
from numpy import matrix
import tensorflow as tf
import numpy as np


########################## tf.constant ##########################

# create constant tensor
scalar = tf.constant(7)

# check dimension
print(scalar.ndim)

# create a vector
vector = tf.constant([10, 10])
print(vector)
print(vector.ndim)

# create a matrix 
matrix = tf.constant([[10, 7], [7, 10]])
print(matrix)
print(matrix.ndim)

another_matrix = tf.constant([[10., 7.],
                              [3., 2.],
                              [8., 9.]], dtype=tf.float16)
print(another_matrix)
print(another_matrix.ndim)


########################## tf.Variable ##########################

changeable_tensor = tf.Variable([10, 7])
unchangeable_tensor = tf.constant([10, 7])
print(changeable_tensor)
print(unchangeable_tensor)

changeable_tensor[0].assign(7)
# unchangeable_tensor[0].assign(7) # unchangeable

print(changeable_tensor)
# print(unchangeable_tensor)

########################## creating random tensors ##########################
random_1 = tf.random.Generator.from_seed(2)
random_1 = random_1.normal(shape=(3, 2))
random_2 = tf.random.Generator.from_seed(2)
random_2 = random_2.normal(shape=(3, 2))

print("random_1", random_1)
print("random_2", random_2)
print("random_1==random_2", random_1==random_2)

########################## shuffleing the order of tensors ##########################
not_shuffled = tf.constant([[10, 7],
                            [3, 4],
                            [2, 5]])
print(not_shuffled.ndim)
print(not_shuffled)
tf.random.set_seed(42) # global level random seed
print("tf.random.shuffle", tf.random.shuffle(not_shuffled, seed=42)) # opeeration level random seed

############################### create a tensor of all ones ###############################
t1 = tf.ones([10, 7])
print(t1)

############################### create a tensor of all zeros ###############################
t0 = tf.zeros([3, 4])
print(t0)

############################### turn numpy arrays into tensor ###############################
numpyA = np.arange(1, 25, dtype=np.int32)
print(numpyA)

A = tf.constant(numpyA, shape=(2, 3, 4))
print(A)

############################### create a rank 4 tensor ###############################
rank_4_tensor = tf.zeros(shape=[2, 3, 4, 5])

print(rank_4_tensor)

# Get various attribute of the tensor
print("Datatype of every element:", rank_4_tensor.dtype)
print("Number of dimensions (rank):", rank_4_tensor.ndim)
print("Shape of temsor:", rank_4_tensor.shape)
print("Elements along the 0 axis:", rank_4_tensor.shape[0])
print("Elements along the last axis:", rank_4_tensor.shape[-1])
print("Total number of elements in our tensor:", tf.size(rank_4_tensor))

############################### indexing tensor ###############################
# get the first 2 elements of each dimension
print(rank_4_tensor[:2, :2, :2, :2])

# get the first element from each dimension from each index except for the final one
print(rank_4_tensor[:1, :1, :1, :])

# get the last item of each of row of the rank 2 tensor
rank_2_tensor = tf.constant([[10, 7],
                             [3, 4]])

print(rank_2_tensor[:, -1])

# add in axtra dimension to the tensor
rank_3_tensor = rank_2_tensor[..., tf.newaxis]
print(rank_3_tensor)

# alternative to tf.newaxis
rank_3_tensor = tf.expand_dims(rank_2_tensor, axis=-1)
print(rank_3_tensor)

############################### manipulating tensor (tensor operation) ###############################
# add
tensor = tf.constant([[10, 7],
                     [3, 4]])
tensorAdd10 = tensor + 10
print(tensorAdd10)

# multiplication
tensorMul10 = tensor * 10
print(tensorMul10)

# substraction
tensorSub10 = tensor - 10
print(tensorSub10)

# you can use the tensorlfow built-in function too
tf.multiply(tensor, 10)

############################### matrix multiplication ###############################
tensorMatmul = tf.matmul(tensor, tensor) # tf.linalg.matmul
print(tensorMatmul)

# create two tensors for matrix multiplication
X = tf.constant([[1, 2], 
                 [3, 4],
                 [5, 6]])

Y = tf.constant([[7, 8, 9], 
                 [10, 11, 12]])

Z = tf.constant([[1, 2], 
                 [3, 4],
                 [5, 6]])

print(X @ tf.reshape(Z, shape=(2, 3)))
print(tf.matmul(X, tf.reshape(Z, shape=(2, 3))))
print(tf.matmul(X, Y))


############################### change the data type ###############################
B = tf.constant([1.7, 7.4])
print(B.dtype)

C = tf.constant([7, 10])
print(C.dtype)

D = tf.cast(B, dtype=tf.float16)
print(D.dtype)

############################### aggregating tensors ###############################
E = tf.constant(np.random.randint(0, 100, size=50))

# find the minmum
print(tf.reduce_min(E))

# find the maximun
print(tf.reduce_max(E))

# find the mean
print(tf.reduce_mean(E))

# find tje sum
print(tf.reduce_sum(E))

# find the variance 
import tensorflow_probability as tfp
print(tfp.stats.variance(E))

# find the standard deviation
print(tf.math.reduce_std(tf.cast(E, dtype=tf.float32)))

############################### find the position maximum and minimum ###############################
tf.random.set_seed(42)
F = tf.random.uniform(shape=[50])
print(F)

# find the positional maximum
print(tf.argmax(F))

# index on our largest value position
print(F[tf.argmax(F)])
print(tf.reduce_max(F))

############################### squeeze a tensor ###############################
tf.random.set_seed(42)
G = tf.constant(tf.random.uniform(shape=[50]), shape=(1, 1, 1, 1, 50))
print(G)
G_squeezed = tf.squeeze(G)
print(G_squeezed, G_squeezed.shape)

############################### one-hot encoding tensor ###############################
# create a list of indices
some_list = [1, 2, 3, 4]

# one hot encode
print(tf.one_hot(some_list, 4))

# specify custom value for noe hot encoding
print(tf.one_hot(some_list, depth=4, on_value="andy", off_value="wu"))

# square 
H = tf.range(1, 10)
print(tf.square(H))

# sqrt
print(tf.sqrt(tf.cast(H, dtype=tf.float32)))

# log
print(tf.math.log(tf.cast(H, dtype=tf.float32)))

############################### tensors and numpy ###############################
# create a tensor from a numpy array
J = tf.constant(np.array([3., 7., 10.]))
print(J)

# convert the tensor back to numpy array
print(np.array(J))
print(J.numpy())

# the default types of each are slightly different
numpy_J = tf.constant(np.array([3., 7., 10.]))
tensor_J = tf.constant([3., 7., 10.])
print(numpy_J.dtype, tensor_J.dtype)

############################### finding access to GPU ###############################
print(tf.config.list_physical_devices("GPU"))