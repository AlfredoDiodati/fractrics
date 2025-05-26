import numpy as np

def partition_map(arr, func=None, num_partitions=2):
    """
    Given a function that maps an array to a real value, it computes an array containing the
    the output of that function for a specified number of ordered partition of an array
    
    :param arr: the input array to compute the partition-wise function
    :param func: a function that maps an array to a real value
    :param num_partitions: n of partitions to split the array in
    """
    
    if func is None:
        func = lambda x: 0    
    
    partition_size = len(arr) // num_partitions
    result = []
    
    for i in range(num_partitions):
        start = i * partition_size
        end = start + partition_size if i < num_partitions - 1 else len(arr)  # Ensure last partition includes remaining elements
        partition = arr[start:end]
        result.append(func(partition))
    
    return np.array(result)

def OLS(x, y, intercept=True):
    """
    Returns array of weights of an OLS regression 
    """
    x = (np.array(x))
    y = np.array(y)
    if(intercept): x = np.c_[x, np.ones(x.shape[0])]
    return np.dot(np.linalg.inv(np.dot(x.T, x)), np.dot(x.T, y))
