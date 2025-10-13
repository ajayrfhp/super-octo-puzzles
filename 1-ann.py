from dataclasses import dataclass
import numpy as np 
import heapq
import time


def euclidean_distance(a, b):
    return np.linalg.norm(a - b)

def get_k_nearest_neighbors_naive(data_points, new_point, k=1):
    """Gets K nearest neigbhors naive
    
    In the worst case, this leads to O(NlogN). Can grow every big if N is large
    Args:
        data_points (_type_): _description_
        new_point (_type_): _description_
        k (int, optional): _description_. Defaults to 1.

    Returns:
        _type_: _description_
    """
    distances = np.linalg.norm(data_points - new_point, axis=1)
    sorted_indices = np.argsort(distances)
    return data_points[sorted_indices][:k]

def get_k_nearest_neighbors_heap(data_points, new_point, k=1):
    """Gets K nearest neigbhors uses heap

    In the worst case, this leads to O(Nlogk)
    Args:
        data_points (_type_): _description_
        new_point (_type_): _description_
        k (int, optional): _description_. Defaults to 1.
    """
    max_heap = []
    distances = np.linalg.norm(data_points - new_point, axis=1)
    for i, data_point in enumerate(data_points):
        distance = distances[i]
        if i < k:
            heapq.heappush(max_heap, (-distance, data_point))
        else:    
            if distance <= -max_heap[0][0]:
                _ = heapq.heappop(max_heap)
                heapq.heappush(max_heap, (-distance, data_point))    

    heapq.heapify(max_heap)
    return [data_point for (_, data_point) in max_heap]


if __name__ == "__main__":
    a = np.array([1, 2, 3])
    b = np.array([4, 5, 6])
    print(f"Euclidean distance between {a} and {b} is {euclidean_distance(a, b)}")

    print(f"Euclidean distance between {a} and itself is {euclidean_distance(a, a)}")


    # Benchmark with 1 million data points
    num_points = 1_000_0000
    dimensions = 10
    k = 5

    print(f"\nGenerating {num_points} random data points for benchmarking...")
    data_points = np.random.rand(num_points, dimensions)
    new_point = np.random.rand(dimensions)

    # Benchmark naive method
    print("Benchmarking get_k_nearest_neighbors_naive...")
    start_time = time.time()
    nearest_points_naive = get_k_nearest_neighbors_naive(data_points, new_point, k)
    end_time = time.time()
    print(f"Naive method took: {end_time - start_time:.4f} seconds")
    # print("Nearest points (naive):", nearest_points_naive)

    # Benchmark heap method
    print("\nBenchmarking get_k_nearest_neighbors_heap...")
    start_time = time.time()
    nearest_points_heap = get_k_nearest_neighbors_heap(data_points, new_point, k)
    end_time = time.time()
    print(f"Heap method took: {end_time - start_time:.4f} seconds")
    # print("Nearest points (heap):", nearest_points_heap)
  
