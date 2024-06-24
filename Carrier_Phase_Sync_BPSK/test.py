

def find_subarray_index(small_array, large_array):
    small_len = len(small_array)
    large_len = len(large_array)
    for i in range(large_len - small_len + 1):
        if large_array[i:i + small_len] == small_array:
            return i
    return -1  # Return -1 if subarray not found

# Example usage:
large_array = [1, 2, 3, 4, 5, 6, 7, 8]
small_array1 = [3, 4, 5]
small_array2 = [8, 9]

print(find_subarray_index(small_array1, large_array))  # Output: 2 (index where [3, 4, 5] starts)
print(find_subarray_index(small_array2, large_array))  # Output: -1 (subarray [8, 9] not found)
