import numpy as np
import threading

result = 0
lock = threading.Lock()

def det_2x2(mat):
    return mat[0][0] * mat[1][1] - mat[0][1] * mat[1][0]

def det_recursive(mat):
    if len(mat) == 1:
        return mat[0][0]
    
    if len(mat) == 2:
        return det_2x2(mat)
    
    det = 0
    for j in range(len(mat)):
        minor = []
        for i in range(1, len(mat)):
            row = []
            for k in range(len(mat)):
                if k != j:
                    row.append(mat[i][k])
            minor.append(row)
        
        cofactor = ((-1) ** j) * mat[0][j]
        det += cofactor * det_recursive(minor)
    
    return det

def thread_task(thread_num, start, end, matrix):
    global result
    
    local_sum = 0
    n = len(matrix)
    
    for j in range(start, end):
        minor = []
        for i in range(1, n):
            row = []
            for k in range(n):
                if k != j:
                    row.append(matrix[i][k])
            minor.append(row)
        
        cofactor = ((-1) ** j) * matrix[0][j]
        det_minor = det_recursive(minor)
        local_sum += cofactor * det_minor
    
    with lock:
        result += local_sum

def mesh_determinant(matrix, mesh_rows, mesh_cols):
    global result
    result = 0
    
    n = len(matrix)
    num_threads = mesh_rows * mesh_cols
    
    cols_per_thread = n // num_threads
    extra = n % num_threads
    
    threads = []
    current_col = 0
    
    for i in range(num_threads):
        start = current_col
        end = start + cols_per_thread
        
        if i < extra:
            end += 1
        
        if start < n:
            t = threading.Thread(target=thread_task, args=(i, start, end, matrix))
            threads.append(t)
            t.start()
            
            current_col = end
    
    for t in threads:
        t.join()
    
    return result


# verification tests
print("="*70)
("test 1 : 4*4 matrix")
print("="*70)

mat1 = [
    [4, 3, 2, 1], 
    [3, 2, 1, 4],
    [2, 1, 4, 3],
    [1, 4, 3, 2]
]

print("mat1:")
for row in mat1:
    print(row)

det1_parallel = mesh_determinant(mat1, mesh_rows=2, mesh_cols=2)
det1_numpy = np.linalg.det(np.array(mat1))

print(f"\n ourcode det:{det1_parallel}")
print(f"verify with NUMPY: {det1_numpy:.6f}")
print(f"{' true!' if abs(det1_parallel - det1_numpy) < 0.0001 else 'false!'}")

print("\n" + "="*70)
print("test 2: 4*4 matrix")
print("="*70)

mat2 = [
    [2, 5, 3, 1],
    [1, 3, 2, 4],
    [4, 1, 5, 2],
    [3, 2, 1, 3]
]

print("mat2:")
for row in mat2:
    print(row)

det2_parallel = mesh_determinant(mat2, mesh_rows=2, mesh_cols=2)
det2_numpy = np.linalg.det(np.array(mat2))

print(f"\n ourcode det:{det2_parallel}")
print(f"verify with NUMPY: {det2_numpy:.6f}")
print(f"{' true!' if abs(det2_parallel - det2_numpy) < 0.0001 else 'false!'}")

print("\n" + "="*70)
print("test 3: 3*3 matrix")
print("="*70)

mat3 = [
    [6, 1, 1],
    [4, -2, 5],
    [2, 8, 7]
]

print("mat3:")
for row in mat3:
    print(row)

det3_parallel = mesh_determinant(mat3, mesh_rows=2, mesh_cols=2)
det3_numpy = np.linalg.det(np.array(mat3))

print(f"\n ourcode det:{det3_parallel}")
print(f"verify with NUMPY: {det3_numpy:.6f}")
print(f"{' true!' if abs(det3_parallel - det3_numpy) < 0.0001 else 'false!'}")

print("\n" + "="*70)
print("test 4: 5*5 matrix")
print("="*70)

mat4 = [
    [1, 2, 3, 4, 5],
    [2, 3, 4, 5, 1],
    [3, 4, 5, 1, 2],
    [4, 5, 1, 2, 3],
    [5, 1, 2, 3, 4]
]

print("mat4:")
for row in mat4:
    print(row)

det4_parallel = mesh_determinant(mat4, mesh_rows=2, mesh_cols=2)
det4_numpy = np.linalg.det(np.array(mat4))

print(f"\n ourcode det:{det4_parallel}")
print(f"verify with NUMPY: {det4_numpy:.6f}")
print(f"{' true!' if abs(det4_parallel - det4_numpy) < 0.0001 else 'false!'}")


