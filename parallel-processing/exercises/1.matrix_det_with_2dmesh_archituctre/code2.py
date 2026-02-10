import threading
import numpy as np  # For verification only

class Processor:
    def __init__(self, row, col, value):
        self.row = row
        self.col = col
        self.value = value
        self.new_value = value
        self.multiplier = 0.0
        
        # Neighbors for local communication
        self.up = None
        self.down = None
        self.left = None
        self.right = None
        
        # Buffers for receiving data in phases
        self.received_pivot = 0.0
        self.received_mult = 0.0
        self.received_pivot_row_val = 0.0

    def __str__(self):
        return f"P({self.row},{self.col}) = {self.value:.3f}"

class MeshOddEvenDeterminant:
    def __init__(self, matrix):
        self.n = len(matrix)
        self.mesh = []
        self.original = [row[:] for row in matrix]

    def build_mesh(self):
        print("\n=== Building Mesh Grid ===")
        print(f"Size: {self.n} x {self.n}\n")

        # Create processors
        for i in range(self.n):
            row_procs = []
            for j in range(self.n):
                p = Processor(i, j, self.original[i][j])
                row_procs.append(p)
            self.mesh.append(row_procs)

        # Connect neighbors
        for i in range(self.n):
            for j in range(self.n):
                if i > 0: self.mesh[i][j].up = self.mesh[i-1][j]
                if i < self.n-1: self.mesh[i][j].down = self.mesh[i+1][j]
                if j > 0: self.mesh[i][j].left = self.mesh[i][j-1]
                if j < self.n-1: self.mesh[i][j].right = self.mesh[i][j+1]

        print("Mesh built.\n")

    def show_matrix(self, title="Matrix"):
        print(f"\n{title}:")
        for i in range(self.n):
            line = [f"{self.mesh[i][j].value:8.3f}" for j in range(self.n)]
            print("[" + " ".join(line) + "]")

    def odd_phase(self, k, barrier):
        def task(p):
            # Odd phase: Processors in odd rows (>k) compute multipliers and updates if applicable
            if (p.row % 2 == 1) and p.row > k:
                # Receive pivot from up (local)
                if p.col == k and p.up:
                    p.received_pivot = p.up.value
                
                if p.col == k and abs(p.received_pivot) > 1e-10:
                    p.multiplier = p.value / p.received_pivot
                    p.new_value = 0.0
                
                # Send multiplier right (local)
                if p.col == k and p.right:
                    p.right.received_mult = p.multiplier
                
                # Update if in submatrix
                if p.col > k:
                    if p.left:
                        p.received_mult = p.left.received_mult
                    mult = p.received_mult
                    if p.up:
                        p.received_pivot_row_val = p.up.value  # Approximate pivot row val from up
                    p.new_value = p.value - mult * p.received_pivot_row_val

            barrier.wait()  # Sync end of odd phase

            # Apply if active
            if p.row > k:
                p.value = p.new_value

            barrier.wait()

        return task

    def even_phase(self, k, barrier):
        def task(p):
            # Even phase: Processors in even rows (>k) compute multipliers and updates if applicable
            if (p.row % 2 == 0) and p.row > k:
                # Receive pivot from up (local)
                if p.col == k and p.up:
                    p.received_pivot = p.up.value
                
                if p.col == k and abs(p.received_pivot) > 1e-10:
                    p.multiplier = p.value / p.received_pivot
                    p.new_value = 0.0
                
                # Send multiplier right (local)
                if p.col == k and p.right:
                    p.right.received_mult = p.multiplier
                
                # Update if in submatrix
                if p.col > k:
                    if p.left:
                        p.received_mult = p.left.received_mult
                    mult = p.received_mult
                    if p.up:
                        p.received_pivot_row_val = p.up.value  # Approximate pivot row val from up
                    p.new_value = p.value - mult * p.received_pivot_row_val

            barrier.wait()  # Sync end of even phase

            # Apply if active
            if p.row > k:
                p.value = p.new_value

            barrier.wait()

        return task

    def compute_det(self):
        self.build_mesh()
        self.show_matrix("Initial Matrix")

        print("\nStarting Odd-Even Reduction...\n")

        while 
        for step in range(self.n - 1):
            print(f"Reduction Step {step+1} (pivot at {step},{step})")

            barrier = threading.Barrier(self.n * self.n)

            # Odd phase first (parallel on threads)
            odd_task = self.odd_phase(step, barrier)
            threads_odd = []
            for i in range(self.n):
                for j in range(self.n):
                    t = threading.Thread(target=odd_task, args=(self.mesh[i][j],))
                    threads_odd.append(t)
                    t.start()
            for t in threads_odd:
                t.join()

            # Then even phase (parallel on threads)
            even_task = self.even_phase(step, barrier)
            threads_even = []
            for i in range(self.n):
                for j in range(self.n):
                    t = threading.Thread(target=even_task, args=(self.mesh[i][j],))
                    threads_even.append(t)
                    t.start()
            for t in threads_even:
                t.join()

            self.show_matrix(f"After Step {step+1}")

        # Compute determinant as product of diagonal
        det = 1.0
        print("\nFinal diagonal elements:")
        for i in range(self.n):
            diag_val = self.mesh[i][i].value
            print(f"  diag[{i}] = {diag_val:.6f}")
            det *= diag_val

        print(f"\nDeterminant = {det:.6f}")
        return det




# ------------------------------
#  Simple tests
# ------------------------------
if __name__ == "__main__":

    # test case 1
    print("\n" + "="*60)
    print("Test 1 - 3x3 matrix")
    print("="*60)

    m1 = [
        [2, 1, 1],
        [1, 3, 2],
        [1, 0, 0]
    ]

    det_calc = MeshOddEvenDeterminant(m1)
    my_det = det_calc.compute_det()

    numpy_det = np.linalg.det(np.array(m1))
    print(f"\nComparison:")
    print(f"  My result    : {my_det:.6f}")
    print(f"  NumPy result : {numpy_det:.6f}")
    print(f"  Difference   : {abs(my_det - numpy_det):.10f}")

    # test case 2
    print("\n" + "="*60)
    print("Test 2 - 4x4 matrix")
    print("="*60)

    m2 = [
        [4, 3, 2, 1],
        [3, 2, 1, 4],
        [2, 1, 4, 3],
        [1, 4, 3, 2]
    ]

    det_calc2 = MeshOddEvenDeterminant(m2)
    my_det2 = det_calc2.compute_det()

    numpy_det2 = np.linalg.det(np.array(m2))
    print(f"\nComparison:")
    print(f"  My result    : {my_det2:.6f}")
    print(f"  NumPy result : {numpy_det2:.6f}")
    print(f"  Difference   : {abs(my_det2 - numpy_det2):.10f}")

    # test case 3
    print("\n" + "="*60)
    print("Test 3 - 10x10 matrix")
    print("="*60)

    np.random.seed(0)
    m3 = np.random.randint(1, 10, size=(10, 10)).tolist()
    det_calc3 = MeshOddEvenDeterminant(m3)
    my_det3 = det_calc3.compute_det()
    numpy_det3 = np.linalg.det(np.array(m3))
    print(f"\nComparison:")
    print(f"  My result    : {my_det3:.6f}")
    print(f"  NumPy result : {numpy_det3:.6f}")
    print(f"  Difference   : {abs(my_det3 - numpy_det3):.10f}")

