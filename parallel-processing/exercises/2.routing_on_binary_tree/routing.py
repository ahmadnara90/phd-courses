import threading
import queue
import math
import time


# ======================================================
# Packet class
# ======================================================
class Packet:
    def __init__(self, src, dst, data="packet"):
        self.src = src
        self.dst = dst
        self.data = data
        self.path = []


# ======================================================
# Tree Node - each node runs as a separate Thread
# ======================================================
class TreeNode(threading.Thread):
    def __init__(self, preorder_idx, level_order_idx):
        super().__init__(daemon=True)
        self.preorder_idx = preorder_idx
        self.level_order_idx = level_order_idx
        self.left = None
        self.right = None
        self.parent = None
        self.inbox = queue.Queue()
        self.running = True
        self.delivered_packets = []
        self.result_event = threading.Event()

    def __repr__(self):
        return f"Node({self.preorder_idx})"

    def get_path_from_root(self):
        path_bits = []
        node = self
        while node.parent is not None:
            if node.parent.left is node:
                path_bits.append('L')
            else:
                path_bits.append('R')
            node = node.parent
        path_bits.reverse()
        return path_bits

    def route_packet(self, packet):
        if packet.dst == self.preorder_idx:
            return 'DELIVER'

        dst_node = preorder_map.get(packet.dst)
        if dst_node is None:
            return 'DROP'

        dst_path = dst_node.get_path_from_root()
        my_path = self.get_path_from_root()

        if len(dst_path) > len(my_path) and dst_path[:len(my_path)] == my_path:
            next_dir = dst_path[len(my_path)]
            if next_dir == 'L':
                return 'LEFT'
            else:
                return 'RIGHT'

        return 'UP'

    def run(self):
        while self.running:
            try:
                packet = self.inbox.get(timeout=0.3)
            except queue.Empty:
                continue

            packet.path.append(self.preorder_idx)
            action = self.route_packet(packet)

            if action == 'DELIVER':
                self.delivered_packets.append(packet)
                self.result_event.set()

            elif action == 'LEFT' and self.left:
                self.left.inbox.put(packet)

            elif action == 'RIGHT' and self.right:
                self.right.inbox.put(packet)

            elif action == 'UP' and self.parent:
                self.parent.inbox.put(packet)

            self.inbox.task_done()

    def stop(self):
        self.running = False


# ======================================================
# Global map: preorder_index -> TreeNode
# ======================================================
preorder_map = {}


# ======================================================
# Build a balanced (possibly incomplete) binary tree
# with level-order structure, then assign Preorder VLR
# ======================================================
def build_balanced_tree(n):
    if n <= 0:
        return []

    # Step 1: create nodes with level-order index (0-based)
    level_nodes = []
    for i in range(n):
        node = TreeNode(preorder_idx=-1, level_order_idx=i)
        level_nodes.append(node)

    # Step 2: connect parent-child (level-order based)
    for i in range(n):
        left_idx = 2 * i + 1
        right_idx = 2 * i + 2
        if left_idx < n:
            level_nodes[i].left = level_nodes[left_idx]
            level_nodes[left_idx].parent = level_nodes[i]
        if right_idx < n:
            level_nodes[i].right = level_nodes[right_idx]
            level_nodes[right_idx].parent = level_nodes[i]

    # Step 3: assign preorder VLR indices
    counter = [0]

    def preorder_assign(node):
        if node is None:
            return
        node.preorder_idx = counter[0]
        preorder_map[counter[0]] = node
        counter[0] += 1
        preorder_assign(node.left)
        preorder_assign(node.right)

    preorder_assign(level_nodes[0])

    return level_nodes


# ======================================================
# Display tree in a readable format
# ======================================================
def display_tree(root, n):
    if root is None:
        print("  (empty tree)")
        return

    height = int(math.log2(n)) + 1 if n > 0 else 0
    max_width = 2 ** height

    current_level = [root]
    level = 0

    print()
    while current_level and level < height:
        gap = max_width // (2 ** level)
        leading = gap // 2

        line_vals = ""
        line_branches = ""

        for i, node in enumerate(current_level):
            if i == 0:
                line_vals += " " * (leading * 3)
            else:
                line_vals += " " * ((gap - 1) * 3)

            if node is not None:
                label = f"[{node.preorder_idx:^3}]"
                line_vals += label
            else:
                line_vals += "     "

        print(line_vals)

        # Print branches
        if level < height - 1:
            branch_leading = leading
            next_level = []
            branch_line = ""
            for i, node in enumerate(current_level):
                if i == 0:
                    branch_line += " " * (branch_leading * 3 - 2)
                else:
                    branch_line += " " * ((gap - 1) * 3 - 4)

                if node is not None and node.left is not None:
                    branch_line += " / "
                else:
                    branch_line += "   "

                branch_line += "   "

                if node is not None and node.right is not None:
                    branch_line += " \\ "
                else:
                    branch_line += "   "

                if node is not None:
                    next_level.append(node.left)
                    next_level.append(node.right)
                else:
                    next_level.append(None)
                    next_level.append(None)

            print(branch_line)
            current_level = next_level
        else:
            break

        level += 1
    print()


# ======================================================
# Alternative simple display (works better for all sizes)
# ======================================================
def simple_display(root, n):
    if root is None:
        return

    levels = []
    current = [root]

    while current:
        level_vals = []
        next_level = []
        all_none = True
        for node in current:
            if node is not None:
                level_vals.append(str(node.preorder_idx))
                all_none = False
                next_level.append(node.left)
                next_level.append(node.right)
            else:
                level_vals.append(".")
                next_level.append(None)
                next_level.append(None)
        if all_none:
            break
        levels.append(level_vals)
        current = next_level

    height = len(levels)
    max_width = 2 ** (height - 1)

    print("\n" + "=" * 60)
    print("  Binary Tree (Preorder VLR Indexing)")
    print("=" * 60)

    for i, level in enumerate(levels):
        spacing = max_width // (2 ** i)
        leading = spacing // 2

        line = ""
        for j, val in enumerate(level):
            if j == 0:
                line += " " * (leading * 5)
            else:
                line += " " * ((spacing - 1) * 5)

            if val != ".":
                line += f" [{val:>2}] "
            else:
                line += "      "

        print(line)

        # branches
        if i < height - 1:
            branch = ""
            for j, val in enumerate(level):
                if j == 0:
                    branch += " " * (leading * 5 - 1)
                else:
                    branch += " " * ((spacing - 1) * 5 - 3)

                left_child = 2 * j
                right_child = 2 * j + 1

                next_lv = levels[i + 1] if i + 1 < height else []

                has_left = left_child < len(next_lv) and next_lv[left_child] != "."
                has_right = right_child < len(next_lv) and next_lv[right_child] != "."

                if has_left:
                    branch += " /  "
                else:
                    branch += "    "

                if has_right:
                    branch += " \\"
                else:
                    branch += "  "

        print(branch) if i < height - 1 else None

    print("=" * 60)


# ======================================================
# Main
# ======================================================
def main():
    global preorder_map
    preorder_map = {}

    print("=" * 60)
    print("  Binary Tree Packet Routing (Multi-threaded)")
    print("  Preorder VLR Indexing")
    print("=" * 60)

    # Get number of nodes
    while True:
        try:
            n = int(input("\nEnter number of nodes: "))
            if n <= 0:
                print("Please enter a positive number.")
                continue
            break
        except ValueError:
            print("Invalid input. Please enter an integer.")

    # Build tree
    nodes = build_balanced_tree(n)
    root = nodes[0] if nodes else None

    # Display tree
    simple_display(root, n)

    # Show preorder mapping
    print("\nPreorder VLR Index -> Level-order Index:")
    for pre_idx in sorted(preorder_map.keys()):
        node = preorder_map[pre_idx]
        print(f"  Preorder {pre_idx} -> Level-order {node.level_order_idx}")

    # Start all threads
    for node in nodes:
        node.start()

    time.sleep(0.2)

    # Get source and destination
    while True:
        print("\n" + "-" * 40)
        try:
            src = int(input("Enter source node index (preorder): "))
            if src not in preorder_map:
                print(f"Node {src} does not exist. Valid: 0 to {n-1}")
                continue

            dst = int(input("Enter destination node index (preorder): "))
            if dst not in preorder_map:
                print(f"Node {dst} does not exist. Valid: 0 to {n-1}")
                continue
            break
        except ValueError:
            print("Invalid input. Please enter an integer.")

    # Create and send packet
    pkt = Packet(src, dst, "test_packet")
    src_node = preorder_map[src]
    dst_node = preorder_map[dst]

    print(f"\nSending packet from Node {src} to Node {dst}...")
    src_node.inbox.put(pkt)

    # Wait for delivery
    dst_node.result_event.wait(timeout=10)
    time.sleep(0.5)

    # Show result
    if dst_node.delivered_packets:
        delivered = dst_node.delivered_packets[0]
        path_str = " -> ".join(str(p) for p in delivered.path)
        print(f"\n{'=' * 60}")
        print(f"  RESULT")
        print(f"{'=' * 60}")
        print(f"  Source:      Node {src}")
        print(f"  Destination: Node {dst}")
        print(f"  Path:        {path_str}")
        print(f"  Hops:        {len(delivered.path) - 1}")
        print(f"{'=' * 60}")
    else:
        print("\nPacket was not delivered (timeout).")

    # Stop threads
    for node in nodes:
        node.stop()
    for node in nodes:
        node.join(timeout=2)

    print("\nDone!")


if __name__ == "__main__":
    main()
