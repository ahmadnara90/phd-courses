import threading
import queue
import math
import time

preorder_map = {}

class Packet:
    def __init__(self, src, dst):
        self.src = src
        self.dst = dst
        self.path = []

class Node(threading.Thread):
    def __init__(self, pre_idx, lvl_idx):
        super().__init__(daemon=True)
        self.pre_idx = pre_idx
        self.lvl_idx = lvl_idx
        self.left = None
        self.right = None
        self.parent = None
        self.q = queue.Queue()
        self.alive = True
        self.result = []
        self.done = threading.Event()

    def path_from_root(self):
        bits = []
        cur = self
        while cur.parent:
            if cur.parent.left is cur:
                bits.append('L')
            else:
                bits.append('R')
            cur = cur.parent
        bits.reverse()
        return bits

    def find_next(self, pkt):
        if pkt.dst == self.pre_idx:
            return 'here'
        target = preorder_map.get(pkt.dst)
        if not target:
            return 'drop'
        t_path = target.path_from_root()
        m_path = self.path_from_root()
        if len(t_path) > len(m_path) and t_path[:len(m_path)] == m_path:
            if t_path[len(m_path)] == 'L':
                return 'left'
            return 'right'
        return 'up'

    def run(self):
        while self.alive:
            try:
                pkt = self.q.get(timeout=0.3)
            except queue.Empty:
                continue
            pkt.path.append(self.pre_idx)
            nxt = self.find_next(pkt)
            if nxt == 'here':
                self.result.append(pkt)
                self.done.set()
            elif nxt == 'left' and self.left:
                self.left.q.put(pkt)
            elif nxt == 'right' and self.right:
                self.right.q.put(pkt)
            elif nxt == 'up' and self.parent:
                self.parent.q.put(pkt)
            self.q.task_done()

    def kill(self):
        self.alive = False


def make_tree(n):
    nodes = []
    for i in range(n):
        nodes.append(Node(-1, i))
    for i in range(n):
        l = 2 * i + 1
        r = 2 * i + 2
        if l < n:
            nodes[i].left = nodes[l]
            nodes[l].parent = nodes[i]
        if r < n:
            nodes[i].right = nodes[r]
            nodes[r].parent = nodes[i]
    cnt = [0]
    def assign(nd):
        if not nd:
            return
        nd.pre_idx = cnt[0]
        preorder_map[cnt[0]] = nd
        cnt[0] += 1
        assign(nd.left)
        assign(nd.right)
    assign(nodes[0])
    return nodes


def print_tree(root, n):
    if not root:
        return
    lvls = []
    cur = [root]
    while cur:
        vals = []
        nxt = []
        empty = True
        for nd in cur:
            if nd:
                vals.append(str(nd.pre_idx))
                empty = False
                nxt.append(nd.left)
                nxt.append(nd.right)
            else:
                vals.append("")
                nxt.append(None)
                nxt.append(None)
        if empty:
            break
        lvls.append(vals)
        cur = nxt

    h = len(lvls)
    w = 2 ** (h - 1)
    print()
    for i, lv in enumerate(lvls):
        sp = w // (2 ** i)
        lead = sp // 2
        line = ""
        for j, v in enumerate(lv):
            if j == 0:
                line += " " * (lead * 5)
            else:
                line += " " * ((sp - 1) * 5)
            if v:
                line += f" [{v:>2}] "
            else:
                line += "      "
        print(line)
        if i < h - 1:
            br = ""
            for j, v in enumerate(lv):
                if j == 0:
                    br += " " * (lead * 5 - 1)
                else:
                    br += " " * ((sp - 1) * 5 - 3)
                nl = lvls[i + 1] if i + 1 < h else []
                li = 2 * j
                ri = 2 * j + 1
                hl = li < len(nl) and nl[li] != ""
                hr = ri < len(nl) and nl[ri] != ""
                br += " /  " if hl else "    "
                br += " \\" if hr else "  "
            print(br)
    print()


def main():
    global preorder_map
    preorder_map = {}

    print("Binary Tree Packet Routing")
    print("-" * 35)

    while True:
        try:
            n = int(input("how many nodes? "))
            if n > 0:
                break
            print("enter a positive number")
        except:
            print("invalid input")

    nodes = make_tree(n)
    print_tree(nodes[0], n)

    for nd in nodes:
        nd.start()
    time.sleep(0.2)

    while True:
        try:
            src = int(input("source node: "))
            if src not in preorder_map:
                print(f"not found, pick from 0 to {n - 1}")
                continue
            dst = int(input("destination node: "))
            if dst not in preorder_map:
                print(f"not found, pick from 0 to {n - 1}")
                continue
            break
        except:
            print("invalid input")

    pkt = Packet(src, dst)
    preorder_map[src].q.put(pkt)

    preorder_map[dst].done.wait(timeout=10)
    time.sleep(0.3)

    if preorder_map[dst].result:
        p = preorder_map[dst].result[0]
        print(f"\npath: {' -> '.join(str(x) for x in p.path)}")
        print(f"hops: {len(p.path) - 1}")
    else:
        print("packet lost")

    for nd in nodes:
        nd.kill()
    for nd in nodes:
        nd.join(timeout=1)


if __name__ == "__main__":
    main()
