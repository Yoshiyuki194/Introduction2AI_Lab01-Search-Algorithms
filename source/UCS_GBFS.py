from queue import PriorityQueue
import os, math
row = [-1, 0, 1, 0]
col = [0, 1, 0, -1]
oo = 100000000
# Number of maps
nMaps = 1
# Read a map and return input values
def readInput(path):
    with open(path, 'r') as inputFile:
        data = inputFile.readlines()
        n = int(data[0])
        r_size = len(data) - n - 1
        c_size = len(data[n + 1]) - 1
        a = [[0 for i in range(c_size)] for j in range(r_size)]
        for i in range(n):
            row = data[i + 1].split()
            x, y, z = int(row[0]), int(row[1]), int(row[2])
            a[x][y] = z
        i, j = 0, 0
        start = (-1, -1)
        dest = (-1, -1)
        for line in data[n + 1:]:
            if i == r_size: break
            j = 0
            for chr in line:
                if j == c_size: break
                a[i][j] = 1 if chr == 'x' else a[i][j]
                if chr == 'S': start = (i, j)
                if a[i][j] == 0 and (i == 0 or i == r_size - 1 or j == 0 or j == c_size -1):
                    dest = (i, j)
                j += 1
            i += 1
    return a, start, dest
# Print result 
def printMaze(a, start, path):
    r_size = len(a)
    c_size = len(a[0])
    with open(path, 'w') as f:
        for i in range(r_size):
            for j in range(c_size):
                if i == start[0] and j == start[1]:
                    f.write('S')
                    continue
                if a[i][j] == 1:
                    f.write('x')
                    continue
                if a[i][j] == 2:
                    f.write('.')
                    continue
                if a[i][j] < 0:
                    f.write('+')
                    continue
                f.write(' ')
            f.write('\n')
# Create a solution
def createPath(trace, start, dest):
    l = [dest]
    while dest != start:
        dest = trace[dest[0]][dest[1]]
        l.append(dest)
    return l[0:][slice(None, None, -1)]
# UCS Algorithm
def UCS(a, start, dest):
    r_size, c_size = len(a), len(a[0])
    d = [[oo for i in range(c_size)] for j in range(r_size)]
    trace = [[None for i in range(c_size)] for j in range(r_size)]
    d[start[0]][start[1]] = 0
    PQ = PriorityQueue(len(a) * len(a[0]))
    PQ.put((0, start))
    while not PQ.empty():
        p, u = PQ.get()
        if (u == dest):
            break
        for k in range(4):
            v = (u[0] + row[k], u[1] + col[k])
            if v[0] < 0 or v[0] > r_size or v[1] < 0 or v[1] > c_size:
                continue
            if a[v[0]][v[1]] == 1 or trace[v[0]][v[1]] != None: 
                continue
            w = 1
            if d[v[0]][v[1]] > p + w:
                trace[v[0]][v[1]] = u
                d[v[0]][v[1]] = p + w
                PQ.put((d[v[0]][v[1]], v))
    if trace[dest[0]][dest[1]] == None: return None
    return createPath(trace, start, dest)
# Greedy Best-first Search Algorithm
# First Heuristic
def GBFS_Heur1(a, start, dest):
    """ Heuristic function: h(x, y) = d((x, y), dest) 
        with d(A, B) is the Euclidean distance of A and B
    """
    r_size, c_size = len(a), len(a[0])
    trace = [[None for i in range(c_size)] for j in range(r_size)]
    PQ = PriorityQueue(r_size * c_size)
    PQ.put((0, start))
    while not PQ.empty():
        p, u = PQ.get()
        if (u == dest):
            break
        for k in range(4):
            v = (u[0] + row[k], u[1] + col[k])
            if v[0] < 0 or v[0] > r_size or v[1] < 0 or v[1] > c_size:
                continue
            if a[v[0]][v[1]] == 1 or trace[v[0]][v[1]] != None: 
                continue
            trace[v[0]][v[1]] = u
            PQ.put((math.sqrt((v[0] - dest[0]) ** 2 + (v[1] - dest[1]) ** 2), v))
    if trace[dest[0]][dest[1]] == None: return None
    return createPath(a, trace, start, dest)
# Second Heuristic
def GBFS_Heur2(a, start, dest):
    r_size, c_size = len(a), len(a[0])
    trace = [[None for i in range(c_size)] for j in range(r_size)]
    PQ = PriorityQueue(r_size * c_size)
    PQ.put((0, start))
    while not PQ.empty():
        p, u = PQ.get()
        if (u == dest):
            break
        for k in range(4):
            v = (u[0] + row[k], u[1] + col[k])
            if v[0] < 0 or v[0] > r_size or v[1] < 0 or v[1] > c_size:
                continue
            if a[v[0]][v[1]] == 1 or trace[v[0]][v[1]] != None: 
                continue
            trace[v[0]][v[1]] = u
            PQ.put((abs(v[0] - dest[0]) + abs(v[1] - dest[1]), v))
    if trace[dest[0]][dest[1]] == None: return None
    return createPath(a, trace, start, dest)
# Main program
# Create output folder
script_path = os.path.realpath(__file__)
dir_path = os.path.dirname(script_path)
print(dir_path)
# new_abs_path = os.path.join(dir_path, 'output')
# if not os.path.exists(new_abs_path):
#     os.mkdir(new_abs_path)
def main():
    for mapNo in range(nMaps):
        a, start, dest = readInput(f'input\\input{mapNo + 1}.txt')
        printMaze(UCS(a, start, dest), start, f'output\\output{mapNo + 1}.txt')
