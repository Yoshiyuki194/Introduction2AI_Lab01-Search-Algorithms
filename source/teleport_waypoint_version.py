from queue import PriorityQueue
# Process maps with teleport waypoints
def read_file(file_name: str):
  f = open(file_name,'r')
  n = int(next(f)[:-1])
  portals = bonus_points = None
  temp = [i for i in map(int, next(f)[:-1].split(' '))]
  if len(temp) == 4:
    portals = [(temp[0], temp[1], temp[2], temp[3])]
  else:
    bonus_points = [(temp[0], temp[1], temp[2])]
  if portals is not None:
    for i in range(n - 1):
      x, y, x1, y1 = map(int, next(f)[:-1].split(' '))
      portals.append((x, y, x1, y1))
  else: 
    for i in range(n - 1):
      x, y, z = map(int, next(f)[:-1].split(' '))
      bonus_points.append((x, y, z))
    pass
  text = f.read()
  matrix = [list(i) for i in text.splitlines()]
  f.close()
  # Insert portals
  if portals is not None:
    for p in portals:
      x, y, x1, y1 = p
      matrix[x][y] = (x1, y1)
      matrix[x1][y1] = (x, y)
  else: # Insert bonus points
    for b in bonus_points:
      x, y, z = b
      matrix[x][y] = z
  # Find start and goal node
  start = goal = None
  for i in range(len(matrix)):
    for j in range(len(matrix[0])):
      if matrix[i][j] == 'S':
        start = (i, j)
      if matrix[i][j] == ' ' and (i == 0 or i == len(matrix) - 1 or j == 0 or j == len(matrix[0]) - 1):
        goal = (i, j)
      if start is not None and goal is not None:
        return matrix, start, goal
# Create a solution
def createPath(trace, start, dest):
    l = [dest]
    while dest != start:
        dest = trace[dest[0]][dest[1]]
        l.append(dest)
    return l[0:][slice(None, None, -1)]
# BFS Algorithm
def BFS(a, start, goal):
  row = [-1, 0, 1, 0]
  col = [0, 1, 0, -1]
  r_size, c_size = len(a), len(a[0])
  trace = [[None for i in range(c_size)] for j in range(r_size)]
  Q = [start]
  op = []
  while Q:
    u = Q.pop(0)
    if (u == goal):
      break
    for k in range(4):
      v = u[0] + row[k], u[1] + col[k]
      if v[0] < 0 or v[0] > r_size or v[1] < 0 or v[1] > c_size:
        continue
      if a[v[0]][v[1]] == 'x' or a[v[0]][v[1]] == 'S' or trace[v[0]][v[1]] is not None: 
        continue
      trace[v[0]][v[1]] = u
      if type(a[v[0]][v[1]]) == tuple:
        w = a[v[0]][v[1]]
        if trace[w[0]][w[1]] is None:
          trace[w[0]][w[1]] = v
          Q.append(w)
          op.append(w)
          continue
      Q.append(v)
      op.append(v)
  if trace[goal[0]][goal[1]] == None: return None
  return createPath(trace, start, goal), op
# A_star Algorithm for maps with bonus points
row = [-1, 0, 1, 0]
col = [0, 1, 0, -1]
oo = 100000000
def A_star(a, start, goal):
  r_size, c_size = len(a), len(a[0])
  d = [[oo for i in range(c_size)] for j in range(r_size)]
  trace = [[None for i in range(c_size)] for j in range(r_size)]
  d[start[0]][start[1]] = 0
  PQ = PriorityQueue(len(a) * len(a[0]))
  PQ.put((0, start))
  while not PQ.empty():
      _, u = PQ.get()
      if (u == goal):
          break
      for k in range(4):
          v = (u[0] + row[k], u[1] + col[k])
          if v[0] < 0 or v[0] > r_size or v[1] < 0 or v[1] > c_size:
              continue
          if a[v[0]][v[1]] == 1 or trace[v[0]][v[1]] != None: 
              continue
          w = a[v[0]][v[1]] if a[v[0]][v[1]] < 0 else 1
          h = abs(v[0] - goal[0]) + abs(v[1] - goal[1])
          if d[v[0]][v[1]] > d[u[0]][u[1]] + w + h:
              trace[v[0]][v[1]] = u
              d[v[0]][v[1]] = d[u[0]][u[1]] + w + h
              PQ.put((d[v[0]][v[1]], v))
  if trace[goal[0]][goal[1]] == None: return None
  return createPath(trace, start, goal)
  pass
# a, start, goal = read_file("input_teleportation.txt")
# printMaze(BFS(a, start, goal), start, "output.txt")

# For testing and debugging
# def printMaze(a, start, path):
#     r_size = len(a)
#     c_size = len(a[0])
#     port = 1
#     with open(path, 'w') as f:
#         for i in range(r_size):
#             for j in range(c_size):
#                 if i == start[0] and j == start[1]:
#                     f.write('S')
#                     continue
#                 if a[i][j] == 'x':
#                     f.write('x')
#                     continue
#                 if a[i][j] == '.':
#                     f.write('.')
#                     continue
#                 if a[i][j] == ' ':
#                     f.write(' ')
#                     continue
#                 if type(a[i][j]) == tuple:
#                     x, y = a[i][j]
#                     a[i][j] = port
#                     a[x][y] = port
#                     port += 1
#                 f.write(str(a[i][j]))
#             f.write('\n')