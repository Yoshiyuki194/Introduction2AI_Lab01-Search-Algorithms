
#from curses import KEY_ENTER
import os
import matplotlib.pyplot as plt
import copy
from heapq import heappop
from heapq import heappush
from queue import PriorityQueue
import math
import pygame
from pyparsing import White
import sys
row = [-1, 0, 1, 0]
col = [0, 1, 0, -1]
weight =1 
oo = 100000000
RED =(255,0,0)
WHITE = (255,255,255)
BLACK = (0,0,0)
GRAY = (54,54,54)
BISQUE =(255,228,196)
BLUE = (135,206,250)
YELLOW = (255,255,0)
KHAKI = (255,246,143)
GREEN = (144,238,144)
weight =500
height = 500

def visualize_maze(matrix, bonus, portal, start, end, route=None, save_path=''):
    """
    Args:
      1. matrix: The matrix read from the input file,
      2. bonus: The array of bonus points,
      3. portal: The array of portals,
      4. start, end: The starting and ending points,
      5. route: The route from the starting point to the ending one, defined by an array of (x, y), e.g. route = [(1, 2), (1, 3), (1, 4)]
    """
    #1. Define walls and array of direction based on the route
    walls=[(i,j) for i in range(len(matrix)) for j in range(len(matrix[0])) if matrix[i][j]=='x']

    if route:
        direction=[]
        for i in range(1,len(route)):
            if route[i][0]-route[i-1][0]>0:
                direction.append('v') #^
            elif route[i][0]-route[i-1][0]<0:
                direction.append('^') #v        
            elif route[i][1]-route[i-1][1]>0:
                direction.append('>')
            else:
                direction.append('<')

        direction.pop(0)

    #2. Drawing the map
    ax=plt.figure(dpi=100).add_subplot(111)

    for i in ['top','bottom','right','left']:
        ax.spines[i].set_visible(False)

    plt.scatter([i[1] for i in walls],[-i[0] for i in walls],
                marker='X',s=100,color='black')
    
    if bonus:
        plt.scatter([i[1] for i in bonus],[-i[0] for i in bonus],
                marker='P',s=100,color='green')

    plt.scatter(start[1],-start[0],marker='*',
                s=100,color='gold')

    if route:
        for i in range(len(route)-2):
            plt.scatter(route[i+1][1],-route[i+1][0],
                        marker=direction[i],color='silver')

    if portal:
        for i in range(len(portal)):
            plt.text(portal[i][1], -portal[i][0], s=chr(i + 49), color='green', fontsize = 10, weight='bold')
            plt.text(portal[i][3], -portal[i][2], s=chr(i + 49), color='green', fontsize = 10, weight='bold')

    plt.text(end[1],-end[0],'EXIT',color='red',
        horizontalalignment='center',
        verticalalignment='center')
    plt.xticks([])
    plt.yticks([])
    plt.savefig(f'{save_path}.png', bbox_inches='tight')

def read_file(file_name: str):
    f = open(file_name,'r')
    n = int(next(f)[:-1])
    portals = bonus_points = None
    if n > 0:
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
    if n > 0: # Insert portals
        if portals is not None:
            for p in portals:
                x, y, x1, y1 = p
                matrix[x][y] = (x1, y1)
                matrix[x1][y1] = (x, y)
        else: # Insert bonus points
            for b in bonus_points:
                x, y, z = b
                matrix[x][y] = z
    return bonus_points, portals, matrix

# Tìm kiếm mù
def dfs(graph, start, end):

    trace={start:(0,0)} #mảng truy vết
    stack=[start]
    open = []
    Free=copy.deepcopy(graph)
    for i in range(len(graph)):
        for j in range(len(graph[0])):
            Free[i][j]=True
    Free[start[0]][start[1]]=False

    while len(stack)>0:   #Duyệt cho đến khi không còn đỉnh còn đường đi 
        top=stack.pop()    #Lấy ra vị trí nằm trên đầu của stack
        if top==end:    
            break
        for neighbor in [(top[0],top[1]+1),(top[0],top[1]-1),(top[0]+1,top[1]),(top[0]-1,top[1])]:     # Xét 4 vị trí liền kề vị trí đang xét top
            if Free[neighbor[0]][neighbor[1]]==True and graph[neighbor[0]][neighbor[1]] == 0: 
                    trace[neighbor]=top  #Truy vết đường đi trước đó của neighbor là top
                    stack.append(neighbor)
                    open.append(neighbor)
                    Free[neighbor[0]][neighbor[1]] =False      
    
    
    path=[]
    f=end
    while trace[f]!=(0,0):
        
        path.append(trace[f])
        f=trace[f]
        
    path.reverse()
    path.append(end)
    return path,open

def bfs(graph, start, end):
    trace={start:(0,0)} #mảng truy vết
    queue=[start]
    open = []
    
    while len(queue)>0:   #Duyệt cho đến khi không còn đỉnh còn đường đi 
        top=queue.pop(0)    #Lấy ra vị trí nằm trên đầu của queue
        if top==end:    
            break
        for neighbor in [(top[0],top[1]+1),(top[0],top[1]-1),(top[0]+1,top[1]),(top[0]-1,top[1])]:     # Xét 4 vị trí liền kề vị trí đang xét top
            if neighbor not in trace and graph[neighbor[0]][neighbor[1]] == 0: 
                    trace[neighbor]=top  #Truy vết đường đi trước đó của neighbor là top
                    queue.append(neighbor)      
                    open.append(neighbor)
    
    path=[]
    f=end
    while trace[f]!=(0,0):
        path.append(trace[f])
        f=trace[f]
        
    path.reverse()
    path.append(end)
    return path,open
def reconstruct_path(trace, start, end):
    current = end
    path = [current]
    while trace:
        current = trace.pop()
        path.append(current)
    path.reverse()
    return path

def Is_Valid_Position(graph, node): 
    if node[0] >= 0:
        if  node[0] < len(graph):
            if node[1] >=0:
                if node[1] <len(graph[0]):
                    return graph[node[0]][node[1]] == 0
    return False

def Heuristic_1(current_node, goal):
    "Manhattan distance"
    current_row, current_col = current_node
    end_row, end_col = goal
    return abs(current_row - end_row) + abs(current_col - end_col)

def Heuristic_2(current_node, goal):
    "Diagonal distance"
    current_row, current_col = current_node
    end_row, end_col = goal
    dx = abs(current_row - end_row)
    dy = abs(current_col - end_col)
    return dx + dy + (math.sqrt(2)-2) * min(dx, dy)

def a_star1(graph, start, end):
    distances = {(start[0], start[1]): 0}
    trace = {start:None}
    visited = set()
    open = [start]
    pq = PriorityQueue()
    pq.put((0, (start[0], start[1])))

    while not pq.empty():
        _, node = pq.get()
        cur_row, cur_col = node
        if node == (end[0], end[1]):
            break
        
        for i in range(0,4):
            new_node = (cur_row + row[i], cur_col + col[i])
            
            if Is_Valid_Position(graph, new_node) and new_node not in visited:
                old_distance = distances.get(new_node, float('inf'))
                new_distance = distances[node] + weight
                open.append(new_node)
                if new_distance < old_distance:
                    distances[new_node] = new_distance
                    priority = new_distance + Heuristic_1(new_node, end)
                    pq.put((priority, new_node))
                    trace[new_node] = node
        
        visited.add(node)

    path=[]
    f=end
    while trace[f]!=None:
        path.append(trace[f])
        f=trace[f]
        
    path.reverse()
    path.append(end)
    return path,open

def a_star2(graph, start, end):
    distances = {(start[0], start[1]): 0}
    trace = {start:None}
    visited = set()
    open = [start]
    pq = PriorityQueue()
    pq.put((0, (start[0], start[1])))

    while not pq.empty():
        _, node = pq.get()
        cur_row, cur_col = node
        if node == (end[0], end[1]):
            break
        
        for i in range(0,4):
            new_node = (cur_row + row[i], cur_col + col[i])
            
            if Is_Valid_Position(graph, new_node) and new_node not in visited:
                old_distance = distances.get(new_node, float('inf'))
                new_distance = distances[node] + weight
                open.append(new_node)
                if new_distance < old_distance:
                    distances[new_node] = new_distance
                    priority = new_distance + Heuristic_2(new_node, end)
                    pq.put((priority, new_node))
                    trace[new_node] = node
        
        visited.add(node)

    path=[]
    f=end
    while trace[f]!=None:
        path.append(trace[f])
        f=trace[f]
        
    path.reverse()
    path.append(end)
    return path,open
def createPath(trace, start, dest):
    l = [dest]
    while dest != start:
        dest = trace[dest[0]][dest[1]]
        l.append(dest)
    return l[0:][slice(None, None, -1)]

def GBFS_Heur1(a, start, dest):
    """ Heuristic function: h(x, y) = d((x, y), dest) 
        with d(A, B) is the Euclidean distance of A and B
    """
    r_size, c_size = len(a), len(a[0])
    trace = [[None for i in range(c_size)] for j in range(r_size)]
    PQ = PriorityQueue(r_size * c_size)
    op = []
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
            op.append(v)
            PQ.put((math.sqrt((v[0] - dest[0]) ** 2 + (v[1] - dest[1]) ** 2), v))
    if trace[dest[0]][dest[1]] == None: return None, op
    return createPath(trace, start, dest), op
# Second Heuristic
def GBFS_Heur2(a, start, dest):
    r_size, c_size = len(a), len(a[0])
    trace = [[None for i in range(c_size)] for j in range(r_size)]
    PQ = PriorityQueue(r_size * c_size)
    op = []
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
            op.append(v)
    if trace[dest[0]][dest[1]] == None: return None, op
    return createPath(trace, start, dest), op

def UCS(a, start, dest):
    r_size, c_size = len(a), len(a[0])
    d = [[oo for i in range(c_size)] for j in range(r_size)]
    trace = [[None for i in range(c_size)] for j in range(r_size)]
    d[start[0]][start[1]] = 0
    op = []
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
                op.append(v)
    if trace[dest[0]][dest[1]] == None: return None, op
    return createPath(trace, start, dest), op
def nearest_bonus_point(u, bonus_points):
    res = None
    for v in bonus_points:
        d = abs(u[0] - v[0]) + abs(u[1] - v[1])
        res = d if res is None else min(res, d)
    return res

def compute_cost(path, bonus_points):
    cost, cnt = 0, 0
    for u in path:
        for v in bonus_points:
            if u[0] == v[0] and u[1] == v[1]:
                cost += v[2]
                cnt += 1
                break
    return cost + len(path) - cnt            

def bonus_astar(matrix, bonus_points, start, goal):
    r_size, c_size = len(matrix), len(matrix[0])
    for u in bonus_points:
        matrix[u[0]][u[1]] = u[2]

    d = [[2 * oo for i in range(c_size)] for j in range(r_size)]
    trace = [[None for i in range(c_size)] for j in range(r_size)]
    d[start[0]][start[1]] = 0
    op = []
    PQ = PriorityQueue(len(matrix) * len(matrix[0]))

    PQ.put((0, start))
    while not PQ.empty():
        _, u = PQ.get()
        if (u == goal):
            break
        for k in range(4):
            v = (u[0] + row[k], u[1] + col[k])
            if v[0] < 0 or v[0] > r_size or v[1] < 0 or v[1] > c_size:
                continue
            if matrix[v[0]][v[1]] == 'x' or matrix[v[0]][v[1]] == 'S' or trace[v[0]][v[1]] != None: 
                continue
            w = 1 if matrix[v[0]][v[1]] == ' ' else matrix[v[0]][v[1]]
            f = d[u[0]][u[1]] + w + nearest_bonus_point(v, bonus_points)
            if d[v[0]][v[1]] > f:
                trace[v[0]][v[1]] = u
                d[v[0]][v[1]] = d[u[0]][u[1]] + w
                PQ.put((f, v))
                op.append(v)

    if trace[goal[0]][goal[1]] == None: return None, op
    return createPath(trace, start, goal), op

def Relax(d, trace, u, v, w):
    res = d[u[0]][u[1]] < oo and d[v[0]][v[1]] > d[u[0]][u[1]] + w
    if res:
        d[v[0]][v[1]] = d[u[0]][u[1]] + w
        trace[v[0]][v[1]] = u
    return d, trace, res

def Bellman_Ford(matrix, bonus_points, start, goal):
    r_size, c_size = len(matrix), len(matrix[0])
    d = [[oo for i in range(c_size)] for j in range(r_size)]
    trace = [[None for i in range(c_size)] for j in range(r_size)]
    d[start[0]][start[1]] = 0 
    for u in bonus_points:
        matrix[u[0]][u[1]] = u[2]
    for cntLoop in range(r_size * c_size - 1):
        stop = True
        for i in range(r_size):
            for j in range(c_size):
                if matrix[i][j] != 'x':
                    u = (i, j)
                    for k in range(4):
                        v = (u[0] + row[k], u[1] + col[k])
                        if v[0] < 0 or v[0] > r_size or v[1] < 0 or v[1] > c_size or matrix[v[0]][v[1]] == 'x' or v == trace[u[0]][u[1]]:
                            continue
                        w = 1 if type(matrix[v[0]][v[1]]) == str else matrix[v[0]][v[1]]
                        d, trace, relax = Relax(d, trace, u, v, w)
                        if relax: stop = False
        if stop: break
    if trace[goal[0]][goal[1]] is None: return None
    return createPath(trace, start, goal)

def BFS_teleport(a, start, goal):
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
class Board:
    def _init_(self):
        self.board = []
        self.selected_piece = None
    def draw_cube(self,win,graph,start,end, bonus, portal):
        row_size=weight//len(graph)
        col_size=height//len(graph[0])
        win.fill(GRAY)
        pygame.draw.polygon(win,YELLOW,((start[1]*col_size, start[0]*row_size),(start[1]*col_size+col_size/2, start[0]*row_size +row_size/2),(start[1]*col_size+col_size, start[0]*row_size)))
        pygame.draw.rect(win,RED,(end[1]*col_size,end[0]*row_size+row_size/4,col_size,col_size))
        pygame.font.init()
        NUM_FONT = pygame.font.Font(pygame.font.get_default_font(), 20)
        if portal:
            count = 0
            for i in portal:
                count = count +1
                text = NUM_FONT.render(str(count),1,GREEN)
                win.blit(text,(i[1]*col_size,i[0]*row_size))
                win.blit(text,(i[3]*col_size,i[2]*row_size))
        if bonus:
            for i in bonus:
                pygame.draw.rect(win,GREEN,(i[1]*col_size,i[0]*row_size,col_size/1.1,row_size/1.1))
        for row in range(len(graph)):
            for col in range(len(graph[0])):
                if graph[row][col]== 1:
                    pygame.draw.rect(win,BISQUE,(col*col_size,row*row_size,col_size/1.1,row_size/1.1))
    def draw_open(self,win,graph,start,end, i):
        row_size=weight//len(graph)
        col_size=height//len(graph[0])
        
        if (i!=start and i!=end):
            pygame.draw.circle(win,KHAKI,(i[1]*col_size + col_size/2,i[0]*row_size+row_size/2), col_size/3)
    def draw_trace(self,win,graph,start,end, open):
        row_size=weight//len(graph)
        col_size=height//len(graph[0])
        for i in open:
            if (i!=start and i!=end):
                pygame.draw.circle(win,BLUE,(i[1]*col_size + col_size/2,i[0]*row_size+row_size/2), col_size/3)
def PGAME(graph,start,end,trace,bonus,portal,open,path,path_temp,algo):
    board = Board()
    path_ch=path+path_temp
    os.chdir(path_ch)
    FPS = 60
    WIN =pygame.display.set_mode((weight,height))
    board.draw_cube(WIN,graph,start,end, bonus, portal) 
    run = True
    row_size=weight//len(graph)
    col_size=height//len(graph[0])
    clock = pygame.time.Clock()
    frame_count = 0
    SCREEN_UPDATE= pygame.USEREVENT
    pygame.time.set_timer(SCREEN_UPDATE,20)
    while run:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run =False
            if event.type == SCREEN_UPDATE:
                if (len(open)==0):
                    board.draw_trace(WIN,graph,start,end,trace)
                    run =False
                    break
                index = open.pop(0)
                board.draw_open(WIN,graph,start,end,index)
        
        frame_count += 1
        pygame.display.update()
        
        pygame.image.save( WIN, "screen_%04d.png" % ( frame_count ) )
        
        clock.tick(FPS)
    pygame.quit()
    s = f'ffmpeg -r 30 -f image2 -s 400x400 -i screen_%04d.png -vcodec libx264 -crf 25 {algo}.mp4'
    os.system(s)
    for i in range(1,frame_count+1):
        os.remove("screen_%04d.png"% ( i ))
    os.chdir(path)
