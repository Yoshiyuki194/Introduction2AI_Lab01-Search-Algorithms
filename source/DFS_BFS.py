def dfs(graph, start, end):

    trace={start:(0,0)} #mảng truy vết
    stack=[start]
    

    while len(stack)>0:   #Duyệt cho đến khi không còn đỉnh còn đường đi 
        top=stack.pop()    #Lấy ra vị trí nằm trên đầu của stack
        if top==end:    
            break
        for neighbor in [(top[0],top[1]+1),(top[0],top[1]-1),(top[0]+1,top[1]),(top[0]-1,top[1])]:     # Xét 4 vị trí liền kề vị trí đang xét top
            if neighbor not in trace and graph[neighbor[0]][neighbor[1]] == 0: 
                    trace[neighbor]=top  #Truy vết đường đi trước đó của neighbor là top
                    stack.append(neighbor)      
    
    path=[]
    f=end
    while trace[f]!=(0,0):
        path.append(trace[f])
        f=trace[f]
        
    path.reverse()
    path.append(end)
    return path

def bfs(graph, start, end):

    trace={start:(0,0)} #mảng truy vết
    queue=[start]
    
    while len(queue)>0:   #Duyệt cho đến khi không còn đỉnh còn đường đi 
        top=queue.pop(0)    #Lấy ra vị trí nằm trên đầu của queue
        if top==end:    
            break
        for neighbor in [(top[0],top[1]+1),(top[0],top[1]-1),(top[0]+1,top[1]),(top[0]-1,top[1])]:     # Xét 4 vị trí liền kề vị trí đang xét top
            if neighbor not in trace and graph[neighbor[0]][neighbor[1]] == 0: 
                    trace[neighbor]=top  #Truy vết đường đi trước đó của neighbor là top
                    queue.append(neighbor)      
    
    
    path=[]
    f=end
    while trace[f]!=(0,0):
        path.append(trace[f])
        f=trace[f]
        
    path.reverse()
    path.append(end)
    return path