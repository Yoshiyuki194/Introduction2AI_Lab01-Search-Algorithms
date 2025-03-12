import os
import ai_pygame as ap
import ai as a

if os.name == 'nt':
    slash = '\\'
else:
    slash = '//'

def getInput(path):
    script_path = os.path.realpath(__file__)
    dir_path = os.path.dirname(script_path)
    dir_path = os.path.dirname(dir_path)
    path = os.path.join(dir_path, path)
    dir = os.scandir(path)
    print(path)
    maps, nMaps = [], 0
    for entry in dir:
        if entry.is_file():
            nMaps += 1
    for mapID in range(nMaps):
        p = f'input{mapID + 1}.txt'
        input_path = os.path.join(path, p)
        maps.append(ap.read_file(input_path))
    return maps

def level_1(maps):
    algos = {'dfs', 'bfs', 'ucs', 'gbfs_heuristic_1', 'gbfs_heuristic_2', 'astar_heuristic_1', 'astar_heuristic_2'}
    script_path = os.path.realpath(__file__)
    dir_path = os.path.dirname(script_path)
    dir_path = os.path.dirname(dir_path)
    new_abs_path = os.path.join(dir_path, 'output')
    if not os.path.exists(new_abs_path):
        os.mkdir(new_abs_path)
    new_abs_path = os.path.join(new_abs_path, 'level_1')
    if not os.path.exists(new_abs_path):
        os.mkdir(new_abs_path)
    for i in range(len(maps)):
        p = f'input{i + 1}'
        new_next_abs_path = os.path.join(new_abs_path, p)
        if not os.path.exists(new_next_abs_path):
            os.mkdir(new_next_abs_path)
        for algo in algos:
            p = f'{algo}'
            alg_abs_path = os.path.join(new_next_abs_path, p)
            if not os.path.exists(alg_abs_path):
                os.mkdir(alg_abs_path)
    mapID = 1
    for obj in maps:
        bonus_points, portals, matrix = obj
        for i in range(len(matrix)):
            for j in range(len(matrix[0])):
                if matrix[i][j] == 'S':
                    start = (i,j)
                elif matrix[i][j] == ' ':
                    if (i == 0) or (i == len(matrix) - 1) or (j == 0) or (j == len(matrix[0]) - 1):
                        goal = (i,j)
                else:
                    continue
        graph = []
        for i in range(len(matrix)):
            adj = []
            for j in range(len(matrix[0])):
                if matrix[i][j] != 'x':
                    adj.append(0)
                else:
                    adj.append(1)
            graph.append(adj)
        
        wayoutDFS, openDFS = ap.dfs(graph, start, goal)
        ap.visualize_maze(matrix, bonus_points, portals, start, goal, wayoutDFS, f'output{slash}level_1{slash}input{mapID}{slash}dfs{slash}dfs')
        with open(f'output{slash}level_1{slash}input{mapID}{slash}dfs{slash}dfs.txt', 'w') as file:
            file.write(f'Cost: {len(wayoutDFS) - 1}')
            file.close()
        
        wayoutBFS, openBFS = ap.bfs(graph, start, goal)
        ap.visualize_maze(matrix, bonus_points, portals, start, goal, wayoutBFS, f'output{slash}level_1{slash}input{mapID}{slash}bfs{slash}bfs')
        with open(f'output{slash}level_1{slash}input{mapID}{slash}bfs{slash}bfs.txt', 'w') as file:
            file.write(f'Cost: {len(wayoutBFS) - 1}')
            file.close()

        wayoutUCS, openUCS = ap.UCS(graph, start, goal)
        ap.visualize_maze(matrix, bonus_points, portals, start, goal, wayoutUCS, f'output{slash}level_1{slash}input{mapID}{slash}ucs{slash}ucs')
        with open(f'output{slash}level_1{slash}input{mapID}{slash}ucs{slash}ucs.txt', 'w') as file:
            file.write(f'Cost: {len(wayoutUCS) - 1}')
            file.close()

        wayoutGBFS_1, openGBFS_1 = ap.GBFS_Heur1(graph, start, goal)
        ap.visualize_maze(matrix, bonus_points, portals, start, goal, wayoutGBFS_1, f'output{slash}level_1{slash}input{mapID}{slash}gbfs_heuristic_1{slash}gbfs_heuristic_1')
        with open(f'output{slash}level_1{slash}input{mapID}{slash}gbfs_heuristic_1{slash}gbfs_heuristic_1.txt', 'w') as file:
            file.write(f'Cost: {len(wayoutGBFS_1) - 1}')
            file.close()

        wayoutGBFS_2, openGBFS_2 = ap.GBFS_Heur2(graph, start, goal)
        ap.visualize_maze(matrix, bonus_points, portals, start, goal, wayoutGBFS_2, f'output{slash}level_1{slash}input{mapID}{slash}gbfs_heuristic_2{slash}gbfs_heuristic_2')
        with open(f'output{slash}level_1{slash}input{mapID}{slash}gbfs_heuristic_2{slash}gbfs_heuristic_2.txt', 'w') as file:
            file.write(f'Cost: {len(wayoutGBFS_2) - 1}')
            file.close()

        wayoutAstar_1, openAstar_1 = ap.a_star1(graph, start, goal)
        ap.visualize_maze(matrix, bonus_points, portals, start, goal, wayoutAstar_1, f'output{slash}level_1{slash}input{mapID}{slash}astar_heuristic_1{slash}astar_heuristic_1')
        with open(f'output{slash}level_1{slash}input{mapID}{slash}astar_heuristic_1{slash}astar_heuristic_1.txt', 'w') as file:
            file.write(f'Cost: {len(wayoutAstar_1) - 1}')
            file.close()

        wayoutAstar_2, openAstar_2 = ap.a_star2(graph, start, goal)
        ap.visualize_maze(matrix, bonus_points, portals, start, goal, wayoutAstar_2, f'output{slash}level_1{slash}input{mapID}{slash}astar_heuristic_2{slash}astar_heuristic_2')
        with open(f'output{slash}level_1{slash}input{mapID}{slash}astar_heuristic_2{slash}astar_heuristic_2.txt', 'w') as file:
            file.write(f'Cost: {len(wayoutAstar_2) - 1}') 
            file.close()
   
        mapID += 1

def level_1_pg(maps):
    algos = {'dfs', 'bfs', 'ucs', 'gbfs_heuristic_1', 'gbfs_heuristic_2', 'astar_heuristic_1', 'astar_heuristic_2'}
    script_path = os.path.realpath(__file__)
    dir_path = os.path.dirname(script_path)
    dir_path = os.path.dirname(dir_path)
    new_abs_path = os.path.join(dir_path, 'output')
    if not os.path.exists(new_abs_path):
        os.mkdir(new_abs_path)
    new_abs_path = os.path.join(new_abs_path, 'level_1')
    if not os.path.exists(new_abs_path):
        os.mkdir(new_abs_path)
    for i in range(len(maps)):
        p = f'input{i + 1}'
        new_next_abs_path = os.path.join(new_abs_path, p)
        if not os.path.exists(new_next_abs_path):
            os.mkdir(new_next_abs_path)
        for algo in algos:
            p = f'{algo}'
            alg_abs_path = os.path.join(new_next_abs_path, p)
            if not os.path.exists(alg_abs_path):
                os.mkdir(alg_abs_path)
    mapID = 1
    for obj in maps:
        bonus_points, portals, matrix = obj
        for i in range(len(matrix)):
            for j in range(len(matrix[0])):
                if matrix[i][j] == 'S':
                    start = (i,j)
                elif matrix[i][j] == ' ':
                    if (i == 0) or (i == len(matrix) - 1) or (j == 0) or (j == len(matrix[0]) - 1):
                        goal = (i,j)
                else:
                    continue
        graph = []
        for i in range(len(matrix)):
            adj = []
            for j in range(len(matrix[0])):
                if matrix[i][j] != 'x':
                    adj.append(0)
                else:
                    adj.append(1)
            graph.append(adj)
        
        wayoutDFS, openDFS = ap.dfs(graph, start, goal)
        path =f'{slash}output{slash}level_1{slash}input{mapID}{slash}dfs'
        ap.PGAME(graph, start, goal, wayoutDFS,bonus_points, portals, openDFS,dir_path,path,'dfs')
        
        wayoutBFS, openBFS = ap.bfs(graph, start, goal)
        path =f'{slash}output{slash}level_1{slash}input{mapID}{slash}bfs'
        ap.PGAME(graph, start, goal, wayoutBFS,bonus_points, portals, openBFS,dir_path,path,'bfs')

        wayoutUCS, openUCS = ap.UCS(graph, start, goal)
        path =f'{slash}output{slash}level_1{slash}input{mapID}{slash}UCS'
        ap.PGAME(graph, start, goal, wayoutUCS,bonus_points, portals, openUCS,dir_path,path, 'ucs')

        wayoutGBFS_1, openGBFS_1 = ap.GBFS_Heur1(graph, start, goal)
        path =f'{slash}output{slash}level_1{slash}input{mapID}{slash}gbfs_heuristic_1'
        ap.PGAME(graph, start, goal, wayoutGBFS_1,bonus_points, portals, openGBFS_1,dir_path,path,'gbfs_heuristic_1')

        wayoutGBFS_2, openGBFS_2 = ap.GBFS_Heur2(graph, start, goal)
        path =f'{slash}output{slash}level_1{slash}input{mapID}{slash}gbfs_heuristic_2'
        ap.PGAME(graph, start, goal, wayoutGBFS_2,bonus_points, portals, openGBFS_2,dir_path,path,'gbfs_heuristic_2')

        wayoutAstar_1, openAstar_1 = ap.a_star1(graph, start, goal)
        path =f'{slash}output{slash}level_1{slash}input{mapID}{slash}astar_heuristic_1'
        ap.PGAME(graph, start, goal, wayoutAstar_1,bonus_points, portals, openAstar_1,dir_path,path,'astar_heuristic_1')


        wayoutAstar_2, openAstar_2 = ap.a_star2(graph, start, goal)
        path =f'{slash}output{slash}level_1{slash}input{mapID}{slash}astar_heuristic_2'
        ap.PGAME(graph, start, goal, wayoutAstar_2,bonus_points, portals, openAstar_2,dir_path,path,'astar_heuristic_2')
   
        mapID += 1

def level_2(maps):
    script_path = os.path.realpath(__file__)
    dir_path = os.path.dirname(script_path)
    dir_path = os.path.dirname(dir_path)
    new_abs_path = os.path.join(dir_path, 'output')
    if not os.path.exists(new_abs_path):
        os.mkdir(new_abs_path)
    new_abs_path = os.path.join(new_abs_path, 'level_2')
    if not os.path.exists(new_abs_path):
        os.mkdir(new_abs_path)
    for i in range(len(maps)):
        p = f'input{i + 1}'
        new_next_abs_path = os.path.join(new_abs_path, p)
        if not os.path.exists(new_next_abs_path):
            os.mkdir(new_next_abs_path)
    mapID = 1
    for obj in maps:
        bonus_points, portals, matrix = obj
        for i in range(len(matrix)):
            for j in range(len(matrix[0])):
                if matrix[i][j] == 'S':
                    start = (i,j)
                elif matrix[i][j] == ' ':
                    if (i == 0) or (i == len(matrix) - 1) or (j == 0) or (j == len(matrix[0]) - 1):
                        goal = (i,j)
                else:
                    continue 
        graph = []
        for i in range(len(matrix)):
            adj = []
            for j in range(len(matrix[0])):
                if matrix[i][j] != 'x':
                    adj.append(0)
                else:
                    adj.append(1)
            graph.append(adj)
        wayout, openSet = ap.bonus_astar(matrix, bonus_points, start, goal)
        ap.visualize_maze(matrix, bonus_points, portals, start, goal, wayout, f'output{slash}level_2{slash}input{mapID}{slash}output{mapID}')
        with open(f'output{slash}level_2{slash}input{mapID}{slash}output{mapID}.txt', 'w') as file:
            file.write(f'Cost: {ap.compute_cost(wayout, bonus_points)}')
            file.close()
        mapID += 1
    pass

def level_2_pg(maps):
    script_path = os.path.realpath(__file__)
    dir_path = os.path.dirname(script_path)
    dir_path = os.path.dirname(dir_path)
    new_abs_path = os.path.join(dir_path, 'output')
    if not os.path.exists(new_abs_path):
        os.mkdir(new_abs_path)
    new_abs_path = os.path.join(new_abs_path, 'level_2')
    if not os.path.exists(new_abs_path):
        os.mkdir(new_abs_path)
    for i in range(len(maps)):
        p = f'input{i + 1}'
        new_next_abs_path = os.path.join(new_abs_path, p)
        if not os.path.exists(new_next_abs_path):
            os.mkdir(new_next_abs_path)
    mapID = 1
    for obj in maps:
        bonus_points, portals, matrix = obj
        for i in range(len(matrix)):
            for j in range(len(matrix[0])):
                if matrix[i][j] == 'S':
                    start = (i,j)
                elif matrix[i][j] == ' ':
                    if (i == 0) or (i == len(matrix) - 1) or (j == 0) or (j == len(matrix[0]) - 1):
                        goal = (i,j)
                else:
                    continue 
        graph = []
        for i in range(len(matrix)):
            adj = []
            for j in range(len(matrix[0])):
                if matrix[i][j] != 'x':
                    adj.append(0)
                else:
                    adj.append(1)
            graph.append(adj)
        wayout, openSet = ap.bonus_astar(matrix, bonus_points, start, goal)
        path =f'{slash}output{slash}level_2{slash}input{mapID}'
        ap.PGAME(graph, start, goal, wayout,bonus_points, portals, openSet,dir_path,path,f'output{mapID}')
        mapID += 1
    pass

def advance(maps):
    script_path = os.path.realpath(__file__)
    dir_path = os.path.dirname(script_path)
    dir_path = os.path.dirname(dir_path)
    new_abs_path = os.path.join(dir_path, 'output')
    if not os.path.exists(new_abs_path):
        os.mkdir(new_abs_path)
    new_abs_path = os.path.join(new_abs_path, 'advance')
    if not os.path.exists(new_abs_path):
        os.mkdir(new_abs_path)
    for i in range(len(maps)):
        p = f'input{i + 1}'
        new_next_abs_path = os.path.join(new_abs_path, p)
        if not os.path.exists(new_next_abs_path):
            os.mkdir(new_next_abs_path)
    mapID = 1
    
    for obj in maps:
        bonus_points, portals, matrix = obj
        for i in range(len(matrix)):
            for j in range(len(matrix[0])):
                if matrix[i][j] == 'S':
                    start = (i,j)
                elif matrix[i][j] == ' ':
                    if (i == 0) or (i == len(matrix) - 1) or (j == 0) or (j == len(matrix[0]) - 1):
                        goal = (i,j)
                else:
                    continue
        graph = []
        for i in range(len(matrix)):
            adj = []
            for j in range(len(matrix[0])):
                if matrix[i][j] != 'x':
                    adj.append(0)
                else:
                    adj.append(1)
            graph.append(adj)     
        wayout, openSet = ap.BFS_teleport(matrix, start, goal)
        ap.visualize_maze(matrix, bonus_points, portals, start, goal, wayout, f'output{slash}advance{slash}input{mapID}{slash}output{mapID}')
        with open(f'output{slash}advance{slash}input{mapID}{slash}output{mapID}.txt', 'w') as file:
            file.write(f'Cost: {len(wayout) - 1}')
            file.close()
        mapID += 1
    pass

def advance_pg(maps):
    script_path = os.path.realpath(__file__)
    dir_path = os.path.dirname(script_path)
    dir_path = os.path.dirname(dir_path)
    new_abs_path = os.path.join(dir_path, 'output')
    if not os.path.exists(new_abs_path):
        os.mkdir(new_abs_path)
    new_abs_path = os.path.join(new_abs_path, 'advance')
    if not os.path.exists(new_abs_path):
        os.mkdir(new_abs_path)
    for i in range(len(maps)):
        p = f'input{i + 1}'
        new_next_abs_path = os.path.join(new_abs_path, p)
        if not os.path.exists(new_next_abs_path):
            os.mkdir(new_next_abs_path)
    mapID = 1
    
    for obj in maps:
        bonus_points, portals, matrix = obj
        for i in range(len(matrix)):
            for j in range(len(matrix[0])):
                if matrix[i][j] == 'S':
                    start = (i,j)
                elif matrix[i][j] == ' ':
                    if (i == 0) or (i == len(matrix) - 1) or (j == 0) or (j == len(matrix[0]) - 1):
                        goal = (i,j)
                else:
                    continue
        graph = []
        for i in range(len(matrix)):
            adj = []
            for j in range(len(matrix[0])):
                if matrix[i][j] != 'x':
                    adj.append(0)
                else:
                    adj.append(1)
            graph.append(adj)     
        wayout, openSet = ap.BFS_teleport(matrix, start, goal)
        path =f'{slash}output{slash}advance{slash}input{mapID}{slash}'
        ap.PGAME(graph, start, goal, wayout,bonus_points, portals, openSet,dir_path,path,f'output{mapID}')
        mapID += 1
    pass

level_1(getInput(f'input{slash}level_1'))
level_2(getInput(f'input{slash}level_2'))
advance(getInput(f'input{slash}advance'))
level_1_pg(getInput(f'input{slash}level_1'))
level_2_pg(getInput(f'input{slash}level_2'))
advance_pg(getInput(f'input{slash}advance'))


