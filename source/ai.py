import os
from re import S
import matplotlib.pyplot as plt
import copy
from heapq import heappop
from heapq import heappush
from queue import PriorityQueue
import math
import pygame
import teleport_waypoint_version as tlp
row = [-1, 0, 1, 0]
col = [0, 1, 0, -1]
weight = 500
height = 500
oo = 100000000

def Heuristic(current_node, goal):
    x1,y1 = current_node
    x2,y2 = goal
    return abs(x1-x2)+abs(y1-y2)

def Is_Middle(graph, a,b,c):
    if a[0]==0 or a[0]==len(graph):
        if abs(a[0]-b[0]) < abs(a[0]-c[0]):
            return 1
    if a[1]==0 or a[1] == len(graph[0]):
        if abs(a[1]-b[1]) < abs(a[1]-c[1]):
            return 1
    return 0

def Heuristic_Bonus(bonus_points, u, end):
    Min= oo
    tmp1 = Heuristic_Bonus(u,end)
    for i in bonus_points:
        tmp = Heuristic_Bonus(u,(i[0],i[1]))
        if tmp < Min:
            Min = tmp
    return Min + tmp1
        
def Get_Bonus(new_node, bonus_points):
    for i in bonus_points:
        if (i[0],i[1]) == new_node:
            return i[2]

def Is_Valid_Position(graph, node): 
    if node[0] >= 0:
        if  node[0] < len(graph):
            if node[1] >=0:
                if node[1] <len(graph[0]):
                    return graph[node[0]][node[1]] == ' '
    return False

def a_star(graph, start, end, bonus_points):
    distances = {(start[0], start[1]): 0}
    trace = {start:None}
    visited = set()
    open=[]


    pq = PriorityQueue()
    pq.put((0, (start[0], start[1])))

    while not pq.empty():
        _, node = pq.get()
        cur_row, cur_col = node
        if node == (end[0],end[1]):
            break
        
        for i in range(0,4):
            new_node = (cur_row + row[i], cur_col + col[i])
            
            if Is_Valid_Position(graph, new_node) and new_node not in visited:
                old_distance = distances.get(new_node, float('inf'))
                if (graph[new_node[0]][new_node[1]]=='+'):
                    new_distance = distances[node] + weight + Get_Bonus(new_node, bonus_points)
                else:
                    new_distance = distances[node] + weight
                open.append(new_node)
                
                if new_distance < old_distance:
                    distances[new_node] = new_distance
                    priority = new_distance + Heuristic_Bonus(bonus_points, new_node, end)
                    pq.put((priority, new_node))
                    trace[new_node] = node
        
        visited.add(node)

    path = []
    f = end
    while trace[f] != None:
        path.append(trace[f])
        f = trace[f]
        
    path.reverse()
    path.append(end)
    return path, open
