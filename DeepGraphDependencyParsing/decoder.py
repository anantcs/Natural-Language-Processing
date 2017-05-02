import numpy as np
import sys
from collections import defaultdict, namedtuple
from operator import itemgetter

def backtrack(bp, i, j, d, c, h):
    if i == j:
        return
    k = bp[i, j, d, c]
    if c == 1:
        if d == 1:
            backtrack(bp, i, k, 1, 0, h)
            backtrack(bp, k, j, 1, 1, h)
        if d == 0:
            backtrack(bp, i, k, 0, 1, h)
            backtrack(bp, k, j, 0, 0, h)
    if c == 0:
        if d == 1:
            h[j] = i
        if d == 0:
            h[i] = j
        backtrack(bp, i, k, 1, 1, h)
        backtrack(bp, k+1, j, 0, 1, h)

def parse_proj(scores, gold=None):
    '''
    Parse using Eisner's algorithm.

    scores - an (n+1) x (n+1) matrix
    gold - the gold arcs
    '''

    #YOUR IMPLEMENTATION GOES HERE
    # 0 for left, 1 for right
    n = len(scores) - 1
    direction = 2
    complete = 2
    dp = np.zeros((n+1, n+1, direction, complete))
    bp = np.ones((n+1, n+1, direction, complete), dtype=int) * -1
    #Initializing the dp matrix
    for i in range(n+1):
        for d in range(direction):
            for c in range(complete):
                dp[i][i][d][c] = 0.0

    #Updating the values in the dp matrix
    for m in range(1, n+1):
        for i in range(0, n-m+1):
            j = i + m
            
            temp = []
            for q in range(i, j):
                temp.append(dp[i,q,1,1] + dp[q+1,j,0,1] + scores[j,i])
            dp[i,j,0,0] = np.max(temp)
            bp[i,j,0,0] = np.argmax(temp) + i
            if not (gold is not None and gold[i] == j):
                dp[i,j,0,0] += 1
            
            temp = []
            for q in range(i, j):
                temp.append(dp[i,q,1,1] + dp[q+1,j,0,1] + scores[i,j])
            dp[i,j,1,0] = np.max(temp)
            bp[i,j,1,0] = np.argmax(temp) + i
            if not (gold is not None and gold[j] == i):
                dp[i,j,1,0] += 1

            temp = []
            for q in range(i,j):
                temp.append(dp[i,q,0,1] + dp[q,j,0,0])
            dp[i,j,0,1] = np.max(temp)
            bp[i,j,0,1] = np.argmax(temp) + i

            temp = []
            for q in range(i+1,j+1):
                temp.append(dp[i,q,1,0] + dp[q,j,1,1])
            dp[i,j,1,1] = np.max(temp)
            bp[i,j,1,1] = np.argmax(temp) + i + 1


    h = [-1]*(n+1)
    backtrack(bp, 0, n, 1, 1, h)
    return h
