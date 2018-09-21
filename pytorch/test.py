import torch
import torch.nn as nn

import numpy as np
import cv2

'''
loss = nn.MSELoss()

input = torch.tensor([[1.0, 1.0], [2, 2],[3,3]], dtype= torch.float32)
target = torch.tensor([[0, 0], [0, 0],[0,0]], dtype= torch.float32)

#input = torch.randn(5, 2, requires_grad=True)
#target = torch.randn(5, 2)
print(input)
print(target)
output = loss(input, target)
print(output)

diff = (input-target)**2
print(diff)



gridSize1 =(50, 50)
gridSize2 = (25, 25)
def makeEyeGrid( gridSize, params):
    gridLen = gridSize[0] * gridSize[1]
    grid = np.zeros([gridLen, ], np.float32)

    indsY = np.array([i // gridSize[0] for i in range(gridLen)])
    indsX = np.array([i % gridSize[0] for i in range(gridLen)])
    condX = np.logical_and(indsX >= params[0], indsX < params[0] + params[2])
    condY = np.logical_and(indsY >= params[1], indsY < params[1] + params[3])
    cond = np.logical_and(condX, condY)
    grid[cond] = params[4]
    return grid

leftEyeGrid = makeEyeGrid(gridSize1, [22.0000,   25.0000,   11.0000,    8.0000,  1])
rightEyeGrid = makeEyeGrid(gridSize1, [8.0000,    26.0000,   11.0000,    8.0000, 1])
faceGrid = makeEyeGrid(gridSize1, [ 5 ,  19 ,   36 ,   27, 1])
eyeGrid = leftEyeGrid+ rightEyeGrid

leftEyeGrid = leftEyeGrid.reshape(50,50)
rightEyeGrid = rightEyeGrid.reshape(50,50)

eyeGrid = eyeGrid.reshape(50,50)
faceGrid = faceGrid.reshape(50,50)

cv2.imshow("1", leftEyeGrid)
cv2.imshow("2", rightEyeGrid)
cv2.imshow("3", eyeGrid)
cv2.imshow("4",faceGrid)
cv2.waitKey(0)
'''

input = torch.tensor([[1.0, 1.0], [2, 2],[3,3]], dtype= torch.float32)
target = torch.tensor([[0, 0], [0, 0],[0,0]], dtype= torch.float32)

lossin = input - target
lossin = torch.mul(lossin,lossin)
lossin = torch.sum(lossin, 1)
lossin = torch.mean(torch.sqrt(lossin))





