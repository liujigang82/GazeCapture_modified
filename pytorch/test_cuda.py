import json
import math, shutil, os, time
import torch.nn as nn
#a = json("D:/GazeCaputreData/00005/appleFace.json")
import torch

'''
with open("D:/GazeCaputreData/00005/appleFace.json",'r') as load_f:
     load_dict = json.load(load_f)
     print(load_dict)


CHECKPOINTS_PATH = '.'

def load_checkpoint(filename='checkpoint.pth.tar'):
    filename = os.path.join(CHECKPOINTS_PATH, filename)
    print(filename)
    if not os.path.isfile(filename):
        return None
    state = torch.load(filename)
    return state


test = torch.cuda.is_available()
print(test)
print(torch.cuda.current_device())
'''

m = nn.Linear(20, 30)
input = torch.randn(128, 20)
output = m(input)
print(output.size())

#state = load_checkpoint()
#print(state)




    for epoch in range(num_epochs):
        best_accuracy = 0
        accuracy = 0
        for n, train_group in enumerate(train_groups):
            data_loader_train = torch.utils.data.DataLoader(dataset=train_group, batch_size=batch_size,
                                                            shuffle=True)
            data_loader_validate = torch.utils.data.DataLoader(dataset=validate_groups[n],
                                                               batch_size=batch_size, shuffle=False)
            train()
            accuracy = accuracy + validation()

        if best_accuracy < accuracy/len(train_group):
            saveModel()
            best_accuracy = accuracy/len(train_group)

