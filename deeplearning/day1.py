import torch
import torch.nn as nn
import torch.optim as optim

#miles
distances = torch.tensor([[1.0],[2.0],[3.0],[4.0]],dtype=torch.float32)

#minutes
times = torch.tensor([[6.96],[12.11],[16.77],[22.21]],dtype=torch.float32)


model = nn.Sequential(nn.Linear(1,1)) #1 input 1 output linear model

loss_function = nn.MSELoss()
optimizer = optim.SGD(model.parameters(),lr=0.01)

for epoch in range(500):
    optimizer.zero_grad() #reset the optimizer
    outputs = model(inputs)
    loss = loss_function(outputs,times)
    loss.backward()
    optimizer.step()

