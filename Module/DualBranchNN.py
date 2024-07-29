import torch
import torch.nn as nn
  
class Net(nn.Module):
    def __init__(self, node_num:int):
        super(Net, self).__init__()

        self.node_num = node_num
        self.fc1_1 = nn.Linear(2, self.node_num)
        self.fc1_2 = nn.Linear(2, self.node_num)
        self.fc2_1 = nn.Linear(self.node_num, 2*self.node_num)
        self.fc2_2 = nn.Linear(self.node_num, 2*self.node_num)
        self.fc3_1 = nn.Linear(2*self.node_num, 2*self.node_num)
        self.fc3_2 = nn.Linear(2*self.node_num, 2*self.node_num)
        self.fc4_1 = nn.Linear(4*node_num, 1)

        self.iter = 0
        self.iter_list = []
        self.loss_list = []
        self.loss_f_list = []
        self.loss_b_list = []
        self.loss_d_list = []
        self.para_ud_list = []

    def forward(self, x, y):
        xy = torch.cat([x,y],dim=1)
        u1 = torch.tanh(self.fc1_1(xy))
        u2 = torch.tanh(self.fc1_2(xy))

        u1 = torch.tanh(self.fc2_1(u1))
        u2 = torch.tanh(self.fc2_2(u2))

        u1 = torch.tanh(self.fc3_1(u1))
        u2 = torch.tanh(self.fc3_2(u2))

        u = self.fc4_1(torch.cat([u1,u2],dim=1))
       
        return u