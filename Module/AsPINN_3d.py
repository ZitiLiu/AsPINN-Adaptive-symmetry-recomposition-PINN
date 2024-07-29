import torch
import torch.nn as nn
  
class Net(nn.Module):
    def __init__(self, node_num:int):
        super(Net, self).__init__()
        self.node_num = node_num
        self.fc1_1 = nn.Linear(1, self.node_num)
        self.fc1_2 = nn.Linear(1, self.node_num)
        self.fc1_3 = nn.Linear(1, self.node_num, bias=False)    
        self.fc1_4 = nn.Linear(1, self.node_num, bias=False)
        self.fc1_5 = nn.Linear(1, self.node_num, bias=False)
        self.fc1_6 = nn.Linear(1, self.node_num, bias=False)
        self.fc2_1 = nn.Linear(self.node_num, self.node_num)
        self.fc2_2 = nn.Linear(self.node_num, self.node_num)
        self.fc2_3 = nn.Linear(self.node_num, self.node_num)    
        self.fc2_4 = nn.Linear(self.node_num, self.node_num)
        self.fc3_1 = nn.Linear(self.node_num, 2*self.node_num)
        self.fc3_2 = nn.Linear(self.node_num, 2*self.node_num)
        self.fc4_1 = nn.Linear(4*self.node_num, 1)
        self.iter = 0
        self.iter_list = []
        self.loss_list = []
        self.loss_f_list = []
        self.loss_b_list = []
        self.loss_d_list = []
        self.para_ud_list = []
    def forward(self, x, y, z):
        u1_1 = torch.tanh(self.fc1_1(x) + self.fc1_3(y) + self.fc1_5(z))
        u1_2 = torch.tanh(self.fc1_1(x) - self.fc1_3(y) + self.fc1_5(z))
        u1_3 = torch.tanh(self.fc1_2(x) + self.fc1_4(y) + self.fc1_6(z))
        u1_4 = torch.tanh(self.fc1_2(x) - self.fc1_4(y) + self.fc1_6(z))
        u2_1 = torch.tanh(self.fc2_1(u1_1) + self.fc2_3(u1_2))
        u2_2 = torch.tanh(self.fc2_3(u1_1) + self.fc2_1(u1_2))
        u2_3 = torch.tanh(self.fc2_2(u1_3) + self.fc2_4(u1_4))
        u2_4 = torch.tanh(self.fc2_4(u1_3) + self.fc2_2(u1_4))
        u3_1 = torch.tanh(self.fc3_1(u2_1) - self.fc3_1(u2_2))
        u3_2 = torch.tanh(self.fc3_2(u2_3) + self.fc3_2(u2_4))
        u = self.fc4_1(torch.cat([u3_1,u3_2],dim=1)) 
       
        return u