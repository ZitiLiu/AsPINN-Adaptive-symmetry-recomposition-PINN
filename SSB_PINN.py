import torch
import torch.nn as nn

class Net(nn.Module):
    def __init__(self, node_num:int):
        super(Net, self).__init__()

        self.node_num = 2*node_num
        self.fc1_2 = nn.Linear(1, self.node_num)   
        self.fc1_4 = nn.Linear(1, self.node_num, bias=False)
        self.fc2_2 = nn.Linear(self.node_num, self.node_num)   
        self.fc2_4 = nn.Linear(self.node_num, self.node_num)  
        self.fc3_2 = nn.Linear(self.node_num, 2*self.node_num)
        self.fc4_1 = nn.Linear(2*self.node_num, 1)
 
        self.iter = 0
        self.iter_list = []
        self.loss_list = []
        self.loss_f_list = []
        self.loss_b_list = []
        self.loss_d_list = []
        self.para_ud_list = []

    def forward(self, x, y):
        u1_3 = torch.tanh(self.fc1_2(x) + self.fc1_4(y))
        u1_4 = torch.tanh(self.fc1_2(x) - self.fc1_4(y))
        u2_3 = torch.tanh(self.fc2_2(u1_3) + self.fc2_4(u1_4))
        u2_4 = torch.tanh(self.fc2_4(u1_3) + self.fc2_2(u1_4))
        u3_2 = torch.tanh(self.fc3_2(u2_3) + self.fc3_2(u2_4))
        u = self.fc4_1(u3_2) 
       
        return u