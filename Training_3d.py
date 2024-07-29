# coding = utf-8
import numpy as np
import torch
import torch.optim as optim
import pandas as pd
import os
import time
import Module.SingleVis as SingleVis
import Module.GroupVis as GroupVis
import Module.AsPINN_3d as AsPINN_3d
import Module.PINN_3d as PINN_3d
import Module.DualBranchNN_3d as DualBranchNN_3d
torch.manual_seed(1234)  
if torch.cuda.is_available():
    device = torch.device('cuda')
    print("GPU is available")
else:
    device = torch.device('cpu')

class model():
    def __init__(self, file_name, ini_num) :
        self.ini_num = ini_num
        self.ini_file_path = './Config/'+ file_name +'_' + str(ini_num)+'.csv'
        self.model_ini_keys = pd.read_csv(self.ini_file_path,header=None).values[:, 0]
        self.model_ini_values = pd.read_csv(self.ini_file_path,dtype={'column_name': object}, header=None)
        string_data = self.model_ini_values.iloc[0, 1]
        float_data = self.model_ini_values.iloc[1:7, 1].astype(float).tolist()
        int_data = self.model_ini_values.iloc[7:17, 1].astype(int).tolist()
        moni_data = self.model_ini_values.iloc[17:, 1].tolist()
        # para_data = self.model_ini_values.iloc[14:, 1].tolist()

        self.model_ini_values = [string_data] + float_data + int_data + moni_data
        self.model_ini_dict = dict(zip(self.model_ini_keys, self.model_ini_values))
        self.node_num = self.model_ini_dict['node_num']
        self.input_num = self.model_ini_dict['input_num']
        self.output_num = self.model_ini_dict['output_num']
        self.layer = [self.input_num] + [2*self.node_num] + [2*self.node_num] + [4*self.node_num] + [self.output_num]
        
        self.learning_rate=1e-4
        self.ques_name = self.model_ini_dict['ques_name']

        self.x_min = self.model_ini_dict['x_min']
        self.x_max = self.model_ini_dict['x_max']
        self.y_min = self.model_ini_dict['y_min']
        self.y_max = self.model_ini_dict['y_max']
        self.z_min = self.model_ini_dict['z_min']
        self.z_max = self.model_ini_dict['z_max']

        self.model_ini_dict['para_ctrl'] = list(map(float, self.model_ini_dict['para_ctrl'].split(',')))
        self.para_ctrl = self.model_ini_dict['para_ctrl']
        self.para_ud_num = self.model_ini_dict['para_ud_num']
        self.model_ini_dict['data_num'] = list(map(str, self.model_ini_dict['data_num'].split(','))) 
        self.data_num = self.model_ini_dict['data_num']
        self.grid_node_num = self.model_ini_dict['grid_node_num']   
        self.monitor_state = bool(self.model_ini_dict['monitor_state']) 
        self.step_num = self.model_ini_dict['step_num'] 
        self.bun_node_num = self.model_ini_dict['bun_node_num']
        self.figure_node_num = self.model_ini_dict['figure_node_num']      

    def net_b(self):    
        loss_b = 0
        if self.input_num == 3:
            y_l = np.linspace(self.y_min, self.y_max, self.bun_node_num)
            z_l = np.linspace(self.z_min, self.z_max, self.bun_node_num)
            y_l, z_l = np.meshgrid(y_l, z_l)
            y_l = torch.tensor(y_l, requires_grad=True).float().to(device).reshape([-1,1])
            z_l = torch.tensor(z_l, requires_grad=True).float().to(device).reshape([-1,1])
            x_l = -torch.ones_like(y_l, requires_grad=True).float().to(device).reshape([-1,1])
            u_l = self.net(x_l, y_l, z_l)
            y_l_1 = np.linspace(self.y_min, self.y_max, self.bun_node_num)
            z_l_1 = np.linspace(self.z_min, self.z_max, self.bun_node_num)
            y_l_1 = torch.tensor(y_l_1, requires_grad=True).float().to(device).reshape([-1,1])
            z_l_1 = torch.tensor(z_l_1, requires_grad=True).float().to(device).reshape([-1,1])
            x_l_1 = -torch.ones_like(y_l_1, requires_grad=True).float().to(device).reshape([-1,1])
            u_l_1 = self.net(x_l_1, y_l_1, z_l_1)
            u_l_2 = self.net(x_l_1, -y_l_1, z_l_1)
            y_c = torch.tensor([self.y_min, (self.y_min + self.y_max)/2, (self.y_min + self.y_max)/2, self.y_max], requires_grad=True).float().to(device).reshape([-1,1])
            z_c = torch.tensor([(self.z_min + self.z_max)/2, self.z_min, self.z_max, (self.z_min + self.z_max)/2], requires_grad=True).float().to(device).reshape([-1,1])
            x_c = -torch.ones_like(y_c, requires_grad=True).float().to(device).reshape([-1,1])
            u_c = self.net(x_c, y_c, z_c)
            y_r = np.linspace(self.y_min, self.y_max, self.bun_node_num)
            z_r = np.linspace(self.z_min, self.z_max, self.bun_node_num)
            y_r, z_r = np.meshgrid(y_r, z_r)
            y_r = torch.tensor(y_r, requires_grad=True).float().to(device).reshape([-1,1])
            z_r = torch.tensor(z_r, requires_grad=True).float().to(device).reshape([-1,1])
            x_r = torch.ones_like(y_r, requires_grad=True).float().to(device).reshape([-1,1])
            u_r = self.net(x_r, y_r, z_r)
            y_r_1 = np.linspace(self.y_min, self.y_max, self.bun_node_num)
            z_r_1 = np.linspace(self.z_min, self.z_max, self.bun_node_num)
            y_r_1 = torch.tensor(y_r_1, requires_grad=True).float().to(device).reshape([-1,1])
            z_r_1 = torch.tensor(z_r_1, requires_grad=True).float().to(device).reshape([-1,1])
            x_r_1 = torch.ones_like(y_r_1, requires_grad=True).float().to(device).reshape([-1,1])
            zero_r = torch.zeros_like(y_r_1, requires_grad=True).float().to(device).reshape([-1,1])
            u_r_1 = self.net(x_r_1, y_r_1, zero_r)
            u_r_2 = self.net(x_r_1, zero_r, z_r_1)
            x_f = np.linspace(self.x_min, self.x_max, self.bun_node_num)
            z_f = np.linspace(self.z_min, self.z_max, self.bun_node_num)
            x_f, z_f = np.meshgrid(x_f, z_f)
            x_f = torch.tensor(x_f, requires_grad=True).float().to(device).reshape([-1,1])
            z_f = torch.tensor(z_f, requires_grad=True).float().to(device).reshape([-1,1])
            y_f = -torch.ones_like(x_f, requires_grad=True).float().to(device).reshape([-1,1])
            u_f = self.net(x_f, y_f, z_f)
            x_b = np.linspace(self.x_min, self.x_max, self.bun_node_num)
            z_b = np.linspace(self.z_min, self.z_max, self.bun_node_num)
            x_b, z_b = np.meshgrid(x_b, z_b)
            x_b = torch.tensor(x_b, requires_grad=True).float().to(device).reshape([-1,1])
            z_b = torch.tensor(z_b, requires_grad=True).float().to(device).reshape([-1,1])
            y_b = torch.ones_like(x_b, requires_grad=True).float().to(device).reshape([-1,1])
            u_b = self.net(x_b, y_b, z_b)
            x_down = np.linspace(self.x_min, self.x_max, self.bun_node_num)
            y_down = np.linspace(self.y_min, self.y_max, self.bun_node_num)
            x_down, y_down = np.meshgrid(x_down, y_down)
            x_down = torch.tensor(x_down, requires_grad=True).float().to(device).reshape([-1,1])
            y_down = torch.tensor(y_down, requires_grad=True).float().to(device).reshape([-1,1])
            z_down = -torch.ones_like(x_down, requires_grad=True).float().to(device).reshape([-1,1])
            u_down = self.net(x_down, y_down, z_down)
            x_up = np.linspace(self.x_min, self.x_max, self.bun_node_num)
            y_up = np.linspace(self.y_min, self.y_max, self.bun_node_num)
            x_up, y_up = np.meshgrid(x_up, y_up)
            x_up = torch.tensor(x_up, requires_grad=True).float().to(device).reshape([-1,1])
            y_up = torch.tensor(y_up, requires_grad=True).float().to(device).reshape([-1,1])
            z_up = torch.ones_like(x_up, requires_grad=True).float().to(device).reshape([-1,1])
            u_up = self.net(x_up, y_up, z_up)

            if 'Thermal_3d' in self.ques_name:
                u_l_1_moni = torch.ones_like(u_l_1)
                loss_b += torch.mean((u_l_1 - u_l_1_moni)**2)
                u_l_2_moni = torch.ones_like(u_l_2)
                loss_b += torch.mean((u_l_2 - u_l_2_moni)**2)
                u_c_moni = torch.zeros_like(u_c)
                loss_b += torch.mean((u_c - u_c_moni)**2)
                u_r_1_moni = torch.zeros_like(u_r_1)
                loss_b += torch.mean((u_r_1 - u_r_1_moni)**2)
                u_r_2_moni = torch.zeros_like(u_r_2)
                loss_b += torch.mean((u_r_2 - u_r_2_moni)**2)
                u_f_y = torch.autograd.grad(u_f, y_f, grad_outputs=torch.ones_like(u_f), retain_graph=True, create_graph=True)[0]
                u_b_y = torch.autograd.grad(u_b, y_b, grad_outputs=torch.ones_like(u_b), retain_graph=True, create_graph=True)[0]
                loss_b += torch.mean((u_f_y)**2)
                loss_b += torch.mean((u_b_y)**2)
                u_down_z = torch.autograd.grad(u_down, z_down, grad_outputs=torch.ones_like(u_down), retain_graph=True, create_graph=True)[0]
                u_up_z = torch.autograd.grad(u_up, z_up, grad_outputs=torch.ones_like(u_up), retain_graph=True, create_graph=True)[0]
                loss_b += torch.mean((u_down_z)**2)
                loss_b += torch.mean((u_up_z)**2)
        else:
            y_b = np.linspace(self.y_min, self.y_max, self.bun_node_num)
            x_b = np.zeros_like(y_b)
            x_b = torch.tensor(x_b, requires_grad=True).float().to(device).reshape([-1,1])
            y_b = torch.tensor(y_b, requires_grad=True).float().to(device).reshape([-1,1])
            u_b = self.net(x_b, y_b)
            x_down= np.linspace(self.x_min, self.x_max, self.bun_node_num)
            y_down = -np.ones_like(x_down)
            x_down = torch.tensor(x_down, requires_grad=True).float().to(device).reshape([-1,1])
            y_down = torch.tensor(y_down, requires_grad=True).float().to(device) .reshape([-1,1])
            u_down = self.net(x_down, y_down)
            x_up = np.linspace(self.x_min, self.x_max, self.bun_node_num)
            y_up = np.ones_like(x_up)
            x_up = torch.tensor(x_up, requires_grad=True).float().to(device).reshape([-1,1])
            y_up = torch.tensor(y_up, requires_grad=True).float().to(device).reshape([-1,1])
            u_up = self.net(x_up, y_up)

            if 'Burgers' in self.ques_name:
                u_b_moni = -torch.sin(torch.pi * y_b) # burgers
                loss_b += torch.mean((u_b - u_b_moni)**2)

                u_down_moni = torch.zeros_like(u_down)  #burgers
                loss_b += torch.mean((u_down - u_down_moni)**2)

                u_up_moni = torch.zeros_like(u_up)  #burgers
                loss_b += torch.mean((u_up - u_up_moni)**2)

            elif 'AC' in self.ques_name:
                u_b_moni = (y_b**2) * torch.cos(torch.pi * y_b) # AC
                loss_b += torch.mean((u_b - u_b_moni)**2)

                u_up = self.net(x_up, y_up)
                loss_b += torch.mean((u_up - u_down)**2)

                u_up_y = torch.autograd.grad(u_up, y_up, grad_outputs=torch.ones_like(u_up), retain_graph=True, create_graph=True)[0] #AC
                u_down_y = torch.autograd.grad(u_down, y_down, grad_outputs=torch.ones_like(u_down), retain_graph=True, create_graph=True)[0]   #AC
                loss_b += torch.mean((u_down_y - u_up_y)**2)

            elif 'Laplace' in self.ques_name:
                u_b_moni = (torch.exp(x_b) + 0.75*torch.exp(-x_b)) * (torch.sin(2 * y_b) + 0.5*torch.cos(2 * y_b))  #laplace
                loss_b += torch.mean((u_b - u_b_moni)**2)
                
                u_down_moni = (torch.exp(x_down) + 0.75*torch.exp(-x_down)) * (torch.sin(2 * y_down) + 0.5*torch.cos(2 * y_down))  #laplace
                loss_b += torch.mean((u_down - u_down_moni)**2)

                u_up_moni = (torch.exp(x_up) + 0.75*torch.exp(-x_up)) * (torch.sin(2 * y_up) + 0.5*torch.cos(2 * y_up))  #laplace
                loss_b += torch.mean((u_up - u_up_moni)**2)
            
        return loss_b
    def mesh_init(self):
        if self.input_num == 3:
            self.x = np.linspace(self.x_min, self.x_max, self.grid_node_num).reshape([-1,1])
            self.y = np.linspace(self.y_min, self.y_max, self.grid_node_num).reshape([-1,1])
            self.z = np.linspace(self.z_min, self.z_max, self.grid_node_num).reshape([-1,1])
            self.x, self.y, self.z = np.meshgrid(self.x, self.y, self.z)
            self.x = torch.tensor(self.x,requires_grad=True).float().to(device).reshape([-1,1])
            self.y = torch.tensor(self.y,requires_grad=True).float().to(device).reshape([-1,1])
            self.z = torch.tensor(self.z,requires_grad=True).float().to(device).reshape([-1,1])
        else:
            self.x = np.linspace(self.x_min, self.x_max, self.grid_node_num).reshape([-1,1])
            self.y = np.linspace(self.y_min, self.y_max, self.grid_node_num).reshape([-1,1])
            self.xy = np.meshgrid(self.x,self.y)
            self.x = self.xy[0]
            self.y = self.xy[1]
            self.x = self.x.reshape([-1, 1])
            self.y = self.y.reshape([-1, 1])
            self.x = torch.tensor(self.x,requires_grad=True).float().to(device)
            self.y = torch.tensor(self.y,requires_grad=True).float().to(device)
        
    def net_d(self,state:bool=False):
        
        current_read = pd.read_csv('./Database/'+self.ques_name + '_data_'+self.data_num[0]+'.csv', header=None).values
        self.database = current_read
        for i in range(1, len(self.data_num)):
            current_read = pd.read_csv('./Database/'+self.ques_name+'_data_'+self.data_num[i]+'.csv', header=None).values
            self.database = np.vstack([self.database,current_read])
        self.x_monitor = self.database[:,0].reshape([-1,1])
        self.y_monitor = self.database[:,1].reshape([-1,1])
        self.u_monitor = self.database[:,2].reshape([-1,1])
        self.x_monitor = torch.tensor(self.x_monitor,requires_grad=True).float().to(device)
        self.y_monitor = torch.tensor(self.y_monitor,requires_grad=True).float().to(device)
        self.u_monitor = torch.tensor(self.u_monitor,requires_grad=True).float().to(device)
        
        return torch.mean((self.net(self.x_monitor,self.y_monitor) - self.u_monitor)**2), state
    def train_adam(self):
        self.para_undetermin = torch.zeros(self.para_ud_num, requires_grad=True).float().to(device)
        self.para_undetermin = torch.nn.Parameter(self.para_undetermin)

        self.optimizer = optim.Adam(list(self.net.parameters()) + [self.para_undetermin], lr=self.learning_rate)  
        
        self.time_start = time.time()

        for iter in range(self.step_num):            
            self.optimizer.zero_grad()     
            if self.input_num == 3:
                u = self.net(self.x,self.y,self.z).cuda()
            else:
                u = self.net(self.x,self.y).cuda()
            
            u_x = torch.autograd.grad(u, self.x, grad_outputs=torch.ones_like(u), retain_graph=True, create_graph=True)[0]    
            u_xx = torch.autograd.grad(u_x, self.x, grad_outputs=torch.ones_like(u_x), retain_graph=True, create_graph=True)[0]
            u_y = torch.autograd.grad(u, self.y, grad_outputs=torch.ones_like(u), retain_graph=True, create_graph=True)[0]
            u_yy = torch.autograd.grad(u_y, self.y, grad_outputs=torch.ones_like(u_y), retain_graph=True, create_graph=True)[0]
            u_yyy = torch.autograd.grad(u_yy, self.y, grad_outputs=torch.ones_like(u_yy), retain_graph=True, create_graph=True)[0]

            if self.input_num == 3:
                u_z = torch.autograd.grad(u, self.z, grad_outputs=torch.ones_like(u), retain_graph=True, create_graph=True)[0]
                u_zz = torch.autograd.grad(u_z, self.z, grad_outputs=torch.ones_like(u_z), retain_graph=True, create_graph=True)[0]
            if self.monitor_state:
                self.loss_d = self.net_d(self.monitor_state)

            self.loss_b = self.net_b()

            if self.ques_name == 'Burgers_inv':
                self.loss_f = torch.mean((u_x + u*u_y - self.para_undetermin[0] * u_yy)**2) 

            elif self.ques_name == 'Burgers':
                self.loss_f = torch.mean((u_x + u*u_y -(0.01/torch.pi)*u_yy)**2) 

            elif self.ques_name == 'Laplace_inv':
                self.loss_f = torch.mean((u_xx + self.para_undetermin[0] * u_yy)**2)  
            
            elif self.ques_name == 'Laplace':
                self.loss_f = torch.mean((u_xx + u_yy)**2)  

            elif self.ques_name == 'AC_inv':
                self.loss_f = torch.mean((u_x - self.para_undetermin[0]*u_yy + self.para_undetermin[1] * u**3 - self.para_undetermin[2]*u)**2)    
            
            elif self.ques_name == 'AC':
                self.loss_f = torch.mean((u_x - 0.0001*u_yy + 5 * u**3 - 5*u)**2)    
            
            elif self.ques_name == 'Thermal_3d':
                self.loss_f = torch.mean((u_xx + u_yy + u_zz)**2)
            
            else:
                raise ValueError("The input is unintegrated or the question name is incorrect. Please check again.")

            if self.monitor_state:
                self.loss = self.loss_d[0] + self.loss_f
            else:
                self.loss = self.loss_f + self.loss_b
            # loss = loss_f

            self.loss.backward(retain_graph=True)      
            self.optimizer.step()   

            self.net.iter += 1
            self.net.iter_list.append(self.net.iter)
            self.net.loss_list.append(self.loss.item())
            self.net.loss_f_list.append(self.loss_f.item())
            self.net.loss_b_list.append(self.loss_b.item())
            if self.monitor_state:
                self.net.para_ud_list.append(self.para_undetermin.tolist())

            if self.monitor_state:
                self.net.loss_d_list.append(self.loss_d[0].item())
                if iter % 100 == 0:
                    print('Iter %d, loss: %.5e, loss_f: %.5e, loss_d: %.5e, loss_b: %.5e' %
                            (iter, self.loss.item(), self.loss_f.item(), self.loss_d[0].item(), self.loss_b.item()))
            else:
                if iter % 100 == 0:
                    print('Iter %d, loss: %.5e, loss_f: %.5e, loss_b: %.5e' %
                            (iter, self.loss.item(), self.loss_f.item(), self.loss_b.item()))
                    # self.model_save(str(iter))

        self.time_end = time.time()
        

        # print(self.para_undetermin[0].item())
        print("\nTime occupied:" + str((self.time_end - self.time_start)/60) + 'min.\n')

    def model_save(self, suffix:str =''):
        self.save_desti = './' + self.ques_name +'_' +str(self.ini_num) + '/'
        if not os.path.exists(self.save_desti):
            os.mkdir(self.save_desti)
        if not os.path.exists(self.save_desti + '/Models/'):       
            os.mkdir(self.save_desti + '/Models/')
        if suffix == '':
            torch.save(self.net.state_dict(), self.save_desti + '/Models/'+ self.ques_name +'_' +str(self.ini_num) + '_' + self.net.__module__.split('.')[-1] + suffix + '.pth')
        else:
            torch.save(self.net.state_dict(), self.save_desti + '/Models/'+ self.ques_name +'_' +str(self.ini_num) + '_' + self.net.__module__.split('.')[-1] +'_step_'+ suffix + '.pth')

        self.control_paras = pd.read_csv(self.ini_file_path)
        self.control_paras.to_csv(self.save_desti + self.ques_name +'_' +str(self.ini_num) + '.csv', index=False)

        self.time_save = pd.DataFrame({self.net.__class__.__module__:[self.time_end - self.time_start]})
        self.time_save.to_csv(self.save_desti + 'Clock time.csv', mode='a', index=False)

        if self.monitor_state:
            loss_data = np.array([self.net.iter_list, self.net.loss_list, self.net.loss_f_list, self.net.loss_d_list, self.net.loss_b_list])
            loss_data = np.transpose(loss_data)
            df_loss_data = pd.DataFrame(loss_data, columns=['iter', 'loss', 'loss_f', 'loss_d', 'loss_b'])
        else:
            loss_data = np.array([self.net.iter_list, self.net.loss_list, self.net.loss_f_list, self.net.loss_b_list])
            loss_data = np.transpose(loss_data)
            df_loss_data = pd.DataFrame(loss_data, columns=['iter', 'loss', 'loss_f', 'loss_b'])
        if not os.path.exists(self.save_desti + '/Loss/'):       
            os.mkdir(self.save_desti + '/Loss/')
        df_loss_data.to_csv(self.save_desti + '/Loss/'+self.ques_name +'_' +str(self.ini_num) + '_loss_'+self.net.__module__.split('.')[-1]+'.csv', index=False)

        if self.para_ud_num:
            iter_list = np.array(self.net.iter_list).reshape([-1,1])
            para_ud = np.array(np.hstack([iter_list, self.net.para_ud_list]))
            # para_ud = np.transpose(para_ud)
            para_ud_columns = ['iter']
            for i in range(self.para_ud_num):
                para_ud_columns.append('parameters_'+str(i+1))
            df_para_ud = pd.DataFrame(para_ud, columns = para_ud_columns)
            if not os.path.exists(self.save_desti + '/Parameters/'):       
                os.mkdir(self.save_desti + '/Parameters/')
            df_para_ud.to_csv(self.save_desti + '/Parameters/'+self.ques_name +'_' +str(self.ini_num) + '_paras_'+self.net.__module__.split('.')[-1]+'.csv', index=False)

    def result_show(self):
        if self.input_num == 3:
            x = np.linspace(self.x_min, self.x_max, self.figure_node_num).reshape([-1,1])
            y = np.linspace(self.y_min, self.y_max, self.figure_node_num).reshape([-1,1])
            z = np.linspace(self.z_min, self.z_max, self.figure_node_num).reshape([-1,1])
            x, y, z = np.meshgrid(x, y, z)
            x = torch.tensor(x).float().to(device).reshape([-1,1])
            y = torch.tensor(y).float().to(device).reshape([-1,1])
            z = torch.tensor(z).float().to(device).reshape([-1,1])

            u = self.net(x, y, z)
            x = x.detach().cpu().numpy()
            y = y.detach().cpu().numpy()
            z = z.detach().cpu().numpy()
            u = u.detach().cpu().numpy()
            u_vis = SingleVis.Vis(self.ques_name, self.ini_num, self.save_desti, self.net.__module__.split('.')[-1], x, y, z, u)
            u_vis.figure_3d()
            loss_vis = SingleVis.Vis(self.ques_name, self.ini_num, self.save_desti, self.net.__module__.split('.')[-1])
            loss_vis.loss_vis()
            if self.para_ud_num:
                para_vis = SingleVis.Vis(self.ques_name, self.ini_num, self.save_desti, self.net.__module__.split('.')[-1])
                para_vis.para_vis()

        else:
            x = np.linspace(self.x_min, self.x_max, self.figure_node_num).reshape([-1,1])
            y = np.linspace(self.y_min, self.y_max, self.figure_node_num).reshape([-1,1])
            x, y = np.meshgrid(x,y)
            x = torch.tensor(x).float().to(device).reshape([-1,1])
            y = torch.tensor(y).float().to(device).reshape([-1,1])

            u = self.net(x, y)
            x = x.detach().cpu().numpy()
            y = y.detach().cpu().numpy()
            u = u.detach().cpu().numpy()
            u_vis = SingleVis.Vis(self.ques_name, self.ini_num, self.save_desti, self.net.__module__.split('.')[-1], x, y,[], u)
            u_vis.figure_2d()
            loss_vis = SingleVis.Vis(self.ques_name, self.ini_num, self.save_desti, self.net.__module__.split('.')[-1])
            loss_vis.loss_vis()
            if self.para_ud_num:
                para_vis = SingleVis.Vis(self.ques_name, self.ini_num, self.save_desti, self.net.__module__.split('.')[-1])
                para_vis.para_vis()

    def workflow(self):
        self.mesh_init()
        self.train_adam()
        self.model_save()
        self.result_show()
    def train(self): 
        if self.model_ini_dict['model'] == 9:
            self.net = AsPINN_3d.Net(self.node_num).float().to(device)
            self.workflow()
        elif self.model_ini_dict['model'] == 90:
            self.save_desti = './' + self.ques_name +'_' +str(self.ini_num) + '/'
            group = GroupVis.Vis(self.ques_name,self.ini_num, self.save_desti)

            self.net = PINN_3d.Net(self.layer).float().to(device)    
            self.workflow()
            group.loss_read(self.net.__module__.split('.')[-1])
            if self.para_ud_num:
                group.para_read(self.net.__module__.split('.')[-1])
            
            self.net = DualBranchNN_3d.Net(self.node_num).float().to(device)
            self.workflow()
            group.loss_read(self.net.__module__.split('.')[-1])
            if self.para_ud_num:
                group.para_read(self.net.__module__.split('.')[-1])
            
            self.net = AsPINN_3d.Net(self.node_num).float().to(device)
            self.workflow()
            group.loss_read(self.net.__module__.split('.')[-1])
            if self.para_ud_num:
                group.para_read(self.net.__module__.split('.')[-1])
        
            group.loss_show()
            if self.para_ud_num:
                group.para_show()

        # 程序默认按照AsPINN进行求解    
        else : 
            print('Attention:  The input model code is wrong, running in default model: AsPINN!\n')
            self.net = AsPINN_3d.Net(self.node_num).float().to(device)
            self.workflow()

