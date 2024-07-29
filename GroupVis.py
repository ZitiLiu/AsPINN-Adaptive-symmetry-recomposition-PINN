import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class Vis():
    def __init__(self, ques_name, ini_num, file_desti, x=[], y=[],u=[]):
        self.x = x
        self.y = y 
        self.u = u
        self.ques_name = ques_name
        self.ini_num = ini_num
        self.file_desti =  file_desti 
        self.module_num = 0
        self.group_loss = []
        self.group_name = []
        self.group_para = []


    def loss_read(self, module_name):
        self.loss_densti = self.file_desti + '/Loss/'
        self.module_name = module_name
        self.group_loss.append([self.group_loss,pd.read_csv(self.loss_densti + self.ques_name + '_' + str(self.ini_num) + '_loss_' + self.module_name + '.csv').values])
        self.loss_header = pd.read_csv(self.loss_densti + self.ques_name + '_'+str(self.ini_num)+'_loss_' + self.module_name + '.csv',  nrows=0).columns
        self.loss_num =  len(self.loss_header)
        self.group_name.append(self.module_name)
        self.module_num+=1

    colors = np.array(['#00CC00','blue','red','#FFA500','#00FFFF'])
    
    def loss_show(self):
        for j in range (len(self.loss_header)-1):
            plt.figure(figsize=(6.6,6))
            for i in range (self.module_num):
                plt.plot(self.group_loss[i][1][:,0], self.group_loss[i][1][:,j+1], label=self.group_name[i], color = self.colors[i])
                font = {'family': 'Times New Roman', 'weight': 'normal', 'size': 16}
                plt.grid()
                plt.legend()
                plt.yscale('log')
                plt.xlabel(self.loss_header[0], fontdict=font)
                plt.ylabel(self.loss_header[j+1], fontdict=font)
            plt.title(self.ques_name + ' ' + self.loss_header[j+1] + ' ' + 'Comparison')
            plt.savefig(self.loss_densti + self.ques_name + '_' +str(self.ini_num) +'_'+ self.loss_header[j+1] + '_' + 'comparison' + '.png', bbox_inches='tight')
            plt.close()     

    def para_read(self, module_name):
        self.para_desti = self.file_desti + '/Parameters/'
        self.module_name = module_name
        self.group_para.append([self.group_para, pd.read_csv(self.para_desti + self.ques_name + '_' + str(self.ini_num) + '_paras_' + self.module_name + '.csv').values])
        self.para_header = pd.read_csv(self.para_desti + self.ques_name + '_'+str(self.ini_num)+'_paras_' + self.module_name + '.csv',  nrows=0).columns
        self.para_num =  len(self.para_header)
    
    def para_show(self):
        for j in range (len(self.para_header)-1):
            plt.figure(figsize=(4.4,4)) 
            for i in range (self.module_num):
                plt.plot(self.group_para[i][1][:,0], self.group_para[i][1][:,j+1], label=self.group_name[i], color = self.colors[i])
                # plt.yscale('log')
                font = {'family': 'Times New Roman', 'weight': 'normal', 'size': 16}
                plt.grid()
                plt.legend()
                # plt.yscale('log')
                plt.xlabel(self.para_header[0], fontdict=font) 
                plt.ylabel(self.para_header[j+1], fontdict=font)
            plt.title(self.ques_name + ' ' + self.para_header[j+1] + ' ' + 'Comparison')
            plt.savefig(self.para_desti + self.ques_name + '_' +str(self.ini_num) +'_'+ self.para_header[j+1] + '_' + 'comparison' + '.png', bbox_inches='tight')
            plt.close()    