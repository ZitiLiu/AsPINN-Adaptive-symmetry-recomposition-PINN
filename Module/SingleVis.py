import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
class Vis():
    plt.rcParams["figure.dpi"] = 300
    plt.rcParams['font.sans-serif'] = ['Times New Roman']
    plt.rcParams['xtick.labelsize'] = 10   
    plt.rcParams['ytick.labelsize'] = 10    
    plt.rcParams['axes.titlesize'] = 17     
    plt.rcParams['axes.labelsize'] = 16     
    plt.rcParams['axes.linewidth'] = 1      

    def __init__(self, ques_name, ini_num, file_desti, module_name, x =[], y = [], z = [], u = []):
        self.x = x
        self.y = y 
        self.z = z
        self.u = u
        self.ques_name = ques_name
        self.ini_num = ini_num
        self.file_densti =  file_desti 
        self.module_name = module_name


    def loss_vis(self):
        self.loss_desti = self.file_densti + '/Loss/'
        df = pd.read_csv(self.loss_desti + self.ques_name + '_' + str(self.ini_num) + '_loss_' + self.module_name + '.csv').values
        header = pd.read_csv(self.loss_desti + self.ques_name + '_' + str(self.ini_num) + '_loss_' + self.module_name + '.csv',  nrows=0).columns
        for j in range (len(header)-1):
            plt.figure(figsize=(3.85, 3.5)) 
            plt.plot(df[:,0],df[:,j+1])
            plt.yscale('log')
            ax = plt.gca()
            ax.ticklabel_format(style='sci', scilimits=(-1,2), axis='x')
            plt.grid()
            plt.xlabel(header[0])
            plt.ylabel(header[j+1])
            plt.title(self.ques_name + ' ' + header[j+1] + ' ' +self.module_name)
            plt.savefig(self.loss_desti + self.ques_name +'_'+str(self.ini_num)+'_' + header[j+1] + '_' +self.module_name + '.png', bbox_inches='tight')
            plt.close()
    
    def figure_2d(self):
        self.figure_desti = self.file_densti + '/Figure/'
        if not os.path.exists(self.figure_desti):
            os.mkdir(self.figure_desti)
        fig, ax = plt.subplots(figsize=(3.85, 3.5)) 
        cf = plt.scatter(self.x, self.y, c=self.u, alpha=1 - 0.1, edgecolors='none', cmap='rainbow',marker='s', s=int(8))
        plt.xlabel('x')
        plt.ylabel('y')
        # plt.grid()
        plt.margins(0)
        plt.title (self.ques_name + ' ' + self.module_name)
        fig.colorbar(cf, fraction=0.046, pad=0.04)
        plt.savefig(self.figure_desti + self.ques_name + '_figure_' + self.module_name +'.png', bbox_inches='tight')
        plt.close()

    def figure_3d(self):
        self.figure_desti = self.file_densti + '/Figure/'
        if not os.path.exists(self.figure_desti):
            os.mkdir(self.figure_desti)
        fig = plt.figure(figsize=(4.4, 4))  
        cf = fig.add_subplot(111, projection='3d')
        # plt.rcParams['font.sans-serif'] = ['Times New Roman']
        scatter = cf.scatter(self.x, self.y, self.z, c= self.u, cmap='rainbow', edgecolors='none', vmin=self.u.min(), vmax=self.u.max())
        cf.set_xlabel('x', style='italic')
        cf.set_ylabel('y', style='italic')
        cf.set_zlabel('z', style='italic')
        cf.view_init(elev=20, azim=160)
        colorbar = plt.colorbar(scatter, fraction=0.04, pad=0.2) 
        colorbar.set_label('T') 
        plt.savefig(self.figure_desti + self.ques_name + '_figure(3D)_' + self.module_name +'.png', bbox_inches='tight', dpi = 300)
        plt.close()

    def para_vis(self):
        self.para_desti = self.file_densti + '/Parameters/'
        df = pd.read_csv(self.para_desti + self.ques_name + '_' + str(self.ini_num) + '_paras_' + self.module_name + '.csv').values
        header = pd.read_csv(self.para_desti + self.ques_name + '_' + str(self.ini_num) + '_paras_' + self.module_name + '.csv',  nrows=0).columns
        for j in range (len(header)-1):
            plt.plot(df[:,0],df[:,j+1])
            # plt.yscale('log')
            font = {'family': 'Times New Roman', 'weight': 'normal', 'size': 16}
            # plt.grid()
            plt.xlabel(header[0], fontdict=font)
            plt.ylabel(header[j+1], fontdict=font)
            plt.title(self.ques_name + ' ' + header[j+1] + ' ' +self.module_name)
            plt.savefig(self.para_desti + self.ques_name +'_'+str(self.ini_num)+'_' + header[j+1] + '_' +self.module_name + '.png', bbox_inches='tight')
            plt.close()

    
    