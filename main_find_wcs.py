# -*- coding: utf-8 -*-
"""
Created on Sun Jun 28 10:36:26 2020

@author: wq_ch
"""
import numpy as np
import global_management as gol
from emukit.multi_fidelity.convert_lists_to_array import convert_xy_lists_to_arrays
import random
import scipy.io as scio
import copy
from function_cst_patch import cst_ven_patch
# WCP阶段
# 读取数据
fillname_fine = "D:\python\patch_test\model\\ven_patch_fine.cst"
fillname_coarse = "D:\python\patch_test\model\\ven_patch_coarse.cst"
fullname_temp = "D:\python\patch_test\model"
dataFile = fullname_temp + '\\E_T_coarse.mat'
data = scio.loadmat(dataFile)
E_T_coarse = data['E_T_coarse']

dataFile = fullname_temp + '\\E_T_fine.mat'
data = scio.loadmat(dataFile)
E_T_fine = data['E_T_fine']

dataFile = fullname_temp + '\\fre_E.mat'
data = scio.loadmat(dataFile)
fre_E = data['fre_E']

dataFile = fullname_temp + '\\X.mat'
data = scio.loadmat(dataFile)
X = data['X']

dataFile = fullname_temp + '\\X_H.mat'
data = scio.loadmat(dataFile)
X_H = data['X_H']
"""============================================建模======================================="""
alph = 11  # 初始样本+设计点的数量
d_x = 7    # 维度
alph_L = 11 # 低保真度数据集的数量 
alph_H = 5  # 高保真度数据集的数量
# 处理初始样本集    
num_1_L = 4  # 3.4G
num_1_H = 6     # 3.6G
num_2_L = 18    # 4.8G
num_2_H = 19    # 4.9G
num_L = np.linspace(num_1_L, num_1_H, num_1_H-num_1_L+1, endpoint=True, retstep=False, dtype=int)
num_H = np.linspace(num_2_L, num_2_H, num_2_H-num_2_L+1, endpoint=True, retstep=False, dtype=int)
num = np.hstack((num_L,num_H))
fre_E = fre_E[num,:]
E_T_coarse = E_T_coarse[num,:]
E_T_fine =  E_T_fine[num,:]

feature = False # 是否加入特征维度
# feature = True
size_f = fre_E.shape[0]
if feature:
    E_T_coarse_wcs0 = E_T_coarse.min(0).reshape(-1,1)
    E_T_fine_wcs0 = E_T_fine.min(0).reshape(-1,1)
    for j in range(alph):
        X_temp = np.hstack((np.tile(X[j, :], (size_f, 1)), fre_E))
        E_temp = E_T_coarse[:,j].reshape(-1,1)
        if j == 0:
            X0 = copy.deepcopy(X_temp)
            E_0 = copy.deepcopy(E_temp)
        else:
            X0 = np.vstack((X0, X_temp))
            E_0 = np.vstack((E_0, E_temp))
    X = copy.deepcopy(X0)
    E_T_coarse_wcs = copy.deepcopy(E_0)
    for j in range(alph_H):
        X_temp = np.hstack((np.tile(X_H[j, :], (size_f, 1)), fre_E))
        E_temp = E_T_fine[:,j].reshape(-1,1)
        if j == 0:
            X0 = copy.deepcopy(X_temp)
            E_0 = copy.deepcopy(E_temp)
        else:
            X0 = np.vstack((X0, X_temp))
            E_0 = np.vstack((E_0, E_temp))
    X_H = copy.deepcopy(X0)
    E_T_fine_wcs = copy.deepcopy(E_0)
    sparse_number = 5      # 加入特征维度后 对于一个样本在不同频率下，选取sparse_number个点
    index2 = np.array(random.sample(range(size_f), sparse_number)).reshape(-1,1)
    for i in range(alph_H-1):
        i += 1
        index20 = random.sample(range(size_f),sparse_number)
        ind = []
        for x in index20:
            ind.append(x + i*size_f)
        index2 = np.vstack((index2.reshape(-1,1), np.array(ind).reshape(-1,1)))      
else:
    E_T_coarse_wcs = E_T_coarse.min(0).reshape(-1,1)
    E_T_fine_wcs = E_T_fine.min(0).reshape(-1,1)

x_train_L = X
y_train_L = E_T_coarse_wcs

if feature:
    x_train_H = X_H[index2[:,0], :]
    Y_trainH0 = E_T_fine_wcs0
    y_train_H_wcs = Y_trainH0
    Y_wcs = y_train_H_wcs.min()
    Y_trainH0 = E_T_fine_wcs[index2[:,0], :]
    y_train_H = Y_trainH0
else:
    x_train_H = X_H
    Y_trainH0 = E_T_fine_wcs
    y_train_H = Y_trainH0
    Y_wcs = y_train_H.min()
max_same_iter = 5
max_iter = 10
Y_wcs_record = np.zeros((max_iter+1,1)).reshape(max_iter+1, 1)
Y_pre_record = np.zeros((max_iter+1,1)).reshape(max_iter+1, 1)
Y_wcs_record[0,0] = Y_wcs
X_train, Y_train = convert_xy_lists_to_arrays([x_train_L, x_train_H], [y_train_L, y_train_H])
n_fidelity = 2           # 代表多保准度模型的数量 这里是2个保真度的模型
"""============================================全局优化算法预测======================================="""
wlcb = 1
count_iter = 0
same_iter = 0
gol.set_map("wlcb", wlcb)
gol.set_map("feature", feature)
gol.set_map("fre_E", fre_E)
gol.set_map("d_x", d_x)
bounds = np.array([
    [5.67, 6.07], [2.75, 3.55], [3.08, 3.48],
    [5.98, 6.38],[33.51, 34.31],[21.63, 22.03],[-0.021, -0.005]]) # 参数范围
gol.set_map("bounds", bounds)
from function_opti import opti_function
nind = 50
maxgen = 50
while count_iter < max_iter and same_iter < max_same_iter:
    # 建模 
    # 预测得到的结果
    x_pre, y_pre = opti_function(nind, maxgen, X_train, Y_train, n_fidelity, fullname_temp, feature, wlcb, fre_E, d_x, bounds)
    x_pre = x_pre.reshape(1,d_x)
    Y_pre_record[count_iter,0] = y_pre
    """===========================================验证======================================"""
    # 进入cst测试阶段
    E_T_coarse, f = cst_ven_patch(x_pre, 1, fillname_coarse, fullname_temp)
    E_T_fine, f = cst_ven_patch(x_pre, 1, fillname_fine, fullname_temp)
    E_T_coarse = E_T_coarse[num,:]
    E_T_fine = E_T_fine[num,:]
    if feature:
        E_T_coarse_wcs0 = E_T_coarse.min(0).reshape(-1,1)
        E_T_fine_wcs0 = E_T_fine.min(0).reshape(-1,1)
        X_temp = np.hstack((np.tile(x_pre[0, :], (size_f, 1)), fre_E))
        E_T_coarse_wcs = E_T_coarse
        E_T_fine_wcs = E_T_fine  
    else:
        X_temp = x_pre
        E_T_coarse_wcs = E_T_coarse.min(0).reshape(-1,1)
        E_T_fine_wcs = E_T_fine.min(0).reshape(-1,1)

    x_train_L = np.vstack([x_train_L, X_temp])
    y_train_L = np.vstack([y_train_L, E_T_coarse_wcs])
    x_train_H = np.vstack([x_train_H, X_temp])
    y_train_H = np.vstack([y_train_H, E_T_fine_wcs])
    x_y = np.hstack((x_train_H,y_train_H))
    if feature:
        y_train_H_wcs = np.vstack([y_train_H_wcs, E_T_fine_wcs0])
        Y_wcs = y_train_H_wcs.min()
        Y_wcs_record[count_iter+1, 0] = Y_wcs
    else:
        Y_wcs = y_train_H.min()
        Y_wcs_record[count_iter+1, 0] = Y_wcs
    X_train, Y_train = convert_xy_lists_to_arrays([x_train_L, x_train_H], [y_train_L, y_train_H])
    if Y_wcs_record[count_iter,0] == Y_wcs_record[count_iter+1,0]:
        same_iter += 1
    else:
        same_iter = 0
    count_iter += 1





