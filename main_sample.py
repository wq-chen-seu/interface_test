# -*- coding: utf-8 -*-
"""
Created on Sun Jun 28 09:41:29 2020

@author: wq_ch
"""
# 1 采样
# 2 建模
# 3 算法
# 4 验证
import numpy as np
from LHSsample_file import LHSample as LHS
import scipy.io as scio
import random
from function_cst_patch import cst_ven_patch
"""============================================采样======================================="""
# 拉丁超立方采样
d_x = 7 # 参数维度
bounds = np.array([
    [5.67, 6.07], [2.75, 3.55], [3.08, 3.48],
    [5.98, 6.38],[33.51, 34.31],[21.63, 22.03],[-0.021, -0.005]]) # 参数范围
alph = 10 # 样本数量
X_sample = LHS(d_x, bounds, alph)
# 给出设计点 
design_point = np.array([5.87, 3.15, 3.28, 6.18, 33.91, 21.83, -0.013]).reshape(1,d_x)
X = np.vstack((X_sample,design_point))

fillname_fine = "D:\python\patch_test\model\\ven_patch_fine.cst"
fillname_coarse = "D:\python\patch_test\model\\ven_patch_coarse.cst"
fullname_temp = "D:\python\patch_test\model"
alph_H = 5
index1 = random.sample(range(alph+1), alph_H)
X_H = X[index1,:]
# 进入cst采样阶段
E_T_coarse, fre_E = cst_ven_patch(X,alph+1,fillname_coarse,fullname_temp)
E_T_fine, fre_E = cst_ven_patch(X_H,alph_H,fillname_fine,fullname_temp)

# 采样数据保存为.mat文件
dataNew = fullname_temp + '\\E_T_coarse.mat'
scio.savemat(dataNew, {'E_T_coarse': E_T_coarse})
dataNew = fullname_temp + '\\E_T_fine.mat'
scio.savemat(dataNew, {'E_T_fine': E_T_fine})
dataNew = fullname_temp + '\\fre_E.mat'
scio.savemat(dataNew, {'fre_E': fre_E.reshape(-1,1)})
dataNew = fullname_temp + '\\X.mat'
scio.savemat(dataNew, {'X': X})
dataNew = fullname_temp + '\\X_H.mat'
scio.savemat(dataNew, {'X_H': X_H})
