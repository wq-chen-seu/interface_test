# -*- coding: utf-8 -*-
"""
Created on Sun Jun 28 10:04:30 2020

@author: wq_ch
"""
# 建立代理模型
import numpy as np
import geatpy as ea
import global_management as gol
from sklearn.externals import joblib
import copy
class MyProblem(ea.Problem): # 继承Problem父类
    def __init__(self):
        name = 'MyProblem'                        # 初始化name（函数名称，可以随意设置）
        M = 1                                     # 初始化M（目标维数）
        maxormins = [-1]                          # 初始化maxormins（目标最小最大化标记列表，1：最小化该目标；-1：最大化该目标）
        Dim = gol.get_map("d_x")                  # 初始化Dim（决策变量维数）
        varTypes = [0] * Dim                      # 初始化varTypes（决策变量的类型，元素为0表示对应的变量是连续的；1表示是离散的）
        bounds = gol.get_map("bounds")
        lb0 = bounds[:, 0].reshape(Dim, 1) # 决策变量下界
        ub0 = bounds[:, 1].reshape(Dim, 1) # 决策变量上界
        lb = []
        ub = []
        for i in range(Dim):
            lb.append(lb0[i][0].tolist())         # 决策变量下界
            ub.append(ub0[i][0].tolist())         # 决策变量上界                              
        lbin = [1] * Dim                          # 决策变量下边界（0表示不包含该变量的下边界，1表示包含）
        ubin = [1] * Dim                          # 决策变量上边界（0表示不包含该变量的上边界，1表示包含）
        # 调用父类构造方法完成实例化
        ea.Problem.__init__(self, name, M, maxormins, Dim, varTypes, lb, ub, lbin, ubin)


    def aimFunc(self, pop): # 目标函数
        model_temp = gol.get_map("model_temp")
        wlcb = gol.get_map("wlcb")
        feature = gol.get_map("feature")
        fre_E = gol.get_map("fre_E")
        nonlin_mf_model = joblib.load(model_temp)   # load代理模型
        x = pop.Phen # 得到决策变量矩阵
        size_x = x.shape[0]
        if feature:
            size_f = fre_E.shape[0]
            x_predict_copy = copy.deepcopy(x)
            for i in range(size_x):
                X_predict_h_test = np.hstack((np.tile(x_predict_copy[i, :], (size_f, 1)), fre_E))
                X_predict_h_test = np.hstack((X_predict_h_test, np.ones((size_f,1)).reshape(-1,1)))
                if i == 0:
                    means, sigmas = nonlin_mf_model.predict(X_predict_h_test)
                    Y_predict0 = means.reshape(-1, 1) - (wlcb * sigmas).reshape(-1, 1)
                    Y_predict0 = np.min(Y_predict0, axis=0).reshape(-1, 1)
                    Y_predict = Y_predict0
                else:
                    means, sigmas = nonlin_mf_model.predict(X_predict_h_test)
                    Y_predict_temp = means.reshape(-1, 1) - (wlcb * sigmas).reshape(-1, 1)
                    Y_predict_temp = np.min(Y_predict_temp, axis=0).reshape(-1, 1)
                    Y_predict = np.vstack((Y_predict, Y_predict_temp))
        else:
            X_predict_h_test = np.hstack([x,np.ones((size_x,1)).reshape(size_x,1)])
            means, sigmas = nonlin_mf_model.predict(X_predict_h_test)
            Y_predict = means.reshape(-1, 1) - (wlcb * sigmas).reshape(-1, 1)
        # X_predict_h_test = np.hstack([x,np.ones((alph_H,1)).reshape(alph_H,1)]).astype(int)

        pop.ObjV = Y_predict
        return pop.ObjV

def opti_function(nind, maxgen, X_train, Y_train, n_fidelity, fullname_temp, feature, wlcb, fre_E, d_x, bounds):
    gol.set_map("feature", feature)
    gol.set_map("wlcb", wlcb)
    gol.set_map("fre_E", fre_E)
    gol.set_map("d_x", d_x)
    gol.set_map("bounds", bounds)
    # from function_nonlinmodel import variable_model
    variable_model(X_train, Y_train, n_fidelity, fullname_temp)
    # 实例化问题对象
    problem = MyProblem()                                                       # 生成问题对象
    # 种群设置
    Encoding = 'BG'                                                             # 编码方式
    NIND = nind                                                                   # 种群规模
    Field = ea.crtfld(Encoding, problem.varTypes, problem.ranges, problem.borders) # 创建区域描述器
    population = ea.Population(Encoding, Field, NIND)                           # 实例化种群对象（此时种群还没被初始化，仅仅是完成种群对象的实例化）
    # 算法参数设置
    myAlgorithm = ea.soea_SEGA_templet(problem, population)                     # 实例化一个算法模板对象
    myAlgorithm.MAXGEN = maxgen                                                    # 最大进化代数
    myAlgorithm.drawing = 0                                                     # 不画图
    # 调用算法模板进行种群进化
    [population, obj_trace, var_trace] = myAlgorithm.run()                      # 执行算法模板
    population.save()                                                           # 把最后一代种群的信息保存到文件中
    # 输出结果
    best_gen = np.argmin(problem.maxormins * obj_trace[:, 1])                   # 记录最优种群个体是在哪一代
    best_ObjV = obj_trace[best_gen, 1]
    
    '''# 遗传算法画图
    try:
        plt.close("Ga 1")
        plt.ion()
        plt.figure("Ga 1")
        plt.plot(obj_trace[:, 0], lw=3, label='Average objective function value')
        plt.plot(obj_trace[:, 1], lw=3, label='Optimal objective function value')
        # plt.grid(True)
        plt.legend(loc=0)  # 图例位置自动
        plt.xlabel('Number of Generation')
        plt.ylabel('Value')
        plt.title('objective function value')
        plt.draw()
    except:
        plt.ion()
        plt.figure("Ga 1")
        plt.plot(obj_trace[:, 0], lw=3, label='Average objective function value')
        plt.plot(obj_trace[:, 1], lw=3, label='Optimal objective function value')
        # plt.grid(True)
        plt.legend(loc=0)  # 图例位置自动
        # plt.axis('tight')
        plt.xlabel('Number of Generation')
        plt.ylabel('Value')
        plt.title('objective function value')
        plt.draw()'''
    return var_trace[best_gen,:].reshape(1,-1), best_ObjV

import GPy
from emukit.multi_fidelity.models.non_linear_multi_fidelity_model import make_non_linear_kernels, NonLinearMultiFidelityModel
def variable_model(x_train, y_train, n_fidelity, fullname_temp):
    base_kernel = GPy.kern.RBF
    kernels = make_non_linear_kernels(base_kernel, 2, x_train.shape[1]-1)
    nonlin_mf_model = NonLinearMultiFidelityModel(x_train, y_train, n_fidelities = n_fidelity, kernels = kernels, 
                                                  verbose=True, optimization_restarts=5)
    
    for m in nonlin_mf_model.models:
        m.Gaussian_noise.variance.fix(0)         
    nonlin_mf_model.optimize()
    
    # 保存模型
    from sklearn.externals import joblib
    model_temp = fullname_temp + "\\nonlin_nf_model.pkl"
    gol.set_map("model_temp",model_temp)
    joblib.dump(nonlin_mf_model, model_temp)
