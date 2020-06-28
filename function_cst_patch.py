# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 18:17:41 2020

@author: wq_ch
"""
"""尝试定义cst的接口函数"""

import os
import sys
import glob
import numpy as np
import cst.interface

def cst_ven_patch(X,N,fullname,fullname_temp):
    global cst
    fullname_temp = fullname_temp + "\\project_1.cst"
    for i in range(N):
        j = 0
        cst = cst.interface.DesignEnvironment()
        mws = cst.open_project(fullname)            # 打开名为 fullname 的文档
        mws.save(fullname_temp)
        modeler = mws.modeler
        # 更新参数
        modeler.add_to_history('StoreParameter','StoreParameter("slot2_w", "%f")' % X[i][j])
        j += 1
        modeler.add_to_history('StoreParameter','StoreParameter("slot1_w", "%f")' % X[i][j])
        j += 1
        modeler.add_to_history('StoreParameter','StoreParameter("slot1_l", "%f")' % X[i][j])
        j += 1
        modeler.add_to_history('StoreParameter','StoreParameter("slot1_offset", "%f")' % X[i][j])
        j += 1
        modeler.add_to_history('StoreParameter','StoreParameter("Patch_l", "%f")' % X[i][j])
        j += 1
        modeler.add_to_history('StoreParameter','StoreParameter("Patch_w", "%f")' % X[i][j])
        j += 1
        modeler.add_to_history('StoreParameter','StoreParameter("move_patch", "%f")' % X[i][j])
        j += 1
    
        modeler.add_to_history('Rebuild','Rebuild')
        #mws.save(fullname_temp)
        modeler. run_solver()                      # 仿真
        mws.save(fullname_temp)
        cst.close()
        import cst.results
        project = cst.results.ProjectFile(fullname_temp)
        e_t = project.get_3d().get_result_item(r"1D Results\Efficiencies\Tot. Efficiency [1]")   # 效率值 复数
        fre_temp = e_t.get_xdata()             
        fre = np.array(fre_temp)
        # S11_temp = 20 * np.log10(abs(np.array([s11.get_ydata()]))).T     # python返回的是S11的复数值，这部操作转变为dB
        E_T_temp_array = abs((np.array([e_t.get_ydata()]))).T    
        if i == 0:
            E_T = E_T_temp_array 
        else:
            E_T_copy = E_T.copy()
            E_T = np.hstack((E_T_copy,E_T_temp_array))
        
        os.remove(fullname_temp)

    return E_T, fre
