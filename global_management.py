# -*- coding: utf-8 -*-

# 拼装成字典构造全局变量  借鉴map  包含变量的增删改查
map = {}
def set_map(key, value):
    map[key] = value
def del_map(key):
    try:
        del map[key]
    except KeyError:
        print("key:"+str(key)+"  不存在")
def get_map(key):
    try:
        if key in "all":
            return map
        return map[key]
    except KeyError as e:
        print("key:"+str(key)+"  不存在")
        