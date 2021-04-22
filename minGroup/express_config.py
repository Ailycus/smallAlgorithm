#!/usr/bin/env python3
# @Time:    2021/4/22 上午11:28
# @License :(C) Copyright 2021, {sjky}
# @Author:  aily
# @File:    express_config.py
# @Project: ID_simple_v2 
# 快递计费参数

class ANE:
    name = '安能'
    base_weight_interval = {0.5, 1, 1.5, 2, 3}
    base_price_interval = {11.27, 11.27, 11.27, 11.27, 11.27}  # 首重kg
    base_weight = 3 # 首重kg
    base_price = 11.27  # 首重价格rmb
    add_price = 0.4 # 续重价格/kg
    pkg_max_weight = 40 # 单个包裹最大重量
    pkg_extra_max_weight = 70 # 单个包裹上楼费最大重量
    extra_price_for_weight = 18 # 上楼费
    extra_price_for_limit = 3 # 额外费用/kg,对于超过上楼费最大重量
    max_weights = float('inf')
    max_pkgs = 7 # 单个订单最大包裹数量

class SF:
    name = '顺丰'
    base_weight_interval = {22}
    base_price_interval = {22}
    base_weight = 22  # 首重kg
    base_price = 27  # 首重价格rmb
    add_price = 1.62  # 续重价格/kg
    pkg_max_weight = float('inf')  # 单个包裹最大重量
    max_weights = 400
    max_pkgs = float('inf')

class HUITONG:
    name = '汇通'
    base_weight_interval = {0.5, 1, 1.5, 2, 3}
    base_price_interval = {1.1, 1.7, 2.1, 2.4, 2.6}
    base_weight = 20  # 首重kg
    base_price = 32.4  # 首重价格rmb
    add_price = 1.62  # 续重价格/kg
    pkg_max_weight = 50 # 单个包裹最大重量
    # max_weights = float('inf')
    # max_pkgs = 2

class YTO:
    name = '圆通'
    base_weight_interval = {0.5, 1, 1.5, 2, 3}
    base_price_interval = {1.5, 1.5, 2.1, 2.4, 2.6}  # 首重kg
    base_weight = 3  # 首重kg
    base_price = 2.6  # 首重价格rmb
    add_price = 1  # 续重价格/kg
    pkg_max_weight = 60  # 单个包裹最大重量
    max_pkgs = 1

class SYTO:
    name = '圆通小包'
    base_weight = 3  # 首重kg
    base_price = 1.75  # 首重价格rmb
    add_price = 0  # 续重价格/kg
    pkg_max_weight = 3  # 单个包裹最大重量
    max_pkgs = 1

class ZTO:
    name = '中通'
    base_weight_interval = {0.5, 1, 1.5, 2, 3}
    base_price_interval = {1.8, 1.8, 2.2, 2.5, 3}  # 首重kg
    base_weight = 3  # 首重kg
    base_price = 3  # 首重价格rmb
    add_price = 3  # 续重价格/kg
    pkg_max_weight = 60  # 单个包裹最大重量
    max_pkgs = 1

class EMS:
    name = '邮政'
    base_weight_interval = {0.5, 1, 1.5, 2, 3}
    base_price_interval = {2.3,2.5, 3.5, 4.5, 6}  # 首重kg
    base_weight = 3  # 首重kg
    base_price = 6  # 首重价格rmb
    add_price = 1  # 续重价格/kg
    pkg_max_weight = 20  # 单个包裹最大重量
    max_pkgs = 1