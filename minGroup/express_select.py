#!/usr/bin/env python3
# @Time:    2021/4/20 下午3:03
# @License :(C) Copyright 2021, {sjky}
# @Author:  aily
# @File:    express_select.py
# @Project: ID_simple_v2
# 目前先不需要考虑体积重量

import numpy as np
import math

# class ExpressType:
#     ANE = '安能'
#     SF = '顺丰'
#     HUITONG = '汇通'
#     YTO = '圆通'
#     SYTO = '圆通小包'
#     ZTO = '中通'
#     EMS = '邮政'
#     # express_list = ['安能', '顺丰', '汇通', '圆通', '圆通小包', '中通', '邮政']
#     express_list = [ANE, SF, HUITONG, YTO, SYTO, ZTO, EMS]

ExpressType = {
    'ANE' : '安能',
    'SF' : '顺丰',
    'HUITONG' : '汇通',
    'YTO' : '圆通',
    'SYTO' : '圆通小包',
    'ZTO' : '中通',
    'EMS' : '邮政',
}

class ANE:
    name = '安能'
    base_weight = 3 # 首重kg
    base_price = 15  # 首重价格rmb
    add_price = 0.7 # 续重价格/kg
    pkg_max_weight = 40 # 单个包裹最大重量
    pkg_extra_max_weight = 70 # 单个包裹上楼费最大重量
    extra_price_for_weight = 18 # 上楼费
    extra_price_for_limit = 3 # 额外费用/kg,对于超过上楼费最大重量
    max_weights = float('inf')
    max_pkgs = 7 # 单个订单最大包裹数量

class SF:
    name = '顺丰'
    base_weight = 20  # 首重kg
    base_price = 32.4  # 首重价格rmb
    add_price = 1.62  # 续重价格/kg
    pkg_max_weight = float('inf')  # 单个包裹最大重量
    max_weights = float('inf')
    max_pkgs = float('inf')

class HUITONG:
    name = '汇通'
    base_weight = 20  # 首重kg
    base_price = 32.4  # 首重价格rmb
    add_price = 1.62  # 续重价格/kg
    pkg_max_weight = 50 # 单个包裹最大重量
    # max_weights = float('inf')
    # max_pkgs = 2

class YTO:
    name = '圆通'
    base_weight_interval = {1, 2, 3}
    base_price_interval = {2.1, 2.4, 2.6}  # 首重kg
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
    base_weight_interval = {1, 2, 3}
    base_price_interval = {3.5, 4.5, 6}  # 首重kg
    base_weight = 3  # 首重kg
    base_price = 6  # 首重价格rmb
    add_price = 1.5  # 续重价格/kg
    pkg_max_weight = 20  # 单个包裹最大重量
    max_pkgs = 1

SmallExpress = [YTO, EMS, ZTO]
BIGExpress = ['圆通', '中通', '邮政', '汇通', '安能', '顺丰']
MulPkgExpress = [HUITONG, ANE, SF]
ExpressList = [YTO, SYTO, ZTO, EMS]

def single_pkg_best_express(weight):
    if weight <= 3:
        min_cost = SYTO.base_price
        express = SYTO.name
        for exp in SmallExpress:
            cost = 0
            for i in exp.base_weight_interval:
                if weight <= i:
                    cost = exp.base_price_interval
                    break
            if cost < min_cost:
                min_cost = cost
                express = exp.name
        return express, min_cost
    else:
        min_cost = 10000
        express = None
        for exp in ExpressList:
            if exp.name in ['圆通', '中通', '邮政', '汇通', '安能', '顺丰']:
                if weight > exp.pkg_max_weight and exp.name != '安能': # 超过单包最大限重
                    continue
                elif weight > exp.pkg_max_weight and exp.name == '安能': # 超过单包最大限重
                    if weight <= exp.pkg_extra_max_weight:
                        cost = exp.base_price + (weight - exp.base_weight) * exp.add_price + exp.extra_price_for_weight
                    else:
                        cost = exp.base_price + (weight - exp.base_weight) * exp.add_price + \
                               (weight - exp.extra_price_for_weight)*exp.extra_price_for_limit
                else: # 未超过单包最大限重
                    cost = exp.base_price + (weight - exp.base_weight) * exp.add_price
                if cost < min_cost:
                    min_cost = cost
                    express = express.name
        return express, min_cost

def double_pkg_best_express(weight):
    # 只有两个包裹，需考虑质量不相同的情况
    for exp in MulPkgExpress:
        pass


def main(weight, num_pkgs):
    if num_pkgs == 1:
        express, total_cost = single_pkg_best_express
    elif num_pkgs <= 2:
        # 先考虑子母单的情况
        min_cost = 10000
        express = None
        for exp in MulPkgExpress:
            # w < baseW, w>baseW, 2*w >baswW, 2*w
            if exp.name != '安能':
                if weight <= exp.pkg_max_weight: # 单个包裹的重量符合要求
                    if (2 * weight) <= exp.base_weight: # 2个包裹总重量<=首重
                        cost = exp.base_price
                    elif (2 * weight) <= exp.max_weights: # 2个包裹总重量大于首重,小于最大重量
                        cost = exp.base_price + (2*weight - exp.base_weight)
                    else:
                        print('[%s]:总重量大于最大限制值，无法组成子母单'%exp.name)
                        continue
                else: # 单个包裹的重量不符合要求
                    print('[%s]:单个包裹的重量大于限制值'%exp.name)
                    continue
                if cost < min_cost:
                    min_cost = cost
                    express = '%s_2'%exp.name
            else:
                if weight <= exp.pkg_max_weight: # 单个包裹的重量符合要求
                    if (2 * weight) <= exp.base_weight: # 2个包裹总重量<=首重
                        cost = exp.base_price
                    else:
                        cost = exp.base_price + (2*weight - exp.base_weight)
                else: # 单个包裹的重量不符合要求
                    if weight <= exp.pkg_extra_max_weight:
                        cost = exp.base_price + (weight - exp.base_weight) * exp.add_price + exp.extra_price_for_weight
                    else:
                        cost = exp.base_price + (2*weight - exp.base_weight) * exp.add_price + \
                               (2*weight - exp.extra_price_for_weight) * exp.extra_price_for_limit
                if cost < min_cost:
                    min_cost = cost
                    express = '%s_2'%exp.name
        # TODO 计算分两个订单的价格
        single_express, single_cost = single_pkg_best_express(weight)
        if single_cost * 2 < min_cost:
            total_cost = single_cost
            print('两个订单')
            result = {'%s'%single_express: [1], '%s'%single_express: [2]}
            return result, total_cost
        else:
            total_cost = min_cost
            result = {'%s'%express.name :['%d/%d'%(i, num_pkgs) for i in range(2)]}
            return result, total_cost
    else:
        # 包裹数大于等于3, 首先考虑子母单的情况
        express = None
        min_cost = 10000
        for exp in MulPkgExpress:
            result = []
            pkgs = math.floor(exp.max_weights / weight) # 一个子母单可以放的包裹数
            if pkgs == 0: # 单个包裹超过最大限重
                if exp.name == '安能':
                    if weight <= exp.pkg_extra_max_weight:
                        single_cost = exp.base_price + (weight - exp.base_weight) * exp.add_price + 18
                    else:
                        single_cost = exp.base_price + (exp.pkg_extra_max_weight - exp.base_weight) * exp.add_price \
                                      + (weight - exp.pkg_extra_max_weight) * x
                    total_cost = single_cost * num_pkgs
                    if total_cost < min_cost:
                        min_cost = total_cost
                        express = exp.name
                        result = []
                else:
                    print('[%s]:单个包裹的重量大于限制值'%exp.name)
                    continue
            elif pkgs >= num_pkgs: # 一个子母单包含全部包裹
                if num_pkgs <= exp.max_pkgs: # 包裹数 < 子母单最大包裹数限制
                    # 一个子母单
                    if weight * num_pkgs <= exp.base_weight:
                        cost = exp.base_price
                    else:
                        cost = exp.base_price + (weight * num_pkgs - exp.base_weight) * exp.add_price
                    result = ['%d/%d'%((i+1), num_pkgs) for i in range(num_pkgs)]
                else: # 包裹数 > 子母单最大包裹数限制
                    express_nums = math.floor(num_pkgs / exp.max_pkgs)
                    remain_pkgs = (num_pkgs - exp.max_pkgs) % exp.max_pkgs

                    if remain_pkgs * weight >= exp.base_weight: # 剩余包裹数重量大于首重
                        cost = exp.base_price + (remain_pkgs * weight - exp.base_weight) * exp.add_price
                        total_cost = cost + express_nums * (exp.base_price + (exp.max_pkgs * weight - exp.base_weight) * exp.add_price)
                        for j in range(express_nums):
                            exp_result = ['%d/%d'%((i+1), exp.max_pkgs) for i in range(exp.max_pkgs)]
                            result.append(exp_result)
                        remain_result = ['%d/%d' % ((i + 1), remain_pkgs) for i in range(remain_pkgs)]
                        result.append(remain_result)
                    else: # 剩余包裹数重量小于首重
                        if exp.max_pkgs > 2:
                            pkgs_basew = math.ceil(exp.base_weight / weight) # 满足首重需要的包裹数
                            need_pkgs = pkgs_basew - remain_pkgs # 满足首重需要添加的包裹数
                            if (exp.max_pkgs - need_pkgs) * weight >= exp.base_weight:
                                # 从最后一个完整的子母单中拆出need_pkgs个包裹放到最后一个子母单中
                                cost1 = exp.base_price + (need_pkgs * weight - exp.base_weight) * exp.add_price
                                cost2 = exp.base_price + ((exp.max_pkgs - need_pkgs) * weight - exp.base_weight) * exp.add_price
                                total_cost = cost1 + cost2 + express_nums * (exp.base_price + (exp.max_pkgs * weight - exp.base_weight) * exp.add_price)
                                for j in range(express_nums - 1):
                                    exp_result = ['%d/%d' % ((i + 1), exp.max_pkgs) for i in range(exp.max_pkgs)]
                                    result.append(exp_result)
                                half_result = ['%d/%d' % ((i + 1), exp.max_pkgs - need_pkgs) for i in range(exp.max_pkgs - need_pkgs)]
                                remain_result = ['%d/%d' % ((i + 1), pkgs_basew) for i in range(pkgs_basew)]
                                result.append(half_result)
                                result.append(remain_result)
                            else:
                                # 最后一个完整的子母单中拆出need_pkgs个包裹后小于首重
                                pass
            else: # 一个子母单无法包含全部包裹
                if pkgs <= exp.max_pkgs:
                    # 一个子母单中可放的包裹数小于最大包裹数限制
                    express_nums = math.floor(num_pkgs / pkgs)
                    remain_pkgs = num_pkgs % pkgs
                    if remain_pkgs * weight >= exp.base_weight: # 剩余包裹数重量大于首重
                        cost = exp.base_price + (pkgs * weight - exp.base_weight) * exp.add_price
                        total_cost = cost + exp.base_price + (remain_pkgs * weight - exp.base_weight) * exp.add_price
                        for i in range(express_nums):
                            exp_result = ['%d/%d' % ((i + 1), pkgs) for i in range(pkgs)]
                            result.append(exp_result)
                        remain_result = ['%d/%d' % ((i + 1), remain_pkgs) for i in range(remain_pkgs)]
                        express = exp.name
                    else: # 剩余包裹数重量小于首重,需要拆分
                        pkgs_basew = math.ceil(exp.base_weight / weight)  # 满足首重需要的包裹数
                        need_pkgs = pkgs_basew - remain_pkgs  # 满足首重需要添加的包裹数
                        if (pkgs - need_pkgs) * weight >= exp.base_weight: # 完整子母单减去拆分包裹后仍然大于首重
                            full_exps = express_nums - 1
                            cost_full = (exp.base_price + (pkgs*weight - exp.base_weight) * exp.add_price) * full_exps
                            cost_half = exp.base_price + ((pkgs - need_pkgs)*weight - exp.base_weight) * exp.add_price
                            cost_last = exp.base_price + (pkgs_basew*weight - exp.base_weight) * exp.add_price
                            total_cost = cost_full + cost_half + cost_last
                            for j in range(full_exps):
                                exp_result = ['%d/%d' % ((i + 1), pkgs) for i in range(pkgs)]
                                result.append(exp_result)
                            half_result = ['%d/%d' % ((i + 1), pkgs - need_pkgs) for i in
                                          range(pkgs - need_pkgs)]
                            remain_result = ['%d/%d' % ((i + 1), pkgs_basew) for i in range(pkgs_basew)]
                            result.append(half_result)
                            result.append(remain_result)
                        else:
                            # 完整子母单减去拆分包裹后小于首重

                            pass
                else:
                    # 一个子母单中可放的包裹数大于最大包裹数限制
                    if exp.max_pkgs * weight >= exp.base_weight:
                        pkgs = exp.max_pkgs




if __name__ == '__main__':
    num_pkgs = 12
    weight = 20
    main(20, 12)
