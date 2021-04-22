#!/usr/bin/env python3
# @Time:    2021/4/22 上午11:27
# @License :(C) Copyright 2021, {sjky}
# @Author:  aily
# @File:    ExpressMatch.py
# @Project: ID_simple_v2 
# 快递预估匹配算法

import math
from express_config import  *
EXPRESS_LIST = [ANE, SF, YTO, SYTO, ZTO, EMS, HUITONG]

def single_pkg_best_express(weight):
    min_cost = 10000
    express = None
    for exp in EXPRESS_LIST:
        cost = 10000
        if weight <= exp.base_weight: # 小于首重
            for i, w in enumerate(exp.base_weight_interval):
                if weight <= w:
                    cost = exp.base_price_interval[i]
                    break
        elif weight <= exp.pkg_max_weight: # 大于首重但小于最大值
            cost = exp.base_price + (weight - exp.base_weight) * exp.add_price
        else: # 大于最大值
            if exp.name == '安能':
                if weight <= exp.pkg_extra_max_weight:
                    cost = exp.base_price + (weight - exp.base_weight) * exp.add_price + exp.extra_price_for_weight
                else:
                    cost = exp.base_price + (weight - exp.base_weight) * exp.add_price + \
                           (weight - exp.extra_price_for_weight) * exp.extra_price_for_limit
            else:
                continue
        if cost < min_cost:
            min_cost = cost
            express = exp.name
    return express, min_cost

def double_motherson_express(weight):
    # weight：单个包裹重量
    t_weight = weight * 2
    # for SF
    SF_cost = 10000
    if t_weight <= SF.base_weight:
        SF_cost = SF.base_price
    elif t_weight <= SF.max_weights:
        SF_cost = SF.base_price + (t_weight - SF.base_weight) * SF.add_price
    else:
        print('[%s]:总重量超过最大重量限值'%SF.name)

    # for ANE
    ANE_cost = 10000
    if t_weight <= ANE.base_weight: # 总重量小于首重
        for i, w in enumerate(ANE.base_weight_interval):
            if t_weight <= w:
                ANE_cost = ANE.base_price_interval[i]
                break
    elif t_weight <= ANE.max_weights: # 总重量小于最大限值
        if weight <= ANE.pkg_max_weight: # <=40 单包不超限重
            ANE_cost = ANE.base_price + (t_weight - ANE.base_weight) * ANE.add_price
        else:
            print('[%s]:单包重量超过单包最大重量限值'%SF.name)

    if SF_cost < ANE_cost:
        return SF.name, SF_cost
    else:
        return ANE.name, ANE_cost

def motherson_express_SF(weight, num_pkgs):
    '''
    单包重量没有限制，没有包数限制，有总重量限制
    返回结果形式：
       {'SF': [[1,1,1,1,1]]} 单个子母单
       {'SF':[[1,1,1,1], [2,2,2,2]]} 两个子母单
       {'SF':[[1,1,1,1,1]], 'EMS': [2,2]}
    '''
    t_weights = weight * num_pkgs
    SF_grp_reslut = list()
    SF_result = {}
    if t_weights <= SF.base_weight:
        SF_cost = SF.base_price
        SF_grp_reslut = [1 for i in range(num_pkgs)]
        SF_result['SF'] = SF_grp_reslut
    elif t_weights <= SF.max_weights: # 总重量小于最大限重，作为一个子母包
        SF_cost = SF.base_price + (t_weights - SF.base_weight) * SF.add_price
        SF_grp_reslut = [1 for i in range(num_pkgs)]
        SF_result['SF'] = SF_grp_reslut
    else: # 总重量超过最大限重，需要划分包裹
        max_pkgs_per_exp = math.floor(SF.max_weights / weight) # 每个子母单最大包裹数
        exp_nums = math.ceil(t_weights / SF.max_weights) # 子母单数量
        remain_weights = SF.max_weights % weight
        remain_pkgs = num_pkgs % max_pkgs_per_exp

        if remain_weights <= SF.base_weight: # 剩余重量小于首重，需要调整
            '''
                两种处理方式：
                    (1)均衡填充每个子母单
                    (2)剩余包裹根据总重量和个数测试其他快递方式
            '''
            # if max_pkgs_per_exp <= 2:  # 完整子母单包裹数无法拆分, 不会存在这种情况

            # 拆分
            pkgs_basew = math.ceil(SF.base_weight / weight)  # 满足首重需要的包裹数
            need_pkgs = pkgs_basew - remain_pkgs  # 满足首重需要添加的包裹数

            # if (max_pkgs_per_exp - need_pkgs) * weight >= SF.base_weight: # 完整子母单减去拆分包裹后仍然大于首重
            full_exps = exp_nums - 1
            cost_full = (SF.base_price + (max_pkgs_per_exp * weight - SF.base_weight) * SF.add_price) * (full_exps - 1)
            cost_half = SF.base_price + ((max_pkgs_per_exp - need_pkgs) * weight - SF.base_weight) * SF.add_price
            cost_last = SF.base_price + (pkgs_basew * weight - SF.base_weight) * SF.add_price
            total_cost_split = cost_full + cost_half + cost_last
            SF_cost = total_cost_split
            # else: # 完整子母单减去拆分包裹后小于首重
            #     print('not exist.')
            if (exp_nums - 2) > 0:
                for i in range(exp_nums - 2):
                    SF_grp_reslut.append([i + 1 for j in range(max_pkgs_per_exp)])
                SF_grp_reslut.append([exp_nums - 2 + 1 for i in range(max_pkgs_per_exp - need_pkgs)])
                SF_grp_reslut.append([exp_nums - 2 + 2 for i in range(need_pkgs)])
                SF_result['SF'] = SF_grp_reslut
            else:
                SF_grp_reslut.append([1 for i in range(max_pkgs_per_exp - need_pkgs)])
                SF_grp_reslut.append([2 for i in range(need_pkgs)])
                SF_result['SF'] = SF_grp_reslut

            if remain_pkgs == 1:
                exp_name, single_cost = single_pkg_best_express(weight)
                cost_full = (SF.base_price + (max_pkgs_per_exp * weight - SF.base_weight) * SF.add_price) * full_exps
                total_cost_two_exp = cost_full + single_cost

                if total_cost_two_exp < total_cost_split:
                    SF_cost = total_cost_two_exp
                    for i in range(exp_nums - 1):
                        SF_grp_reslut.append([i + 1 for j in range(max_pkgs_per_exp)])
                    SF_result['SF'] = SF_grp_reslut
                    SF_result[exp_name] = [exp_nums]
        else: # 剩余重量大于首重，不需要调整
            SF_cost_full_exp = SF.base_price + (max_pkgs_per_exp * weight - SF.base_weight) * SF.add_price
            SF_cost_remain = SF.base_price + (remain_weights - SF.base_weight) * SF.add_price
            SF_cost = SF_cost_full_exp * (exp_nums - 1) + SF_cost_remain
            for i in range(exp_nums - 1):
                SF_grp_reslut.append([i+1 for j in range(max_pkgs_per_exp)])
            SF_grp_reslut.append([exp_nums for i in range(remain_pkgs)])
            SF_result['SF'] = SF_grp_reslut

    return SF_result, SF_cost

def motherson_express_ANE(weight, num_pkgs):
    '''
    单包重量有限制，有包数限制，没有总重量限制
    返回结果形式：
       {'SF': [[1,1,1,1,1]]} 单个子母单
       {'SF':[[1,1,1,1], [2,2,2,2]]} 两个子母单
       {'SF':[[1,1,1,1,1]], 'EMS': [2,2]}
    '''
    ANE_cost = 10000
    t_weights = weight * num_pkgs
    ANE_grp_reslut = list()
    ANE_result = {}
    if weight > ANE.pkg_max_weight: # 单包重量超过限制
        print('[%s]:单包重量超过限制'%ANE.name)
        return ANE_result, ANE_cost
    else:
        if num_pkgs <= ANE.max_pkgs: # 包裹数量小于限制值
            ANE_cost = ANE.base_price + (t_weights - ANE.base_weight) * ANE.add_price
        else: # 包裹数量大于限制值
            num_express = math.ceil(num_pkgs / ANE.max_pkgs)
            remain_pkgs = num_pkgs % ANE.max_pkgs

            if remain_pkgs * weight >= ANE.base_weight: # 剩余重量大于首重，不需要调整
                full_exp_cost = ANE.base_price + (ANE.max_pkgs * weight - ANE.base_weight) * ANE.add_price
                remain_cost = ANE.base_price + (remain_pkgs * weight - ANE.base_weight) * ANE.add_price
                ANE_cost = full_exp_cost + remain_cost
            else: # 剩余重量小于首重，需要调整
                basew_pkgs = math.floor(ANE.base_weight / weight)
                need_pkgs = basew_pkgs - remain_pkgs

                if (ANE.max_pkgs - need_pkgs) * weight >= ANE.base_weight: # 拆分后的子母单总重量大于首重
                    pass
                else: # 拆分后的子母单总重量小于首重
                    split_pkgs = math.floor((ANE.max_pkgs * weight - ANE.base_weight) / weight)
                    if (num_express - 1) * split_pkgs  < need_pkgs:
                    # if split_pkgs == 0: # 完整子母单无法拆分
                        # 剩余不足首重作为一个子母单
                        full_exp_cost = ANE.base_price + (ANE.max_pkgs * weight - ANE.base_weight) * ANE.add_price
                        remain_cost = ANE.base_price

                        # 计算作为单个快递的累加费用
                        exp_name, sing_cost = single_pkg_best_express(weight)
                        sum_single_cost = sing_cost * remain_pkgs

                        # 计算作为一个顺丰子母单的费用
                        # 对于单个包裹的重量较小的情况，才会出现无法分割的情况，此时作为顺丰子母单，肯定是一个子母单
                        SF_result, SF_cost = motherson_express_SF(weight, remain_pkgs)

                        min_cost = min(remain_cost, sum_single_cost, SF_cost)
                        ANE_cost = full_exp_cost + min_cost
                    # else: # 完整子母单可拆分
                    #     mean_pkgs = math.floor(num_pkgs / num_express)
                    #     mean_need_pkgs = basew_pkgs * num_express # 平均分配，都满足首重，所需要的总的包裹数
                    #     if num_pkgs < mean_need_pkgs: # 平均分配都无法满足,则，不考虑拆分
                    else:  # 完整子母单可拆分








def main(weight, num_pkgs):
    if num_pkgs == 1:
        total_cost, express = single_pkg_best_express(weight)
    elif num_pkgs == 2:
        single_exp_cost, single_express = single_pkg_best_express(weight)
        # 计算作为子母包的费用
        mothersong_cost, MC_express = double_motherson_express(weight)
        if (single_exp_cost * num_pkgs) < mothersong_cost:
            total_cost = single_exp_cost * num_pkgs
            express = single_express
        else:
            total_cost = mothersong_cost
            express = MC_express
    else: # 大于等于3个包裹
        # 计算单个快递时的费用
        single_exp_cost, single_express = single_pkg_best_express(weight)
        # 计算作为子母包的费用
        mothersong_cost, MC_express = motherson_express_SF(weight, num_pkgs)



if __name__ == '__main__':
    weight = 30
    N=5
    main(weight, N)