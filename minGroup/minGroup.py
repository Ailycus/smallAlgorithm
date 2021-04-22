#!/usr/bin/env python3
# @Time:    2020/12/7 上午10:24
# @Author:  aily
# @File:    minGroup.py
# @Content: 给定一组数据，将其划分为多个小组，要求每组数据之和小于等于max_th，每组数值优先填满基础阈值base_th，求最少分组数及每组的数值。
from collections import Counter

stop = False

def epoch_group(data, max_th):
    if sum(data) <= max_th:
        loss = max_th - sum(data)
        index = list(range(len(data)))
        return [data], [index], [loss]

    # 从大到小依次获取数值放入init_grp中，作为后续优化的基础
    init_diff = max_th
    init_grp = []
    init_index = []
    one_groups = []
    one_indexs = []
    one_dist = []

    # 从大到小依次获取数值放入init_grp中，作为后续优化的基础
    for i, v in enumerate(data):
        if init_diff >= v:
            init_grp.append(v)
            init_diff = init_diff - v
            init_index.append(i)
        elif init_diff < data[-1]:
            break
    # 记录init_grp的信息至one_**中
    cur_init_grp = init_grp.copy()
    one_groups.append(cur_init_grp)
    one_indexs.append(init_index)
    one_dist.append(init_diff)


    # 根据init_grp结果进行算法优化
    cur_base_diff = init_diff
    min_diff = init_diff
    index = init_index.copy()
    while len(init_grp) > 0:
        value = init_grp.pop(-1)  # 拿出最后一个数值
        diff = cur_base_diff + value
        cur_base_diff = diff
        id = index.pop(-1)

        if sum(data[id + 1:]) >= diff:
            # 当前值后面的数累加和大于等于diff+当前值，才有可能替换当前值
            first_id = id
            while first_id < (len(data) - 1):
                tmp_grp = []
                tmp_index = []
                diff = cur_base_diff
                for i, v in enumerate(data):
                    if i > first_id and diff >= v:
                        tmp_grp.append(v)
                        tmp_index.append(i)
                        diff = diff - v
                    elif diff < data[-1]:
                        break
                # 结束循环，获取一组排列及对应dist
                if diff <= min_diff or diff < data[-1]:
                    min_diff = diff
                    tmp_best_grp = init_grp + tmp_grp
                    tmp_best_index = index + tmp_index
                    one_groups.append(tmp_best_grp)
                    one_indexs.append(tmp_best_index)
                    one_dist.append(diff)
                first_id = tmp_index[0]
        else:
            # 无法优化该值，继续往前找
            continue

    return one_groups, one_indexs, one_dist


def minGroup_exhaustion(data, max_th, base_th, best_grp):
    '''
        最优解算法
        多层循环实现实现穷举
        该方法有问题，无法确定要写几层循环
    '''
    count = len(data)

    init_grps, init_loss = init_groups(data, max_th)
    if len(init_grps) == best_grp:  # 已获取最优分组数
        result = averageResult(init_grps, max_th, base_th)
        print('共有%d个数据，划分为%d组' % (count, len(result)))
        for i, grp in enumerate(result):
            dist = max_th - sum(grp)
            print("distance: {:.2f}, {}".format(dist, grp))

        return 0

    all_groups = list()
    all_dists = list()
    groups0 = list()  # 最终分组结果

    # 对数据由大到小排序
    data = sorted(data, reverse=True)

    best_loss = 100

    # 如果第一个数值大于max_th，则单独分为一组
    while len(data) > 0:
        if data[0] >= max_th:
            groups0.append([data[0]])
            del data[0]
        else:
            break
    data_copy = data.copy()

    epoch_groups, epoch_indexs, epoch_dists = epoch_group(data_copy, max_th)

    for i, ids in enumerate(epoch_indexs):
        data_copy = data.copy()
        groups = groups0.copy()
        groups.append(epoch_groups[i])
        loss = epoch_dists[i]
        if loss > best_loss:
            continue
        # 对每种组合重新计算下一轮
        for id in ids[::-1]:
            del data_copy[id]
        if sum(data_copy) <= max_th:
            groups.append(data_copy)
            all_groups.append(groups)
            loss = loss + (max_th - sum(data_copy))
            all_dists.append(loss)
        else:
            tmp_groups, tmp_indexs, tmp_dists = epoch_group(data_copy, max_th)
            for j, ids in enumerate(tmp_indexs):
                data_copy1 = data_copy.copy()
                groups1 = groups + [tmp_groups[j]]
                loss1 = loss + tmp_dists[j]
                if loss1 > best_loss:
                    continue
                # 对每种组合重新计算下一轮
                for id in ids[::-1]:
                    del data_copy1[id]
                if sum(data_copy1) <= max_th:
                    groups1.append(data_copy1)
                    loss1 = loss1 + (max_th - sum(data_copy1))
                    if loss1 > best_loss:
                        continue
                    all_groups.append(groups1)
                    all_dists.append(loss1)
                else:
                    # pass
                    tmp_groups1, tmp_indexs1, tmp_dists1 = epoch_group(data_copy1, max_th)
                    for j, ids in enumerate(tmp_indexs1):
                        data_copy2 = data_copy1.copy()
                        groups2 = groups1 + [tmp_groups1[j]]
                        loss2 = loss1 + tmp_dists1[j]
                        if loss2 > best_loss:
                            continue
                        # 对每种组合重新计算下一轮
                        for id in ids[::-1]:
                            del data_copy2[id]
                        if sum(data_copy2) <= max_th:
                            groups2.append(data_copy2)
                            loss2 = loss2 + (max_th - sum(data_copy2))
                            if loss2 > best_loss:
                                continue
                            all_groups.append(groups2)
                            all_dists.append(loss2)
                        else:
                            tmp_groups2, tmp_indexs2, tmp_dists2 = epoch_group(data_copy2, max_th)
                            for j, ids in enumerate(tmp_indexs2):
                                data_copy3 = data_copy2.copy()
                                groups3 = groups2 + [tmp_groups2[j]]
                                loss3 = loss2 + tmp_dists2[j]
                                if loss3 > best_loss:
                                    continue
                                # 对每种组合重新计算下一轮
                                for id in ids[::-1]:
                                    del data_copy3[id]
                                if sum(data_copy3) <= max_th:
                                    groups3.append(data_copy3)
                                    loss3 = loss3 + (max_th - sum(data_copy3))
                                    if loss3 > best_loss:
                                        continue
                                    all_groups.append(groups3)
                                    all_dists.append(loss3)
                                else:
                                    pass
    if len(all_dists) > 0:
        dist_sort_index = sorted(range(len(all_dists)), key=lambda k: all_dists[k])  # 从小到大
        groups=all_groups[dist_sort_index[0]]

        # print('all groups: {}, dis:{}'.format(all_groups, all_dists))
        print('共有%d个数据，划分为%d组' % (count, len(groups)))
        for i, grp in enumerate(groups):
            dist = max_th - sum(grp)
            print("distance: {:.2f}, {}".format(dist, grp))


def minGroupFullSearch(data, max_th, base_th, best_grp):
    '''
        尝试从后往前搜素优化算法
        一组一组的确定，获取每组的最优解，一直到数据全部用完
    '''
    # 对数据由大到小排序
    data = sorted(data, reverse=True)
    count = len(data)
    groups = []

    while len(data) > 0:
        init_grp = []
        init_diff = max_th
        # 如果最大值大于max_th，则将其单独作为一个组
        if data[0] >= max_th:
            groups.append([data[0]])
            del data[0]
            continue

        # 剩余数据总和小于等于max_th，则将全部数据加入groups之中， 并结束循环
        if init_diff == max_th and sum(data) <= max_th:
            groups.append(data)
            break

        # 从大到小依次获取数值放入init_grp中，作为后续优化的基础
        init_index = []
        for i, v in enumerate(data):
            if init_diff >= v:
                init_grp.append(v)
                init_diff = init_diff - v
                init_index.append(i)
            elif init_diff < data[-1]:
                break

        # 如果init_grp的diff为0，则为最优，不进行后续优化搜索
        if init_diff == 0:
            groups.append(init_grp)
            for i in init_index[::-1]:
                del data[i]
        else:
            # 当前init_grp并非最优结果，根据此结果进行算法优化
            cur_base_diff = init_diff
            min_diff = init_diff # 记录最小loss
            tmp_best_grp = init_grp.copy() # 用于记录最优组合
            tmp_best_index = init_index.copy() # 用于记录最优组合数据索引
            while len(init_grp) > 0:
                value = init_grp.pop(-1) # 拿出最后一个数值
                diff = cur_base_diff + value # 更新loss
                cur_base_diff = diff
                id = init_index.pop(-1) # 最后一个值的索引

                if sum(data[id+1:]) >= diff:
                    # 当前值后面的数累加和大于等于diff+当前值，才有可能替换当前值
                    first_id = id
                    best_flag = False
                    while first_id < (len(data) - 1):
                        tmp_grp = []
                        tmp_index = []
                        diff = cur_base_diff
                        for i, v in enumerate(data):
                            if i > first_id and diff >= v:
                                tmp_grp.append(v)
                                tmp_index.append(i)
                                diff = diff - v
                            elif diff < data[-1]:
                                break
                        # 结束循环，获取一组排列及对应dist
                        if diff == 0: # 如果已获取最优值
                            best_flag = True
                            tmp_best_grp = init_grp + tmp_grp
                            tmp_best_index = init_index + tmp_index
                            break # 结束本轮
                        elif diff < min_diff: # 结果优于init_diff则保留
                            min_diff = diff
                            tmp_best_grp = init_grp + tmp_grp
                            tmp_best_index = init_index + tmp_index
                        first_id = tmp_index[0]

                    if best_flag:
                        break
                else:
                    # 无法优化该值，继续往前找
                    continue
            # 优化搜索结束，获取最优解
            groups.append(tmp_best_grp)
            for i in tmp_best_index[::-1]:
                del data[i]

    print('共有%d个数据，划分为%d组' % (count, len(groups)))
    result = averageResult(groups, max_th, base_th)
    print_result(result, max_th)

#-------------------------------------
def find_first_grp(data, max_th, init_loss):
    '''
        功能：从数据列表中，找到第一个分组的所有组合
        以列表的形式范围，列表中每一个元素都是一个节点Node类型

        如果当前组的diff大于init_loss（即当前一个组的loss就大于全部loss，则不保留该组）

        init_loss: 初始化分组总loss值
    '''
    # 剩余数据之和小于max_th,则全部作为一个group
    res_list = []
    if sum(data) < max_th:
        res_node = Node(data, (max_th-sum(data)), [])
        res_list.append(res_node)
        return  res_list

    # 从大到小依次获取数值放入init_grp中，作为后续优化的基础
    init_diff = max_th
    init_grp = []
    init_index = []

    groups = []
    indexs = []
    dist = []

    # 从大到小依次获取数值放入init_grp中，作为后续优化的基础
    for i, v in enumerate(data):
        if init_diff >= v:
            init_grp.append(v)
            init_diff = init_diff - v
            init_index.append(i)
        elif init_diff < data[-1]:
            break
    # 记录init_grp的信息至one_**中
    cur_init_grp = init_grp.copy()
    groups.append(cur_init_grp)
    indexs.append(init_index)
    dist.append(init_diff)

    # 根据init_grp结果进行算法优化
    cur_base_diff = init_diff # 记录第一组的diff值
    min_diff = init_diff # 作为初始最小值
    index = init_index.copy() # 第一组的所有元素位置索引

    # 从后往前依次替换init_grp中的值
    while len(init_grp) > 0:
        value = init_grp.pop(-1)  # 拿出最后一个数值
        diff = cur_base_diff + value # 当前可用空间
        cur_base_diff = diff # 记录可用空间
        id = index.pop(-1) # init_grp中拿出的元素对应位置索引

        if sum(data[id + 1:]) >= diff:
            # 当前值后面的数累加和大于等于diff+当前值，才有可能替换当前值
            first_id = id # 记录从init_grp中拿出的像素的位置
            while first_id < (len(data) - 1):
                tmp_grp = [] # 记录当前组可以换组合的数据
                tmp_index = [] # 记录当前组可以换组合的数据位置索引
                diff = cur_base_diff # 初始化diff
                for i, v in enumerate(data): # 从拿出元素的后一个元素开始循环，查找符合要求的数据
                    if i > first_id and diff >= v:
                        tmp_grp.append(v)
                        tmp_index.append(i)
                        diff = diff - v # add
                    elif diff < data[-1]:
                        break
                # 结束循环，获取一组排列及对应dist,保留小于min_diff的组，或者diff小于数据最小值，这是一个限制条件，可以修改
                if diff <= min_diff or diff < init_loss: # 如果当前组的diff小于于init_loss
                    min_diff = diff
                    tmp_best_grp = init_grp + tmp_grp
                    tmp_best_index = index + tmp_index
                    groups.append(tmp_best_grp)
                    indexs.append(tmp_best_index)
                    dist.append(diff)
                first_id = tmp_index[0]
        else:
            # 无法优化该值，继续往前找
            continue

    # 将结果保存为Node的形式并返回
    res_list = list()
    for i, grp in enumerate(groups):
        res_node = Node(grp, dist[i], [])
        res_list.append(res_node)

    return res_list


def group(data, max_th, base_th):
    '''
        数据从大到小排序
        依次将序列的最大值加入分组，如果当前最大值小于可用空间，则循环判断依次向后查找能够放入该组的最大值并放入，直到当前组剩余空间小于目前序列的最小值。
    '''
    count = len(data)
    result = []
    res = []
    diff = max_th

    while len(data) > 0:  # 还有数据
        if max_th <= data[0]:
            result.append([data])
            del data[0]
            continue
        if diff == max_th and sum(data) <= max_th:
            result.append(data)
            break
        if diff >= data[0]:  # 剩余空间大于当前最大值，则加入
            res.append(data[0])
            diff = diff - data[0]  # 剩余空间
            del data[0]  # 删除该值
            continue
        else:   # 最大值大于diff
            if diff >= data[-1]:
                index = []
                for i, v in enumerate(data):
                    if diff >= v:
                        res.append(v)
                        diff = diff - v
                        index.append(i)
                    elif diff < data[-1]:
                        break
                data = [v for i, v in enumerate(data) if i not in index]

            result.append(res) # 剩余空间无法装任何数据时，结束该组
            res = []
            diff = max_th

    # 计算距离
    result = averageResult(result, max_th, base_th)
    print('共有%d个数据，划分为%d组' % (count, len(result)))
    print_result(result, max_th)


def minGroupExhanstion(data, max_th, base_th):
    '''
        基于递归方法实现穷举
        暂时不考虑单个数值大于max_th的情况
    '''
    count = len(data)

    big_data_id = [i for i, v in enumerate(data) if v >= max_th]
    single_grps = list() # 单个数据最为一组的数量
    if len(big_data_id) > 0:
        for i in big_data_id[::-1]:
            single_grps.append([data[i]])
            data.pop(i)
    single_grp_num = len(big_data_id) # 单个数据分组的数量

    # ------以下处理不包含大数值-------
    min_space = max_th - sum(data) % max_th # 理论最小loss
    if sum(data) % max_th == 0:
        best_grp = int(sum(data) / max_th)
    else:
        best_grp = int(sum(data) / max_th + 1)

    # best_grp = best_grp + single_grp_num

    init_grps, init_loss = init_groups(data, max_th)
    if len(init_grps) == best_grp: # 已获取最优分组数
        if single_grp_num > 0:
            init_grps = single_grps + init_grps
        result = averageResult(init_grps, max_th, base_th)
        print('共有%d个数据，划分为%d组' % (count, len(result)))
        print_result(result, max_th)

        return 0

    # 对数据由大到小排序
    data = sorted(data, reverse=True)

    root_node = Node() # 创建根节点
    loss = 0
    generate_tree(root_node, data, max_th, min_space, init_loss, loss)  # 创建树

    result_list = [] # 所有组合结果
    group = []
    traverse(root_node, group, result_list, best_grp) # 遍历树

    print('共有%d组结果' % len(result_list))

    if len(result_list) > 0:
        res = result_list[0]
        if single_grp_num > 0:
            res = single_grps + res
        result = averageResult(res, max_th, base_th)
        print('共有%d个数据，划分为%d组' % (count, len(result)))
        print_result(result, max_th)

    else:
        if single_grp_num > 0:
            init_grps = single_grps + init_grps
        result = averageResult(init_grps, max_th, base_th)
        print('共有%d个数据，划分为%d组' % (count, len(result)))
        print_result(result, max_th)
    print('^_^ '*6)


def traverse(node, groups, result, best_grp):
    if node.is_stop:
        return 0
    for i, nd in enumerate(node.subNode):
        # 如果子节点的is_break为True则放弃该子分支
        if i > 0 and node.subNode[i-1].is_stop:
            node.is_stop = True
            break
        if nd.is_break == True:
            continue

        cur_group = groups.copy()
        cur_group.append(nd.grp)

        if len(nd.subNode) == 0:
            result.append(cur_group)
            if len(cur_group) == best_grp:
                node.is_stop = True
                break
            continue
        traverse(nd, cur_group, result, best_grp)


def generate_tree(node, data, max_th, min_space, init_loss, cur_loss):
    # 查找data数据划分第一组的所有组合, res_list是一个列表，列表中每一个元素均是一个Node类型

    if node.stop_generate:
        return 0
    res_list = find_first_grp(data, max_th, init_loss) # 结果列表中每个元素（一个组的组合）中包含当前组的loss值
    for i, res in enumerate(res_list):
        if i > 0 and node.subNode[i - 1].stop_generate:
            node.stop_generate = True
            break
        node.add_subnode_ele(res)
        loss = cur_loss
        cur_data = data.copy()
        loss += res.cur_loss
        # if loss >= init_loss:
        if round(loss, 2) > round(min_space, 2) or round(loss, 2) >= round(init_loss, 2):
            # 放弃该分支
            # node.is_break = True
            node.subNode[-1].is_break = True
            continue
        for v in res.grp:
            cur_data.remove(v)
        if len(cur_data) == 0:
            if round(loss, 2) == round(min_space,2):
                node.stop_generate = True
                break
            continue
        generate_tree(res, cur_data, max_th, min_space, init_loss, loss)


class Node():
    def __init__(self, cur_data=None, loss=0, sub_node=[]):
        self.grp = cur_data
        self.cur_loss = loss
        self.is_break = False
        self.is_stop = False
        self.stop_generate = False
        self.subNode = sub_node

    def add_node(self, sub_nd_list):
        self.subNode = sub_nd_list

    def add_subnode_ele(self, list_ele):
        self.subNode.append(list_ele)


def minGroupFixSearch(data, max_th, base_th):
    '''
        每组必包含的当前数据中的最大值，剩余空间以分组中第二个数进行变换搜索
        得到的并不是最优解，因为只更换了第二个数的迭代搜索
    '''

    # 对数据由大到小排序
    data = sorted(data, reverse=True)
    count = len(data)

    groups = []

    while len(data) > 0 :
        first_id = -1
        grp = []
        if max_th >= data[0]:
            grp.append(data[0])
            diff = max_th - data[0]
            del data[0]
        else:
            groups.append([data[0]])
            del data[0]
            continue
        min_diff= 100
        tmp_grp = []
        index = []
        tmp_best_grp = []
        tmp_best_index = []
        base_diff = diff

        while first_id < len(data) - 1:
            for i, v in enumerate(data):
                if i > first_id and diff >= v:
                    tmp_grp.append(v)
                    index.append(i)
                    diff = diff - v
                elif diff < data[-1]:
                    break
            if diff == 0:
                tmp_best_grp = tmp_grp
                tmp_best_index = index
                break
            if diff < min_diff:
                min_diff = diff
                tmp_best_grp=tmp_grp
                tmp_best_index = index
            first_id = index[0]
            tmp_grp = []
            index = []
            diff = base_diff
        grp = grp + tmp_best_grp
        groups.append(grp)
        for i in tmp_best_index[::-1]:
            del data[i]

    result = averageResult(groups, max_th, base_th)
    print('共有%d个数据，划分为%d组' % (count, len(result)))
    print_result(result, max_th)



def minGroup(data, max_th, base_th):
    '''
    params:
        data: 待分组的数值
        max_th: 每组数值之和的最大值
        base_th: 每组的基础阈值，每组优先填满该阈值
    return:
        groups: 最终分组结果
    method:
        对数据由大到小排序，如果当前分组剩余空间大于一半则依次将列表中最大值放入分组中，\
        当前分组剩余空间小于一半或者剩余空间小于列表最大值值时，则向后搜索合适的数值加入分组，并且寻找dist最小的组合。
    '''
    count = len(data) # 数据总量，用于打印信息
    groups = list()# 最终分组结果
    grp = list() # 组内数值
    diff = max_th # 剩余空间

    # 对数据由大到小排序
    data = sorted(data, reverse=True)

    while len(data) > 0: # 待分组数据大于0且当前分组剩余空间大于带分组数据的最小值，则循环
        if max_th < data[0]:  # 单个数值大于max_th, 则该数值单独分为一组
            groups.append([data[0]])
            diff = max_th  # 初始化
            del data[0]  # 删除该值
            continue
        elif diff == max_th and sum(data) <= max_th: # 剩余数据总和小于等于max_th，则将全部数据加入groups之中， 并结束循环
            groups.append(data)
            break
        elif diff >= data[0] and sum(grp) < max_th/2: # 当前组可用空间大于待分组最大值，则将该最大值划分到该组
            grp.append(data[0])
            diff = diff - data[0]
            del data[0]
        else: # 当前组可用空间大于待分组最大值但当前组数值之和大于max_th/2，或者当前组可用空间小于待分组最大值，则向后搜索合适的数值
            if sum(data) <= diff: # data中剩余元素之和小于当前分组可用空间，则将剩余元素加入当前分组，并结束循环
                for v in data:
                    grp.append(v)
                break

            if diff < data[-1]:
                groups.append(grp)
                diff = max_th  # 开启下一次分组过程
                grp = list()
                continue

            # 在当前组剩余空间内，执行类似于穷举法，记录多种划分结果以及每组划分结果的distance，选择distance最小的一组作为最优组
            init_diff = diff # 当前分组内剩余可用空间
            first_id = -1 # 记录每次搜索第一个数值的位置，下次循环从该位置的下一个位置开始搜索
            min_diff = 100 # 初始化最小可用空间即距离
            best_grp = list() # 最小距离对应的最优分组剩余元素
            best_index = list() # # 最小距离对应的最优分组剩余元素位置索引

            while first_id < (len(data)-1):
                tmp_grp = list()
                tmp_index = list()
                diff = init_diff
                for i, v in enumerate(data):# 循环往后寻找可加入当前分组的元素
                    if i > first_id and diff >= v:
                        tmp_grp.append(v)
                        diff = diff - v # 更新当前分组可用空间
                        tmp_index.append(i) # 记录元素位置索引
                    elif diff < data[-1]: # 无元素可加入时，终止循环
                        break

                # 结束循环，获取一组排列及对应dist
                if diff == 0: # 当前分组已完全填满
                    best_grp = grp + tmp_grp
                    best_index = tmp_index
                    break
                elif diff < min_diff: # 当前循环的diff小于min_diff，更新min_diff
                    min_diff = diff
                    best_grp = grp + tmp_grp
                    best_index = tmp_index
                first_id = tmp_index[0] # 记录当前训练第一个元素位置，下次循环从该位置的下一个位置开始
            # 循环结束，更新最优分组, 将目前最优分组加入结果列表中
            groups.append(best_grp)
            for i in best_index[::-1]: # 将所填元素在列表中删除
                data.pop(i)
            diff = max_th # 开启下一次分组过程
            grp = list()

    if len(grp) > 0:
        groups.append(grp)

    result = averageResult(groups, max_th, base_th)
    print('共有%d个数据，划分为%d组' % (count, len(result)))
    print_result(result, max_th)


def averageResult(groups, max_th, base_th):
    k = 0
    for i, grp in enumerate(groups):
        if sum(grp) < base_th:
            print('The %d group is too small.'%i)
            diff = base_th - sum(grp)
            for v in groups[k]:
                if (v > diff) and (sum(groups[k]) - v) > base_th and (sum(grp) + v <= max_th):
                    groups[i].append(v)
                    groups[k].remove(v)
                    break
            k += 1

    return groups

def print_result(result, max_th):
    for i, grp in enumerate(result):
        dist = max_th - sum(grp)
        print("distance: {:.2f}, {}".format(dist, grp))

def init_groups(data, max_th):
    '''
        数据从大到小排序
        依次将序列的最大值加入分组，如果当前最大值小于可用空间，则循环判断依次向后查找能够放入该组的最大值并放入，直到当前组剩余空间小于目前序列的最小值。
    '''
    result = []
    res = []
    diff = max_th
    loss = 0 # 总loss值

    # 对数据由大到小排序
    sort_data = sorted(data, reverse=True)

    while len(sort_data) > 0:
        if max_th <= sort_data[0]: # 最大值大于max_th，单独作为一组
            res.append(sort_data[0])
            # loss += (max_th-sort_data[0]) # 如果单个数值大于max_th，则不计loss
            del sort_data[0]
            continue
        elif diff == max_th and sum(sort_data) <= max_th: # 剩余数据之和小于max_th，作为组后一组
            result.append(sort_data)
            loss += (max_th - sum(sort_data))
            break
        elif diff >= sort_data[0]:  # 剩余空间大于当前最大值，则加入
            res.append(sort_data[0])
            diff = diff - sort_data[0]  # 剩余空间
            del sort_data[0]  # 删除该值
        else:   # 最大值大于diff
            del_index = []
            if diff >= sort_data[-1]:
                for i, v in enumerate(sort_data):
                    if diff < sort_data[-1]:
                        break
                    if diff >= v:
                        res.append(v)
                        diff = diff - v
                        del_index.append(i)
                sort_data = [v for i, v in enumerate(sort_data) if i not in del_index]

            result.append(res)
            loss += (max_th - sum(res))
            res = []
            diff = max_th

    # 返回分组结果，总loss以及单个数据的组数
    return result, loss


def minGroupDivide(data, max_th, best_grp):
    '''
        数据从小到大排序，依次将最大值放入子组中，如果空间不足则另分一组
        计算每个分组的距离，并按升序进行排序
        将第一个分组的数据按从大到小的顺序排序，循环判断是否能够加入到其他组内，此过程称为分裂
        Ref: https://blog.csdn.net/weixin_33894640/article/details/93742399

        注意：该方法并不能得到最优解，甚至是次优解
    '''
    num = len(data)
    res = []
    result = []

    # 数据从小到大排序，依次将最大值放入子组中，如果空间不足则另分一组
    data = sorted(data)  # 由小到大
    sum_group = 0
    for i, v in enumerate(data):
        sum_group += v
        if sum_group <= max_th:
            res.append(v)
        else:
            result.append(res)
            res = [v]
            sum_group = v
    result.append(res)

    # # 计算距离
    distance = []
    for i, grp in enumerate(result):
        distance.append(max_th - sum(grp))

    if len(result) > best_grp:
        def distSort(grps, dists, id):
            sort_grp = grps[id+1:]
            sort_dist = dists[id+1:]

            dist_sort_index = sorted(range(len(sort_dist)), key=lambda k: sort_dist[k])  # 从小到大
            sort_dist = sorted(sort_dist)
            sort_grp = [sort_grp[i] for i in dist_sort_index]

            new_grps = grps[:id+1] + sort_grp
            new_dist = dists[:id+1] + sort_dist

            return new_grps, new_dist


        def division(groups, distance, start_id):
            # 距离信息从小到大排序
            groups, distance = distSort(groups, distance, start_id)

            first_grp = groups[start_id]
            first_grp = sorted(first_grp, reverse=True) # big -> small
            for i, v in enumerate(first_grp):
                if v <= distance[-1]:
                    for j, grp in enumerate(groups):
                        if j > start_id and v <= distance[j]:
                            grp.append(v)
                            distance[j] -= v
                            first_grp[i] = 0
                            groups, distance = distSort(groups, distance, start_id)
                            break
                else:
                    continue
            groups.pop(start_id)
            groups.insert(start_id, first_grp)

            return groups, distance

        # 分裂过程
        for i, grp in enumerate(result):
            result, distance = division(result, distance, i)
            if str(result[i]).count("0") < len(result):
                break

        new_result = []
        for i, grp in enumerate(result):
            zero_num = Counter(grp)[0]
            if zero_num == len(grp):
                continue
            elif zero_num > 0:
                for j in range(zero_num):
                    grp.remove(0)
                new_result.append(grp)
            else:
                new_result.append(grp)


        # print('#' * 20)
        print('共有%d个数据，划分为%d组' % (num, len(new_result)))
        print_result(new_result, max_th)
    else:
        # 计算距离
        print_result(result, max_th)



def generate_random(a, b, n, isFloat=False):
    import random
    res = []
    for i in range(n):
        if isFloat:
            res.append(round(random.uniform(a, b), 2))
        else:
            res.append(random.randint(a, b))

    return res



if __name__ == '__main__':
    import random
    import time
    import sys

    max_num = int(sys.argv[1])

    data1 = [16.6,16.62,16.7,17.37,13.4,13.4,13.35,13.42,13.35,13.42,13.4,13.4,13.42,13.37,13.35,13.42,13.3,13.38,13.35,13.42,6.67]
    data2 = [22.65,22.07,21.97,7.47,22.2,22.59,22.47,7.55,18.62,18.75,18.75,18.77,18.67,18.44,18.72,18.92,18.37,19.1,18.65,18.7,18.67,\
            18.64,19.85,89.7676,16.1,15.95,15.98,14.87,16.02,14.85,14.95,14.85,14.7,14.95,14.92,14.95,14.8,14.92,16.35,16.27,11.15,\
            18.82,18.62,19,18.77,18.8,18.82,18.8,18.7,18.69,18.7,18.42,18.72,12.57,16.64,16.59,16.6,16.62,16.7,17.37,13.4,13.4,13.35]#,\
            # 13.42,13.35,13.42,13.4,13.4,13.42,13.37,13.35,13.42,13.3,13.38,13.35,13.42,6.67]
    # data = [9, 22, 25, 7, 25, 18, 13, 16, 12, 8, 13, 25, 8, 24, 7, 16, 23]
    # data = [23.95, 14.44, 13.19, 22.75, 24.31, 18.56, 15.96, 10.37, 8.7, 16.83, 9.4, 9.86, 12.53, 24.74, 21.81, 17.55, 13.72]
    # data = [7.32, 21.53, 19.96, 17.05, 11.52, 8.29, 8.54, 15.1, 12.89, 23.69, 21.81, 23.51, 21.29, 14.5, 17.33, 15.32, 19.39]
    # data = [17.65, 18.85, 18.59, 15.87, 10.01, 24.82, 24.93, 15.06, 23.56, 15.94, 21.15, 19.36, 20.8, 14.25, 15.77, 19.18, 7.41]
    # data = [23.19, 17.91, 12.49, 12.44, 12.05, 17.33, 23.1, 17.24, 20.34, 24.07, 13.18, 8.91, 20.36, 19.65, 17.23]
    # data = [13.9, 9.96, 9.98, 22.11, 16.68, 22.47, 10.46, 9.99, 20.16, 16.92, 15.23, 18.8, 15.45, 8.96, 24.34]
    # data = [22.85, 18.08, 17.5, 22.44, 7.01, 24.29, 7.36, 22.55, 18.17, 18.04, 18.51, 18.94, 21.58, 17.45, 20.18]
    # data = [24.67, 20.95, 24.9, 21.15, 14.74, 14.54, 22.71, 13.57, 12.84, 24.36, 23.29, 8.97, 20.46, 14.2, 18.97, 22.85]
    # data = [11.76, 17.23, 25.0, 7.47, 7.71, 23.61, 16.96, 16.38, 15.41, 12.9, 23.64, 21.94, 10.63, 18.95, 12.25, 20.09, 8.89, 7.21]
    # data = [22.28, 17.26, 21.98, 18.16, 7.86, 11.19, 11.22, 20.88, 23.82, 8.01, 22.75, 14.57, 19.94, 12.16, 20.06, 9.46, 12.24, 19.19]
    # data = [15.67, 12.02, 22.01, 11.66, 13.55, 13.83, 24.79, 8.92, 14.83, 9.56, 9.65, 7.37, 21.67, 9.35, 23.21, 16.27, 21.54, 23.56]
    # data = [24.31, 23.92, 19.49, 23.78, 15.05, 14.98, 13.64, 23.73, 23.26, 20.02, 23.37, 19.41, 19.05, 8.11, 19.9, 18.63, 17.82, 16.95]
    # data = generate_random(6, 35, max_num, isFloat=True)
    # data = [22.47, 18.99, 7.5, 22.79, 20.0, 9.54, 14.32, 7.87, 21.13, 9.49, 13.32, 12.76, 18.85, 20.74, 16.54, 18.05, 10.63, 11.63]
    max_value = 70
    base_th = 20

    min_space = 100
    while min_space > 5:
        data = generate_random(3, 35, max_num, isFloat=True)
        min_space = max_value - sum(data) % max_value

    if sum(data) % max_value == 0:
        best_grp = int(sum(data) / max_value)
    else:
        best_grp = int(sum(data) / max_value + 1)
    min_space = max_value - sum(data) % max_value
    print("共有%d个数据,总和为%.2f, 最优组数: %d, 剩余空间: %.2f\n" % (
    len(data), sum(data), best_grp, round(min_space,2)))

    sort_data = sorted(data, reverse=True)  # 由大到小
    print("orignal: {}".format(data))
    # print("sort   : {}".format(sort_data))
    print('------------正确穷举------------')
    t0 = time.time()
    minGroupExhanstion(data, max_value, base_th)
    print('正确穷举 running time is : %f' % (time.time() - t0))

    print('\n------------依次填充-----------')
    t1 = time.time()
    group(sort_data, max_value, base_th)
    print('依次填充 running time is : %f' % (time.time() - t1))

    print('\n------------分裂------------')
    t2 = time.time()
    minGroupDivide(data, max_value,best_grp)
    print('分裂 running time is : %f' % (time.time() - t2))

    print('\n------------搜索一半空间------------')
    t3 = time.time()
    minGroup(data, max_value, base_th)
    print('搜索一半空间 running time is : %f'%(time.time() - t3))

    # print('\n------------method 穷举------------')
    # t4 = time.time()
    # minGroup_exhaustion(data, max_value, base_th, dis)
    # print('Method 4 running time is : %f' % (time.time() - t4))

    # print('\n------------fix搜索------------')
    # t5 = time.time()
    # minGroupFixSearch(data, max_value, base_th)
    # print('fix搜索 running time is : %f' % (time.time() - t5))
    #
    print('\n------------从后往前不完全搜索------------')
    t6 = time.time()
    minGroupFullSearch(data, max_value, base_th, best_grp)
    print('从后往前不完全搜索 running time is : %f' % (time.time() - t6))

