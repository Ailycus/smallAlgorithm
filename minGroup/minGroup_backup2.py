#!/usr/bin/env python3
# @Time:    2020/12/7 上午10:24
# @Author:  aily
# @File:    minGroup.py
# @Content: 给定一组数据，将其划分为多个小组，要求每组数据之和小于等于max_th，每组数值优先填满基础阈值base_th，求最少分组数及每组的数值。
from collections import Counter

def epoch_group(data, max_th):

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
    # print('init result: {}'.format(init_grp))

    if len(init_grp) == len(data):
        print('数组已经分完')
        return one_groups, one_indexs, one_dist

    # 根据init_grp结果进行算法优化
    cur_base_diff = init_diff
    # diff = init_diff
    min_diff = init_diff
    index = init_index.copy()
    # cur_init_grp = init_grp.copy()
    # tmp_best_grp = []
    # tmp_best_index = []
    while len(init_grp) > 0:
        value = init_grp.pop(-1)  # 拿出最后一个数值
        diff = cur_base_diff + value
        # diff += value
        cur_base_diff = diff
        id = index.pop(-1)

        if sum(data[id + 1:]) >= diff:
            # 当前值后面的数累加和大于等于diff+当前值，才有可能替换当前值
            first_id = id
            # best_flag = False
            while first_id < (len(data) - 1):
                tmp_grp = []
                tmp_index = []
                diff = cur_base_diff
                for i, v in enumerate(data):
                    if i <= first_id:
                        continue
                    elif diff >= v:
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

def minGroup_exhaustion(data, max_th, base_th, best_loss):
    '''
        最优解算法

        当寻找到一个最小dist时，不停止，而是继续寻找所有的dist组合
    '''
    count = len(data)
    all_groups = list()
    all_dists = list()
    groups0 = list()  # 最终分组结果
    # diff = max_th  # 剩余空间

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

    # 每个元素值小于max_th

    data_copy = data.copy()    

    loss = 0
    # while sum(data_copy) > max_th:
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
            dist = max_value - sum(grp)
            print("distance: {:.2f}, {}".format(dist, grp))


def minGroup_test(data, max_th, base_th):
    '''
        尝试从后往前搜素优化算法
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
        # 从大到小依次获取数值放入init_grp中，作为后续优化的基础
        index = []
        for i, v in enumerate(data):
            if init_diff >= v:
                init_grp.append(v)
                init_diff = init_diff - v
                index.append(i)
            elif init_diff < data[-1]:
                break
        # print('init result: {}'.format(init_grp))

        if len(data) == len(init_grp):
            groups.append(init_grp)
            break

        # 如果init_grp的diff为0，则为最优，不进行后续优化搜索
        if init_diff == 0:
            groups.append(init_grp)
            for i in index[::-1]:
                del data[i]
            # print('==已获得最优解，无需优化==')
        else:
            # 当前init_grp并非最优结果，根据此结果进行算法优化
            cur_base_diff = init_diff
            min_diff = init_diff
            init_index = index.copy()
            cur_init_grp = init_grp.copy()
            tmp_best_grp = []
            tmp_best_index = []
            while len(init_grp) > 0:
                value = init_grp.pop(-1) # 拿出最后一个数值
                diff = cur_base_diff + value
                cur_base_diff = diff
                id = index.pop(-1)

                if sum(data[id+1:]) >= diff:
                    # 当前值后面的数累加和大于等于diff+当前值，才有可能替换当前值
                    first_id = id
                    best_flag = False
                    while first_id < (len(data) - 1):
                        tmp_grp = []
                        tmp_index = []
                        diff = cur_base_diff
                        for i, v in enumerate(data):
                            if i <= first_id:
                                continue
                            elif diff >= v:
                                tmp_grp.append(v)
                                tmp_index.append(i)
                                diff = diff - v
                            elif diff < data[-1]:
                                break
                        # 结束循环，获取一组排列及对应dist
                        if diff == 0: # 如果已获取最优值
                            best_flag = True
                            min_diff = diff
                            tmp_best_grp = init_grp + tmp_grp
                            tmp_best_index = index + tmp_index
                            break # 结束本轮
                        elif diff < min_diff: # 结果优于init_diff则保留
                            min_diff = diff
                            tmp_best_grp = init_grp + tmp_grp
                            tmp_best_index = index + tmp_index
                        first_id = tmp_index[0]

                    if best_flag:
                        break
                else:
                    # 无法优化该值，继续往前找
                    continue
            # 优化搜索结束，获取最优解
            if min_diff >= init_diff: #表示init_grp即是最优
                groups.append(cur_init_grp)
                for i in init_index[::-1]:
                    del data[i]
                # print('init_grp即是最优：{}'.format(cur_init_grp))
            else:
                groups.append(tmp_best_grp)
                for i in tmp_best_index[::-1]:
                    del data[i]
                # print('优化搜索结果：{}'.format(tmp_best_grp))

    print('共有%d个数据，划分为%d组' % (count, len(groups)))
    for i, grp in enumerate(groups):
        dist = max_value - sum(grp)
        print("distance: {:.2f}, {}".format(dist, grp))






def minGroup_test2(data, max_th, base_th):
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
                if i <= first_id:
                    continue
                elif diff >= v:
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
    print('共有%d个数据，划分为%d组' % (count, len(groups)))
    for i, grp in enumerate(groups):
        dist = max_value - sum(grp)
        print("distance: {:.2f}, {}".format(dist, grp))


# 第一版v1.0
def find_first_grp(data, max_th):
    '''
        功能：从数据列表中，找到第一个分组的所有组合
        以列表的形式范围，列表中每一个元素都是一个节点Node类型
    '''
    # 剩余数据之和小于max_th,则全部作为一个group
    res_list = []
    if sum(data) <= max_th:
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
                if diff <= min_diff or diff < data[-1]:
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
    res_list = []
    for i, grp in enumerate(groups):
        res_node = Node(grp, dist[i], [])
        res_list.append(res_node)

    return res_list

# 第一版v1.0
def minGroupExhanstion(data, max_th, base_th):
    count = len(data)
    all_groups = list()
    all_dists = list()
    groups0 = list()  # 最终分组结果
    # diff = max_th  # 剩余空间

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

    root_node = Node()
    generate_tree(root_node, data, max_th)

    result_list = []
    group = []
    traverse(root_node, group, result_list)

    num_list = [len(res) for res in result_list]
    best_num = min(num_list)
    print('共有%d个最优解, 总个数：%d'%(Counter(num_list)[best_num], len(num_list)))

    i = 0
    for res in result_list:
        if i <=3:
            if len(res) <= best_num:
                print('共有%d个数据，划分为%d组' % (count, len(res)))
                for i, grp in enumerate(res):
                    dist = max_value - sum(grp)
                    print("distance: {:.2f}, {}".format(dist, grp))
                i += 1


    print('-_- '*6)


# 第一版v1.0
def traverse(node, groups, result):
    # 遍历树，获取所有组合结果
    # if len(node.grp) == 0:
    #     groups = list()

    for nd in node.subNode:
        cur_group = groups.copy()
        cur_group.append(nd.grp)
        if len(nd.subNode) == 0:
            result.append(cur_group)
            continue
        traverse(nd, cur_group, result)

# 第一版v1.0
def generate_tree(node, data, max_th):
    # 查找data数据划分第一组的所有组合, res_list是一个列表，列表中每一个元素均是一个Node类型
    res_list = find_first_grp(data, max_th)
    node.add_node(res_list) # 将结果加入到当前节点
    for res in res_list:
        cur_data = data.copy()
        for v in res.grp:
            cur_data.remove(v)
        if len(cur_data) == 0:
            continue
        generate_tree(res, cur_data, max_th)

# 第一版v1.0
class Node():
    def __init__(self, cur_list=[], loss=0, sub_node=[]):
        self.grp = cur_list
        self.cur_loss = loss
        self.subNode = sub_node

    def add_node(self, nodes):
        self.subNode = nodes


# ------------------第二版递归穷举---------------------
def init_groups(data, max_th):
    '''
        数据从大到小排序
        依次将序列的最大值加入分组，如果当前最大值小于可用空间，则循环判断依次向后查找能够放入该组的最大值并放入，直到当前组剩余空间小于目前序列的最小值。
    '''
    result = []
    res = []
    diff = max_th
    loss = 0 # 总loss值
    # del_index = []
    # 对数据由大到小排序
    sort_data = sorted(data, reverse=True)

    # while len(sort_data) > 0 and diff >= sort_data[-1]:
    while len(sort_data) > 0:
        if max_th <= sort_data[0]: # 最大值大于max_th，单独作为一组
            res.append(sort_data[0])
            loss += (max_th-sort_data[0])
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
            # while diff >= sort_data[-1]:
            if diff >= sort_data[-1]:
                for i, v in enumerate(sort_data):
                    if diff < sort_data[-1]:
                        break
                    if diff >= v:
                        res.append(v)
                        diff = diff - v
                        del_index.append(i)
                        # del sort_data[i]
                        # break
                sort_data = [v for i, v in enumerate(sort_data) if i not in del_index]

            result.append(res)
            loss += (max_th - sum(res))
            res = []
            diff = max_th

    # 返回该组的距离值
    return result, loss

def minGroupExhanstion_v2(data, max_th, base_th, best_grp, min_space):
    count = len(data)
    groups0 = list()  # 最终分组结果

    init_grps, init_loss = init_groups(data, max_th)
    if len(init_grps) == best_grp: # 已获取最优分组数
        result = averageResult(init_grps, max_th, base_th)
        print('共有%d个数据，划分为%d组' % (count, len(result)))
        for i, grp in enumerate(result):
            dist = max_value - sum(grp)
            print("distance: {:.2f}, {}".format(dist, grp))

        return 0

    # 对数据由大到小排序
    data = sorted(data, reverse=True)

    # 如果第一个数值大于max_th，则单独分为一组
    while len(data) > 0:
        if data[0] >= max_th:
            groups0.append([data[0]])
            del data[0]
        else:
            break

    root_node = Node_v2() # 创建根节点
    # generate_tree(root_node, data, max_th) # 创建树
    loss = 0
    generate_tree_v2(root_node, data, max_th, min_space, init_loss, loss)  # 创建树

    result_list = [] # 所有组合结果
    group = []
    traverse_v2(root_node, group, result_list, best_grp) # 遍历树

    print('共有%d组结果'%len(result_list))
    num_list = [len(res) for res in result_list]
    best_num = min(num_list) if len(num_list) > 0 else len(init_grps) # 最优分组数
    

    if len(result_list) > 0:
        res = result_list[0]
        print('共有%d个数据，划分为%d组' % (count, len(res)))
        for i, grp in enumerate(res):
            dist = max_value - sum(grp)
            print("distance: {:.2f}, {}".format(dist, grp))
    else:
        print('共有%d个数据，划分为%d组' % (count, len(init_grps)))
        for i, grp in enumerate(init_grps):
            dist = max_value - sum(grp)
            print("distance: {:.2f}, {}".format(dist, grp))

    print('-_- '*6)


def traverse_v2(node, groups, result, best_grp):
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
        traverse_v2(nd, cur_group, result, best_grp)


def generate_tree_v2(node, data, max_th, min_space, init_loss, cur_loss):
    # 查找data数据划分第一组的所有组合, res_list是一个列表，列表中每一个元素均是一个Node类型
    # res_list = find_first_grp(data, max_th)
    if node.stop_generate:
        return 0
    res_list = find_first_grp_v2(data, max_th, init_loss) # 结果列表中每个元素（一个组的组合）中包含当前组的loss值
    for i, res in enumerate(res_list):
        if i > 0 and node.subNode[i - 1].stop_generate:
            node.stop_generate = True
            break
        node.add_subnode_ele(res)
        loss = cur_loss
        cur_data = data.copy()
        loss += res.cur_loss
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
        generate_tree_v2(res, cur_data, max_th, min_space, init_loss, loss)

class Node_v2():
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

def find_first_grp_v2(data, max_th, init_loss):
    '''
        功能：从数据列表中，找到第一个分组的所有组合
        以列表的形式范围，列表中每一个元素都是一个节点Node类型

        如果当前组的diff大于init_loss（即当前一个组的loss就大于全部loss，则不保留该组）

        init_loss: 初始化分组总loss值
    '''
    # 剩余数据之和小于max_th,则全部作为一个group
    res_list = []
    if sum(data) < max_th:
        res_node = Node_v2(data, (max_th-sum(data)), [])
        res_list.append(res_node)
        return  res_list

    # 从大到小依次获取数值放入init_grp中，作为后续优化的基础
    init_diff = max_th
    init_grp = []
    init_index = []

    groups = []
    indexs = []
    dist = []
    loss = 0
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
                # if diff <= min_diff or diff < data[-1]:
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
        res_node = Node_v2(grp, dist[i], [])
        res_list.append(res_node)

    return res_list

#------------------------------------------
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

    while len(data) > 0 and diff >= data[-1]: # 待分组数据大于0且当前分组剩余空间大于带分组数据的最小值，则循环
        if max_th < data[0]:  # 单个数值大于max_th, 则该数值单独分为一组
            groups.append([data[0]])
            diff = max_th  # 初始化
            del data[0]  # 删除该值
        elif diff >= data[0] and sum(grp) < max_th/2: # 当前组可用空间大于待分组最大值，则将该最大值划分到该组
            grp.append(data[0])
            diff = diff - data[0]
            del data[0]
        else: # 当前组可用空间大于待分组最大值但当前组数值之和大于max_th/2，或者当前组可用空间小于待分组最大值，则向后搜索合适的数值
            if sum(data) <= diff: # data中剩余元素之和小于当前分组可用空间，则将剩余元素加入当前分组，并结束循环
                for v in data:
                    grp.append(v)
                # groups.append(grp)
                break

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

    # 计算距离
    distance = []
    for i, grp in enumerate(groups):
        distance.append(max_th - sum(grp))

    print('共有%d个数据，划分为%d组' % (count, len(groups)))
    for i, grp in enumerate(groups):
        print("distance: {:.2f}, {}".format(distance[i], grp))

    # 处理distance大于(max_th-base_th)的分组
    # k = 0
    # for i, dis in enumerate(distance):
    #     if dis > (max_th - base_th):
    #         print("The %d group is too small."%i)
    #         diff = base_th - sum(groups[i])
    #         for v in groups[k]:
    #             if v > diff and (distance[0] + v) < (max_th - base_th):
    #                 groups[i].append(v)
    #                 groups[k].remove(v)
    #                 break
    #         k += 1
    #
    # if k > 0:
    #     print('共有%d个数据，划分为%d组' % (count, len(groups)))
    #     for i, grp in enumerate(groups):
    #         dist = max_value - sum(grp)
    #         print("distance: {:.2f}, {}".format(dist, grp))




def minGroup_1(data, max_th, base_th):
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

    while len(data)>0:
        while len(data) > 0 and diff >= data[-1]: # 待分组数据大于0且当前分组剩余空间大于带分组数据的最小值，则循环
            if max_th < data[0]:  # 单个数值大于max_th, 则该数值单独分为一组
                groups.append([data[0]])
                diff = max_th  # 初始化
                del data[0]  # 删除该值
                break  # 开始新一组划分
            elif diff >= data[0] and sum(grp) < max_th/2: # 当前组可用空间大于待分组最大值，则将该最大值划分到该组
                grp.append(data[0])
                diff = diff - data[0]
                del data[0]
            else: # 当前组可用空间大于待分组最大值但当前组数值之和大于max_th/2，或者当前组可用空间小于待分组最大值，则向后搜索合适的数值
                # 在当前组剩余空间内，执行类似于穷举法，记录多种划分结果以及每组划分结果的distance，选择distance最小的一组作为最优组。
                tmp_dist = list() # 当前组剩余空间内：分组数值
                tmp_grps = list() # 当前组剩余空间内：多种分组的结果
                del_data_list = list() # 当前组剩余空间内：记录多种分组内数值的索引值，用于删除data的数值
                tmp_diff = diff # 当前分组内剩余可用空间
                first_id = -1
                best_flag = False

                while first_id < (len(data)-1):
                    tmp_grp = list()
                    del_data = list()
                    best_flag = False # 是否找到最优分组的标志
                    data_copy = data.copy()
                    for i, v in enumerate(data_copy):
                        if i > first_id and diff >= v:
                            tmp_grp.append(v)
                            diff = diff - v
                            del_data.append(i)
                        elif diff < data[-1]:
                            break

                        # if i <= first_id:
                        #     continue
                        # elif diff >= v:
                        #     tmp_grp.append(v)
                        #     diff = diff - v
                        #     del_data.append(i)
                        # elif diff < data[-1] or diff < 1:
                        #     break
                    if diff <= 1: # 表示最优分组的处理
                        best_flag = True
                        groups.append(grp+tmp_grp)
                        grp = []
                        diff = max_th
                        for v in del_data[::-1]:
                            data.pop(v)
                        break
                    else: # 记录多种划分结果
                        tmp_grps.append(tmp_grp)
                        tmp_dist.append(tmp_diff - sum(tmp_grp))
                        diff = tmp_diff
                        del_data_list.append(del_data)
                        first_id = del_data[0]
                if len(tmp_dist) > 0 and best_flag==False:
                    dist_sort_index = sorted(range(len(tmp_dist)), key=lambda k: tmp_dist[k])  # 从小到大
                    groups.append(grp+(tmp_grps[dist_sort_index[0]]))
                    grp = list()
                    diff = max_th
                    for v in del_data_list[dist_sort_index[0]][::-1]:
                        data.pop(v)
        if len(grp) > 0:
            groups.append(grp)
            grp = list()
            diff = max_th
    # 计算距离
    distance = []
    for i, grp in enumerate(groups):
        distance.append(max_th - sum(grp))

    print('共有%d个数据，划分为%d组' % (count, len(groups)))
    for i, grp in enumerate(groups):
        print("distance: {:.2f}, {}".format(distance[i], grp))

    # 处理distance大于(max_th-base_th)的分组
    k = 0
    for i, dis in enumerate(distance):
        if dis > (max_th - base_th):
            print("The %d group is too small."%i)
            diff = base_th - sum(groups[i])
            for v in groups[k]:
                if v > diff and (distance[0] + v) < (max_th - base_th):
                    groups[i].append(v)
                    groups[k].remove(v)
                    break
            k += 1

    if k==0:
        print('The optimal grouping has been obtained.')
    else:
        print('共有%d个数据，划分为%d组' % (count, len(groups)))
        for i, grp in enumerate(groups):
            dist = max_value - sum(grp)
            print("distance: {:.2f}, {}".format(dist, grp))




def group(data, max_th, base_th, best_grp):
    '''
        数据从大到小排序
        依次将序列的最大值加入分组，如果当前最大值小于可用空间，则循环判断依次向后查找能够放入该组的最大值并放入，直到当前组剩余空间小于目前序列的最小值。
        对相邻组的数据进行优化（替换），目的是让距离小的值尽量更小
        此方法存在问题，即相邻分组之间优化的思路完全是考虑到个别情况而处理的
    '''
    count = len(data)
    result = []
    res = []
    diff = max_th

    while len(data) > 0:  # 还有数据
        while len(data) > 0 and diff >= data[-1]:
            if diff >= data[0]:  # 剩余空间大于当前最大值，则加入
                now_value = data[0]
                res.append(now_value)
                diff = diff - now_value  # 剩余空间
                del data[0]  # 删除该值
                continue
            else:   # 最大值大于diff
                while diff >= data[-1]:
                    index = -1
                    for i, v in enumerate(data):
                        if diff >= v:
                            res.append(v)
                            index = i
                            break
                    diff = diff - data[index]
                    del data[index]
                result.append(res)
                res = []
                diff = max_th
        result.append(res)  # 剩余空间无法装任何数据时，结束该组
        res = []
        diff = max_th
        continue

    # 计算距离
    distance = []
    for i, grp in enumerate(result):
        distance.append(max_th - sum(grp))

    print('共有%d个数据，划分为%d组' % (count, len(result)))
    for i, grp in enumerate(result):
        print("distance: {:.2f}, {}".format(distance[i], grp))

    '''
    # 当前分组数量为最优分组，则不再进行其他优化
    if len(result) == best_grp:
        # 最后一组的值小于首重，则进行处理
        if sum(result[-1]) < base_th:
            print('最后一组小于首重')
            diff = max_th - distance[-1]
            for i, grp in enumerate(result):
                if diff > grp[0]:
                    result[-1].append(grp[0])
                    del grp[0]
                    break

        print('共有%d个数据，划分为%d组' % (count, len(result)))
        for i, grp in enumerate(result):
            print("distance: {:.2f}, {}".format((max_th - sum(grp)), grp))
        # print('-----result is best, over.-----')
    
    else:
        new_result = []
        # 判断是否有分组的distance为0,有则去除
        good_index = [i for i, v in enumerate(distance) if v==0]

        # 保留分组内dist为0的组
        for i in good_index[::-1]:
            new_result.append(result[i])
            del result[i]
            del distance[i]

        #距离信息从小到大排序依次查找可以替换的数据，目的是优先将距离小的组尽量使距离达到最小
        while True:
            # 距离信息从小到大排序
            distance_sort_index = sorted(range(len(distance)), key=lambda k: distance[k])  # 从小到大
            distance_sort = sorted(distance)
            result_sort = [result[i] for i in distance_sort_index]

            # 相邻分组内数据替换
            dis = distance_sort[0]
            for k, v in enumerate(result_sort[1]):
                for j, cur in enumerate(result_sort[0]):
                    if dis == v:# add one
                        result_sort[0].append(v)
                        result_sort[1].remove(v)
                        new_result.append(result_sort[0])
                        del result_sort[0]
                        if len(result_sort[0]) == 0:
                            del result_sort[0]
                        break
                    if cur + dis == v: # replace one
                        result_sort[0].append(v)
                        result_sort[1].remove(v)
                        result_sort[1].append(cur)
                        result_sort[0].remove(cur)
                        new_result.append(result_sort[0])
                        del result_sort[0]
                        if len(result_sort[0]) == 0:
                            del result_sort[0]
                        break

            if len(result_sort) == 0:
                break

        # 最后一组的值小于首重，则进行处理
        if sum(new_result[-1]) < base_th:
            print('最后一组小于首重')
            diff = max_th - distance[-1]
            for i, grp in enumerate(new_result):
                if diff > grp[0]:
                    new_result[-1].append(grp[0])
                    del grp[0]
                    break

        # print('-' * 20)
        # print('共有%d个数据，划分为%d组' % (count, len(new_result)))
        # for i, grp in enumerate(new_result):
        #     print("distance: {:.2f}, {}".format((max_value - sum(grp)), grp))
    '''

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

    # 计算距离
    distance = []
    for i, grp in enumerate(result):
        distance.append(max_th - sum(grp))

    # print('共有%d个数据，划分为%d组' % (num, len(result)))
    # for i, grp in enumerate(result):
    #     print("distance: {:.2f}, {}".format(distance[i], grp))

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
        for i, grp in enumerate(new_result):
            print("distance: {:.2f}, {}".format((max_th - sum(grp)), grp))
        return sum([max_th-sum(grp) for grp in new_result])
    else:
        # 计算距离
        distance = []
        for i, grp in enumerate(result):
            distance.append(max_th - sum(grp))

        print('共有%d个数据，划分为%d组' % (num, len(result)))
        for i, grp in enumerate(result):
            print("distance: {:.2f}, {}".format(distance[i], grp))

        return sum(distance)

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

    # data = [17.78,17.89,17.95,17.8,17.89,18.01,18.42,7.65,17.91,17.84,17.9,18.06,17.97,18.03,17.93]
    # data = [17.78,13.89,12.95,7.8,17.89,8.01,5.42,7.65,27.91,17.84,9.9,18.06,7.97,18.03,17.93]
    # data = [26, 7, 3, 19, 1, 10, 16, 12, 8, 27, 21, 2, 15, 17, 25]
    # data = [6, 28, 13, 10, 2, 8, 27, 3, 23, 5, 22, 1, 26, 15, 20, 11, 17]
    # data = [15, 16, 18, 28, 25, 11, 17, 20, 10, 21, 24, 12, 27, 13, 23]
    # data = [17, 21, 8, 26, 23, 12, 27, 28, 2, 22, 9, 11, 20, 13, 3, 25, 10]
    # data = [7, 17, 21, 26, 29, 27, 9, 15, 1, 23, 8, 6, 24, 16, 20, 11, 19]
    # data = [15.08, 7.45, 13.69, 9.25, 17.77, 15.47, 10.25, 18.66, 19.51, 13.02, 16.72, 14.33, 14.94, 8.59, 16.22]
    # data = [10, 10, 18, 11, 18, 11, 15, 18, 10, 15, 10, 15, 19, 8, 19]
    data1 = [16.6,16.62,16.7,17.37,13.4,13.4,13.35,13.42,13.35,13.42,13.4,13.4,13.42,13.37,13.35,13.42,13.3,13.38,13.35,13.42,6.67]
    # data = [22.65,22.07,21.97,7.47,22.2,22.59,22.47,7.55,18.62,18.75,18.75,18.77,18.67,18.44,18.72,18.92,18.37,19.1,18.65,18.7,18.67,\
    #         18.64,19.85,89.7676,16.1,15.95,15.98,14.87,16.02,14.85,14.95,14.85,14.7,14.95,14.92,14.95,14.8,14.92,16.35,16.27,11.15,\
    #         18.82,18.62,19,18.77,18.8,18.82,18.8,18.7,18.69,18.7,18.42,18.72,12.57,16.64,16.59]
    data2 = [22.65,22.07,21.97,7.47,22.2,22.59,22.47,7.55,18.62,18.75,18.75,18.77,18.67,18.44,18.72,18.92,18.37,19.1,18.65,18.7,18.67,\
            18.64,19.85,89.7676,16.1,15.95,15.98,14.87,16.02,14.85,14.95,14.85,14.7,14.95,14.92,14.95,14.8,14.92,16.35,16.27,11.15,\
            18.82,18.62,19,18.77,18.8,18.82,18.8,18.7,18.69,18.7,18.42,18.72,12.57,16.64,16.59,16.6,16.62,16.7,17.37,13.4,13.4,13.35]#,\
            # 13.42,13.35,13.42,13.4,13.4,13.42,13.37,13.35,13.42,13.3,13.38,13.35,13.42,6.67]
    # data = [9, 22, 25, 7, 25, 18, 13, 16, 12, 8, 13, 25, 8, 24, 7, 16, 23]
    # data = [23.95, 14.44, 13.19, 22.75, 24.31, 18.56, 15.96, 10.37, 8.7, 16.83, 9.4, 9.86, 12.53, 24.74, 21.81, 17.55, 13.72]
    # data = generate_random(7, 25, max_num, isFloat=True)
    # data = [7.32, 21.53, 19.96, 17.05, 11.52, 8.29, 8.54, 15.1, 12.89, 23.69, 21.81, 23.51, 21.29, 14.5, 17.33, 15.32, 19.39]
    # data = [17.65, 18.85, 18.59, 15.87, 10.01, 24.82, 24.93, 15.06, 23.56, 15.94, 21.15, 19.36, 20.8, 14.25, 15.77, 19.18, 7.41]
    # data = [23.19, 17.91, 12.49, 12.44, 12.05, 17.33, 23.1, 17.24, 20.34, 24.07, 13.18, 8.91, 20.36, 19.65, 17.23]
    # data = [24.31, 23.92, 19.49, 23.78, 15.05, 14.98, 13.64, 23.73, 23.26, 20.02, 23.37, 19.41, 19.05, 8.11, 19.9, 18.63, 17.82, 16.95]
    # data = [15.67, 12.02, 22.01, 11.66, 13.55, 13.83, 24.79, 8.92, 14.83, 9.56, 9.65, 7.37, 21.67, 9.35, 23.21, 16.27, 21.54, 23.56]
    max_value = 70
    base_th = 20

    min_space = 100
    while min_space > 5:
        data = generate_random(7, 25, max_num, isFloat=True)
        min_space = max_value - sum(data) % max_value

    if sum(data) % max_value == 0:
        best_grp = int(sum(data) / max_value)
    else:
        best_grp = int(sum(data) / max_value + 1)
    # min_space = max_value - sum(data) % max_value
    print("共有%d个数据,总和为%.2f, 最优组数: %d, 剩余空间: %.2f\n" % (
    len(data), sum(data), best_grp, min_space))

    sort_data = sorted(data, reverse=True)  # 由大到小
    print("orignal: {}".format(data))
    # print("sort   : {}".format(sort_data))

    print('\n------------method 正确穷举------------')
    t0 = time.time()
    minGroupExhanstion_v2(data, max_value, base_th, best_grp, min_space)
    print('Method 正确穷举 running time is : %f' % (time.time() - t0))

    # print('------------method 从大到小依次填充，优化还有问题------------')
    # t1 = time.time()
    # group(sort_data, max_value, base_th, best_grp)
    # print('Method 1 running time is : %f' % (time.time() - t1))

    print('\n------------method 分裂------------')
    t2 = time.time()
    dis = minGroupDivide(data, max_value,best_grp)
    print('Method 2 running time is : %f' % (time.time() - t2))

    print('\n------------method 搜索一半空间------------')
    t3 = time.time()
    minGroup(data, max_value, base_th)
    print('Method 3 running time is : %f'%(time.time() - t3))

    print('\n------------method 穷举------------')
    t4 = time.time()
    minGroup_exhaustion(data, max_value, base_th, dis)
    print('Method 4 running time is : %f' % (time.time() - t4))

    # print('\n------------method test2------------')
    # t5 = time.time()
    # minGroup_test2(data, max_value, base_th)
    # print('Method test2 running time is : %f' % (time.time() - t5))
    #
    print('\n------------method 从后往前不完全搜索------------')
    t6 = time.time()
    minGroup_test(data, max_value, base_th)
    print('Method exhaustion running time is : %f' % (time.time() - t6))

    # print('over')
