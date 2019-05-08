from typing import List, Any, Callable, Tuple
import toml
from config import data_dir
import os
import json
from functools import reduce

'''
所有的直觉模糊度节点类
'''


class Node:
    name: str
    membership: float
    non_membership: float
    hesitation: float
    # 这三个是原始数据, 在测试数据集合里面是没有的.
    good: int
    bad: int
    unknown: int
    # 表示其在模糊一致性矩阵的当前矩阵内的offset位置
    group_offset: int

    def __init__(self, name, membership=0.5, non_membership=0.3, hesitation=0.2):
        # 隶属度, 非隶属度, 犹豫度.
        self.name = name
        self.membership = membership
        self.non_membership = non_membership
        self.hesitation = hesitation
        self.good = 0
        self.bad = 0
        self.unknown = 0
        self.group_offset = 0

    def __repr__(self) -> str:
        return "{}: ({:.4f} {:.4f} {:.4f})".format(self.name, self.membership, self.non_membership, self.hesitation)

    def from_vec(self, vec: List[float]):
        self.membership, self.non_membership, self.hesitation = vec

    def into_vec(self) -> List[float]:
        return [self.membership, self.non_membership, self.hesitation]

    def from_origin_data_vec(self, vec: List[int]):
        self.good, self.bad, self.unknown = vec
        v_total = reduce(lambda x, y: x + y, vec)
        self.from_vec([vec[0] / v_total, vec[1] / v_total, vec[2] / v_total])


'''
关联节点类型基类
'''


class RelationNode(Node):
    node_a: Node
    node_b: Node

    def __init__(self, node_a: Node, node_b: Node, f: Callable[[Node, Node], List[float]] = None):
        self.node_a = node_a
        self.node_b = node_b
        super(RelationNode, self).__init__(node_a.name + "_" + node_b.name)
        if f is not None:
            self.membership, self.non_membership, self.hesitation = f(node_a, node_b)
        else:
            self.membership, self.non_membership, self.hesitation = self.default_relation_func(node_a, node_b)

    @staticmethod
    def default_relation_func(node_a: Node, node_b: Node) -> List[float]:
        ret = [
            (1 + node_a.membership - node_b.membership) / 2,  # membership
            (1 + node_a.non_membership - node_b.non_membership) / 2,  # non_membership
            0
        ]
        ret[2] = 1 - ret[0] - ret[1]
        return ret


class BaseLevelMatrix:
    # 当前层的配置读取名字.
    name: str

    # 秩, 也就是这个matrix内包含了多少个level node.
    rank: int

    # 存放node的列表.
    nodes: List[Node]

    # 组, 因为一层内的level node可以不是相互两两作用的.
    # 这里只存索引.
    groups: List[List[int]]

    # 存放relationship Node矩阵,即直觉模糊判断矩阵
    # 三层list对照group, 最外层是分组, 一组内需要两个纬度.
    # 这个存放的是原始的值, 不一定满足一致性检测.
    matrix: List[List[List[RelationNode]]]

    # 直觉模糊一致性判断矩阵值，可能不满足一致性检测.
    # 这个矩阵的论域: f[y][x]表示x相对于y这个指标的优秀程度.
    fix_matrix: List[List[List[RelationNode]]]

    # 经过转换后的直觉模糊一致性判断矩阵值, 这个位置存放最后达到一致性的矩阵.
    # 这个矩阵的论域: f[y][x]表示x相对于y这个指标的优秀程度.
    refined_matrix: List[List[List[RelationNode]]]

    #   最终的权重矩阵, 在计算好refined matrix之后就可以被计算出来.
    weight_list: List[List[Tuple[float, float]]]

    # conf, toml的解析对象, 方便随时读取.
    conf: Any

    # data, 数据文件的目录.
    data: Any

    # 保存调整后的修正因子. alpha同时可以表示matrix有没有被初始化ok.
    # 如果一开始还没有matrix, 那么alpha表示
    alphas: List[float]
    delta: float

    def __init__(self):
        self.rank = 0
        self.nodes = []
        self.groups = []
        self.matrix = []
        self.fix_matrix = []
        self.refined_matrix = []
        self.weight_list = []
        self.conf = None
        self.data = None
        self.alpha = []
        self.delta = 0.01
        self.name = "unknown"

    def init(self, conf_file: str, case_name: str) -> bool:

        ret = self.init_conf(conf_file, case_name)
        if not ret:
            return ret
        return self.init_nodes_from_conf()

    def init_conf(self, conf_file: str, case_name: str) -> bool:
        with open(conf_file, "r") as conf_file:
            self.conf = toml.load(conf_file)[self.name]
            with open(os.path.join(data_dir, self.conf["data_file"]), "r") as data_file:
                self.data = json.load(data_file)[case_name]
        return True

    # 这个函数初始化nodes和groups这两个矩阵.
    def init_nodes_from_conf(self) -> bool:

        for node_info in self.conf['nodes']:
            name: str = node_info['dname']
            group_id: int = node_info['group_id']
            node = Node(name)
            # 这里给测试留了逻辑, 允许不读取数据.
            if self.data.get("node_data") is not None:
                stat_data = self.data["node_data"][name]
                node.from_origin_data_vec(stat_data)
            self.nodes.append(node)

            # TODO 目前要求数据的传入是按照index卡着来的, 对异常数据处理不足.
            if len(self.groups) > group_id:
                self.groups[group_id].append(len(self.nodes) - 1)
                node.group_offset = len(self.groups[group_id])
            else:
                self.groups.append([len(self.nodes) - 1])
                node.group_offset = 1
        return True

    # 拟合该层直到符合一致性矩阵判断标准.
    def fix(self):
        # TODO 一个更好的表示是否已经产生数据的方法.
        if len(self.matrix) == 0:
            if not self.calc_matrix():
                raise RuntimeError("In this case, construct fix matrix fail!")

        if len(self.fix_matrix) == 0:
            if not self.calc_fix_matrix():
                raise RuntimeError("In this case, construct fix matrix fail!")

        if len(self.refined_matrix) == 0:
            if not self.calc_refined_matrix():
                raise RuntimeError("In this case, construct refined matrix fail!")

        if not self.calc_weight_list():
            raise RuntimeError("In this case, construct refined matrix fail!")

        return True

    @staticmethod
    # 检查自己的matrix
    def check_consistency(origin_matrix: List[List[RelationNode]], target_matrix: List[List[RelationNode]]) \
            -> (bool, float):
        if len(origin_matrix) == 0:
            raise RuntimeError("check consistency given a empty matrix!")
        if len(origin_matrix) != len(target_matrix):
            raise RuntimeError("origin matrix size != target matrix size, o = {}, t = {}".format(len(origin_matrix),
                                                                                                 len(target_matrix)))
        if len(origin_matrix[0]) != len(target_matrix[0]):
            raise RuntimeError(
                "origin element matrix size != target element matrix size, o = {}, t = {}".format(len(origin_matrix[0]),
                                                                                                  len(target_matrix[
                                                                                                          0])))
        d = 0
        for i in range(len(origin_matrix)):
            # 遍历矩阵内元素
            for j in range(len(origin_matrix[i])):
                mem = abs(target_matrix[i][j].membership - origin_matrix[i][j].membership)
                nonmem = abs(target_matrix[i][j].non_membership - origin_matrix[i][j].non_membership)
                hesi = abs(target_matrix[i][j].hesitation - origin_matrix[i][j].hesitation)
                d += (mem + nonmem + hesi)
        n = len(origin_matrix)
        if n == 2:
            return True, 0
        d = d / (2 * (n - 1) * (n - 2))
        return d < 0.1, d

    # 这个函数从初始化好的nodes和groups里计算matrix.
    def calc_matrix(
            self,
            func: Callable[[Node, Node], List[float]] = None) -> bool:
        for group in self.groups:
            # 初始化该单元matrix的空间.
            n = len(group)
            matrix = [[] for _ in range(n)]
            for i in range(n):
                matrix[i] = [None for _ in range(n)]

            for index_i in range(len(group)):
                for index_j in range(len(group)):
                    relation_node = RelationNode(self.nodes[group[index_i]], self.nodes[group[index_j]], func)
                    matrix[index_i][index_j] = relation_node

            self.matrix.append(matrix)
        return True

    # 计算直觉模糊一致性矩阵
    def calc_fix_matrix(self) -> bool:
        m1, m2, nm1, nm2 = 1, 1, 1, 1
        # 遍历group
        for group_id in range(len(self.matrix)):
            self.fix_matrix.append([])
            # 遍历矩阵内元素
            for i in range(len(self.matrix[group_id])):
                self.fix_matrix[group_id].append([])
                for j in range(len(self.matrix[group_id][i])):
                    assert len(self.fix_matrix[group_id][i]) == j, \
                        "Fix Matrix Size Error! j = %d but len = %d" % (j, len(self.fix_matrix[group_id][i]))
                    self.fix_matrix[group_id][i].append(
                        RelationNode(self.matrix[group_id][i][j].node_a, self.matrix[group_id][i][j].node_b))

                    if j == i + 1:
                        self.fix_matrix[group_id][i][j].from_vec(self.matrix[group_id][i][j].into_vec())
                    elif j > i + 1:
                        for t in range(i + 1, j):
                            m1 *= pow(
                                self.matrix[group_id][i][t].membership * self.matrix[group_id][t][j].membership,
                                1 / (j - i - 1))
                            m2 *= pow((1 - self.matrix[group_id][i][t].membership) * (
                                    1 - self.matrix[group_id][t][j].membership),
                                      1 / (j - i - 1))
                            nm1 *= pow(
                                self.matrix[group_id][i][t].non_membership * self.matrix[group_id][t][j].non_membership,
                                1 / (j - i - 1))
                            nm2 *= pow(
                                (1 - self.matrix[group_id][i][t].non_membership) * (
                                        1 - self.matrix[group_id][t][j].non_membership),
                                1 / (j - i - 1))

                        self.fix_matrix[group_id][i][j].from_vec([
                            m1 / (m1 + m2),
                            nm1 / (nm1 + nm2),
                            1 - m1 / (m1 + m2) - nm1 / (nm1 + nm2)
                        ])
                        m1, m2, nm1, nm2 = 1, 1, 1, 1
                    elif j < i:
                        self.fix_matrix[group_id][i][j].from_vec([
                            self.fix_matrix[group_id][j][i].non_membership,
                            self.fix_matrix[group_id][j][i].membership,
                            1 - self.fix_matrix[group_id][j][i].non_membership - self.fix_matrix[group_id][j][
                                i].membership
                        ])
                    else:
                        self.fix_matrix[group_id][i][j].from_vec([0.5, 0.5, 0])
        return True

    # 计算得到满足一致性的直觉模糊一致性矩阵.
    def calc_refined_matrix(self) -> bool:
        # 每个group有自己的权重.
        self.alpha = [1 for _ in range(len(self.matrix))]

        for group_id in range(len(self.matrix)):
            # 遍历矩阵内元素
            self.refined_matrix.append([])
            while 0 <= self.alpha[group_id] <= 1:
                self.refined_matrix[group_id] = []
                for i in range(len(self.matrix[group_id])):
                    self.refined_matrix[group_id].append([])
                    for j in range(len(self.matrix[group_id][i])):
                        assert len(self.refined_matrix[group_id][i]) == j, \
                            "Refine Matrix Size Error! j = %d but len = %d" % (j, len(self.refined_matrix[group_id][i]))
                        self.refined_matrix[group_id][i].append(
                            RelationNode(self.matrix[group_id][i][j].node_a, self.matrix[group_id][i][j].node_b))

                        refine_m1 = pow(self.matrix[group_id][i][j].membership, (1 - self.alpha[group_id])) \
                                    * pow(self.fix_matrix[group_id][i][j].membership, self.alpha[group_id])
                        refine_m2 = pow((1 - self.matrix[group_id][i][j].membership), (1 - self.alpha[group_id])) \
                                    * pow((1 - self.fix_matrix[group_id][i][j].membership), self.alpha[group_id])
                        refine_nm1 = pow(self.matrix[group_id][i][j].non_membership, (1 - self.alpha[group_id])) \
                                     * pow(self.fix_matrix[group_id][i][j].non_membership, self.alpha[group_id])
                        refine_nm2 = pow((1 - self.matrix[group_id][i][j].non_membership), (1 - self.alpha[group_id])) \
                                     * pow((1 - self.fix_matrix[group_id][i][j].non_membership), self.alpha[group_id])
                        self.refined_matrix[group_id][i][j].from_vec([
                            refine_m1 / (refine_m1 + refine_m2),
                            refine_nm1 / (refine_nm1 + refine_nm2),
                            1 - refine_m1 / (refine_m1 + refine_m2) - refine_nm1 / (refine_nm1 + refine_nm2)
                        ])
                if self.check_consistency(self.matrix[group_id], self.refined_matrix[group_id])[0]:
                    break
                self.alpha[group_id] -= self.delta

            assert self.alpha[group_id] >= 0, "Refine Matrix Error, Can't refine it! alpha = %.4f, group id = %d " \
                                              % (self.alpha[group_id], group_id)

        return True

    def calc_weight_list(self) -> bool:
        s1, s2, m, nm = 0, 0, 0, 0
        M, NM = [], []
        for group_id in range(len(self.refined_matrix)):
            self.weight_list.append([])
            M.append([])
            NM.append([])
            for i in range(len(self.refined_matrix[group_id])):
                # 这一步仅仅是为了扩充list的内存长度.
                self.weight_list[group_id].append((0, 0))
                for j in range(len(self.refined_matrix[group_id][i])):
                    m += self.refined_matrix[group_id][i][j].membership
                    nm += (1 - self.refined_matrix[group_id][i][j].non_membership)
                    s1 += self.refined_matrix[group_id][i][j].membership
                    s2 += (1 - self.refined_matrix[group_id][i][j].non_membership)
                M[group_id].append(m)
                NM[group_id].append(nm)
                nm, m = 0, 0

            for i in range(len(self.weight_list[group_id])):
                self.weight_list[group_id][i] = (M[group_id][i] / s2, 1 - NM[group_id][i] / s1)
            s1, s2, m, nm = 0, 0, 0, 0

        return True
