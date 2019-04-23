from typing import List, Mapping, Any, Callable, Tuple, Dict
import toml
from config import data_dir
import os
import json
import abc

'''
所有的直觉模糊度节点基类
'''


class BaseNode:
    __slots__ = ('membership', 'non_membership', 'hesitation', 'name')

    name: str
    membership: float
    non_membership: float
    hesitation: float

    def __init__(self, name, membership=0.5, non_membership=0.3, hesitation=0.2):
        # 隶属度, 非隶属度, 犹豫度.
        self.name = name
        self.membership = membership
        self.non_membership = non_membership
        self.hesitation = hesitation

    def get_node_value_from_conf(self, json_conf: Mapping, case_name: str = 'case1'):
        try:
            return json_conf[case_name][self.name]
        except Exception as e:
            raise RuntimeError("Get Node Value From Conf Error! name = " + self.name + ", Exception = " + e.__str__())

    def __format__(self, format_spec: str) -> str:
        return "{}: ({} {} {})".format(self.name, self.membership, self.non_membership, self.hesitation)

    def into_vec(self):
        return [self.membership, self.non_membership, self.hesitation]


'''
关联节点类型基类
'''


class BaseRelationNode(BaseNode):

    def __init__(self, name, membership=0.5, non_membership=0.3, hesitation=0.2):
        super(BaseRelationNode, self).__init__(name, membership, non_membership, hesitation)


class BaseLevelMatrix:
    # 当前层的配置读取名字.
    name: str

    # 秩, 也就是这个matrix内包含了多少个level node.
    rank: int

    # 存放node的列表.
    nodes: List[BaseNode]

    # 组, 因为一层内的level node可以不是相互两两作用的.
    # 这里只存索引.
    groups: List[List[int]]

    # 存放relationship Node矩阵,即直觉模糊判断矩阵
    # 三层list对照group, 最外层是分组, 一组内需要两个纬度.
    # 这个存放的是原始的值, 不一定满足一致性检测.
    matrix: List[List[List[BaseRelationNode]]]

    # 直觉模糊一致性判断矩阵值，可能不满足一致性检测.
    # 这个矩阵的论域: f[y][x]表示x相对于y这个指标的优秀程度.
    fix_matrix: List[List[List[BaseRelationNode]]]

    # 经过转换后的直觉模糊一致性判断矩阵值, 这个位置存放最后达到一致性的矩阵.
    # 这个矩阵的论域: f[y][x]表示x相对于y这个指标的优秀程度.
    refined_matrix: List[List[List[BaseRelationNode]]]

    # conf, toml的解析对象, 方便随时读取.
    conf: Any

    # data, 数据文件的目录.
    data: Any

    # 保存调整后的修正因子. alpha同时可以表示matrix有没有被初始化ok.
    # 如果一开始还没有matrix, 那么alpha表示
    alpha: float
    delta: float

    # 只有决策层和准则层有
    # 对应下层的哪个group
    group_id: int

    def __init__(self):
        self.rank = 0
        self.nodes = []
        self.groups = []
        self.matrix = []
        self.conf = None
        self.data = None
        self.alpha = -1
        self.delta = 0.01
        self.name = "unknown"
        self.group_id = -1

    # 有个问题！一致性检查过程：(第一次和后面几次检查一致性的两个矩阵不相同，建议函数入口传入两个矩阵参数方便后面调用)
    # step 1 : check_consistency(matrix , fix_matrix)
    #          if 满足一致性 : stop
    #          else : step 2
    # step 2 : calc_define_matrix(alpha-0.01)
    # step 3 : check_consistency(matrix , refine_matrix)
    #          if 满足一致性：stop
    #          else : go to step 2

    # 检查自己的matrix
    def check_consistency(self) -> bool:
        # 一致性检查必须在由原始数据计算得到一致性判断矩阵之后，才能进行
        # if self.alpha < 0:
        # raise RuntimeError("check consistency before construct matrix!")

        # 每次遍历一个group.
        for i in range(len(self.matrix)):
            # 遍历矩阵内元素
            for j in range(len(self.matrix[i])):
                for k in range(len(self.matrix[i][j])):
                    mem = abs(self.fix_matrix[i][j][k].membership - self.matrix[i][j][k].membership)
                    nonmem = abs(self.fix_matrix[i][j][k].non_membership - self.matrix[i][j][k].non_membership)
                    hesi = abs(self.fix_matrix[i][j][k].hesitation - self.matrix[i][j][k].hesitation)
                    D = (mem + nonmem + hesi) / 2 * (len(self.matrix[i]))
        return D < 0.1

    def init(self, conf_file: str) -> bool:

        ret = self._init_conf(conf_file)
        if not ret:
            return ret
        return self._init_nodes_from_conf()

    def _init_conf(self, conf_file: str) -> bool:
        with open(conf_file, "r") as conf_file:
            self.conf = toml.load(conf_file)[self.name]
            with open(os.path.join(data_dir, self.conf["data_file"]), "r") as data_file:
                # TODO 目前默认为case1.
                self.data = json.load(data_file)["case1"]
        return True

    # 这个函数初始化nodes和groups这两个矩阵.
    @abc.abstractmethod
    def _init_nodes_from_conf(self) -> bool:
        for node in self.conf['nodes']:
            name: str = node['dname']
            group_id: str = node['group_id']


    # 这个函数从初始化好的nodes和groups里计算matrix.
    def __construct_matrix(
            self,
            func: Callable[[List[BaseNode], List[List[int]], Any, Any], Tuple[bool, List[List[List[BaseRelationNode]]]]]
            = lambda x, y: (False, [])
    ) -> bool:
        ret = func(self.nodes, self.groups)
        if not ret[0]:
            return False
        self.matrix = ret[1]
        self.alpha = 1
        return True

    # 拟合该层直到符合一致性矩阵判断标准.
    def fix(self):
        # TODO 一个更好的表示是否已经产生数据的方法.
        if len(self.matrix) == 0:
            if not self.calc_matrix():
                raise RuntimeError("In this case, construct fix matrix fail!")

        if len(self.fix_matrix) == 0:
            if not self._calc_fix_matrix():
                raise RuntimeError("In this case, construct fix matrix fail!")

        if len(self.refined_matrix) == 0:
            if not self._calc_refined_matrix():
                raise RuntimeError("In this case, construct refined matrix fail!")

    def calc_matrix(self) -> bool:
        return True

    def _calc_fix_matrix(self) -> bool:
        m1, m2, nm1, nm2 = 1, 1, 1, 1
        # 遍历group
        for i in range(len(self.matrix)):
            # 遍历矩阵内元素
            for j in range(len(self.matrix[i])):
                for k in range(len(self.matrix[i][j])):
<<<<<<< Updated upstream
                    if k == j + 1:
                        self.fix_matrix[i][j][k] = self.matrix[i][j][k]
                    elif k > j + 1:
                        for t in range(j + 1, k - 1):
                            m1 = m1 * pow(self.matrix[i][j][t].membership * self.matrix[i][t][k].membership,
                                          1 / (j - i - 1))
                            m2 = m2 * pow((1 - self.matrix[i][j][t].membership) * (1 - self.matrix[i][t][k].membership),
                                          1 / (j - i - 1))
                            nm1 = nm1 * pow(self.matrix[i][j][t].non_membership * self.matrix[i][t][k].membership,
                                            1 / (j - i - 1))
                            nm2 = nm1 * pow(
                                (1 - self.matrix[i][j][t].non_membership) * (1 - self.matrix[i][t][k].non_membership),
                                1 / (j - i - 1))
                            self.fix_matrix[i][j][k].membership = m1 / (m1 + m2)
                            self.fix_matrix[i][j][k].non_membership = nm1 / (nm1 + nm2)
                            self.fix_matrix[i][j][k].hesitation = (
                                        1 - self.fix_matrix[i][j][k].membership - self.fix_matrix[i][j][
                                    k].non_membership)
=======
                    if k == j+1:
                        self.fix_matrix[i][j][k] = self.matrix[i][j][k]
                    elif k>j+1:
                        for t in range(j+1,k-1):
                            m1 = m1*pow(self.matrix[i][j][t].membership * self.matrix[i][t][k].membership,1/(j-i-1))
                            m2 = m2*pow((1-self.matrix[i][j][t].membership) * (1-self.matrix[i][t][k].membership),1/(j-i-1))
                            nm1 = nm1*pow(self.matrix[i][j][t].non_membership * self.matrix[i][t][k].membership,1/(j-i-1))
                            nm2 = nm1*pow((1-self.matrix[i][j][t].non_membership) * (1-self.matrix[i][t][k].non_membership),1/(j-i-1))
                            self.fix_matrix[i][j][k].membership = m1/(m1+m2)
                            self.fix_matrix[i][j][k].non_membership = nm1/(nm1+nm2)
                            self.fix_matrix[i][j][k].hesitation = (1-self.fix_matrix[i][j][k].membership-self.fix_matrix[i][j][k].non_membership)
>>>>>>> Stashed changes
                    else:
                        self.fix_matrix[i][j][k].membership = 0.5
                        self.fix_matrix[i][j][k].non_membership = 0.5
                        self.fix_matrix[i][j][k].hesitation = 0
        return True

    def _calc_refined_matrix(self) -> bool:
        while 0 <= self.alpha <= 1:
            self.alpha -= 0.01
            for i in range(len(self.matrix)):
                # 遍历矩阵内元素
                for j in range(len(self.matrix[i])):
                    for k in range(len(self.matrix[i][j])):
                        refine_m1 = pow(self.matrix[i][j][k].membership,(1-self.alpha)) * pow(self.fix_matrix[i][j][k].membership , self.alpha)
                        refine_m2 = pow((1-self.matrix[i][j][k].membership),(1-self.alpha)) * pow((1-self.fix_matrix[i][j][k].membership) , self.alpha)
                        refine_nm1 = pow(self.matrix[i][j][k].non_membership,(1-self.alpha)) * pow(self.fix_matrix[i][j][k].non_membership , self.alpha)
                        refine_nm2 = pow((1-self.matrix[i][j][k].non_membership),(1-self.alpha)) * pow((1-self.fix_matrix[i][j][k].non_membership) , self.alpha)
                        refine_m =  refine_m1/(refine_m1+refine_m2)
                        refine_nm = refine_nm1 / (refine_nm1 + refine_nm2)
            # 注意！check_consistency()内容是matrix与fix_matrix的计算，此处应该为matrix与refine_matrix的计算,应作修改
            if self.check_consistency():
                self.consistancy = True
                return True
        return False
