from level.target_level_matrix import TargetLeveLMatrix
from level.criterion_level_matrix import CriterionLevelMatrix
from level.index_level_matrix import IndexLevelMatrix
from typing import Tuple, List
from level.component import RelationNode
from level import util
from functools import reduce


class Framework:
    target_level: TargetLeveLMatrix
    criterion_level: CriterionLevelMatrix
    index_level: IndexLevelMatrix
    final_weight: Tuple[float, float]
    case_name: str

    def __init__(self, conf_file: str, case_name: str):
        self.case_name = case_name
        self.target_level = TargetLeveLMatrix()
        ret = self.target_level.init(conf_file, self.case_name)
        if not ret:
            raise RuntimeError("target level init error!")
        self.criterion_level = CriterionLevelMatrix()
        ret = self.criterion_level.init(conf_file, self.case_name)
        if not ret:
            raise RuntimeError("criterion level init error!")
        self.index_level = IndexLevelMatrix()
        ret = self.index_level.init(conf_file, self.case_name)
        if not ret:
            raise RuntimeError("index level init error!")

    # 求解直觉模糊矩阵的合理值.计算矩阵的权重.
    def build(self):
        # 计算模糊矩阵部分.
        ret = self.criterion_level.fix()
        if not ret:
            raise RuntimeError("criterion level fix error!")
        ret = self.index_level.fix()
        if not ret:
            raise RuntimeError("index level fix error!")

        # 生成最终权重阶段  阶段
        final_weight_set = []
        criterion_nodes_cnt = len(self.criterion_level.nodes)
        for criterion_id in range(criterion_nodes_cnt):
            for index_id in range(len(self.index_level.weight_list[criterion_id])):
                final_weight_set.append(util.weight_mul(
                    self.criterion_level.weight_list[0][criterion_id],
                    self.index_level.weight_list[criterion_id][index_id]
                ))
        self.final_weight = reduce(lambda x, y: util.weight_add(x, y), final_weight_set)

    def after_build(self) -> bool:
        # 各种check是否合法.
        ret = True
        ret &= len(self.criterion_level.nodes) == len(self.criterion_level.weight_list[0])

        return ret

    def __repr__(self):
        def final_result_calc(arg: Tuple[float, float]) -> float:
            return (1 - arg[1]) / (1 + 1 - arg[0] - arg[1])

        def format_matrix(m: List[List[List[RelationNode]]]) -> str:
            _ff = ""
            for matrix in m:
                _f = []
                for i in range(len(matrix)):
                    _line = []
                    for j in range(len(matrix)):
                        _line.append("[{:.4f},{:.4f}]".format(matrix[i][j].membership, matrix[i][j].non_membership))
                    _f.append(" ".join(_line))
                _ff += "\n".join(_f) + "\n--------------------------\n"
            return _ff

        ret = "======{}====== \n".format(self.case_name)
        ret += "Level 0 weight = {} \n".format("{}".format(self.final_weight))
        ret += "Level 1 Origin Matrix = \n{}".format(format_matrix(self.criterion_level.matrix))
        ret += "Level 1 Refined Matrix = \n{}".format(format_matrix(self.criterion_level.refined_matrix))
        ret += "Level 1 Weight = {} \n".format("{}".format(self.criterion_level.weight_list))
        ret += "Level 2 Origin Matrix = \n{}".format(format_matrix(self.index_level.matrix))
        ret += "Level 2 Refined Matrix = \n{}".format(format_matrix(self.index_level.refined_matrix))
        ret += "Level 2 Weight = {} \n".format("{}".format(self.index_level.weight_list))
        ret += "Final Result = {} \n".format(final_result_calc(self.final_weight))

        return ret
