import unittest
from config import test_conf_file
from level.criterion_level_matrix import CriterionLevelMatrix

'''
针对第一份数据准则层的测试用例
测试点:
1. 数据API读取的正确性
2. 验证一致性的函数正确性.
3. 验证迭代过程的正确性.
'''


class CriterionLevelTest(unittest.TestCase):
    m: CriterionLevelMatrix

    def setUp(self):
        self.m = CriterionLevelMatrix()
        # init了group, nodes(值没有初始化), 还有额外的matrix.
        self.assertEqual(self.m.init(test_conf_file), True)

    # 测试数据是否初始化正确.
    def test_read(self):
        self.assertEqual(len(self.m.matrix), 1)
        self.assertEqual(len(self.m.matrix[0]), 4)
        node = self.m.matrix[0][1][2]
        self.assertEqual(node.into_vec(), [0.56, 0.31, 0.13])

    def test_fix_matrix(self):
        self.m.calc_fix_matrix()
        self.assertEqual(len(self.m.matrix), 1)
        self.assertEqual(len(self.m.fix_matrix), 1)
        for i in range(len(self.m.matrix[0])):
            for j in range(len(self.m.fix_matrix[0])):
                name = self.m.fix_matrix[0][i][j].name
                actual_val = self.m.fix_matrix[0][i][j].into_vec()
                theory_val = self.m.data["fix_matrix"][name]
                self.assertAlmostEqual(actual_val[0], theory_val[0], places=4)
                self.assertAlmostEqual(actual_val[1], theory_val[1], places=4)

    def test_check_consistency(self):
        if len(self.m.fix_matrix) == 0:
            self.m.calc_fix_matrix()

        ret = self.m.check_consistency(self.m.matrix[0], self.m.fix_matrix[0])
        self.assertEqual(ret[0], False)
        self.assertAlmostEqual(ret[1], 0.2608, places=4)

    def test_refine_matrix(self):
        if len(self.m.fix_matrix) == 0:
            self.m.calc_fix_matrix()

        self.assertEqual(self.m.calc_refined_matrix(), True)
        self.assertAlmostEqual(self.m.alpha[0], 0.33, places=2)
        ret = self.m.check_consistency(self.m.matrix[0], self.m.refined_matrix[0])
        self.assertEqual(ret[0], True)
        self.assertAlmostEqual(ret[1], 0.09996, places=4)
        for i in range(len(self.m.matrix[0])):
            for j in range(len(self.m.refined_matrix[0])):
                name = self.m.refined_matrix[0][i][j].name
                actual_val = self.m.refined_matrix[0][i][j].into_vec()
                theory_val = self.m.data["refine_matrix"][name]
                self.assertAlmostEqual(actual_val[0], theory_val[0], places=4)
                self.assertAlmostEqual(actual_val[1], theory_val[1], places=4)
