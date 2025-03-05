"""
FiberTree 全面测试脚本 - 修复版
这个脚本测试 FiberTree 的所有关键功能和性能
"""

import os
import time
import random
import tempfile
import unittest
import shutil
from typing import List, Dict, Any

# 导入 FiberTree 相关类（假设优化版本已在 main.py 中实现）
from main import Move, Fiber, FiberTree, create_tree, load_tree

class FiberTreeTest(unittest.TestCase):
    """FiberTree 的全面测试套件"""
    
    def setUp(self):
        # 创建测试目录
        self.test_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.test_dir, "test_db.sqlite")
        
        # 创建一个内存树和一个SQLite树用于测试
        self.memory_tree = create_tree()
        self.sqlite_tree = create_tree(storage_type='sqlite', db_path=self.db_path)
    
    def tearDown(self):
        # 清理测试资源
        shutil.rmtree(self.test_dir)
    
    def _add_test_paths(self, tree: FiberTree, count: int = 10) -> None:
        """向树中添加测试路径"""
        for i in range(count):
            tree.start_path()
            # 添加1-5个随机移动
            moves_count = random.randint(1, 5)
            for j in range(moves_count):
                move_value = random.choice(["A", "B", "C", "D", "E"])
                tree.add_move(Move(move_value))
            
            # 记录随机结果
            outcome = random.choice(['win', 'loss', 'draw'])
            tree.record_outcome(outcome)
            tree.end_path()
    
    def test_basic_functionality(self):
        """测试基本功能：创建、添加、统计"""
        # 测试内存树
        self._test_basic_on_tree(self.memory_tree, "Memory Tree")
        
        # 测试SQLite树
        self._test_basic_on_tree(self.sqlite_tree, "SQLite Tree")
    
    def _test_basic_on_tree(self, tree: FiberTree, tree_name: str):
        """在指定的树上测试基本功能"""
        # 1. 添加路径并记录结果
        tree.start_path()
        tree.add_move(Move("A"))
        tree.add_move(Move("B"))
        tree.add_move(Move("C"))
        tree.record_outcome('win')
        tree.end_path()
        
        # 2. 验证可以再次添加相同路径
        tree.start_path()
        tree.add_move(Move("A"))
        tree.add_move(Move("B"))
        tree.add_move(Move("C"))
        tree.record_outcome('win')
        tree.end_path()
        
        # 3. 添加一个分支路径
        tree.start_path()
        tree.add_move(Move("A"))
        tree.add_move(Move("B"))
        tree.add_move(Move("D"))  # 不同的第三步
        tree.record_outcome('loss')
        tree.end_path()
        
        # 4. 获取统计信息
        # 检查第一个路径的统计
        path1_id = tree.find_path([Move("A"), Move("B"), Move("C")])
        stats1 = tree.get_statistics(path1_id)
        self.assertEqual(stats1['visit_count'], 2, f"{tree_name}: 第一个路径应该有2次访问")
        self.assertEqual(stats1['win_count'], 2, f"{tree_name}: 第一个路径应该有2次胜利")
        
        # 检查第二个路径的统计
        path2_id = tree.find_path([Move("A"), Move("B"), Move("D")])
        stats2 = tree.get_statistics(path2_id)
        self.assertEqual(stats2['visit_count'], 1, f"{tree_name}: 第二个路径应该有1次访问")
        self.assertEqual(stats2['loss_count'], 1, f"{tree_name}: 第二个路径应该有1次失败")
        
        # 检查公共前缀 A->B 的统计
        common_id = tree.find_path([Move("A"), Move("B")])
        common_stats = tree.get_statistics(common_id)
        self.assertEqual(common_stats['visit_count'], 3, f"{tree_name}: 共同前缀应该有3次访问")
        
        # 5. 测试批量添加移动
        tree.start_path()
        moves = [Move("X"), Move("Y"), Move("Z")]
        result = tree.add_moves(moves)
        self.assertTrue(result, f"{tree_name}: 批量添加移动应该成功")
        tree.record_outcome('win')
        tree.end_path()
        
        # 验证批量添加的路径
        path3_id = tree.find_path(moves)
        self.assertIsNotNone(path3_id, f"{tree_name}: 应该能找到批量添加的路径")
    
    def test_advanced_features(self):
        """测试高级功能：最佳后续移动、频率分析、热图等"""
        # 先添加一些测试数据
        tree = self.memory_tree  # 使用内存树测试高级功能
        
        # 添加一些有规律的路径
        # 路径 A->B 有高胜率
        for _ in range(10):
            tree.start_path()
            tree.add_move(Move("A"))
            tree.add_move(Move("B"))
            tree.record_outcome('win')
            tree.end_path()
        
        # 路径 A->C 有低胜率
        for _ in range(10):
            tree.start_path()
            tree.add_move(Move("A"))
            tree.add_move(Move("C"))
            # 8次失败，2次胜利
            outcome = 'win' if random.random() < 0.2 else 'loss'
            tree.record_outcome(outcome)
            tree.end_path()
        
        # 1. 测试获取最佳后续移动
        best_moves = tree.get_best_continuation([Move("A")], top_n=2)
        self.assertEqual(len(best_moves), 2, "应该返回两个后续移动")
        
        # 验证排序 - 胜率最高的应该排在前面
        self.assertEqual(best_moves[0]['move'].value, "B", "A之后最佳移动应该是B")
        self.assertGreater(best_moves[0]['win_rate'], best_moves[1]['win_rate'], 
                         "最佳移动的胜率应该更高")
        
        # 2. 测试移动频率分析
        freq = tree.get_move_frequency(depth=1)
        self.assertIn("A", freq, "第一步最常见的移动应该是A")
        self.assertEqual(freq["A"], 20, "A应该出现20次（现在统计总访问次数）")
        
        # 3. 添加一些棋盘位置数据进行热图测试
        for i in range(9):  # 3x3棋盘
            # 更频繁地选择中心和角落
            visits = 3 if i in [0, 2, 4, 6, 8] else 1
            for _ in range(visits):
                tree.start_path()
                tree.add_move(Move(i))
                tree.record_outcome('win')
                tree.end_path()
        
        heatmap = tree.generate_move_heatmap(board_size=3)
        self.assertEqual(len(heatmap), 3, "热图应该是3x3")
        # 中心位置(1,1)应该有最高的频率 - 索引是4对应(1,1)
        self.assertEqual(heatmap[1][1], 3, "中心位置应该访问3次")
        
        # 4. 测试路径多样性分析
        diversity = tree.analyze_path_diversity()
        self.assertGreater(diversity['total_fibers'], 0, "应该有一些fibers")
        self.assertIn('depth_distribution', diversity, "应该包含深度分布")
        
        # 5. 测试树修剪
        original_count = len(tree)
        removed = tree.prune_tree(min_visits=5)
        new_count = len(tree)
        self.assertEqual(original_count - removed, new_count, "修剪后节点数应该减少")
        
        # 6. 测试模拟路径
        fiber_id = tree.simulate_path([Move("X"), Move("Y")], outcome='win', visits=5)
        self.assertIsNotNone(fiber_id, "模拟路径应该成功")
        stats = tree.get_statistics(fiber_id)
        self.assertEqual(stats['visit_count'], 5, "模拟应该记录5次访问")
        self.assertEqual(stats['win_count'], 5, "模拟应该记录5次胜利")
    
    def test_storage_operations(self):
        """测试存储相关操作：保存、加载、合并"""
        # 1. 测试JSON保存和加载
        tree = self.memory_tree
        self._add_test_paths(tree, 20)
        
        # 保存到JSON文件
        json_path = os.path.join(self.test_dir, "test_tree.json")
        tree.save(json_path)
        self.assertTrue(os.path.exists(json_path), "JSON文件应该被创建")
        
        # 加载JSON文件
        loaded_tree = load_tree(json_path)
        self.assertEqual(len(tree), len(loaded_tree), "加载的树应该与原树大小相同")
        
        # 2. 测试二进制保存和加载
        bin_path = os.path.join(self.test_dir, "test_tree.bin")
        actual_bin_path = tree.export_binary(bin_path, compress=False)
        self.assertTrue(os.path.exists(actual_bin_path), "二进制文件应该被创建")
        
        # 加载二进制文件
        bin_loaded_tree = FiberTree.import_binary(actual_bin_path)
        self.assertEqual(len(tree), len(bin_loaded_tree), "二进制加载的树应该与原树大小相同")
        
        # 3. 测试压缩存储
        gz_path = os.path.join(self.test_dir, "test_tree.json.gz")
        tree.save(gz_path, compress=True)
        self.assertTrue(os.path.exists(gz_path), "压缩文件应该被创建")
        
        # 4. 测试树合并
        tree1 = create_tree()
        tree2 = create_tree()
        
        # 在两棵树中添加一些数据
        self._add_test_paths(tree1, 5)
        self._add_test_paths(tree2, 5)
        
        # 记录合并前的大小
        size_before = len(tree1)
        
        # 合并树
        merged_count = tree1.merge(tree2)
        self.assertGreater(merged_count, 0, "应该合并一些路径")
        self.assertGreater(len(tree1), size_before, "合并后树应该更大")
    
    def test_performance(self):
        """测试性能：大量添加、查询、内存使用"""
        # 1. 大量路径添加性能
        tree = self.memory_tree
        start_time = time.time()
        
        # 添加1000条路径
        for i in range(1000):
            tree.start_path()
            # 3-4步的路径
            path_len = random.randint(3, 4)
            for j in range(path_len):
                # 从10个可能的移动中选择
                move_value = random.choice(["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"])
                tree.add_move(Move(move_value))
            tree.record_outcome(random.choice(['win', 'loss', 'draw']))
            tree.end_path()
        
        add_time = time.time() - start_time
        print(f"添加1000条路径耗时: {add_time:.3f}秒")
        
        # 2. 查询性能
        start_time = time.time()
        
        # 执行1000次随机路径查找
        for i in range(1000):
            path_len = random.randint(1, 4)
            test_path = [Move(random.choice(["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"])) 
                        for _ in range(path_len)]
            tree.find_path(test_path)
        
        query_time = time.time() - start_time
        print(f"执行1000次路径查找耗时: {query_time:.3f}秒")
        
        # 3. 统计分析性能
        start_time = time.time()
        
        for i in range(20):
            tree.get_common_path_statistics(min_visits=2)
            tree.analyze_path_diversity()
            tree.get_move_frequency(depth=2)
        
        analysis_time = time.time() - start_time
        print(f"执行20次统计分析耗时: {analysis_time:.3f}秒")
        
        # 4. 序列化性能比较
        # JSON序列化
        json_path = os.path.join(self.test_dir, "perf_tree.json")
        start_time = time.time()
        tree.save(json_path)
        json_save_time = time.time() - start_time
        
        # 二进制序列化
        bin_path = os.path.join(self.test_dir, "perf_tree.bin")
        start_time = time.time()
        actual_bin_path = tree.export_binary(bin_path)  # 获取实际保存的路径
        bin_save_time = time.time() - start_time
        
        print(f"JSON存储耗时: {json_save_time:.3f}秒")
        print(f"二进制存储耗时: {bin_save_time:.3f}秒")
        
        if actual_bin_path and os.path.exists(actual_bin_path):
            # 反序列化比较
            start_time = time.time()
            load_tree(json_path)
            json_load_time = time.time() - start_time
            
            start_time = time.time()
            FiberTree.import_binary(actual_bin_path)  # 使用实际的文件路径
            bin_load_time = time.time() - start_time
            
            print(f"JSON加载耗时: {json_load_time:.3f}秒")
            print(f"二进制加载耗时: {bin_load_time:.3f}秒")
            
            # 简单的性能断言
            self.assertLess(bin_save_time, json_save_time * 2, "二进制存储不应该比JSON慢太多")
        else:
            print("二进制文件未成功创建，跳过加载测试")
    
    """
    改进的边缘情况测试，专门处理不可哈希类型的情况
    """

    def test_edge_cases(self):
        """测试边缘情况：空路径、长路径、错误处理等"""
        tree = self.memory_tree
        
        # 1. 空路径测试
        tree.start_path()
        tree.record_outcome('win')
        tree.end_path()
        # 应该正常运行，不会崩溃
        
        # 2. 非常长的路径
        tree.start_path()
        for i in range(100):  # 100步的长路径
            tree.add_move(Move(f"LONG_{i}"))
        tree.record_outcome('win')
        tree.end_path()
        
        # 验证可以找到这个长路径
        long_path = [Move(f"LONG_{i}") for i in range(100)]
        path_id = tree.find_path(long_path)
        self.assertIsNotNone(path_id, "应该能找到长路径")
        
        # 3. 添加带有复杂值和元数据的移动
        tree.start_path()
        tree.add_move(Move("complex_string", metadata={"extra": "info"}))  # 使用可哈希的字符串
        tree.add_move(Move(123, metadata={"numbers": True}))  # 使用数字
        tree.record_outcome('win')
        tree.end_path()
        
        # 4. 操作顺序错误测试
        # 在结束添加模式后添加移动
        tree.end_path()
        result = tree.add_move(Move("Z"))
        self.assertFalse(result, "在添加模式外添加移动应该返回False")
        
        # 5. 测试重复添加路径并更新统计
        tree.start_path()
        tree.add_move(Move("A"))
        tree.add_move(Move("B"))
        tree.record_outcome('win')
        tree.end_path()
        
        tree.start_path()
        tree.add_move(Move("A"))
        tree.add_move(Move("B"))
        tree.record_outcome('loss')
        tree.end_path()
        
        path_id = tree.find_path([Move("A"), Move("B")])
        stats = tree.get_statistics(path_id)
        self.assertEqual(stats['visit_count'], 2, "路径应该有2次访问")
        self.assertEqual(stats['win_count'], 1, "路径应该有1次胜利")
        self.assertEqual(stats['loss_count'], 1, "路径应该有1次失败")
        
        # 6. 测试不可哈希类型 - 安全地处理
        print("\n测试不可哈希类型...")
        success = True
        try:
            # 首先测试字典类型
            print("  测试字典类型移动值...")
            tree.start_path()
            dict_move = Move({"complex": "value"}, metadata={"extra": "info"})
            tree.add_move(dict_move)
            tree.record_outcome('win')
            tree.end_path()
            print("  成功添加字典类型移动值")
            
            # 测试列表类型
            print("  测试列表类型移动值...")
            tree.start_path()
            list_move = Move([1, 2, 3], metadata={"list": "values"})
            tree.add_move(list_move)
            tree.record_outcome('win')
            tree.end_path()
            print("  成功添加列表类型移动值")
            
            # 测试集合类型
            print("  测试集合类型移动值...")
            tree.start_path()
            set_move = Move(set([1, 2, 3]), metadata={"set": "values"})
            tree.add_move(set_move)
            tree.record_outcome('win')
            tree.end_path()
            print("  成功添加集合类型移动值")
        except Exception as e:
            success = False
            print(f"  错误: {str(e)}")
            self.fail(f"处理不可哈希类型时出错: {str(e)}")
        
        self.assertTrue(success, "应该能够处理不可哈希类型的移动值")


class MemoryUsageTest(unittest.TestCase):
    """测试内存使用情况"""
    
    def test_memory_usage(self):
        """在大规模操作下测试内存使用"""
        try:
            import psutil
            process = psutil.Process(os.getpid())
            
            # 记录初始内存使用
            initial_memory = process.memory_info().rss / 1024 / 1024  # MB
            
            # 创建树并添加大量数据
            tree = create_tree()
            for i in range(10000):  # 添加10000条路径
                tree.start_path()
                path_len = random.randint(1, 5)
                for j in range(path_len):
                    move_value = random.choice(["A", "B", "C", "D", "E"])
                    tree.add_move(Move(move_value))
                tree.record_outcome(random.choice(['win', 'loss']))
                tree.end_path()
            
            # 执行一些统计操作
            tree.get_common_path_statistics()
            tree.analyze_path_diversity()
            
            # 记录最终内存使用
            final_memory = process.memory_info().rss / 1024 / 1024  # MB
            memory_used = final_memory - initial_memory
            
            print(f"10000条路径的内存使用: {memory_used:.2f} MB")
            
            # 简单的内存使用断言 - 根据具体情况调整阈值
            self.assertLess(memory_used, 200, "10000条路径的内存使用应该小于200MB")
            
        except ImportError:
            print("psutil模块不可用，跳过内存使用测试")
            return


def compare_with_original():
    """比较优化版与原始版的性能（如果可用）"""
    try:
        # 尝试导入原始版本（如果有）
        import sys
        sys.path.append("..")
        from origin import FiberTree as OriginalFiberTree, Move as OriginalMove
        
        # 路径数量
        path_count = 1000
        
        # 测试优化版
        start_time = time.time()
        optimized_tree = create_tree()
        for i in range(path_count):
            optimized_tree.start_path()
            path_len = random.randint(1, 5)
            for j in range(path_len):
                move_value = random.choice(["A", "B", "C", "D", "E"])
                optimized_tree.add_move(Move(move_value))
            optimized_tree.record_outcome('win')
            optimized_tree.end_path()
        optimized_time = time.time() - start_time
        
        # 测试原始版
        start_time = time.time()
        original_tree = OriginalFiberTree()
        for i in range(path_count):
            original_tree.start_adding_mode()
            path_len = random.randint(1, 5)
            for j in range(path_len):
                move_value = random.choice(["A", "B", "C", "D", "E"])
                original_tree.add_move(OriginalMove(move_value))
            original_tree.update_statistics('win')
            original_tree.end_adding_mode()
        original_time = time.time() - start_time
        
        print(f"\n性能比较 ({path_count}条路径):")
        print(f"优化版耗时: {optimized_time:.3f}秒")
        print(f"原始版耗时: {original_time:.3f}秒")
        print(f"提升倍数: {original_time/optimized_time:.2f}x")
        
    except (ImportError, ModuleNotFoundError):
        print("\n原始版本不可用，跳过性能比较。")


if __name__ == "__main__":
    # 运行测试
    unittest.main(verbosity=2, exit=False)
    
    # 运行性能比较（如果可能）
    compare_with_original()