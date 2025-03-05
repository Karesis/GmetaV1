import unittest
from rule import GomokuRule

class TestGomokuRule(unittest.TestCase):
    def setUp(self):
        """
        在每个测试方法之前创建一个新的游戏实例
        """
        self.game = GomokuRule()
    
    def test_initial_state(self):
        """
        测试游戏的初始状态
        """
        # 验证初始玩家是黑子
        self.assertEqual(self.game.current_player, 1, "初始玩家应该是黑子")
        
        # 验证初始棋盘状态
        board_state = self.game.get_board_state()
        self.assertEqual(len(board_state), 15, "棋盘应该是15x15")
        self.assertEqual(len(board_state[0]), 15, "棋盘应该是15x15")
        
        # 验证初始棋盘全为空白
        for row in board_state:
            for cell in row:
                self.assertEqual(cell, 0, "初始棋盘应该全为空白")
    
    def test_make_move(self):
        """
        测试落子功能
        """
        # 验证第一步落子
        self.assertTrue(self.game.make_move(7, 7), "应该可以在(7,7)落子")
        
        # 验证落子后棋盘状态
        board_state = self.game.get_board_state()
        self.assertEqual(board_state[7][7], 1, "(7,7)应该是黑子")
        
        # 验证玩家已切换
        self.assertEqual(self.game.current_player, -1, "下一个应该是白子")
        
        # 验证重复落子失败
        self.assertFalse(self.game.make_move(7, 7), "不能在已有棋子的位置落子")
    
    def test_valid_moves(self):
        """
        测试合法落子
        """
        # 测试棋盘边界
        self.assertTrue(self.game.make_move(0, 0), "应该可以在(0,0)落子")
        self.assertTrue(self.game.make_move(14, 14), "应该可以在(14,14)落子")
        
        # 测试超出棋盘范围的落子
        with self.assertRaises(ValueError, msg="落子超出棋盘范围应该抛出异常"):
            self.game.board.set_piece(15, 15, self.game.board.get_piece(0, 0))
    
    def test_winner_detection(self):
        """
        测试获胜判定
        """
        # 模拟五连
        moves = [
            (7, 7), (7, 8),  # 黑白轮流
            (8, 7), (8, 8),
            (9, 7), (9, 8),
            (10, 7), (10, 8),
            (11, 7), (11, 8)  # 构造黑子五连
        ]
        
        for x, y in moves:
            self.game.make_move(x, y)
        
        # 检查获胜
        winner = self.game.check_winner()
        self.assertEqual(winner, 1, "黑子应该获胜")
        self.assertTrue(self.game.game_over, "游戏应该结束")
    
    def test_diagonal_win(self):
        """
        测试对角线获胜
        """
        # 对角线五连
        moves = [
            (0, 0), (0, 1),
            (1, 1), (0, 2),
            (2, 2), (0, 3),
            (3, 3), (0, 4),
            (4, 4), (0, 5)  # 黑子对角线五连
        ]
        
        for x, y in moves:
            self.game.make_move(x, y)
        
        winner = self.game.check_winner()
        self.assertEqual(winner, 1, "黑子对角线五连应该获胜")
    
    def test_undo_move(self):
        """
        测试悔棋功能
        """
        # 落子
        self.game.make_move(7, 7)
        self.game.make_move(7, 8)
        
        # 悔棋
        self.assertTrue(self.game.undo_move(), "应该可以悔棋")
        
        # 验证棋盘状态
        board_state = self.game.get_board_state()
        self.assertEqual(board_state[7][8], 0, "悔棋后该位置应该为空")
        
        # 验证玩家已切换回来
        self.assertEqual(self.game.current_player, -1, "玩家应该切换回来")
    
    def test_game_end(self):
        """
        测试游戏结束后的状态
        """
        # 模拟一个获胜局面
        moves = [
            (0, 0), (0, 1),
            (1, 0), (1, 1),
            (2, 0), (2, 1),
            (3, 0), (3, 1),
            (4, 0), (4, 1)  # 黑子获胜
        ]
        
        for x, y in moves:
            self.game.make_move(x, y)
        
        # 检查游戏状态
        self.assertTrue(self.game.game_over, "游戏应该结束")
        self.assertEqual(self.game.winner, 1, "黑子应该获胜")
        
        # 尝试继续落子应该失败
        self.assertFalse(self.game.make_move(5, 0), "游戏结束后不应该继续落子")

def main():
    """
    运行所有测试
    """
    # 创建测试加载器
    suite = unittest.TestLoader().loadTestsFromTestCase(TestGomokuRule)
    
    # 创建测试运行器
    runner = unittest.TextTestRunner(verbosity=2)
    
    # 运行测试
    result = runner.run(suite)
    
    # 返回测试是否全部通过
    return result.wasSuccessful()

if __name__ == "__main__":
    # 运行测试
    success = main()
    # 如果测试失败，以非零状态退出
    exit(0 if success else 1)