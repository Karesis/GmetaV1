
from board import Board, Piece
from rule import GomokuRule

def test_board_display():
    """
    测试棋盘显示功能
    """
    # 控制台显示测试
    print("控制台棋盘显示测试:")
    # 创建空棋盘
    board = Board()
    board.console_visualize()
    
    # 在不同位置放置棋子
    test_positions = [
        (0, 0, 1),    # 黑子
        (7, 7, 1),    # 黑子
        (14, 14, -1), # 白子
        (0, 14, -1),  # 白子
        (7, 0, 1),    # 黑子
        (0, 7, -1)    # 白子
    ]
    
    for x, y, color in test_positions:
        board.set_piece(x, y, Piece(color))
    
    print("\n放置特定位置的棋子后的棋盘:")
    board.console_visualize()
    
    # 使用游戏规则测试
    print("\n游戏规则下的棋盘显示:")
    game = GomokuRule()
    
    # 模拟一些落子
    moves = [
        (7, 7),   # 黑子
        (7, 8),   # 白子
        (8, 7),   # 黑子
        (8, 8),   # 白子
        (6, 7),   # 黑子
        (9, 7),   # 白子
    ]
    
    for x, y in moves:
        game.make_move(x, y)
        
    print("\n落子后的棋盘:")
    print(game)

def test_win_visualization():
    """
    测试获胜情况的棋盘显示
    """
    print("\n测试获胜情况的棋盘显示:")
    game = GomokuRule()
    
    # 创建一个获胜局面 - 黑子横向五连
    for i in range(5):
        game.make_move(7, i+5)  # 黑子
        if i < 4:  # 不需要第5个白子
            game.make_move(8, i+5)  # 白子
    
    print("黑子横向五连获胜局面:")
    print(game)
    print(f"游戏结束: {game.game_over}, 获胜方: {'黑子' if game.winner == 1 else '白子' if game.winner == -1 else '无'}")

def test_move_undo():
    """
    测试悔棋功能
    """
    print("\n测试悔棋功能:")
    game = GomokuRule()
    
    # 落几个子
    moves = [(7, 7), (8, 8), (9, 9)]
    for x, y in moves:
        game.make_move(x, y)
    
    print("三步落子后的棋盘:")
    print(game)
    
    # 悔棋两次
    game.undo_move()
    game.undo_move()
    
    print("\n悔棋两次后的棋盘:")
    print(game)

# 如果直接运行这个脚本
if __name__ == "__main__":
    test_board_display()
    test_win_visualization()
    test_move_undo()