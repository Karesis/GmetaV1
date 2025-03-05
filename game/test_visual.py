from board import Board, Piece
from rule import GomokuRule

def test_board_display():
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
    
    # HTML显示测试
    print("\nHTML棋盘显示测试:")
    html_board = board.html_visualize()
    print(html_board)
    
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
    print("\nHTML显示:")
    print(game.board.html_visualize())
    print("\n控制台显示:")
    print(game)

# 如果直接运行这个脚本
if __name__ == "__main__":
    test_board_display()