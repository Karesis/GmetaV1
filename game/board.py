class Piece:
    def __init__(self, color=0):
        """
        初始化棋子
        
        Args:
            color: 棋子颜色 
            0: 空白
            1: 黑子 
            -1: 白子
        """
        self.color = color
    
    def __str__(self):
        """
        返回棋子的字符串表示
        
        Returns:
            str: 棋子的符号 
        """
        if self.color == 1:
            return "@"
        elif self.color == -1:
            return "O"
        else:
            return "+"
    
    def __repr__(self):
        """
        返回棋子的开发者友好表示
        
        Returns:
            str: 棋子的详细描述
        """
        return f"Piece(color={self.color})"

class Board:
    def __init__(self, size=15):
        """
        初始化棋盘
        
        Args:
            size: 棋盘大小，默认为15x15
        """
        self.size = size
        self.board = [[Piece() for _ in range(size)] for _ in range(size)]
    
    def is_valid_position(self, x, y):
        """
        检查坐标是否在棋盘范围内
        
        Args:
            x: x坐标
            y: y坐标
        
        Returns:
            bool: 坐标是否有效
        """
        return 0 <= x < self.size and 0 <= y < self.size
    
    def get_piece(self, x, y):
        """
        获取指定位置的棋子
        
        Args:
            x: x坐标
            y: y坐标
        
        Returns:
            Piece: 指定位置的棋子
        """
        if not self.is_valid_position(x, y):
            raise ValueError(f"无效的棋盘坐标: ({x}, {y})")
        return self.board[x][y]
    
    def set_piece(self, x, y, piece):
        """
        在指定位置放置棋子
        
        Args:
            x: x坐标
            y: y坐标
            piece: 要放置的棋子
        """
        if not self.is_valid_position(x, y):
            raise ValueError(f"无效的棋盘坐标: ({x}, {y})")
        self.board[x][y] = piece
    
    def clear_board(self):
        """
        清空棋盘
        """
        self.board = [[Piece() for _ in range(self.size)] for _ in range(self.size)]
    
    def __str__(self):
        """
        生成棋盘的简洁字符串表示
        
        Returns:
            str: 棋盘的简化文本可视化
        """
        # 优化宽度比例，使棋盘视觉上更方正
        row_width = 2  # 行号宽度
        col_width = 2  # 棋子宽度
        
        # 创建列标题（第一行）
        header = " " * row_width  # 左上角留空
        for i in range(self.size):
            header += f"{i:^{col_width}}"
        
        # 添加每一行，减少宽度使棋盘更方正
        rows = [header]
        for i in range(self.size):
            row = f"{i:{row_width}d}"  # 行号
            for j in range(self.size):
                row += f"{str(self.board[i][j]):^{col_width}}"
            rows.append(row)
        
        return "\n".join(rows)
    
    def to_list(self):
        """
        将棋盘转换为二维列表，每个元素是棋子的颜色
        
        Returns:
            list: 棋盘状态的二维列表
        """
        return [[piece.color for piece in row] for row in self.board]
    
    def console_visualize(self):
        """
        在控制台打印棋盘
        """
        print(str(self))