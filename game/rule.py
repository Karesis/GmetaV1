from .board import Board, Piece

class GomokuRule:
    def __init__(self, board_size=15):
        """
        初始化五子棋规则
        
        Args:
            board_size: 棋盘大小，默认15x15
        """
        self.board = Board(size=board_size)
        self.current_player = 1  # 黑子先手
        self.move_history = []
        self.game_over = False
        self.winner = 0
    
    def is_valid_move(self, x, y):
        """
        检查落子是否合法
        
        Args:
            x: x坐标
            y: y坐标
        
        Returns:
            bool: 是否可以落子
        """
        # 检查游戏是否已结束
        if self.game_over:
            return False
        
        # 检查坐标是否在棋盘内
        if not self.board.is_valid_position(x, y):
            return False
        
        # 检查该位置是否已经有棋子
        piece = self.board.get_piece(x, y)
        return piece.color == 0  # 只有空位可以落子
    
    def make_move(self, x, y):
        """
        落子
        
        Args:
            x: x坐标
            y: y坐标
        
        Returns:
            bool: 落子是否成功
        """
        if not self.is_valid_move(x, y):
            return False
        
        # 放置棋子
        self.board.set_piece(x, y, Piece(self.current_player))
        
        # 记录落子历史
        self.move_history.append((x, y, self.current_player))
        
        # 检查是否有获胜者
        winner = self.check_winner()
        if winner != 0:
            self.game_over = True
            self.winner = winner
        
        # 切换玩家
        self.current_player = -self.current_player
        
        return True
    
    def check_winner(self):
        """
        检查是否有玩家获胜
        
        Returns:
            int: 获胜玩家（1为黑棋，-1为白棋，0为未分出胜负）
        """
        directions = [
            (0, 1),   # 水平
            (1, 0),   # 垂直
            (1, 1),   # 正对角线
            (1, -1)   # 反对角线
        ]
        
        for x in range(self.board.size):
            for y in range(self.board.size):
                piece = self.board.get_piece(x, y)
                if piece.color == 0:
                    continue
                
                # 检查四个方向是否有五连
                for dx, dy in directions:
                    if self._check_line(x, y, dx, dy):
                        return piece.color
        
        return 0
    
    def _check_line(self, x, y, dx, dy):
        """
        检查某个方向是否有五连
        
        Args:
            x: 起始x坐标
            y: 起始y坐标
            dx: x方向增量
            dy: y方向增量
        
        Returns:
            bool: 是否有五连
        """
        color = self.board.get_piece(x, y).color
        count = 1
        
        # 正方向
        for i in range(1, 5):
            nx, ny = x + i * dx, y + i * dy
            if (not self.board.is_valid_position(nx, ny) or 
                self.board.get_piece(nx, ny).color != color):
                break
            count += 1
        
        # 反方向
        for i in range(1, 5):
            nx, ny = x - i * dx, y - i * dy
            if (not self.board.is_valid_position(nx, ny) or 
                self.board.get_piece(nx, ny).color != color):
                break
            count += 1
        
        return count >= 5
    
    def undo_move(self):
        """
        悔棋
        
        Returns:
            bool: 是否成功悔棋
        """
        if not self.move_history:
            return False
        
        # 重置游戏结束状态
        self.game_over = False
        self.winner = 0
        
        x, y, player = self.move_history.pop()
        self.board.set_piece(x, y, Piece())  # 重置为空白
        self.current_player = player  # 恢复上一个玩家
        
        return True
    
    def get_board_state(self):
        """
        获取当前棋盘状态
        
        Returns:
            list: 当前棋盘状态的二维列表
        """
        return self.board.to_list()
    
    def __str__(self):
        """
        返回棋盘的字符串表示
        
        Returns:
            str: 棋盘的文本可视化
        """
        return str(self.board)