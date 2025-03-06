#!/usr/bin/env python3

import torch
import numpy as np
import os
import time
from fbtree import create_tree, Move
from game.rule import GomokuRule
from game.board import Piece
from model import GomokuLSTMModel  # 确保model.py在同一目录下


class GomokuAIPlayer:
    def __init__(self, model_path="gomoku_final_model.pth", board_size=15):
        """
        初始化AI玩家
        
        Args:
            model_path: 训练好的模型路径
            board_size: 棋盘大小
        """
        self.board_size = board_size
        
        # 创建并加载模型
        self.model = self._load_model(model_path)
        
        # 创建FiberTree用于记录游戏历史
        self.tree = create_tree()
        
        # 设置坐标和一维位置的映射
        self._setup_position_mappings()

    def _load_model(self, model_path):
        """加载训练好的模型"""
        model = GomokuLSTMModel(board_size=self.board_size)
        
        if os.path.exists(model_path):
            print(f"加载模型: {model_path}")
            model.load_state_dict(torch.load(model_path))
            model.eval()  # 设置为评估模式
        else:
            print(f"警告: 模型文件 {model_path} 不存在。使用未训练的模型。")
        
        return model
    
    def _setup_position_mappings(self):
        """设置坐标和一维位置之间的映射"""
        self.coord_to_pos = {}  # (x, y) -> position
        self.pos_to_coord = {}  # position -> (x, y)
        
        position = 1  # 从1开始计数
        for x in range(self.board_size):
            for y in range(self.board_size):
                self.coord_to_pos[(x, y)] = position
                self.pos_to_coord[position] = (x, y)
                position += 1
                
    def _board_to_occupied_positions(self, board_state):
        """将棋盘状态转换为已占用位置的集合"""
        occupied = set()
        
        for x in range(self.board_size):
            for y in range(self.board_size):
                if board_state[x][y] != 0:
                    pos = self.coord_to_pos[(x, y)]
                    occupied.add(pos)
                    
        return occupied
    
    def start_new_game(self):
        """开始一个新游戏"""
        self.tree.start_path()
    
    def make_move(self, game, player):
        """
        让AI做出一步移动
        
        Args:
            game: GomokuRule实例
            player: 1表示黑，-1表示白
            
        Returns:
            (x, y): 选择的坐标
        """
        # 获取当前游戏历史
        move_history = [m.value for m in self.tree.get_complete_path()]
        
        # 获取棋盘状态和已占用位置
        board_state = game.get_board_state()
        occupied_positions = self._board_to_occupied_positions(board_state)
        
        # 准备模型输入
        with torch.no_grad():
            tensor_history = torch.tensor([move_history], dtype=torch.long)
            
            # 获取模型预测
            output = self.model(tensor_history)
            
            # 掩盖已占用位置
            mask = torch.ones(self.board_size * self.board_size, dtype=torch.bool)
            for pos in occupied_positions:
                mask[pos-1] = False  # 转换为0索引
            
            # 应用掩码
            masked_output = output.clone()
            masked_output[0, ~mask] = float('-inf')
            
            # 选择最高分数的位置
            position = torch.argmax(masked_output).item() + 1  # 转换回1索引
            
            # 转换为坐标
            x, y = self.pos_to_coord[position]
            
            # 记录到树中
            self.tree.add_move(Move(player * position))
            
            return x, y
    
    def record_game_result(self, winner):
        """记录游戏结果到FiberTree"""
        if winner == 1:
            self.tree.record_outcome('win')
        elif winner == -1:
            self.tree.record_outcome('loss')
        else:
            self.tree.record_outcome('draw')
        
        self.tree.end_path()


def display_board(game, last_move=None):
    """美化显示棋盘"""
    board_state = game.get_board_state()
    size = len(board_state)
    
    # 打印列坐标
    print("  ", end="")
    for i in range(size):
        print(f"{i:2d}", end="")
    print()
    
    # 打印棋盘和行坐标
    for i in range(size):
        print(f"{i:2d}", end="")
        for j in range(size):
            # 高亮最后一步
            if last_move and last_move == (i, j):
                if board_state[i][j] == 1:
                    print("\033[1;32mX\033[0m ", end="")  # 绿色高亮X表示黑子
                else:
                    print("\033[1;31mO\033[0m ", end="")  # 红色高亮O表示白子
            else:
                if board_state[i][j] == 1:
                    print("X ", end="")  # X表示黑子
                elif board_state[i][j] == -1:
                    print("O ", end="")  # O表示白子
                else:
                    print(". ", end="")  # .表示空位
        print()
    print()


def get_player_move(game):
    """获取玩家的移动"""
    while True:
        try:
            move = input("请输入你的移动 (格式: x,y): ")
            if move.lower() in ['q', 'quit', 'exit']:
                return None
                
            parts = move.strip().split(',')
            if len(parts) != 2:
                print("无效的格式。请使用'x,y'格式。")
                continue
                
            x, y = int(parts[0]), int(parts[1])
            
            if not game.is_valid_move(x, y):
                print("无效的移动。该位置已被占用或超出棋盘范围。")
                continue
                
            return x, y
        except ValueError:
            print("无效的输入。请输入数字坐标。")
        except Exception as e:
            print(f"错误: {e}")


def main():
    # 初始化游戏和AI
    print("欢迎来到五子棋游戏！")
    print("你将与训练好的AI模型对战。")
    print("输入格式: x,y （如 7,7 表示中心位置）")
    print("输入 q, quit 或 exit 退出游戏\n")
    
    # 选择先手或后手
    while True:
        choice = input("你想先手(1)还是后手(2)? ")
        if choice == '1':
            player_color = 1  # 黑子
            ai_color = -1     # 白子
            break
        elif choice == '2':
            player_color = -1  # 白子
            ai_color = 1       # 黑子
            break
        else:
            print("请输入1或2。")
    
    # 创建游戏和AI
    game = GomokuRule()
    ai = GomokuAIPlayer()
    ai.start_new_game()
    
    last_move = None
    
    # 游戏主循环
    while not game.game_over:
        # 显示当前棋盘
        display_board(game, last_move)
        
        # 当前玩家
        current_player = game.current_player
        
        if current_player == player_color:
            # 玩家回合
            print("轮到你了（" + ("黑子X" if player_color == 1 else "白子O") + "）")
            move = get_player_move(game)
            
            if move is None:
                print("游戏已退出。")
                return
                
            x, y = move
            success = game.make_move(x, y)
            
            if success:
                # 记录玩家移动到AI的历史中
                position = ai.coord_to_pos[(x, y)]
                ai.tree.add_move(Move(player_color * position))
                last_move = (x, y)
            
        else:
            # AI回合
            print(f"轮到AI了（" + ("黑子X" if ai_color == 1 else "白子O") + "）")
            print("AI思考中...")
            time.sleep(0.5)  # 增加短暂延迟，使体验更自然
            
            x, y = ai.make_move(game, ai_color)
            success = game.make_move(x, y)
            
            if success:
                print(f"AI选择了位置: {x},{y}")
                last_move = (x, y)
    
    # 游戏结束，显示最终棋盘和结果
    display_board(game, last_move)
    
    if game.winner == player_color:
        print("恭喜！你赢了！")
    elif game.winner == ai_color:
        print("AI赢了！再接再厉！")
    else:
        print("平局！")
    
    # 记录游戏结果
    ai.record_game_result(game.winner)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n游戏已中断。")
    except Exception as e:
        print(f"游戏出错: {e}")