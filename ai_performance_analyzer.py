#!/usr/bin/env python3

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import LinearSegmentedColormap
import os
import time
from tqdm import tqdm
import seaborn as sns
from collections import Counter, defaultdict
import random
import argparse

# 设置Matplotlib使用Agg后端以解决显示问题
import matplotlib
matplotlib.use('Agg')

# 导入必要的模块
from fbtree import create_tree, Move
from game.rule import GomokuRule
from model import GomokuLSTMModel


class GomokuAnalyzer:
    def __init__(self, model_path="gomoku_final_model.pth", board_size=15):
        """
        初始化分析器
        
        Args:
            model_path: 模型路径
            board_size: 棋盘大小
        """
        self.board_size = board_size
        
        # 加载模型
        self.model = self._load_model(model_path)
        self.model_path = model_path
        
        # 创建坐标映射
        self._setup_position_mappings()
        
        # 创建FiberTree
        self.tree = create_tree()
        
        # 存储所有的游戏记录
        self.all_games = []
        
        # 定义输出目录
        self.output_dir = "analysis_output"
        os.makedirs(self.output_dir, exist_ok=True)
    
    def _load_model(self, model_path):
        """加载训练好的模型"""
        model = GomokuLSTMModel(board_size=self.board_size)
        
        if os.path.exists(model_path):
            print(f"加载模型: {model_path}")
            model.load_state_dict(torch.load(model_path))
        else:
            print(f"警告: 模型文件 {model_path} 不存在。使用未训练的模型。")
        
        model.eval()  # 设置为评估模式
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
    
    def get_model_prediction(self, move_history, board_state):
        """
        获取模型对下一步的预测
        
        Args:
            move_history: 移动历史
            board_state: 当前棋盘状态
            
        Returns:
            预测分数和对应的坐标
        """
        # 获取已占用位置
        occupied_positions = self._board_to_occupied_positions(board_state)
        
        # 准备模型输入
        with torch.no_grad():
            tensor_history = torch.tensor([move_history], dtype=torch.long)
            
            # 获取模型预测
            output = self.model(tensor_history)
            
            # 创建掩码
            mask = torch.ones(self.board_size * self.board_size, dtype=torch.bool)
            for pos in occupied_positions:
                mask[pos-1] = False  # 转换为0索引
            
            # 应用掩码
            masked_output = output.clone()
            masked_output[0, ~mask] = float('-inf')
            
            # 获取预测分数
            scores = torch.softmax(masked_output, dim=1)[0].cpu().numpy()
            
            # 创建坐标到分数的映射
            coord_scores = {}
            for pos in range(1, self.board_size * self.board_size + 1):
                if pos not in occupied_positions:
                    x, y = self.pos_to_coord[pos]
                    coord_scores[(x, y)] = scores[pos-1]
            
            return coord_scores
    
    def _select_move(self, move_history, board_state, player, temperature=0.1):
        """
        根据模型预测选择下一步
        
        Args:
            move_history: 移动历史
            board_state: 当前棋盘状态
            player: 当前玩家 (1 or -1)
            temperature: 控制随机性的温度参数
            
        Returns:
            (x, y), position: 选择的坐标和对应的位置
        """
        # 获取模型预测分数
        coord_scores = self.get_model_prediction(move_history, board_state)
        
        if temperature <= 0.01:
            # 确定性选择
            x, y = max(coord_scores.items(), key=lambda item: item[1])[0]
        else:
            # 按概率采样
            coords = list(coord_scores.keys())
            scores = np.array([coord_scores[c] for c in coords])
            
            # 应用温度
            scores = np.power(scores, 1.0 / temperature)
            # 归一化
            scores = scores / np.sum(scores)
            
            # 采样
            idx = np.random.choice(len(coords), p=scores)
            x, y = coords[idx]
        
        position = self.coord_to_pos[(x, y)]
        signed_position = player * position
        
        return (x, y), signed_position
    
    def self_play_game(self, temperature=0.1, record=True):
        """
        进行一场自对弈
        
        Args:
            temperature: 控制随机性的温度参数
            record: 是否记录游戏
            
        Returns:
            game_record: 游戏记录
        """
        # 初始化游戏
        game = GomokuRule(board_size=self.board_size)
        
        # 如果记录，开始一个新路径
        if record:
            self.tree.start_path()
        
        # 游戏记录
        game_record = {
            'moves': [],
            'board_states': [np.zeros((self.board_size, self.board_size))],
            'predictions': []
        }
        
        # 游戏循环
        while not game.game_over:
            # 当前玩家
            current_player = game.current_player
            
            # 获取当前棋盘状态
            board_state = game.get_board_state()
            
            # 获取移动历史
            if record:
                move_history = [m.value for m in self.tree.get_complete_path()]
            else:
                move_history = [m[1] for m in game_record['moves']]
            
            # 获取模型预测并选择移动
            coord_scores = self.get_model_prediction(move_history, board_state)
            (x, y), signed_position = self._select_move(move_history, board_state, current_player, temperature)
            
            # 记录预测
            game_record['predictions'].append(coord_scores)
            
            # 执行移动
            success = game.make_move(x, y)
            
            if success:
                # 记录移动
                game_record['moves'].append(((x, y), signed_position))
                game_record['board_states'].append(np.array(game.get_board_state()))
                
                # 记录到FiberTree
                if record:
                    self.tree.add_move(Move(signed_position))
            
        # 记录胜者
        game_record['winner'] = game.winner
        
        # 如果记录，结束路径并记录结果
        if record:
            outcome = 'win' if game.winner == 1 else 'loss' if game.winner == -1 else 'draw'
            self.tree.record_outcome(outcome)
            self.tree.end_path()
        
        # 保存游戏记录
        self.all_games.append(game_record)
        
        return game_record
    
    def visualize_game(self, game_idx=None):
        """
        可视化一局游戏
        
        Args:
            game_idx: 游戏索引，None表示最新的游戏
        """
        if not self.all_games:
            print("没有游戏记录")
            return
        
        if game_idx is None:
            game_idx = len(self.all_games) - 1
        
        game = self.all_games[game_idx]
        
        # 创建输出目录
        game_dir = os.path.join(self.output_dir, f"game_{game_idx}")
        os.makedirs(game_dir, exist_ok=True)
        
        # 为每一步生成图像
        for step in range(len(game['moves']) + 1):
            plt.figure(figsize=(10, 10))
            
            # 获取棋盘状态
            if step == 0:
                board = np.zeros((self.board_size, self.board_size))
                last_move = None
            else:
                board = game['board_states'][step]
                last_move = game['moves'][step-1][0]
            
            # 获取预测
            if step < len(game['predictions']):
                predictions = game['predictions'][step]
                
                # 创建热图数据
                heatmap_data = np.zeros((self.board_size, self.board_size))
                for coord, score in predictions.items():
                    x, y = coord
                    heatmap_data[x, y] = score
                
                # 绘制热图
                plt.imshow(heatmap_data, cmap='Blues', alpha=0.5)
            
            # 绘制棋盘网格
            for i in range(self.board_size):
                plt.axhline(i - 0.5, color='black', alpha=0.3)
                plt.axvline(i - 0.5, color='black', alpha=0.3)
            
            # 绘制棋子
            for i in range(self.board_size):
                for j in range(self.board_size):
                    if board[i, j] == 1:  # 黑子
                        plt.plot(j, i, 'o', markersize=20, markerfacecolor='black', markeredgecolor='black')
                    elif board[i, j] == -1:  # 白子
                        plt.plot(j, i, 'o', markersize=20, markerfacecolor='white', markeredgecolor='black')
            
            # 高亮显示最后一步
            if last_move:
                x, y = last_move
                plt.plot(y, x, 'X', markersize=10, markerfacecolor='red', markeredgecolor='red')
            
            # 设置坐标轴
            plt.xticks(range(self.board_size))
            plt.yticks(range(self.board_size))
            
            # 当前玩家
            player = "Black" if step % 2 == 0 else "White"
            
            # 设置标题
            plt.title(f"Game {game_idx}, Step {step}, {player}'s turn")
            
            # 保存图像
            plt.savefig(os.path.join(game_dir, f"step_{step:03d}.png"))
            plt.close()
            
        print(f"已生成游戏 {game_idx} 的所有步骤图像")
        print(f"输出目录: {game_dir}")
    
    def run_multiple_games(self, num_games=100, temperature=0.1, show_progress=True):
        """
        运行多场自对弈游戏
        
        Args:
            num_games: 游戏数量
            temperature: 温度参数
            show_progress: 是否显示进度条
            
        Returns:
            统计信息
        """
        # 清空之前的游戏记录
        self.all_games = []
        
        # 进度条
        game_iter = tqdm(range(num_games)) if show_progress else range(num_games)
        
        for _ in game_iter:
            self.self_play_game(temperature=temperature)
        
        # 返回统计信息
        return self.analyze_games()
    
    def analyze_games(self):
        """
        分析所有游戏的统计信息
        
        Returns:
            统计信息字典
        """
        if not self.all_games:
            return {}
        
        stats = {}
        
        # 胜率统计
        wins_black = sum(1 for game in self.all_games if game['winner'] == 1)
        wins_white = sum(1 for game in self.all_games if game['winner'] == -1)
        draws = sum(1 for game in self.all_games if game['winner'] == 0)
        
        stats['win_rate_black'] = wins_black / len(self.all_games)
        stats['win_rate_white'] = wins_white / len(self.all_games)
        stats['draw_rate'] = draws / len(self.all_games)
        
        # 游戏长度统计
        game_lengths = [len(game['moves']) for game in self.all_games]
        stats['avg_game_length'] = np.mean(game_lengths)
        stats['min_game_length'] = min(game_lengths)
        stats['max_game_length'] = max(game_lengths)
        stats['game_lengths'] = game_lengths
        
        # 创建统计图
        self._create_stats_plots(stats, game_lengths)
        
        return stats
    
    def _create_stats_plots(self, stats, game_lengths):
        """
        创建统计图表
        
        Args:
            stats: 统计信息
            game_lengths: 游戏长度列表
        """
        # 1. 胜率饼图
        plt.figure(figsize=(10, 6))
        plt.pie([stats['win_rate_black'], stats['win_rate_white'], stats['draw_rate']],
               labels=['Black', 'White', 'Draw'],
               autopct='%1.1f%%',
               colors=['#333333', '#DDDDDD', '#AAAAFF'])
        plt.title('Win Rate Distribution')
        plt.savefig(os.path.join(self.output_dir, 'win_rate_pie.png'))
        plt.close()
        
        # 2. 游戏长度分布
        plt.figure(figsize=(10, 6))
        sns.histplot(game_lengths, kde=True, bins=20)
        plt.axvline(stats['avg_game_length'], color='r', linestyle='--', 
                   label=f"Avg: {stats['avg_game_length']:.1f}")
        plt.title('Game Length Distribution')
        plt.xlabel('Number of Moves')
        plt.ylabel('Frequency')
        plt.legend()
        plt.savefig(os.path.join(self.output_dir, 'game_length_dist.png'))
        plt.close()
        
        # 3. 落子热图
        self._create_move_heatmaps()
    
    def _create_move_heatmaps(self):
        """生成落子热图"""
        if not self.all_games:
            return
            
        # 创建热图数据
        black_heatmap = np.zeros((self.board_size, self.board_size))
        white_heatmap = np.zeros((self.board_size, self.board_size))
        
        # 累计每个位置的落子次数
        for game in self.all_games:
            for (x, y), signed_pos in game['moves']:
                if signed_pos > 0:  # 黑子
                    black_heatmap[x, y] += 1
                else:  # 白子
                    white_heatmap[x, y] += 1
        
        # 归一化
        if np.max(black_heatmap) > 0:
            black_heatmap = black_heatmap / np.max(black_heatmap)
        if np.max(white_heatmap) > 0:
            white_heatmap = white_heatmap / np.max(white_heatmap)
        
        # 创建图形
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
        
        # 绘制黑子热图
        im1 = ax1.imshow(black_heatmap, cmap='Blues')
        ax1.set_title('Black Moves Heatmap')
        ax1.set_xticks(range(self.board_size))
        ax1.set_yticks(range(self.board_size))
        fig.colorbar(im1, ax=ax1)
        
        # 绘制白子热图
        im2 = ax2.imshow(white_heatmap, cmap='Reds')
        ax2.set_title('White Moves Heatmap')
        ax2.set_xticks(range(self.board_size))
        ax2.set_yticks(range(self.board_size))
        fig.colorbar(im2, ax=ax2)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'move_heatmaps.png'))
        plt.close()
    
    def compare_with_random(self, num_games=20):
        """
        与随机代理进行比较
        
        Args:
            num_games: 比赛数量
        """
        print(f"AI vs Random Agent ({num_games} games):")
        
        # 比赛记录
        results = {
            'ai_wins': 0,
            'random_wins': 0,
            'draws': 0,
            'game_lengths': []
        }
        
        for game_idx in tqdm(range(num_games)):
            # 随机决定AI是黑子还是白子
            ai_is_black = random.choice([True, False])
            ai_color = 1 if ai_is_black else -1
            random_color = -1 if ai_is_black else 1
            
            # 创建游戏
            game = GomokuRule(board_size=self.board_size)
            
            # 使用FiberTree记录
            self.tree.start_path()
            
            # 游戏历史
            move_history = []
            
            # 游戏循环
            while not game.game_over:
                current_player = game.current_player
                
                if current_player == ai_color:
                    # AI回合
                    board_state = game.get_board_state()
                    coord_scores = self.get_model_prediction(move_history, board_state)
                    (x, y), signed_position = self._select_move(move_history, board_state, current_player, temperature=0.1)
                else:
                    # 随机代理回合
                    # 获取所有合法移动
                    legal_moves = []
                    for x in range(self.board_size):
                        for y in range(self.board_size):
                            if game.is_valid_move(x, y):
                                legal_moves.append((x, y))
                    
                    # 随机选择
                    x, y = random.choice(legal_moves)
                    position = self.coord_to_pos[(x, y)]
                    signed_position = random_color * position
                
                # 执行移动
                success = game.make_move(x, y)
                
                if success:
                    # 更新历史
                    move_history.append(signed_position)
                    self.tree.add_move(Move(signed_position))
            
            # 记录结果
            if game.winner == ai_color:
                results['ai_wins'] += 1
            elif game.winner == random_color:
                results['random_wins'] += 1
            else:
                results['draws'] += 1
            
            results['game_lengths'].append(len(move_history))
            
            # 结束记录
            outcome = 'win' if game.winner == 1 else 'loss' if game.winner == -1 else 'draw'
            self.tree.record_outcome(outcome)
            self.tree.end_path()
        
        # 计算胜率
        ai_win_rate = results['ai_wins'] / num_games
        random_win_rate = results['random_wins'] / num_games
        draw_rate = results['draws'] / num_games
        
        print(f"AI Win Rate: {ai_win_rate:.1%}")
        print(f"Random Win Rate: {random_win_rate:.1%}")
        print(f"Draw Rate: {draw_rate:.1%}")
        print(f"Average Game Length: {np.mean(results['game_lengths']):.1f} moves")
        
        # 创建比较图
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # 胜率饼图
        ax1.pie([ai_win_rate, random_win_rate, draw_rate],
               labels=['AI', 'Random', 'Draw'],
               autopct='%1.1f%%',
               colors=['royalblue', 'lightcoral', 'lightgrey'])
        ax1.set_title('Win Rate Distribution')
        
        # 游戏长度分布
        sns.histplot(results['game_lengths'], kde=True, bins=10, ax=ax2)
        ax2.set_title('Game Length Distribution')
        ax2.set_xlabel('Number of Moves')
        ax2.axvline(np.mean(results['game_lengths']), color='r', linestyle='--',
                  label=f"Avg: {np.mean(results['game_lengths']):.1f}")
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'ai_vs_random.png'))
        plt.close()
        
        return results


def main():
    parser = argparse.ArgumentParser(description='Gomoku AI Analysis Tool')
    parser.add_argument('--model', type=str, default='gomoku_final_model.pth',
                       help='Model path')
    parser.add_argument('--num-games', type=int, default=50,
                       help='Number of self-play games')
    parser.add_argument('--visualize-game', action='store_true',
                       help='Visualize a game after analysis')
    parser.add_argument('--vs-random', action='store_true',
                       help='Compare with random agent')
    parser.add_argument('--output', type=str, default='analysis_output',
                       help='Output directory')
    
    args = parser.parse_args()
    
    # 创建分析器
    analyzer = GomokuAnalyzer(model_path=args.model)
    analyzer.output_dir = args.output
    os.makedirs(args.output, exist_ok=True)
    
    # 运行自对弈
    print(f"Running {args.num_games} self-play games...")
    stats = analyzer.run_multiple_games(num_games=args.num_games)
    
    # 打印统计信息
    print("\nStatistics:")
    print(f"Black Win Rate: {stats['win_rate_black']:.1%}")
    print(f"White Win Rate: {stats['win_rate_white']:.1%}")
    print(f"Draw Rate: {stats['draw_rate']:.1%}")
    print(f"Average Game Length: {stats['avg_game_length']:.1f} moves")
    print(f"Shortest Game: {stats['min_game_length']} moves")
    print(f"Longest Game: {stats['max_game_length']} moves")
    
    # 与随机代理比较
    if args.vs_random:
        print("\nComparing with random agent...")
        analyzer.compare_with_random(num_games=20)
    
    # 可视化游戏
    if args.visualize_game:
        print("\nVisualizing a game...")
        analyzer.visualize_game()
    
    print(f"\nAnalysis complete! Results saved to: {args.output}")


if __name__ == "__main__":
    main()