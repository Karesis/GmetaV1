import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import defaultdict
import copy
import os
import time

from fbtree import create_tree, Move
from game.rule import GomokuRule
from game.board import Piece

# 导入之前定义的LSTM模型
from model import GomokuLSTMModel


class GomokuSelfPlayTrainer:
    def __init__(self, model, board_size=15, discount_factor=0.95, learning_rate=0.001):
        """
        自对弈训练器
        
        Args:
            model: 神经网络模型
            board_size: 棋盘大小
            discount_factor: 奖励折扣因子 - 用于从游戏结束往前递减奖励
            learning_rate: 学习率
        """
        self.model = model
        self.board_size = board_size
        self.discount_factor = discount_factor
        
        # 优化器
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        
        # FiberTree用于存储游戏历史
        self.tree = create_tree()
        
        # 五子棋游戏规则
        self.game = GomokuRule(board_size=board_size)
        
        # 转换棋盘坐标到一维位置和反向映射的工具函数
        self._setup_position_mappings()
    
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
    
    def _board_to_occupied_positions(self):
        """将当前棋盘状态转换为已占用位置的集合"""
        occupied = set()
        board_state = self.game.get_board_state()
        
        for x in range(self.board_size):
            for y in range(self.board_size):
                if board_state[x][y] != 0:
                    pos = self.coord_to_pos[(x, y)]
                    occupied.add(pos)
                    
        return occupied
    
    def _get_legal_moves(self):
        """获取所有合法的移动"""
        legal_moves = []
        board_state = self.game.get_board_state()
        
        for x in range(self.board_size):
            for y in range(self.board_size):
                if board_state[x][y] == 0:
                    legal_moves.append((x, y))
                    
        return legal_moves
    
    def _select_move(self, move_history, exploration_temp=1.0):
        """
        基于模型预测选择下一步移动
        
        Args:
            move_history: 当前的移动历史（符号整数序列）
            exploration_temp: 探索温度参数，控制探索与利用的平衡
                             较高的值增加探索，较低的值增加利用
        
        Returns:
            (x, y): 选择的坐标
            move_position: 对应的一维位置（正负表示黑白）
            move_probs: 模型预测的概率分布
        """
        # 获取已占用的位置
        occupied_positions = self._board_to_occupied_positions()
        
        # 如果是游戏第一步（没有历史），选择靠近中心的随机位置
        if len(move_history) == 0:
            # 创建合法移动列表
            legal_moves = self._get_legal_moves()
            
            # 倾向于选择靠近中心的位置
            center = self.board_size // 2
            center_pos = (center, center)
            
            # 按照到中心距离排序移动
            def distance_to_center(move):
                return ((move[0] - center) ** 2 + (move[1] - center) ** 2) ** 0.5
            
            sorted_moves = sorted(legal_moves, key=distance_to_center)
            
            # 从靠近中心的3个位置中随机选择一个
            available_center_moves = sorted_moves[:min(9, len(sorted_moves))]
            x, y = random.choice(available_center_moves)
            
            # 转换为位置
            move_position = self.coord_to_pos[(x, y)]
            # 创建均匀概率分布（用于学习）
            # 注意：这里不需要梯度，因为是随机选择
            move_probs = torch.ones(self.board_size * self.board_size) / (self.board_size * self.board_size)
            
            # 根据当前玩家确定符号
            player = self.game.current_player  # 1为黑，-1为白
            signed_position = player * move_position
            
            return (x, y), signed_position, move_probs
        
        # 正常情况：使用模型预测
        # 准备输入数据
        tensor_history = torch.tensor([move_history], dtype=torch.long)
        
        # 获取模型预测
        self.model.eval()  # 评估模式
        with torch.no_grad():  # 不计算梯度
            output = self.model(tensor_history)
            
        # 创建一个表示法律移动的掩码
        legal_mask = torch.ones(self.board_size * self.board_size, dtype=torch.bool)
        for pos in occupied_positions:
            legal_mask[pos-1] = False  # 转换为0索引
        
        # 应用掩码并获取概率分布
        masked_output = output.clone()
        masked_output[0, ~legal_mask] = float('-inf')
        
        # 应用温度参数
        if exploration_temp != 1.0:
            masked_output = masked_output / exploration_temp
            
        # 转换为概率分布
        move_probs = torch.softmax(masked_output, dim=1)[0].detach()  # 分离梯度，因为这是选择阶段
        
        # 根据概率分布采样或选择最高概率的移动
        if exploration_temp > 0.01:  # 有探索
            # 转换为numpy进行采样
            move_probs_np = move_probs.numpy()
            # 如果总和不为1，重新归一化
            if not np.isclose(np.sum(move_probs_np), 1.0):
                move_probs_np = move_probs_np / np.sum(move_probs_np)
            chosen_idx = np.random.choice(len(move_probs_np), p=move_probs_np)
        else:  # 无探索，选择最佳移动
            chosen_idx = torch.argmax(move_probs).item()
        
        # 转换为棋盘坐标（加1因为pos_to_coord从1开始）
        move_position = chosen_idx + 1
        x, y = self.pos_to_coord[move_position]
        
        # 根据当前玩家（黑/白）确定位置的符号
        player = self.game.current_player  # 1为黑，-1为白
        signed_position = player * move_position
        
        # 为训练创建新的概率分布（带梯度）
        train_probs = torch.softmax(output, dim=1)[0]
        
        return (x, y), signed_position, train_probs

        legal_mask = torch.ones(self.board_size * self.board_size, dtype=torch.bool)
        for pos in occupied_positions:
            legal_mask[pos-1] = False  # 转换为0索引
        
        # 应用掩码并获取概率分布
        masked_output = output.clone()
        masked_output[0, ~legal_mask] = float('-inf')
        
        # 应用温度参数
        if exploration_temp != 1.0:
            masked_output = masked_output / exploration_temp
            
        # 转换为概率分布
        move_probs = torch.softmax(masked_output, dim=1)[0]
        
        # 根据概率分布采样或选择最高概率的移动
        if exploration_temp > 0.01:  # 有探索
            # 转换为numpy进行采样
            move_probs_np = move_probs.numpy()
            chosen_idx = np.random.choice(len(move_probs_np), p=move_probs_np)
        else:  # 无探索，选择最佳移动
            chosen_idx = torch.argmax(move_probs).item()
        
        # 转换为棋盘坐标（加1因为pos_to_coord从1开始）
        move_position = chosen_idx + 1
        x, y = self.pos_to_coord[move_position]
        
        # 根据当前玩家（黑/白）确定位置的符号
        player = self.game.current_player  # 1为黑，-1为白
        signed_position = player * move_position
        
        return (x, y), signed_position, move_probs
        
    def play_game(self, exploration_temp=1.0, max_moves=225):
        """
        进行一局自对弈游戏
        
        Args:
            exploration_temp: 探索温度
            max_moves: 最大移动次数（防止无限循环）
            
        Returns:
            game_moves: 游戏中的所有移动
            move_probs: 每一步的概率分布
            winner: 获胜方 (1: 黑方, -1: 白方, 0: 平局)
        """
        # 重置游戏
        self.game = GomokuRule(board_size=self.board_size)
        
        # 使用FiberTree跟踪游戏路径
        self.tree.start_path()
        
        # 游戏移动历史和对应的概率分布
        game_moves = []
        move_probs = []
        
        # 进行游戏直到结束或达到最大移动次数
        for _ in range(max_moves):
            # 获取当前移动历史作为模型输入
            current_path = [m.value for m in self.tree.get_complete_path()]
            
            # 模型选择移动
            (x, y), signed_position, probs = self._select_move(current_path, exploration_temp)
            
            # 执行移动
            success = self.game.make_move(x, y)
            if not success:
                print(f"警告：无效移动 ({x}, {y})")
                break
                
            # 记录移动和概率分布
            game_moves.append(signed_position)
            move_probs.append(probs)
            
            # 将移动添加到FiberTree
            self.tree.add_move(Move(signed_position))
            
            # 检查游戏是否结束
            if self.game.game_over:
                break
                
        # 记录胜者
        winner = self.game.winner
        
        # 如果游戏未结束但达到最大移动数，则视为平局
        if not self.game.game_over and len(game_moves) >= max_moves:
            winner = 0
            
        # 结束FiberTree路径并记录结果
        outcome = 'win' if winner == 1 else 'loss' if winner == -1 else 'draw'
        self.tree.record_outcome(outcome)
        self.tree.end_path()
            
        return game_moves, move_probs, winner
        
    def compute_rewards(self, moves, winner):
        """
        计算每一步的奖励
        
        Args:
            moves: 游戏中的所有移动
            winner: 获胜方 (1: 黑方, -1: 白方, 0: 平局)
            
        Returns:
            rewards: 每一步的奖励值
        """
        rewards = []
        
        # 如果平局，两方都获得小的负奖励
        if winner == 0:
            rewards = [-0.1] * len(moves)
            return rewards
            
        # 为每一步分配奖励，从后向前递减
        for i, move in enumerate(moves):
            player = 1 if move > 0 else -1  # 根据移动的符号确定玩家
            
            # 如果这步是由获胜方落子，给予正奖励；否则，给予负奖励
            base_reward = 1.0 if player == winner else -1.0
            
            # 根据移动的时间点应用折扣因子
            # 游戏越早的步骤，奖励越低（因为对最终结果的影响较小）
            discount = self.discount_factor ** (len(moves) - i - 1)
            rewards.append(base_reward * discount)
            
        return rewards
    
    def train_step(self, moves, probs, rewards):
        """
        执行一步训练
        
        Args:
            moves: 游戏中的所有移动
            probs: 每一步的模型预测概率分布
            rewards: 每一步的奖励值
            
        Returns:
            loss: 训练损失值
        """
        self.model.train()  # 设置为训练模式
        self.optimizer.zero_grad()  # 清除旧的梯度
        
        # 检查是否有有效的移动和概率分布用于训练
        if len(moves) == 0 or len(probs) == 0:
            return 0.0  # 如果没有有效的数据，返回零损失
        
        # 创建一个新的计算图用于训练
        # 这是必要的，因为原始的概率分布可能已经分离了梯度
        batch_size = len(moves)
        batch_loss = 0.0
        
        # 为每个移动准备输入序列
        for step, (move, prob, reward) in enumerate(zip(moves, probs, rewards)):
            # 获取当前步骤之前的序列
            if step == 0:
                # 第一步没有历史，使用空序列
                history = torch.tensor([[]], dtype=torch.long)
            else:
                # 使用之前的步骤作为历史
                history = torch.tensor([moves[:step]], dtype=torch.long)
            
            # 前向传播获取模型输出（确保这一次是在训练模式下）
            model_output = self.model(history)
            log_probs = torch.log_softmax(model_output, dim=1)[0]
            
            # 获取实际选择的位置
            position = abs(move) - 1  # 转换为0索引
            
            # 计算该位置的负对数概率，乘以奖励
            # 这样，如果奖励为正，我们会最小化负对数概率（增加选择该位置的概率）
            # 如果奖励为负，我们会最大化负对数概率（减少选择该位置的概率）
            if position < len(log_probs):
                action_loss = -log_probs[position] * reward
                batch_loss += action_loss
            
        # 计算平均损失
        if batch_size > 0:
            batch_loss = batch_loss / batch_size
            
        # 总损失
        loss = batch_loss
        
        # 反向传播
        batch_loss.backward()
        
        # 梯度裁剪以防止梯度爆炸
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        
        # 更新模型参数
        self.optimizer.step()
        
        return loss.item()
        
        # 总损失
        loss = policy_loss + entropy_loss
        
        # 反向传播
        loss.backward()
        
        # 梯度裁剪以防止梯度爆炸
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        
        # 更新模型参数
        self.optimizer.step()
        
        return loss.item()
    
    def train(self, num_games=1000, exploration_schedule=None):
        """
        训练模型
        
        Args:
            num_games: 要玩的自对弈游戏数量
            exploration_schedule: 探索温度随时间变化的调度函数
                                 None表示使用默认的线性衰减
                                 
        Returns:
            training_stats: 训练统计信息
        """
        if exploration_schedule is None:
            # 默认线性衰减温度，从1.0到0.1
            exploration_schedule = lambda game_idx: max(0.1, 1.0 - 0.9 * game_idx / num_games)
        
        training_stats = {
            'game_lengths': [],
            'winners': [],
            'losses': [],
            'exploration_temps': []
        }
        
        for game_idx in range(num_games):
            # 根据调度获取当前探索温度
            exploration_temp = exploration_schedule(game_idx)
            training_stats['exploration_temps'].append(exploration_temp)
            
            # 进行一局自对弈游戏
            moves, probs, winner = self.play_game(exploration_temp=exploration_temp)
            
            # 记录游戏统计信息
            training_stats['game_lengths'].append(len(moves))
            training_stats['winners'].append(winner)
            
            # 如果游戏过早结束（可能是模型bug导致无效移动），则跳过训练
            if len(moves) < 5:
                print(f"游戏 {game_idx+1} 过早结束，跳过训练")
                continue
                
            # 计算奖励
            rewards = self.compute_rewards(moves, winner)
            
            # 执行训练
            loss = self.train_step(moves, probs, rewards)
            training_stats['losses'].append(loss)
            
            # 定期打印训练进度
            if (game_idx + 1) % 10 == 0:
                print(f"游戏 {game_idx+1}/{num_games}, 长度: {len(moves)}, 赢家: {winner}, 损失: {loss:.4f}, 温度: {exploration_temp:.2f}")
            
            # 定期保存模型
            if (game_idx + 1) % 100 == 0:
                self.save_model(f"gomoku_model_game_{game_idx+1}.pth")
                
        return training_stats
    
    def save_model(self, filepath):
        """保存模型到文件"""
        torch.save(self.model.state_dict(), filepath)
    
    def load_model(self, filepath):
        """从文件加载模型"""
        self.model.load_state_dict(torch.load(filepath))


def main():
    # 创建模型
    model = GomokuLSTMModel(board_size=15, embedding_dim=128, hidden_dim=256, num_layers=2)
    
    # 创建训练器
    trainer = GomokuSelfPlayTrainer(model, board_size=15, discount_factor=0.95, learning_rate=0.001)
    
    # 定义自定义探索调度 - 开始时高温度，然后逐渐降低
    def custom_schedule(game_idx, num_games=1000):
        # 前20%的游戏使用高温度促进探索
        if game_idx < num_games * 0.2:
            return 1.5
        # 中间60%的游戏线性降低温度
        elif game_idx < num_games * 0.8:
            progress = (game_idx - num_games * 0.2) / (num_games * 0.6)
            return 1.5 - progress * 1.4  # 从1.5降至0.1
        # 最后20%的游戏使用低温度以利用学到的策略
        else:
            return 0.1
    
    try:
        # 进行训练
        print("开始训练...")
        # 先尝试运行少量游戏进行测试
        print("测试阶段：进行10局游戏确保系统正常工作")
        test_stats = trainer.train(num_games=10, exploration_schedule=lambda _: 1.0)
        print("测试成功，开始完整训练")
        
        # 如果测试成功，进行完整训练
        schedule_fn = lambda game_idx: custom_schedule(game_idx, num_games=1000)
        training_stats = trainer.train(num_games=1000, exploration_schedule=schedule_fn)
        
        # 保存最终模型
        trainer.save_model("gomoku_final_model.pth")
        
        print("训练完成!")
    except Exception as e:
        print(f"训练出错: {e}")
        import traceback
        traceback.print_exc()
        # 即使出错，也尝试保存当前模型
        try:
            trainer.save_model("gomoku_model_error_recovery.pth")
            print("已保存恢复模型")
        except:
            print("保存恢复模型失败")


if __name__ == "__main__":
    main()