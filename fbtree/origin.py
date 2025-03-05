"""
FiberTree: A path-oriented database for storing and analyzing sequential decision paths
"""

from typing import List, Tuple, Dict, Optional, Any, Iterator, Union, Generic, TypeVar
from dataclasses import dataclass
import json
import sqlite3
import os
import uuid
import logging
from collections import Counter
import datetime

T = TypeVar('T')

@dataclass
class Move(Generic[T]):
    """表示一个移动/决策的通用类"""
    value: T
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
    
    def __eq__(self, other):
        if not isinstance(other, Move):
            return False
        return self.value == other.value
    
    def __hash__(self):
        return hash(self.value)
    
    def __str__(self):
        return str(self.value)
    
    def to_dict(self) -> Dict[str, Any]:
        """将Move转换为字典表示"""
        return {
            'value': self.value,
            'metadata': self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Move':
        """从字典创建Move对象"""
        return cls(
            value=data['value'],
            metadata=data.get('metadata', {})
        )


class Fiber:
    """表示决策路径的容器类"""
    
    def __init__(self, 
                 moves: List[Move],
                 fiber_id: str = None,
                 parent_id: str = None,
                 metadata: Dict[str, Any] = None):
        """
        初始化一个Fiber
        
        Args:
            moves: 此Fiber包含的移动列表
            fiber_id: 唯一标识符（如果为None则自动生成）
            parent_id: 父Fiber的ID（如果有）
            metadata: 与此Fiber相关的任意元数据
        """
        self.moves = moves.copy()
        self.fiber_id = fiber_id if fiber_id else str(uuid.uuid4())
        self.parent_id = parent_id
        self.metadata = metadata if metadata else {}
        self.stats = {
            'visit_count': 0,
            'win_count': 0,
            'loss_count': 0,
            'draw_count': 0
        }
    
    def is_empty(self) -> bool:
        """检查是否为空Fiber"""
        return len(self.moves) == 0
    
    def __len__(self) -> int:
        """获取Fiber长度（移动数量）"""
        return len(self.moves)
    
    def __getitem__(self, index) -> Move:
        """通过索引访问移动"""
        return self.moves[index]
    
    def get_win_rate(self) -> float:
        """计算胜率"""
        if self.stats['visit_count'] == 0:
            return 0.0
        return self.stats['win_count'] / self.stats['visit_count']
    
    def update_stats(self, outcome: str):
        """
        更新统计信息
        
        Args:
            outcome: 'win', 'loss', 'draw' 之一
        """
        self.stats['visit_count'] += 1
        if outcome == 'win':
            self.stats['win_count'] += 1
        elif outcome == 'loss':
            self.stats['loss_count'] += 1
        elif outcome == 'draw':
            self.stats['draw_count'] += 1
    
    def add_metadata(self, key: str, value: Any):
        """添加或更新元数据"""
        self.metadata[key] = value
    
    def to_dict(self) -> Dict[str, Any]:
        """将Fiber转换为字典表示"""
        return {
            'fiber_id': self.fiber_id,
            'parent_id': self.parent_id,
            'moves': [move.to_dict() for move in self.moves],
            'metadata': self.metadata,
            'stats': self.stats
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Fiber':
        """从字典创建Fiber对象"""
        fiber = cls(
            moves=[Move.from_dict(m) for m in data['moves']],
            fiber_id=data['fiber_id'],
            parent_id=data['parent_id'],
            metadata=data.get('metadata', {})
        )
        fiber.stats = data.get('stats', {
            'visit_count': 0,
            'win_count': 0,
            'loss_count': 0,
            'draw_count': 0
        })
        return fiber


class FiberTree:
    """管理决策路径集合的树形数据库"""
    
    def __init__(self, 
                 storage_type: str = 'memory',
                 db_path: str = None,
                 max_cache_size: int = 1000,
                 logger: logging.Logger = None):
        """
        初始化FiberTree
        
        Args:
            storage_type: 'memory' 或 'sqlite'
            db_path: SQLite数据库路径（当storage_type='sqlite'时需要）
            max_cache_size: 内存缓存的最大项数
            logger: 可选的日志记录器
        """
        self.storage_type = storage_type
        self.db_path = db_path
        self.max_cache_size = max_cache_size
        self.logger = logger or logging.getLogger('FiberTree')
        
        # 内存存储
        self._memory_store: Dict[str, Fiber] = {}
        self._fiber_cache: Dict[str, Fiber] = {}
        self._children_map: Dict[str, List[str]] = {}
        
        # 当前正在构建的路径
        self.current_path: List[Move] = []
        self.current_fiber: Optional[Fiber] = None
        self.adding_mode = False
        
        # 创建根Fiber
        self.root = Fiber(moves=[], fiber_id='root')
        self._memory_store['root'] = self.root
        
        # 如果使用SQLite，初始化数据库
        if storage_type == 'sqlite':
            self._init_database()
    
    def _init_database(self):
        """初始化SQLite数据库"""
        if not self.db_path:
            raise ValueError("使用SQLite存储时必须提供db_path")
            
        os.makedirs(os.path.dirname(os.path.abspath(self.db_path)), exist_ok=True)
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # 创建fibers表
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS fibers (
            fiber_id TEXT PRIMARY KEY,
            parent_id TEXT,
            data TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        ''')
        
        # 创建索引
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_parent_id ON fibers(parent_id)')
        
        # 确保根节点存在
        cursor.execute('SELECT fiber_id FROM fibers WHERE fiber_id = ?', ('root',))
        if not cursor.fetchone():
            root_data = json.dumps(self.root.to_dict())
            cursor.execute(
                'INSERT INTO fibers (fiber_id, parent_id, data) VALUES (?, ?, ?)',
                ('root', None, root_data)
            )
        
        conn.commit()
        conn.close()
    
    def _get_fiber(self, fiber_id: str) -> Optional[Fiber]:
        """获取指定ID的Fiber"""
        # 首先检查缓存
        if fiber_id in self._fiber_cache:
            return self._fiber_cache[fiber_id]
        
        # 然后检查内存存储
        if fiber_id in self._memory_store:
            fiber = self._memory_store[fiber_id]
            self._update_cache(fiber_id, fiber)
            return fiber
        
        # 如果使用SQLite存储，从数据库获取
        if self.storage_type == 'sqlite':
            return self._get_fiber_from_db(fiber_id)
        
        return None
    
    def _get_fiber_from_db(self, fiber_id: str) -> Optional[Fiber]:
        """从SQLite数据库获取Fiber"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT data FROM fibers WHERE fiber_id = ?', (fiber_id,))
        result = cursor.fetchone()
        conn.close()
        
        if result:
            fiber_data = json.loads(result[0])
            fiber = Fiber.from_dict(fiber_data)
            self._update_cache(fiber_id, fiber)
            return fiber
        
        return None
    
    def _save_fiber(self, fiber: Fiber):
        """保存Fiber到存储"""
        if self.storage_type == 'memory':
            self._memory_store[fiber.fiber_id] = fiber
            self._update_cache(fiber.fiber_id, fiber)
            
            # 更新父子关系映射
            if fiber.parent_id:
                if fiber.parent_id not in self._children_map:
                    self._children_map[fiber.parent_id] = []
                if fiber.fiber_id not in self._children_map[fiber.parent_id]:
                    self._children_map[fiber.parent_id].append(fiber.fiber_id)
        else:
            self._save_fiber_to_db(fiber)
    
    def _save_fiber_to_db(self, fiber: Fiber):
        """保存Fiber到SQLite数据库"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        fiber_data = json.dumps(fiber.to_dict())
        
        cursor.execute(
            '''
            INSERT OR REPLACE INTO fibers (fiber_id, parent_id, data, last_updated)
            VALUES (?, ?, ?, CURRENT_TIMESTAMP)
            ''',
            (fiber.fiber_id, fiber.parent_id, fiber_data)
        )
        
        conn.commit()
        conn.close()
        
        self._update_cache(fiber.fiber_id, fiber)
    
    def _update_cache(self, fiber_id: str, fiber: Fiber):
        """更新缓存"""
        # 如果缓存已满，移除最早添加的项
        if len(self._fiber_cache) >= self.max_cache_size:
            oldest_key = next(iter(self._fiber_cache))
            del self._fiber_cache[oldest_key]
        
        self._fiber_cache[fiber_id] = fiber
    
    def _get_children(self, fiber_id: str) -> List[str]:
        """获取指定Fiber的所有子Fiber ID"""
        if self.storage_type == 'memory':
            return self._children_map.get(fiber_id, [])
        else:
            return self._get_children_from_db(fiber_id)
    
    def _get_children_from_db(self, fiber_id: str) -> List[str]:
        """从数据库获取子Fiber ID"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT fiber_id FROM fibers WHERE parent_id = ?', (fiber_id,))
        results = cursor.fetchall()
        conn.close()
        
        return [r[0] for r in results]
    
    def _find_matching_fiber(self, current_fiber_id: str, path: List[Move], 
                             start_idx: int) -> Tuple[Optional[str], int]:
        """
        查找匹配给定路径的下一个fiber
        
        Args:
            current_fiber_id: 当前Fiber ID
            path: 要匹配的完整路径
            start_idx: 路径中的起始索引
            
        Returns:
            Tuple[Optional[str], int]: 匹配的Fiber ID和匹配的移动数量
        """
        if start_idx >= len(path):
            return None, 0
            
        remaining_moves = path[start_idx:]
        child_ids = self._get_children(current_fiber_id)
        
        best_match_id = None
        best_match_len = 0
        
        for child_id in child_ids:
            child = self._get_fiber(child_id)
            if not child:
                continue
                
            # 计算最长匹配前缀
            match_len = 0
            for i in range(min(len(child.moves), len(remaining_moves))):
                if child.moves[i] == remaining_moves[i]:
                    match_len += 1
                else:
                    break
            
            if match_len > 0 and match_len > best_match_len:
                best_match_id = child_id
                best_match_len = match_len
        
        return best_match_id, best_match_len
    
    def start_adding_mode(self):
        """开始添加模式以构建新路径"""
        self.adding_mode = True
        self.current_fiber = self.root
        self.current_path = []
    
    def end_adding_mode(self):
        """结束添加模式"""
        self.adding_mode = False
    
    def add_move(self, move: Move) -> bool:
        """
        添加一个移动到当前路径
        
        Args:
            move: 要添加的移动
            
        Returns:
            bool: 操作是否成功
        """
        if not self.adding_mode:
            self.logger.warning("在添加模式外调用add_move")
            return False
        
        self.current_path.append(move)
        
        # 获取当前Fiber的子节点
        children_ids = self._get_children(self.current_fiber.fiber_id)
        matching_child_id = None
        
        # 检查是否有匹配的子节点
        for child_id in children_ids:
            child = self._get_fiber(child_id)
            if not child or not child.moves:
                continue
                
            if child.moves[0] == move:
                matching_child_id = child_id
                break
        
        if matching_child_id:
            # 找到匹配的子节点，使用它
            self.current_fiber = self._get_fiber(matching_child_id)
        else:
            # 创建新的子Fiber
            new_fiber = Fiber(
                moves=[move],
                parent_id=self.current_fiber.fiber_id
            )
            self._save_fiber(new_fiber)
            self.current_fiber = new_fiber
        
        return True
    
    def add_moves(self, moves: List[Move]) -> bool:
        """
        批量添加多个移动
        
        Args:
            moves: 要添加的移动列表
            
        Returns:
            bool: 操作是否成功
        """
        if not self.adding_mode:
            self.start_adding_mode()
            
        success = True
        for move in moves:
            if not self.add_move(move):
                success = False
                
        return success
    
    def update_statistics(self, outcome: str):
        """
        更新当前路径的统计信息
        
        Args:
            outcome: 'win', 'loss', 或 'draw'
        """
        if not self.current_fiber:
            self.logger.warning("尝试更新统计信息，但没有当前Fiber")
            return
            
        # 从当前Fiber开始，向上更新所有父Fiber
        fiber = self.current_fiber
        updated_ids = set()
        
        while fiber and fiber.fiber_id not in updated_ids:
            fiber.update_stats(outcome)
            self._save_fiber(fiber)
            updated_ids.add(fiber.fiber_id)
            
            if fiber.parent_id and fiber.parent_id != 'root':
                fiber = self._get_fiber(fiber.parent_id)
            else:
                # 更新根节点
                if fiber.fiber_id != 'root':
                    self.root.update_stats(outcome)
                    self._save_fiber(self.root)
                break
    
    def get_complete_path(self) -> List[Move]:
        """
        获取从根节点到当前节点的完整移动序列
        
        Returns:
            List[Move]: 完整的移动序列
        """
        if not self.current_fiber:
            return []
            
        # 向上追踪路径
        complete_moves = []
        fiber = self.current_fiber
        fiber_chain = []
        
        while fiber and fiber.fiber_id != 'root':
            fiber_chain.append(fiber)
            fiber = self._get_fiber(fiber.parent_id)
        
        # 从上到下构建路径
        for fiber in reversed(fiber_chain):
            complete_moves.extend(fiber.moves)
            
        return complete_moves
    
    def find_path(self, moves: List[Move]) -> Optional[str]:
        """
        查找匹配给定移动序列的路径
        
        Args:
            moves: 要查找的移动序列
            
        Returns:
            Optional[str]: 匹配的Fiber ID，如果未找到则为None
        """
        current_fiber_id = 'root'
        remaining_idx = 0
        
        while remaining_idx < len(moves):
            next_fiber_id, match_len = self._find_matching_fiber(
                current_fiber_id, moves, remaining_idx
            )
            
            if not next_fiber_id or match_len == 0:
                return None
                
            current_fiber_id = next_fiber_id
            remaining_idx += match_len
            
        return current_fiber_id if remaining_idx == len(moves) else None
    
    def get_statistics(self, fiber_id: str = None) -> Dict[str, Any]:
        """
        获取指定Fiber的统计信息
        
        Args:
            fiber_id: 要查询的Fiber ID，默认为当前Fiber
            
        Returns:
            Dict: 统计信息
        """
        if not fiber_id:
            if self.current_fiber:
                fiber_id = self.current_fiber.fiber_id
            else:
                fiber_id = 'root'
                
        fiber = self._get_fiber(fiber_id)
        if not fiber:
            return {}
            
        result = fiber.stats.copy()
        result['win_rate'] = fiber.get_win_rate()
        result['fiber_id'] = fiber.fiber_id
        result['parent_id'] = fiber.parent_id
        result['moves_count'] = len(fiber.moves)
        
        return result
    
    def get_common_path_statistics(self, min_visits: int = 5) -> List[Dict[str, Any]]:
        """
        获取常用路径的统计信息
        
        Args:
            min_visits: 最小访问次数阈值
            
        Returns:
            List[Dict]: 常用路径统计信息列表
        """
        result = []
        
        # 根据存储类型使用不同的查询方法
        if self.storage_type == 'memory':
            for fiber_id, fiber in self._memory_store.items():
                if fiber.stats['visit_count'] >= min_visits:
                    path = self._get_path_to_fiber(fiber_id)
                    result.append({
                        'fiber_id': fiber_id,
                        'path': path,
                        'visits': fiber.stats['visit_count'],
                        'win_rate': fiber.get_win_rate(),
                        'depth': len(path)
                    })
        else:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute(
                'SELECT fiber_id, data FROM fibers'
            )
            
            for row in cursor.fetchall():
                fiber_id, data_json = row
                fiber_data = json.loads(data_json)
                fiber = Fiber.from_dict(fiber_data)
                
                if fiber.stats.get('visit_count', 0) >= min_visits:
                    path = self._get_path_to_fiber(fiber_id)
                    result.append({
                        'fiber_id': fiber_id,
                        'path': path,
                        'visits': fiber.stats['visit_count'],
                        'win_rate': fiber.get_win_rate(),
                        'depth': len(path)
                    })
            
            conn.close()
        
        # 按访问次数排序
        return sorted(result, key=lambda x: x['visits'], reverse=True)
    
    def _get_path_to_fiber(self, fiber_id: str) -> List[Move]:
        """获取从根节点到指定Fiber的路径"""
        path = []
        current_id = fiber_id
        
        visited = set()  # 防止循环引用
        
        while current_id and current_id != 'root' and current_id not in visited:
            visited.add(current_id)
            fiber = self._get_fiber(current_id)
            if not fiber:
                break
                
            path = fiber.moves + path
            current_id = fiber.parent_id
            
        return path
    
    def prune_tree(self, min_visits: int = 1, max_depth: int = None) -> int:
        """
        修剪树，删除很少访问或超过最大深度的分支
        
        Args:
            min_visits: 最小访问次数阈值
            max_depth: 最大深度阈值
            
        Returns:
            int: 删除的Fiber数量
        """
        removed_count = 0
        
        if self.storage_type == 'memory':
            to_remove = []
            
            # 标记需要删除的Fiber
            for fiber_id, fiber in self._memory_store.items():
                if fiber_id == 'root':
                    continue
                    
                path_len = len(self._get_path_to_fiber(fiber_id))
                if (fiber.stats['visit_count'] < min_visits or 
                    (max_depth and path_len > max_depth)):
                    to_remove.append(fiber_id)
            
            # 删除标记的Fiber
            for fiber_id in to_remove:
                self._remove_fiber(fiber_id)
                removed_count += 1
                
        else:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # 获取所有非根Fiber
            cursor.execute('SELECT fiber_id, data FROM fibers WHERE fiber_id != "root"')
            
            to_remove = []
            for row in cursor.fetchall():
                fiber_id, data_json = row
                fiber_data = json.loads(data_json)
                fiber = Fiber.from_dict(fiber_data)
                
                path_len = len(self._get_path_to_fiber(fiber_id))
                if (fiber.stats.get('visit_count', 0) < min_visits or 
                    (max_depth and path_len > max_depth)):
                    to_remove.append(fiber_id)
            
            # 删除标记的Fiber
            for fiber_id in to_remove:
                cursor.execute('DELETE FROM fibers WHERE fiber_id = ?', (fiber_id,))
                removed_count += 1
                
                # 从缓存中删除
                if fiber_id in self._fiber_cache:
                    del self._fiber_cache[fiber_id]
            
            conn.commit()
            conn.close()
            
        self.logger.info(f"修剪了{removed_count}个Fiber")
        return removed_count
    
    def _remove_fiber(self, fiber_id: str):
        """从存储中删除Fiber"""
        if fiber_id == 'root':
            self.logger.warning("尝试删除根Fiber，操作被拒绝")
            return
            
        # 递归删除所有子Fiber
        children = self._get_children(fiber_id)
        for child_id in children:
            self._remove_fiber(child_id)
            
        # 从内存存储中删除
        if fiber_id in self._memory_store:
            fiber = self._memory_store[fiber_id]
            parent_id = fiber.parent_id
            
            del self._memory_store[fiber_id]
            
            # 更新父子关系映射
            if parent_id in self._children_map and fiber_id in self._children_map[parent_id]:
                self._children_map[parent_id].remove(fiber_id)
                
        # 从缓存中删除
        if fiber_id in self._fiber_cache:
            del self._fiber_cache[fiber_id]
            
    def get_move_frequency(self, depth: int = 1, min_visits: int = 0) -> Dict[str, int]:
        """
        获取特定深度的移动频率统计
        
        Args:
            depth: 要分析的深度（从1开始）
            min_visits: 最小访问次数阈值
            
        Returns:
            Dict[str, int]: 移动值到频率的映射
        """
        counter = Counter()
        
        # 从根节点的子节点开始，递归收集指定深度的移动
        def collect_moves_at_depth(fiber_id: str, current_depth: int = 0):
            if current_depth == depth:
                fiber = self._get_fiber(fiber_id)
                if fiber and fiber.stats['visit_count'] >= min_visits:
                    # 如果这是我们要找的深度，计算这个移动
                    if fiber.moves:
                        counter[str(fiber.moves[0].value)] += 1
                return
                
            # 递归检查子节点
            children = self._get_children(fiber_id)
            for child_id in children:
                collect_moves_at_depth(child_id, current_depth + 1)
                
        # 从深度0（根节点）开始收集
        collect_moves_at_depth('root')
        return dict(counter)
    
    def generate_move_heatmap(self, board_size: int) -> List[List[int]]:
        """
        为棋盘游戏生成移动热图
        
        Args:
            board_size: 棋盘大小
            
        Returns:
            List[List[int]]: 二维热图数组
        """
        # 初始化热图
        heatmap = [[0 for _ in range(board_size)] for _ in range(board_size)]
        
        # 收集所有移动
        all_moves = self.get_move_frequency(min_visits=1)
        
        # 填充热图
        for move_str, frequency in all_moves.items():
            try:
                move_value = int(move_str)
                row = move_value // board_size
                col = move_value % board_size
                if 0 <= row < board_size and 0 <= col < board_size:
                    heatmap[row][col] += frequency
            except (ValueError, TypeError):
                continue
                
        return heatmap
    
    def get_best_continuation(self, 
                             current_path: List[Move], 
                             top_n: int = 3, 
                             min_visits: int = 5) -> List[Dict[str, Any]]:
        """
        获取当前路径的最佳后续移动
        
        Args:
            current_path: 当前移动路径
            top_n: 返回的最佳后续移动数量
            min_visits: 最小访问次数阈值
            
        Returns:
            List[Dict]: 最佳后续移动列表
        """
        # 首先找到匹配当前路径的Fiber
        fiber_id = self.find_path(current_path)
        if not fiber_id:
            return []
            
        # 获取所有子Fiber
        children_ids = self._get_children(fiber_id)
        continuations = []
        
        for child_id in children_ids:
            child = self._get_fiber(child_id)
            if not child or child.stats['visit_count'] < min_visits:
                continue
                
            if not child.moves:
                continue
                
            continuations.append({
                'move': child.moves[0],
                'win_rate': child.get_win_rate(),
                'visits': child.stats['visit_count'],
                'fiber_id': child_id
            })
            
        # 按胜率排序
        sorted_continuations = sorted(
            continuations,
            key=lambda x: (x['win_rate'], x['visits']),
            reverse=True
        )
        
        return sorted_continuations[:top_n]
    
    def export_to_json(self, file_path: str):
        """
        将整个树导出为JSON文件
        
        Args:
            file_path: 输出JSON文件路径
        """
        tree_data = {
            'metadata': {
                'version': '1.0',
                'storage_type': self.storage_type,
                'exported_at': datetime.datetime.now().isoformat()
            },
            'fibers': {}
        }
        
        # 收集所有Fiber
        if self.storage_type == 'memory':
            for fiber_id, fiber in self._memory_store.items():
                tree_data['fibers'][fiber_id] = fiber.to_dict()
        else:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('SELECT fiber_id, data FROM fibers')
            for row in cursor.fetchall():
                fiber_id, data_json = row
                fiber_data = json.loads(data_json)
                tree_data['fibers'][fiber_id] = fiber_data
            
            conn.close()
        
        # 创建目录（如果不存在）
        os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
        
        # 写入JSON文件
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(tree_data, f, indent=2, ensure_ascii=False)
            
        self.logger.info(f"已将树导出到 {file_path}")

    @classmethod
    def import_from_json(cls, file_path: str, 
                        storage_type: str = 'memory',
                        db_path: str = None) -> 'FiberTree':
        """
        从JSON文件导入树
        
        Args:
            file_path: JSON文件路径
            storage_type: 目标存储类型
            db_path: 如果storage_type='sqlite'，则为数据库路径
            
        Returns:
            FiberTree: 导入的树实例
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            tree_data = json.load(f)
            
        # 创建新的FiberTree实例
        tree = cls(storage_type=storage_type, db_path=db_path)
        
        # 导入所有Fiber
        for fiber_id, fiber_data in tree_data['fibers'].items():
            fiber = Fiber.from_dict(fiber_data)
            tree._save_fiber(fiber)
            
            # 如果是root，更新树的root引用
            if fiber_id == 'root':
                tree.root = fiber
        
        tree.logger.info(f"已从 {file_path} 导入树")
        return tree

    def _import_subtree(self, source_tree: 'FiberTree', source_id: str, target_parent_id: str) -> int:
        """从另一棵树导入子树"""
        count = 0
        source_fiber = source_tree._get_fiber(source_id)
        
        if not source_fiber:
            return count
            
        # 创建新Fiber
        new_fiber = Fiber(
            moves=source_fiber.moves,
            parent_id=target_parent_id,
            metadata=source_fiber.metadata.copy()
        )
        new_fiber.stats = source_fiber.stats.copy()
        self._save_fiber(new_fiber)
        count += 1
        
        # 递归处理子节点
        source_children = source_tree._get_children(source_id)
        for child_id in source_children:
            sub_count = self._import_subtree(source_tree, child_id, new_fiber.fiber_id)
            count += sub_count
            
        return count

    def merge(self, other_tree: 'FiberTree', conflict_strategy: str = 'stats_sum') -> int:
        """
        合并另一个树到当前树
        
        Args:
            other_tree: 要合并的树
            conflict_strategy: 冲突处理策略:
                - 'stats_sum': 合并统计数据
                - 'keep_max': 保留访问次数最多的
                - 'keep_current': 保留当前树的数据
                
        Returns:
            int: 合并的Fiber数量
        """
        # 跟踪合并的节点数量
        merged_count = 0
        
        # 首先处理根节点
        root_fiber = self._get_fiber('root')
        other_root = other_tree._get_fiber('root')
        
        if conflict_strategy == 'stats_sum':
            for key in ['visit_count', 'win_count', 'loss_count', 'draw_count']:
                root_fiber.stats[key] += other_root.stats.get(key, 0)
            self._save_fiber(root_fiber)
        
        # 处理所有非根节点
        if self.storage_type == 'memory' and other_tree.storage_type == 'memory':
            # 高效的内存模式合并
            for other_id, other_fiber in other_tree._memory_store.items():
                if other_id == 'root':
                    continue  # 根节点已处理
                    
                # 获取路径
                path = other_tree._get_path_to_fiber(other_id)
                if not path:
                    continue
                    
                # 在当前树中查找或创建路径
                self.start_adding_mode()
                for move in path:
                    self.add_move(move)
                
                # 更新统计信息
                fiber_id = self.current_fiber.fiber_id
                current_fiber = self._get_fiber(fiber_id)
                
                if conflict_strategy == 'stats_sum':
                    for key in ['visit_count', 'win_count', 'loss_count', 'draw_count']:
                        current_fiber.stats[key] += other_fiber.stats.get(key, 0)
                elif conflict_strategy == 'keep_max':
                    if other_fiber.stats.get('visit_count', 0) > current_fiber.stats.get('visit_count', 0):
                        current_fiber.stats = other_fiber.stats.copy()
                
                self._save_fiber(current_fiber)
                self.end_adding_mode()
                
                merged_count += 1
        else:
            # 通用方法，适用于SQLite模式
            paths = []
            for fiber_id, fiber in other_tree:
                if fiber_id == 'root':
                    continue
                path = other_tree._get_path_to_fiber(fiber_id)
                if path:
                    paths.append((path, fiber.stats))
            
            # 添加所有路径
            for path, stats in paths:
                self.start_adding_mode()
                for move in path:
                    self.add_move(move)
                    
                fiber_id = self.current_fiber.fiber_id
                current_fiber = self._get_fiber(fiber_id)
                
                if conflict_strategy == 'stats_sum':
                    for key in ['visit_count', 'win_count', 'loss_count', 'draw_count']:
                        current_fiber.stats[key] += stats.get(key, 0)
                elif conflict_strategy == 'keep_max':
                    if stats.get('visit_count', 0) > current_fiber.stats.get('visit_count', 0):
                        current_fiber.stats = stats.copy()
                        
                self._save_fiber(current_fiber)
                self.end_adding_mode()
                
                merged_count += 1
        
        self.logger.info(f"合并了 {merged_count} 个Fiber")
        return merged_count

    def analyze_path_diversity(self) -> Dict[str, Any]:
        """
        分析路径多样性和分布情况
        
        Returns:
            Dict: 多样性分析统计
        """
        stats = {
            'total_fibers': 0,
            'max_depth': 0,
            'avg_branching_factor': 0,
            'leaf_nodes': 0,
            'depth_distribution': {},
            'most_visited_paths': []
        }
        
        # 收集所有非空节点
        all_fibers = []
        if self.storage_type == 'memory':
            all_fibers = [(fid, self._get_fiber(fid)) for fid in self._memory_store.keys()]
        else:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute('SELECT fiber_id, data FROM fibers')
            
            for row in cursor.fetchall():
                fiber_id, data_json = row
                all_fibers.append((fiber_id, Fiber.from_dict(json.loads(data_json))))
                
            conn.close()
            
        # 计算基本统计
        stats['total_fibers'] = len(all_fibers)
        
        # 计算分支因子和深度分布
        depth_counts = {}
        branching_factors = []
        leaf_count = 0
        
        for fiber_id, _ in all_fibers:
            if fiber_id == 'root':
                continue
                
            # 计算深度
            path = self._get_path_to_fiber(fiber_id)
            depth = len(path)
            depth_counts[depth] = depth_counts.get(depth, 0) + 1
            
            # 更新最大深度
            stats['max_depth'] = max(stats['max_depth'], depth)
            
            # 计算分支因子
            children = self._get_children(fiber_id)
            if not children:
                leaf_count += 1
            else:
                branching_factors.append(len(children))
        
        # 叶节点统计
        stats['leaf_nodes'] = leaf_count
        
        # 深度分布
        stats['depth_distribution'] = depth_counts
        
        # 平均分支因子
        if branching_factors:
            stats['avg_branching_factor'] = sum(branching_factors) / len(branching_factors)
        
        # 最常访问路径
        most_visited = self.get_common_path_statistics(min_visits=10)
        stats['most_visited_paths'] = most_visited[:10]  # 前10个
        
        return stats

    def simulate_path(self, 
                    path: List[Move], 
                    outcome: str,
                    visits: int = 1,
                    update_stats: bool = True) -> str:
        """
        模拟一个路径的多次访问，用于初始化或增强树的知识
        
        Args:
            path: 要模拟的移动路径
            outcome: 'win', 'loss' 或 'draw'
            visits: 模拟的访问次数
            update_stats: 是否更新统计数据
            
        Returns:
            str: 创建或更新的最终Fiber ID
        """
        if not path:
            return 'root'
            
        self.start_adding_mode()
        for move in path:
            self.add_move(move)
            
        final_fiber_id = self.current_fiber.fiber_id
        
        if update_stats:
            for _ in range(visits):
                self.update_statistics(outcome)
                
        self.end_adding_mode()
        return final_fiber_id

    def visualize(self, max_depth: int = 5, output_format: str = 'text'):
        """
        可视化树结构
        
        Args:
            max_depth: 最大显示深度
            output_format: 'text' 或 'graphviz'
        """
        if output_format == 'text':
            self._visualize_text(max_depth)
        elif output_format == 'graphviz':
            return self._generate_graphviz(max_depth)
        else:
            self.logger.warning(f"不支持的输出格式: {output_format}")

    def _visualize_text(self, max_depth: int = 5):
        """树形文本可视化"""
        def print_node(fiber_id: str, prefix: str = "", depth: int = 0):
            if depth > max_depth:
                print(f"{prefix}└── ...")
                return
                
            fiber = self._get_fiber(fiber_id)
            if not fiber:
                return
                
            # 节点信息
            if fiber_id == 'root':
                print("Root")
            else:
                moves_str = ", ".join(str(m) for m in fiber.moves)
                stats_str = f"[访问: {fiber.stats['visit_count']}, 胜率: {fiber.get_win_rate():.2f}]"
                print(f"{prefix}└── {moves_str} {stats_str}")
            
            # 处理子节点
            children = self._get_children(fiber_id)
            new_prefix = prefix + "    "
            
            for i, child_id in enumerate(children):
                print_node(child_id, new_prefix, depth + 1)
                
                # 限制显示的子节点数量
                if i >= 9 and len(children) > 10:
                    remaining = len(children) - i - 1
                    print(f"{new_prefix}└── ... 还有 {remaining} 个分支")
                    break
        
        # 从根节点开始打印
        print("\n树结构可视化:")
        print("Root")  # 修复：确保打印根节点
        
        # 处理根节点的子节点
        children = self._get_children('root')
        prefix = "    "
        for i, child_id in enumerate(children):
            print_node(child_id, prefix, 1)
            
            if i >= 9 and len(children) > 10:
                remaining = len(children) - i - 1
                print(f"{prefix}└── ... 还有 {remaining} 个分支")
                break

    def _generate_graphviz(self, max_depth: int = 5) -> str:
        """
        生成Graphviz DOT格式的图形表示
        
        Returns:
            str: DOT格式字符串
        """
        lines = ['digraph FiberTree {', 
                '  node [shape=box, style=filled];',
                '  edge [color=gray50];',
                '  rankdir=LR;']
        
        # 修复：确保添加根节点
        lines.append('  "root" [label="Root", fillcolor="#E5F5E0"];')
        
        visited = set(['root'])  # 标记根节点为已访问
        
        def process_node(fiber_id: str, depth: int = 0):
            if depth > max_depth or fiber_id in visited:
                return
                
            visited.add(fiber_id)
            fiber = self._get_fiber(fiber_id)
            if not fiber:
                return
                
            # 节点样式
            if fiber_id != 'root':  # 根节点已经处理过
                label = fiber_id
                color = '"#E5F5E0"'  # 默认颜色
                
                move_str = ", ".join(str(m.value) for m in fiber.moves)
                visits = fiber.stats['visit_count']
                win_rate = fiber.get_win_rate()
                
                label = f"{move_str}\\nVisits: {visits}\\nWin Rate: {win_rate:.2f}"
                
                # 根据胜率设置颜色
                if win_rate > 0.7:
                    color = '"#A1D99B"'  # 高胜率
                elif win_rate > 0.4:
                    color = '"#E5F5E0"'  # 中等胜率
                else:
                    color = '"#FEE0D2"'  # 低胜率
                    
                # 添加节点定义
                lines.append(f'  "{fiber_id}" [label="{label}", fillcolor={color}];')
                
            # 处理子节点
            children = self._get_children(fiber_id)
            for child_id in children:
                if child_id not in visited:
                    child = self._get_fiber(child_id)
                    if child and child.moves:
                        edge_label = str(child.moves[0].value)
                        lines.append(f'  "{fiber_id}" -> "{child_id}" [label="{edge_label}"];')
                        process_node(child_id, depth + 1)
        
        # 从根节点开始处理子节点
        children = self._get_children('root')
        for child_id in children:
            child = self._get_fiber(child_id)
            if child and child.moves:
                edge_label = str(child.moves[0].value)
                lines.append(f'  "root" -> "{child_id}" [label="{edge_label}"];')
                process_node(child_id, 1)
            
        lines.append('}')
        
        return '\n'.join(lines)

    def __len__(self) -> int:
        """获取树中的Fiber总数"""
        if self.storage_type == 'memory':
            return len(self._memory_store)
        else:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute('SELECT COUNT(*) FROM fibers')
            count = cursor.fetchone()[0]
            conn.close()
            return count

    def __iter__(self) -> Iterator[Tuple[str, Fiber]]:
        """迭代树中的所有Fiber"""
        if self.storage_type == 'memory':
            for fiber_id, fiber in self._memory_store.items():
                yield fiber_id, fiber
        else:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute('SELECT fiber_id, data FROM fibers')
            
            for row in cursor.fetchall():
                fiber_id, data_json = row
                fiber = Fiber.from_dict(json.loads(data_json))
                yield fiber_id, fiber
                
            conn.close()