"""
FiberTree: 用于存储和分析顺序决策路径的数据库系统

FiberTree 帮助您跟踪、分析和优化决策过程，通过记录决策路径（fibers）及其结果。

基本用法:
    from fbtree import create_tree, Move
    
    # 创建一个树
    tree = create_tree()
    
    # 开始构建路径
    tree.start_path()
    
    # 添加移动到路径
    tree.add_move(Move("A"))
    tree.add_move(Move("B"))
    tree.add_move(Move("C"))
    
    # 记录结果
    tree.record_outcome('win')
    
    # 结束路径
    tree.end_path()
    
    # 获取统计信息
    stats = tree.get_statistics()
"""

# 从主模块导出核心类
from .main import FiberTree, Fiber, Move

# 提供简化的创建接口
def create_tree(storage_type='memory', db_path=None, max_cache_size=1000):
    """
    创建一个新的 FiberTree，参数简化
    
    Args:
        storage_type: 'memory'（更快，非持久化）或 'sqlite'（持久化）
        db_path: SQLite 数据库文件路径（storage_type='sqlite'时需要）
        max_cache_size: 内存中缓存的最大 fiber 数量
        
    Returns:
        FiberTree: 一个新的树实例
    """
    return FiberTree(storage_type=storage_type, db_path=db_path, max_cache_size=max_cache_size)

def load_tree(file_path, storage_type='memory', db_path=None):
    """
    从文件加载 FiberTree
    
    Args:
        file_path: 要加载的文件路径（.json、.bin、.gz）
        storage_type: 加载树的 'memory' 或 'sqlite' 存储类型
        db_path: SQLite 数据库路径（storage_type='sqlite'时需要）
        
    Returns:
        FiberTree: 加载的树实例
    """
    if file_path.endswith('.pickle') or file_path.endswith('.pkl') or file_path.endswith('.bin') or file_path.endswith('.gz'):
        return FiberTree.import_binary(file_path, storage_type, db_path)
    else:
        return FiberTree.import_from_json(file_path, storage_type, db_path)

# 为 FiberTree 类添加便捷方法别名
FiberTree.start_path = FiberTree.start_adding_mode if hasattr(FiberTree, 'start_adding_mode') else FiberTree.start_path
FiberTree.end_path = FiberTree.end_adding_mode if hasattr(FiberTree, 'end_adding_mode') else FiberTree.end_path
FiberTree.record_outcome = FiberTree.update_statistics if hasattr(FiberTree, 'update_statistics') else FiberTree.record_outcome
FiberTree.save = FiberTree.export_to_json if hasattr(FiberTree, 'export_to_json') else FiberTree.save

# 版本信息
__version__ = "1.1.0"