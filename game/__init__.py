"""
Gomoku (五子棋) 游戏模块
"""

from .board import Board, Piece
from .rule import GomokuRule

__all__ = ['Board', 'Piece', 'GomokuRule']