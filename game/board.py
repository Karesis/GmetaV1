class Piece:
    def __init__(self, color=0):
        self.color=color
    
    def __str__(self):
        if self.color == 1:
            return "●"
        elif self.color == -1:
            return "○"
        else:
            return "十"
