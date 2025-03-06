import torch
import torch.nn as nn

class GomokuLSTMModel(nn.Module):
    def __init__(self, board_size=15, embedding_dim=128, hidden_dim=256, num_layers=2, dropout=0.2):
        """
        Neural network model for predicting the next Gomoku (Five-in-a-Row) move.
        
        Args:
            board_size: Size of the Gomoku board (15x15 standard)
            embedding_dim: Dimension of the embedding vectors
            hidden_dim: Dimension of the hidden state in LSTM
            num_layers: Number of LSTM layers
            dropout: Dropout probability for regularization
        """
        super(GomokuLSTMModel, self).__init__()
        
        # Number of possible positions on the board
        self.num_positions = board_size * board_size
        
        # Total vocabulary size: black moves + white moves + special tokens
        # We'll use indices 0 to num_positions-1 for black moves
        # and indices num_positions to 2*num_positions-1 for white moves
        # Add 1 for special start token
        self.vocab_size = 2 * self.num_positions + 1
        self.start_token_idx = 2 * self.num_positions  # Index for start token
        
        # Embedding layer to convert move indices to vector representations
        self.embedding = nn.Embedding(self.vocab_size, embedding_dim)
        
        # LSTM to process the sequence of embedded moves
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Fully connected layer to predict the next move position
        self.fc = nn.Linear(hidden_dim, self.num_positions)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape (batch_size, sequence_length)
               Each element is a move represented as a signed integer.
               Positive integers are black moves, negative integers are white moves.
               The absolute value represents the position on the board.
        
        Returns:
            Tensor of shape (batch_size, num_positions) representing the 
            probability/score for each possible next move position.
        """
        # Handle empty sequences by providing a "start of game" token
        if x.size(1) == 0:
            # Create a tensor with the start token
            x = torch.full((x.size(0), 1), self.start_token_idx, device=x.device, dtype=torch.long)
        
        # Convert signed position indices to embedding vocabulary indices
        x_indices = torch.zeros_like(x, dtype=torch.long)
        
        # For the start token (if present)
        start_mask = (x == self.start_token_idx)
        x_indices[start_mask] = self.start_token_idx
        
        # For black moves (positive), we'll use position index directly
        black_mask = x > 0
        x_indices[black_mask] = x[black_mask] - 1  # -1 to convert to 0-indexed
        
        # For white moves (negative), we'll offset by num_positions
        white_mask = x < 0
        x_indices[white_mask] = self.num_positions + (-x[white_mask] - 1)  # -1 to convert to 0-indexed
        
        # Embed the moves
        embedded = self.embedding(x_indices)
        
        # Apply dropout for regularization
        embedded = self.dropout(embedded)
        
        # Process the sequence with LSTM
        lstm_out, _ = self.lstm(embedded)
        
        # Get the final output from the LSTM (for each sequence in the batch)
        final_hidden = lstm_out[:, -1, :]
        
        # Apply dropout before the final layer
        final_hidden = self.dropout(final_hidden)
        
        # Predict scores for each possible next move position
        output = self.fc(final_hidden)
        
        return output
    
    def predict_next_move(self, move_sequence, occupied_positions=None):
        """
        Predict the next move given a sequence of previous moves.
        
        Args:
            move_sequence: List of moves (signed integers) representing the game history
            occupied_positions: Set of positions that are already occupied (optional)
            
        Returns:
            The predicted next move position (1 to num_positions)
        """
        # Convert move sequence to tensor
        with torch.no_grad():
            # Add batch dimension
            x = torch.tensor([move_sequence], dtype=torch.long)
            
            # Forward pass
            output = self(x)
            
            # If we know which positions are occupied, mask them out
            if occupied_positions is not None:
                mask = torch.ones(self.num_positions, dtype=torch.bool)
                for pos in occupied_positions:
                    if 1 <= pos <= self.num_positions:
                        mask[pos-1] = False  # Convert to 0-indexed
                
                # Apply mask (set occupied positions to -inf)
                masked_output = output.clone()
                masked_output[0, ~mask] = float('-inf')
                
                # Get the position with the highest score
                next_pos = torch.argmax(masked_output).item() + 1  # +1 to convert back to 1-indexed
            else:
                # Get the position with the highest score
                next_pos = torch.argmax(output).item() + 1  # +1 to convert back to 1-indexed
                
            return next_pos

def convert_board_to_occupied_positions(board):
    """
    Convert a board state to a set of occupied positions.
    
    Args:
        board: 2D list representing the board state (0: empty, 1: black, -1: white)
    
    Returns:
        Set of positions (1 to board_size^2) that are occupied
    """
    occupied = set()
    board_size = len(board)
    
    for i in range(board_size):
        for j in range(board_size):
            if board[i][j] != 0:
                # Convert (i,j) to position (1-indexed)
                pos = i * board_size + j + 1
                occupied.add(pos)
                
    return occupied