import re
import json
import random
import asyncio
from typing import Tuple, List
from gomoku import Agent
from gomoku.llm import OpenAIGomokuClient
from gomoku.core.models import Player, GameState

try:
    from google.colab import userdata
except ImportError:
    import os
    class userdata:
        @staticmethod
        def get(key):
            return os.environ.get(key)

class AdvancedLLMAgent(Agent):
    """An improved LLM agent with better strategic prompting and fallback logic."""

    def __init__(self, agent_id: str):
        super().__init__(agent_id)
        self._setup()
        print(f"🚀 Created AdvancedLLMAgent: {agent_id}")

    def _setup(self):
        """Setup our LLM client and prompts."""
        self.system_prompt = self._create_advanced_system_prompt()

        try:
            api_key = userdata.get('Groq_API_l1')
            if not api_key:
                print("WARNING: Groq_API_l1 not found, using strategic fallback mode")
                self.llm_client = None
                return
                
            self.llm_client = OpenAIGomokuClient(
                api_key=api_key,
                model="gemma2-9b-it",
                endpoint="https://api.groq.com/openai/v1",
            )
        except Exception as e:
            print(f"Error setting up LLM client: {e}")
            self.llm_client = None

    def _create_advanced_system_prompt(self) -> str:
        """Create an advanced system prompt with better strategic guidance."""
        return """
You are an EXPERT Gomoku strategist with deep tactical knowledge. You play on an 8x8 board where 5-in-a-row wins.

PRIORITY SYSTEM (Check in this exact order):

1. INSTANT WIN: If you can make 5-in-a-row, play it IMMEDIATELY
2. CRITICAL BLOCK: If opponent can win next turn, block them
3. CREATE DOUBLE THREAT: Force opponent into unwinnable position
4. BLOCK OPPONENT DOUBLE THREAT: Prevent their forcing moves
5. BUILD STRONG ATTACK: Create multiple winning paths
6. POSITIONAL ADVANTAGE: Control center and key squares

ADVANCED TACTICAL PATTERNS:
- FORK: Create two 3-in-a-rows that share a winning square
- TRIANGLE: Build connected 3s forming a triangle shape
- BRIDGE: Connect distant pieces with intermediate stones
- CROSS: Create intersecting threats that multiply your chances

BOARD ANALYSIS FRAMEWORK:
- Count all 2s, 3s, 4s for both players
- Identify weak points in opponent's position
- Find squares that serve multiple purposes
- Evaluate potential counter-attacks after each move

OPENING PRINCIPLES:
- Control center: (3,3), (3,4), (4,3), (4,4)
- Build toward corners and edges for space
- Create multiple development paths
- Force opponent to react to your threats

ENDGAME MASTERY:
- Calculate all forcing sequences
- Prioritize moves that create the most threats
- Block with moves that also attack
- Look for sacrificial tactics that lead to wins

You must respond with this exact JSON format:
{
    "win_check": "Can I win immediately? [YES/NO and where]",
    "threat_analysis": "What are opponent's biggest threats?",
    "tactical_plan": "What pattern am I trying to create?",
    "move_evaluation": "Why this move is optimal",
    "row": <number>,
    "col": <number>
}

Think like a grandmaster: Every move should either threaten victory or prevent defeat while improving your position.
""".strip()

    async def get_move(self, game_state: GameState) -> Tuple[int, int]:
        """Main method: Get the next move from our advanced LLM."""
        print(f"\n{self.agent_id} is analyzing the position...")

        # Use strategic fallback if no LLM client
        if not self.llm_client:
            print("Using advanced strategic fallback")
            return self._get_strategic_move(game_state)

        try:
            board_str = self._create_detailed_board_representation(game_state)
            analysis = self._analyze_position(game_state)
            
            move_count = len(game_state.move_history)
            
            user_prompt = f"""
POSITION ANALYSIS:
{board_str}

TACTICAL SITUATION:
{analysis}

GAME CONTEXT:
- Move #{move_count + 1}
- You are: {game_state.current_player.value}
- Phase: {'Opening' if move_count < 10 else 'Middle Game' if move_count < 20 else 'Endgame'}

Apply your expert knowledge and provide the best move as JSON.
"""

            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_prompt}
            ]

            print("🧠 Advanced analysis in progress...")
            response = await self.llm_client.complete(messages)
            print(f"💭 LLM Strategy: {response[:100]}...")

            if match := re.search(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", response, re.DOTALL):
                json_data = json.loads(match.group(0).strip())
                row, col = json_data["row"], json_data["col"]
                
                if game_state.is_valid_move(row, col):
                    print(f"🎯 Strategic choice: ({row}, {col})")
                    return (row, col)

        except Exception as e:
            print(f"LLM error: {e}")

        return self._get_strategic_move(game_state)

    def _analyze_position(self, game_state: GameState) -> str:
        """Analyze the current position for threats and opportunities."""
        board = game_state.board
        my_player = game_state.current_player
        opp_player = Player.O if my_player == Player.X else Player.X
        
        # Count sequences
        my_twos = self._count_sequences(board, my_player, 2)
        my_threes = self._count_sequences(board, my_player, 3)
        opp_twos = self._count_sequences(board, opp_player, 2)
        opp_threes = self._count_sequences(board, opp_player, 3)
        
        return f"""
My position: {my_threes} threes, {my_twos} twos
Opponent position: {opp_threes} threes, {opp_twos} twos
Tactical balance: {'Attacking' if my_threes > opp_threes else 'Defending' if opp_threes > my_threes else 'Equal'}
"""

    def _count_sequences(self, board, player, length):
        """Count sequences of a given length for a player."""
        # Simplified implementation - you could make this more sophisticated
        count = 0
        directions = [(0,1), (1,0), (1,1), (1,-1)]
        
        for row in range(len(board)):
            for col in range(len(board[0])):
                for dr, dc in directions:
                    sequence = 0
                    r, c = row, col
                    while (0 <= r < len(board) and 0 <= c < len(board[0]) and 
                           board[r][c] == player):
                        sequence += 1
                        r, c = r + dr, c + dc
                    if sequence == length:
                        count += 1
        return count

    def _create_detailed_board_representation(self, game_state: GameState) -> str:
        """Create a detailed string representation of the board."""
        board = game_state.board
        result = "  0 1 2 3 4 5 6 7\n"
        for i in range(len(board)):
            result += f"{i} "
            for j in range(len(board[i])):
                if board[i][j] == Player.X:
                    result += "X "
                elif board[i][j] == Player.O:
                    result += "O "
                else:
                    result += ". "
            result += "\n"
        return result

    def _get_strategic_move(self, game_state: GameState) -> Tuple[int, int]:
        """Advanced fallback strategy when LLM is unavailable."""
        legal_moves = game_state.get_legal_moves()
        board = game_state.board
        my_player = game_state.current_player
        
        # 1. Check for immediate wins
        for move in legal_moves:
            if self._creates_five_in_row(board, move, my_player):
                print(f"🏆 Winning move found: {move}")
                return move
        
        # 2. Block opponent wins
        opp_player = Player.O if my_player == Player.X else Player.X
        for move in legal_moves:
            if self._creates_five_in_row(board, move, opp_player):
                print(f"🛡️ Blocking opponent win: {move}")
                return move
        
        # 3. Play center if available
        center_moves = [(3, 3), (3, 4), (4, 3), (4, 4)]
        for move in center_moves:
            if move in legal_moves:
                print(f"📍 Taking center: {move}")
                return move
        
        # 4. Look for good tactical moves
        scored_moves = []
        for move in legal_moves:
            score = self._evaluate_move(board, move, my_player)
            scored_moves.append((score, move))
        
        scored_moves.sort(reverse=True)
        best_move = scored_moves[0][1]
        print(f"🎲 Best tactical move: {best_move}")
        return best_move

    def _creates_five_in_row(self, board, move, player):
        """Check if a move creates five in a row."""
        row, col = move
        if board[row][col] is not None:
            return False
            
        # Simulate the move
        board[row][col] = player
        
        directions = [(0,1), (1,0), (1,1), (1,-1)]
        for dr, dc in directions:
            count = 1  # Count the piece we just placed
            
            # Count in positive direction
            r, c = row + dr, col + dc
            while (0 <= r < len(board) and 0 <= c < len(board[0]) and 
                   board[r][c] == player):
                count += 1
                r, c = r + dr, c + dc
            
            # Count in negative direction
            r, c = row - dr, col - dc
            while (0 <= r < len(board) and 0 <= c < len(board[0]) and 
                   board[r][c] == player):
                count += 1
                r, c = r - dr, c - dc
            
            if count >= 5:
                board[row][col] = None  # Undo the simulation
                return True
        
        board[row][col] = None  # Undo the simulation
        return False

    def _evaluate_move(self, board, move, player):
        """Evaluate a move's tactical value."""
        row, col = move
        score = 0
        
        # Prefer center
        center_distance = abs(row - 3.5) + abs(col - 3.5)
        score += max(0, 7 - center_distance)
        
        # Prefer moves near existing pieces
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr == 0 and dc == 0:
                    continue
                r, c = row + dr, col + dc
                if (0 <= r < len(board) and 0 <= c < len(board[0]) and 
                    board[r][c] is not None):
                    score += 2
        
        return score
