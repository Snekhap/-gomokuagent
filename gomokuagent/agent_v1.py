import re
import json
import random
import asyncio
from typing import Tuple
from gomoku import Agent
from gomoku.llm import OpenAIGomokuClient
from gomoku.core.models import Player, GameState

# Add this if userdata is from Colab
try:
    from google.colab import userdata
except ImportError:
    import os
    class userdata:
        @staticmethod
        def get(key):
            return os.environ.get(key)

class StudentLLMAgent(Agent):
    """An educational LLM agent that students will build step by step."""

    def __init__(self, agent_id: str):
        super().__init__(agent_id)
        self._setup()
        print(f"🎓 Created StudentLLMAgent: {agent_id}")

    def _setup(self):
        """Setup our LLM client and prompts."""
        self.system_prompt = self._create_system_prompt()

        try:
            api_key = userdata.get('Groq_API_l1')  # Fixed to match your actual key name
            if not api_key:
                print("WARNING: Groq_API_l1 not found, using fallback mode")
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

    def _create_system_prompt(self) -> str:
        """Create the system prompt that teaches the LLM how to play Gomoku."""
        return """
You are a MASTER Gomoku player with perfect strategic vision. You play on an 8x8 board where 5-in-a-row wins.

CRITICAL ANALYSIS FRAMEWORK - Always follow this order:

1. IMMEDIATE WIN CHECK: Scan for ANY 4-in-a-row you can complete
   - Check all directions: horizontal, vertical, both diagonals
   - If found, play it IMMEDIATELY

2. IMMEDIATE THREAT DEFENSE: Scan for opponent's 4-in-a-row threats
   - Check all their sequences of 4 with one gap
   - Block the MOST DANGEROUS threat first
   - If multiple threats exist, block the one that gives you counter-attack potential

3. TACTICAL OPPORTUNITIES (if no immediate win/threat):
   - DOUBLE THREAT CREATION: Can you create two 3-in-a-rows simultaneously?
   - FORK ATTACKS: Create multiple winning paths they cannot block
   - OPEN THREE: Build 3-in-a-row with both ends open (.XXX.)
   - SEMI-OPEN THREE: Build 3-in-a-row with one end open (OXXX. or .XXXO)

4. POSITIONAL STRATEGY:
   - CONTROL CENTER: Positions (3,3), (3,4), (4,3), (4,4) are strongest
   - BUILD CONNECTED STRUCTURES: Don't scatter pieces randomly
   - FORCE OPPONENT TO DEFEND: Make threats they must respond to

RESPONSE FORMAT - You MUST respond with valid JSON:
{
    "analysis": "Detailed step-by-step analysis following the framework above",
    "strategy": "Your chosen strategy (WIN/BLOCK/ATTACK/POSITION)",
    "reasoning": "Why this specific move beats other options",
    "row": <number>,
    "col": <number>
}
""".strip()

    async def get_move(self, game_state: GameState) -> Tuple[int, int]:
        """Main method: Get the next move from our LLM."""
        print(f"\n{self.agent_id} is thinking...")

        # Fallback to random if no LLM client
        if not self.llm_client:
            print("No LLM client available, using random move")
            return self._get_fallback_move(game_state)

        try:
            board_str = self._create_board_representation(game_state)
            
            # Add game context
            move_count = len(game_state.move_history)
            
            # Create comprehensive prompt
            user_prompt = f"""
CURRENT GAME SITUATION:
{board_str}

GAME CONTEXT:
- Move #{move_count + 1}
- You are player: {game_state.current_player.value}
- Legal moves available: {len(game_state.get_legal_moves())}

Apply the strategic framework step by step and provide your best move as JSON.
"""

            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_prompt}
            ]

            print("Prompt Being Sent...")
            response = await self.llm_client.complete(messages)
            print(f"LLM Response: {response}")

            # Extract JSON
            if match := re.search(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", response, re.DOTALL):
                json_data = json.loads(match.group(0).strip())
                row, col = json_data["row"], json_data["col"]
                
                if game_state.is_valid_move(row, col):
                    print(f"AI chose: ({row}, {col})")
                    return (row, col)

        except Exception as e:
            print(f"Error: {e}")

        return self._get_fallback_move(game_state)

    def _create_board_representation(self, game_state: GameState) -> str:
        """Create a string representation of the board."""
        # This is a placeholder - implement based on your GameState structure
        board = game_state.board
        board_str = ""
        for i in range(len(board)):
            for j in range(len(board[i])):
                if board[i][j] == Player.X:
                    board_str += "X "
                elif board[i][j] == Player.O:
                    board_str += "O "
                else:
                    board_str += ". "
            board_str += "\n"
        return board_str

    def _get_fallback_move(self, game_state: GameState) -> Tuple[int, int]:
        """Simple fallback when LLM fails."""
        legal_moves = game_state.get_legal_moves()
        return random.choice(legal_moves)
