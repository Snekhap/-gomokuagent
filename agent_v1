import re
import json
from gomoku import Agent
from gomoku.llm import OpenAIGomokuClient
from gomoku.core.models import Player

class StudentLLMAgent(Agent):
    """An educational LLM agent that students will build step by step."""

    def init(self, agent_id: str):
        super().init(agent_id)
        print(f"🎓 Created StudentLLMAgent: {agent_id}")

    def _setup(self):
        """Setup our LLM client and prompts."""

        # We'll add the LLM client setup here
        # For now, let's define our system prompt
        self.system_prompt = self._create_system_prompt()

        self.llm_client = OpenAIGomokuClient(
            api_key=userdata.get('Groq_API_l1'),
            model="gemma2-9b-it",
            endpoint="https://api.groq.com/openai/v1",
        )


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

ADVANCED PATTERN RECOGNITION:
- Count sequences: Look for your 2s, 3s, 4s and opponent's
- Identify WEAK SPOTS: Where can you break their formations?
- Find INTERSECTION POINTS: Moves that serve multiple purposes

OPENING STRATEGY:
- First move: Always center (3,3) or (4,4)
- Second move: Adjacent to center or create diagonal
- Build toward multiple directions from strong positions

DEFENSIVE PRIORITIES:
1. Block immediate wins (their 4-in-a-row)
2. Block double threats (their two 3-in-a-rows)
3. Block open threes (.XXX.)
4. Contest strong positions

BOARD READING GUIDE:
- Your pieces: X
- Opponent pieces: O  
- Empty spaces: .
- Coordinates: (0,0) is top-left, (7,7) is bottom-right

RESPONSE FORMAT - You MUST respond with valid JSON:
{
    "analysis": "Detailed step-by-step analysis following the framework above",
    "strategy": "Your chosen strategy (WIN/BLOCK/ATTACK/POSITION)",
    "reasoning": "Why this specific move beats other options",
    "row": <number>,
    "col": <number>
}

WINNING MINDSET: 
- Think 2-3 moves ahead
- Every move should either threaten victory or prevent defeat
- Force your opponent into reactive play
- Create complexity where you have the advantage

Remember: In Gomoku, the player who controls the initiative usually wins. Make moves that put pressure on your opponent!
""".strip()

    async def get_move(self, game_state: GameState) -> Tuple[int, int]:
        """Main method: Get the next move from our LLM."""
        print(f"\n{self.agent_id} is thinking...")

        try:

            board_str = self._create_enhanced_board_representation(game_state)
            
            # Add game context
            move_count = len(game_state.move_history)
            game_phase = self._determine_game_phase(move_count)
            
            # Create comprehensive prompt
            user_prompt = f"""
CURRENT GAME SITUATION:
{board_str}

GAME CONTEXT:
- Move #{move_count + 1}
- You are player: {game_state.current_player.value}
- Game phase: {game_phase}
- Legal moves available: {len(game_state.get_legal_moves())}

RECENT MOVES:
{self._format_recent_moves(game_state)}

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
            print(e)

        return self._get_fallback_move(game_state)

    def _get_fallback_move(self, game_state: GameState) -> Tuple[int, int]:
        """Simple fallback when LLM fails."""
        return random.choice(game_state.get_legal_moves())
