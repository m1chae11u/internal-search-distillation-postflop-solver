from typing import List, Optional, Union
from pydantic import BaseModel, Field

class ActionEvaluation(BaseModel):
    """
    Represents an evaluation of a single action, either for Hero or Opponent,
    or an abstracted chance outcome.
    """
    action_description: str  # Canonical string for the action (e.g., "CHECK", "BET 10.5BB", "CALL", "FOLD")
                             # Or for chance nodes: "FLUSH_CARD", "BLANK_CARD"
    ev_for_hero: float       # Expected value from Hero's perspective if this action/outcome occurs.
    probability: Optional[float] = None # Probability of this action/outcome occurring.
                                     # For Hero's initial decision, this might be None if we only get EVs.
                                     # For Opponent's response, this is their GTO probability.
                                     # For Chance nodes, this is the probability of the abstracted outcome.

class HeroDecisionOutput(BaseModel):
    """
    Output when the solver evaluates a state where it's Hero's turn to act.
    """
    node_type: str = Field(default="hero_decision", frozen=True)
    # List of possible actions Hero can take and their immediate EV.
    # The 'probability' field in ActionEvaluation might be filled if Hero has a mixed strategy.
    possible_actions: List[ActionEvaluation]

class OpponentDecisionOutput(BaseModel):
    """
    Output when the solver evaluates a state where it's Opponent's turn to act
    (typically in response to a prior Hero action).
    """
    node_type: str = Field(default="opponent_decision", frozen=True)
    # List of actions Opponent might take, their GTO probabilities, and the resulting EV for Hero.
    possible_actions: List[ActionEvaluation]

class ChanceNodeOutput(BaseModel):
    """
    Output when the solver evaluates a state where the next event is a community card being dealt.
    """
    node_type: str = Field(default="chance_node", frozen=True)
    # List of abstracted card outcomes, their probabilities, and the resulting EV for Hero.
    # action_description in ActionEvaluation here would be like "FLUSH_DRAW_COMPLETES", "BOARD_PAIRS", "BLANK_OFFSUIT".
    abstracted_outcomes: List[ActionEvaluation]

# A wrapper to allow the FFI to return one of these types, perhaps identified by node_type.
# The actual JSON returned by Rust might directly be one of the above three.
# query_solver.py can then parse based on an expected type or a 'node_type' field in the JSON. 