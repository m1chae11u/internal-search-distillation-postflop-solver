"""
    poker_search_builder.py

    A class for dynamically building internal search structures for poker decisions.
    Only expands the highest EV path at each decision point.
"""

import random
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
from placeholderpokersolver import PlaceholderPokerSolver
# from pokersolver import PokerSolver
# in the future, we can use the actual solver

@dataclass
class SearchNode:
    """Represents a node in the search tree"""
    id: str
    street: str
    pot: float
    effective_stack: float
    hero_hand: List[str]
    board: List[str]
    actions: List[Dict[str, Any]] = None
    range_description: str = ""
    all_legal_actions: List[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.actions is None:
            self.actions = []
        if self.all_legal_actions is None:
            self.all_legal_actions = []

class PokerSearchBuilder:
    """Class for building internal search structures for poker decisions"""
    
    def __init__(self, solver=None):
        """Initialize with a poker solver"""
        self.solver = solver or PlaceholderPokerSolver()
        
        # Standard bet sizes to consider
        self.bet_sizes = {
            "small": 1/3,  # 1/3 pot
            "medium": 2/3,  # 2/3 pot
            "large": 1.0,   # 1x pot
            "overbet": 1.5  # 1.5x pot
        }
        
        # Card buckets for turn and river cards
        self.card_buckets = [
            {"name": "high_cards", "description": "Broadway cards (A, K, Q, J, T)", "representative": "Qh", "probability": 0.27},
            {"name": "middle_cards", "description": "Middle cards (9-7)", "representative": "8c", "probability": 0.18},
            {"name": "low_cards", "description": "Low cards (6-2)", "representative": "4d", "probability": 0.25},
            {"name": "flush_draw_completers", "description": "Cards that complete flush draws", "representative": "9h", "probability": 0.15},
            {"name": "straight_draw_completers", "description": "Cards that complete straight draws", "representative": "Jd", "probability": 0.15}
        ]
    
    def _get_available_actions(self, node: SearchNode) -> List[Dict[str, Any]]:
        """Get available actions for the current node"""
        actions = []
        
        # Always include check/fold if applicable
        if node.street != "preflop":  # Can't check preflop unless you're BB
            actions.append({"type": "check", "size": 0, "amount": 0})
        
        # Include standard bet sizes if we have chips
        for name, size in self.bet_sizes.items():
            bet_amount = round(node.pot * size)
            if bet_amount <= node.effective_stack:
                actions.append({"type": "bet", "size": f"{name}", "amount": bet_amount})
        
        # Include all-in if we have less than 1.5x pot
        if node.effective_stack < node.pot * 1.5:
            actions.append({"type": "bet", "size": "all_in", "amount": node.effective_stack})
            
        return actions
    
    def _evaluate_actions(self, node: SearchNode, actions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Evaluate actions using the solver and return them with EVs"""
        # Call the solver to get EVs for each action
        solver_results = self.solver.solve(
            node.hero_hand, 
            node.board, 
            pot_size=node.pot, 
            effective_stack=node.effective_stack
        )
        
        # Use the solver's action results to update our actions
        result_actions = []
        for action in actions:
            # Find matching action type in solver results
            action_type = action["type"]
            for solver_action in solver_results["actions"]:
                if solver_action.action_type == action_type:
                    # Found matching action type
                    if action_type == "bet":
                        # For bet actions, also match the size
                        # Use string comparison to match "small"/"medium" etc.
                        if action["size"] != solver_action.size:
                            continue
                    
                    # Copy values from solver action
                    action["ev"] = solver_action.ev
                    action["frequency"] = solver_action.frequency
                    result_actions.append(action)
                    break
        
        # Sort actions by EV, highest first
        return sorted(result_actions, key=lambda a: a["ev"], reverse=True)
    
    def _get_opponent_responses(self, node: SearchNode, hero_action: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate likely opponent responses to hero's action"""
        # Convert our action to a format the solver understands
        action_str = hero_action["type"]
        if hero_action["type"] == "bet":
            action_str += f"_{hero_action['amount']}"
        
        # Get opponent responses from the solver
        opponent_responses = self.solver.get_opponent_actions(action_str, node.pot)
        
        # Convert to our format
        results = []
        for response in opponent_responses:
            result = {
                "type": response["type"],
                "probability": response["probability"],
                "ev": response["ev"]
            }
            
            if response["type"] in ["bet", "raise"]:
                result["size"] = response["size"]
                result["amount"] = response["amount"]
            
            results.append(result)
        
        # Sort by EV
        return sorted(results, key=lambda r: r["ev"], reverse=True)
    
    def _select_card_bucket(self, current_board: List[str]) -> Dict[str, Any]:
        """Select a representative card bucket based on the current board"""
        # In a real implementation, we'd analyze the board and select a
        # relevant bucket. For now, we'll randomly select one.
        return random.choice(self.card_buckets)
    
    def _get_next_street(self, street: str) -> str:
        """Get the name of the next street"""
        streets = ["preflop", "flop", "turn", "river"]
        try:
            idx = streets.index(street)
            if idx < len(streets) - 1:
                return streets[idx + 1]
        except ValueError:
            pass
        return None
    
    def _add_next_street_card(self, board: List[str], street: str) -> Tuple[List[str], str]:
        """Add a card for the next street and return updated board and card"""
        card_bucket = self._select_card_bucket(board)
        card = card_bucket["representative"]
        
        # Make sure we don't duplicate a card that's already on the board
        while card in board:
            # Generate a new card if there's a collision
            suits = ["h", "d", "c", "s"]
            ranks = ["2", "3", "4", "5", "6", "7", "8", "9", "T", "J", "Q", "K", "A"]
            card = random.choice(ranks) + random.choice(suits)
        
        new_board = board.copy()
        new_board.append(card)
        return new_board, card
    
    def _generate_range_description(self, hero_hand: List[str], board: List[str], street: str) -> str:
        """Generate a range description for the current game state"""
        # In a real implementation, we'd analyze the board and generate a 
        # meaningful range description. For now, we'll return a placeholder.
        return "Top pairs (30%), overpairs (15%), draws (25%), air (30%)"
    
    def _build_search_tree(self, node: SearchNode) -> Dict[str, Any]:
        """Build a search tree starting from the given node, only expanding highest EV paths"""
        # Get available actions
        actions = self._get_available_actions(node)
        
        # Evaluate actions to get EVs
        evaluated_actions = self._evaluate_actions(node, actions)
        
        # Save all legal actions for display
        node.all_legal_actions = evaluated_actions.copy()
        
        # Initialize actions in the node
        node_actions = []
        
        # Process top 3 actions (or fewer if not enough actions)
        for i, action in enumerate(evaluated_actions[:min(3, len(evaluated_actions))]):
            action_copy = action.copy()
            action_copy["id"] = f"a{i+1}"
            
            # Mark the highest EV action
            if i == 0:
                action_copy["best"] = True
                
                # Only expand the highest EV action further
                # Get opponent responses
                opponent_responses = self._get_opponent_responses(node, action)
                
                # Only take the highest EV opponent response
                if opponent_responses:
                    best_response = opponent_responses[0]
                    
                    # If not at river yet, continue to next street
                    next_street = self._get_next_street(node.street)
                    if next_street:
                        # Add card for next street
                        new_board, card_added = self._add_next_street_card(node.board, next_street)
                        card_bucket = self._select_card_bucket(node.board)
                        
                        # Create child node for next street
                        child_node = SearchNode(
                            id=f"{action_copy['id']}-1",
                            street=next_street,
                            pot=node.pot + (action.get("amount", 0) if action["type"] == "bet" else 0),
                            effective_stack=node.effective_stack - (action.get("amount", 0) if action["type"] == "bet" else 0),
                            hero_hand=node.hero_hand,
                            board=new_board
                        )
                        
                        # Recursively build child node
                        child_tree = self._build_search_tree(child_node)
                        
                        # Add next street information to the opponent response
                        best_response["next_street"] = {
                            "street": next_street,
                            "card": card_added,
                            "probability": card_bucket["probability"],
                            "node": child_tree
                        }
                    
                    # Add the best opponent response to the action
                    action_copy["opponent_action"] = best_response
            
            # Add the action to the node's actions
            node_actions.append(action_copy)
        
        # Return the node with all its data
        return {
            "id": node.id,
            "street": node.street,
            "pot": node.pot,
            "effective_stack": node.effective_stack,
            "hero_hand": node.hero_hand,
            "board": node.board,
            "range_description": node.range_description,
            "actions": node_actions,
            "all_legal_actions": node.all_legal_actions  # Add all legal actions
        }
    
    def _format_search_tree(self, tree_dict, indent=0) -> str:
        """Format the search tree as a string"""
        indent_str = "  " * indent
        result = []
        
        # Start the node
        result.append(f"{indent_str}<node id=\"{tree_dict['id']}\" street=\"{tree_dict['street']}\" pot=\"{tree_dict['pot']}\" effective_stack=\"{tree_dict['effective_stack']}\">")
        
        # Add hero hand
        result.append(f"{indent_str}  <hero_hand>{' '.join(tree_dict['hero_hand'])}</hero_hand>")
        
        # Add board
        result.append(f"{indent_str}  <board>{' '.join(tree_dict['board'])}</board>")
        
        # Add range description if available
        if tree_dict.get('range_description'):
            result.append(f"{indent_str}  <range>{tree_dict['range_description']}</range>")
        
        # Display all legal actions and their EVs
        result.append(f"{indent_str}  <legal_actions>")
        if tree_dict.get('all_legal_actions'):
            for action in tree_dict.get('all_legal_actions', []):
                action_desc = f"type=\"{action['type']}\""
                if action['type'] == "bet":
                    action_desc += f" size=\"{action['size']}\" amount=\"{action['amount']}\""
                result.append(f"{indent_str}    <legal_action {action_desc} ev=\"{action['ev']}\"/>")
        else:
            result.append(f"{indent_str}    <!-- No legal actions available -->")
        result.append(f"{indent_str}  </legal_actions>")
        
        # Add actions
        if tree_dict.get('actions'):
            result.append(f"{indent_str}  <actions>")
            
            # Add a comment for clarity
            if len(tree_dict['actions']) > 0:
                result.append(f"{indent_str}    <!-- List top {len(tree_dict['actions'])} actions by EV but only fully expand the highest one -->")
            
            # Process each action
            for action in tree_dict['actions']:
                action_attr = f"id=\"{action['id']}\" type=\"{action['type']}\""
                if action['type'] == "bet":
                    action_attr += f" size=\"{action['size']}\" amount=\"{action['amount']}\""
                action_attr += f" ev=\"{action['ev']}\""
                
                if action.get('best'):
                    action_attr += " best=\"true\""
                
                result.append(f"{indent_str}    <action {action_attr}>")
                
                # If it's the best action, expand it
                if action.get('opponent_action'):
                    opponent = action['opponent_action']
                    op_attr = f"type=\"{opponent['type']}\" probability=\"{opponent['probability']}\" ev=\"{opponent['ev']}\""
                    
                    if opponent['type'] in ["bet", "raise"]:
                        op_attr += f" size=\"{opponent['size']}\" amount=\"{opponent['amount']}\""
                    
                    result.append(f"{indent_str}      <opponent_action {op_attr}>")
                    
                    # If there's a next street, recursively format it
                    if opponent.get('next_street'):
                        next_street = opponent['next_street']
                        result.append(f"{indent_str}        <{next_street['street']} card=\"{next_street['card']}\" probability=\"{next_street['probability']}\">")
                        
                        # Recursively format the child node
                        child_str = self._format_search_tree(next_street['node'], indent + 4)
                        result.append(child_str)
                        
                        result.append(f"{indent_str}        </{next_street['street']}>")
                    
                    result.append(f"{indent_str}      </opponent_action>")
                
                # If this is not the best action, just close it immediately
                if not action.get('best'):
                    result.append(f"{indent_str}      <!-- Not expanded since it's not the highest EV action -->")
                
                # Close the action tag
                result.append(f"{indent_str}    </action>")
            
            result.append(f"{indent_str}  </actions>")
        
        # Close the node
        result.append(f"{indent_str}</node>")
        
        return "\n".join(result)
    
    def build_search(self, hero_hand: List[str], board: List[str], pot: float, 
                   effective_stack: float, street: str, position: str = "OOP") -> str:
        """
        Build a search tree for the given game state.
        
        Args:
            hero_hand: List of hero's hole cards (e.g., ["As", "Ks"])
            board: List of community cards (e.g., ["Ts", "7h", "2d"])
            pot: Current pot size
            effective_stack: Hero's effective stack
            street: Current street ("preflop", "flop", "turn", "river")
            position: Hero's position ("OOP" or "IP")
            
        Returns:
            String representation of the search tree
        """
        # Create root node
        root = SearchNode(
            id="root",
            street=street,
            pot=pot,
            effective_stack=effective_stack,
            hero_hand=hero_hand,
            board=board,
            range_description=self._generate_range_description(hero_hand, board, street)
        )
        
        # Build the search tree
        tree_dict = self._build_search_tree(root)
        
        # Format the search tree as a string
        search_str = "<search>\n"
        search_str += self._format_search_tree(tree_dict, indent=1)
        search_str += "\n</search>"
        
        return search_str


if __name__ == "__main__":
    # Example usage
    builder = PokerSearchBuilder()
    
    search = builder.build_search(
        hero_hand=["As", "Ks"],
        board=["Ts", "7h", "2d"],
        pot=100,
        effective_stack=900,
        street="flop"
    )
    
    print(search)
