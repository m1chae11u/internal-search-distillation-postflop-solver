"""
generate_search_examples.py

Generates example internal search structures for poker decisions.
"""

import json
import random
from typing import List, Dict
from poker_search_builder import PokerSearchBuilder
from placeholderpokersolver import PlaceholderPokerSolver
import os

# Sample hands for examples
SAMPLE_HANDS = [
    ["Ah", "Kh"],  # AK suited hearts
    ["Js", "Jc"],  # JJ pocket pair
    ["Td", "9d"],  # T9 suited diamonds
    ["As", "Qc"],  # AQ offsuit
    ["5h", "5s"],  # 55 pocket pair
    ["Kc", "Qc"],  # KQ suited clubs
    ["8d", "7d"],  # 87 suited diamonds
    ["Ac", "Ts"],  # AT offsuit
    ["6h", "6c"],  # 66 pocket pair
    ["Qh", "Jh"]   # QJ suited hearts
]

# Sample flops
SAMPLE_FLOPS = [
    ["Ts", "7h", "2d"],  # T72 rainbow
    ["Ah", "Kh", "4h"],  # AK4 flush
    ["8s", "8c", "3d"],  # 883 paired
    ["Jd", "Td", "2s"],  # JT2 two diamonds
    ["Qs", "9s", "5c"],  # Q95 two spades
    ["7c", "6d", "2h"],  # 762 rainbow
    ["Kc", "Qc", "9d"],  # KQ9 two clubs
    ["4s", "3s", "2s"],  # 432 flush
    ["Ac", "Jd", "7h"],  # AJ7 rainbow 
    ["9h", "8h", "6c"]   # 986 two hearts
]

# Sample turn cards
SAMPLE_TURNS = ["Qh", "3c", "Th", "5s", "Jd", "As", "8d", "2c", "Kd", "7s"]

# Sample river cards
SAMPLE_RIVERS = ["4d", "9c", "6h", "Ks", "5d", "Qc", "Jh", "3s", "Ad", "7c"]

def generate_random_board(street: str) -> List[str]:
    """Generate a random board for the given street"""
    if street == "flop":
        return random.choice(SAMPLE_FLOPS)
    elif street == "turn":
        flop = random.choice(SAMPLE_FLOPS)
        turn = random.choice([c for c in SAMPLE_TURNS if c not in flop])
        return flop + [turn]
    elif street == "river":
        flop = random.choice(SAMPLE_FLOPS)
        turn = random.choice([c for c in SAMPLE_TURNS if c not in flop])
        river = random.choice([c for c in SAMPLE_RIVERS if c not in flop + [turn]])
        return flop + [turn] + [river]
    return []

def generate_examples(num_examples: int = 10) -> List[Dict]:
    """Generate a set of internal search examples"""
    examples = []
    builder = PokerSearchBuilder(PlaceholderPokerSolver())
    
    for i in range(num_examples):
        # Randomly select a street
        street = random.choice(["flop", "turn", "river"])
        
        # Randomly select a hand
        hand = random.choice(SAMPLE_HANDS)
        
        # Generate a board for the given street
        board = generate_random_board(street)
        
        # Ensure hand and board don't have duplicated cards
        while any(card in board for card in hand):
            hand = random.choice(SAMPLE_HANDS)
        
        # Generate random pot and stack sizes
        pot_size = random.choice([50, 75, 100, 150, 200, 300])
        effective_stack = random.choice([300, 500, 750, 1000, 1500, 2000])
        
        # Build the search structure
        search_str = builder.build_search(
            hero_hand=hand,
            board=board,
            pot=pot_size,
            effective_stack=effective_stack,
            street=street
        )
        
        examples.append({
            "id": f"example_{i+1}",
            "hero_hand": hand,
            "board": board,
            "pot_size": pot_size,
            "effective_stack": effective_stack,
            "street": street,
            "search_structure": search_str
        })
    
    return examples

def main():
    # Generate examples
    examples = generate_examples(10)
    
    # Ensure data directory exists
    data_dir = "internal_search/data"
    os.makedirs(data_dir, exist_ok=True)
    
    # Save to JSON file
    json_path = os.path.join(data_dir, "internal_search_examples.json")
    with open(json_path, "w") as f:
        json.dump(examples, f, indent=2)
    
    print(f"All 10 examples saved to {json_path}")
    
    # Print 3 examples
    print("Generated 10 examples. Here are 3 samples:\n")
    for i in range(3):
        print(f"Example {i+1}:")
        print(f"Hero Hand: {examples[i]['hero_hand']}")
        print(f"Board: {examples[i]['board']}")
        print(f"Street: {examples[i]['street']}")
        print(f"Pot Size: {examples[i]['pot_size']}")
        print(f"Effective Stack: {examples[i]['effective_stack']}")
        print("\nSearch Structure:")
        print(examples[i]['search_structure'])
        print("\n" + "="*80 + "\n")

if __name__ == "__main__":
    main()
