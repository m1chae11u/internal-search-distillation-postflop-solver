import pandas as pd
import os
import re
from typing import Optional, Dict, Tuple

# Attempt to import from the expected location
# If this script is run from the root, 'dataset_generator' should be a sibling to 'data_processing' directory
try:
    from dataset_generator.action_parser import calculate_stacks_after_actions, INITIAL_STACK_SIZE
except ImportError:
    # Fallback for different execution contexts, assuming action_parser.py might be found elsewhere or needs specific path setup
    # For this exercise, we'll assume direct import works or print a clearer error.
    print("Error: Could not import from dataset_generator.action_parser.")
    print("Ensure that the script is run from a context where 'dataset_generator' is discoverable (e.g., from the project root 'internal-search-distillation-postflop-solver').")
    print("Alternatively, adjust PYTHONPATH or the import statement if action_parser.py is located elsewhere.")
    # As a simple fallback for this specific file, if direct import fails, we might need to copy or define essentials.
    # For now, we proceed hoping the import works. If not, execution will fail here.
    # A more robust solution would involve setting up the project as a package or adjusting sys.path.
    # Let's define them here as a last resort if above fails, based on viewed content.
    INITIAL_STACK_SIZE = 200.0

    def parse_action_amount(action_part: str) -> float:
        action_part = action_part.strip()
        action_upper = action_part.upper()
        bb_match = re.search(r"(\d*\.?\d+)BB", action_upper)
        if bb_match:
            try: return float(bb_match.group(1))
            except ValueError: pass
        parts = action_upper.split('_')
        if len(parts) > 1:
            try: return float(parts[-1])
            except ValueError:
                if parts[0] == "RAISE" and parts[-2] == "TO":
                    try: return float(parts[-1])
                    except ValueError: pass
                elif parts[0] in ["BET", "RAISE"] and len(parts) == 1:
                    numeric_part = "".join(filter(lambda c: c.isdigit() or c == '.', parts[0]))
                    if numeric_part:
                        try: return float(numeric_part)
                        except ValueError: pass
        try: return float(action_part)
        except ValueError: pass
        return 0.0

    def _parse_preflop_commitment(preflop_action_str: Optional[str]) -> float:
        if not preflop_action_str: return 0.0
        actions = preflop_action_str.strip().split('/')
        if not actions: return 0.0
        if actions[-1].upper() == "CALL" and len(actions) > 1:
            bb_match = re.search(r"(\d*\.?\d+)BB", actions[-2].upper())
            if bb_match:
                try: return float(bb_match.group(1))
                except ValueError: pass
        max_committed_bet = 0.0
        for action in actions:
            if "BB" in action.upper() and "CALL" not in action.upper() and "FOLD" not in action.upper():
                amount = parse_action_amount(action)
                if amount > max_committed_bet: max_committed_bet = amount
        if actions[-1].upper() == "CALL" and max_committed_bet > 0: return max_committed_bet
        if max_committed_bet > 0: return max_committed_bet
        if "CALL" in actions[-1].upper() and len(actions) ==1 and actions[0].upper() == "SB": return 1.0
        if actions == ['SB', 'BB'] or actions == ['BTN','SB','BB'] and 'CALL' not in preflop_action_str.upper() and 'RAISE' not in preflop_action_str.upper(): return 1.0
        return 0.0 # Default

    def calculate_stacks_after_actions(
        hero_is_oop: bool, 
        preflop_action_str: Optional[str],
        postflop_action_history_str: Optional[str], 
        initial_hero_stack: float = INITIAL_STACK_SIZE,
        initial_villain_stack: float = INITIAL_STACK_SIZE
    ) -> Dict[str, float]:
        hero_stack = initial_hero_stack
        villain_stack = initial_villain_stack
        pot_from_actions = 0.0
        preflop_commitment_each = _parse_preflop_commitment(preflop_action_str)
        if preflop_commitment_each > 0:
            actual_hero_preflop_commit = min(preflop_commitment_each, hero_stack)
            actual_villain_preflop_commit = min(preflop_commitment_each, villain_stack)
            hero_stack -= actual_hero_preflop_commit
            villain_stack -= actual_villain_preflop_commit
            pot_from_actions += (actual_hero_preflop_commit + actual_villain_preflop_commit)
        
        hero_role_for_parser = "OOP" if hero_is_oop else "IP"
        if postflop_action_history_str:
            actions = postflop_action_history_str.split('/')
            current_bet_to_match_on_street = 0.0
            hero_invested_this_street = 0.0
            villain_invested_this_street = 0.0
            for action_part in actions:
                action_part = action_part.strip()
                if not action_part or "dealcards" in action_part.lower():
                    current_bet_to_match_on_street = 0.0
                    hero_invested_this_street = 0.0
                    villain_invested_this_street = 0.0
                    continue
                actor_role_from_str, action_verb_and_amount = "", ""
                if action_part.startswith("OOP_"): actor_role_from_str, action_verb_and_amount = "OOP", action_part[len("OOP_"):].strip()
                elif action_part.startswith("IP_"): actor_role_from_str, action_verb_and_amount = "IP", action_part[len("IP_"):].strip()
                else: continue
                is_hero_acting = (actor_role_from_str == hero_role_for_parser)
                action_amount_parsed = parse_action_amount(action_verb_and_amount)
                actor_current_stack = hero_stack if is_hero_acting else villain_stack
                actor_invested_this_street_val = hero_invested_this_street if is_hero_acting else villain_invested_this_street

                if "CALL" in action_verb_and_amount.upper():
                    amount_to_call = max(0, current_bet_to_match_on_street - actor_invested_this_street_val)
                    actual_call_amount = min(amount_to_call, actor_current_stack)
                    if is_hero_acting: hero_stack -= actual_call_amount; hero_invested_this_street += actual_call_amount
                    else: villain_stack -= actual_call_amount; villain_invested_this_street += actual_call_amount
                    pot_from_actions += actual_call_amount
                elif "BET" in action_verb_and_amount.upper():
                    bet_amount = min(action_amount_parsed, actor_current_stack)
                    if is_hero_acting: hero_stack -= bet_amount; hero_invested_this_street += bet_amount
                    else: villain_stack -= bet_amount; villain_invested_this_street += bet_amount
                    pot_from_actions += bet_amount
                    current_bet_to_match_on_street = hero_invested_this_street if is_hero_acting else villain_invested_this_street
                elif "RAISE" in action_verb_and_amount.upper():
                    new_total_bet_for_street = action_amount_parsed
                    amount_needed_for_raise = max(0, new_total_bet_for_street - actor_invested_this_street_val)
                    actual_raise_contribution = min(amount_needed_for_raise, actor_current_stack)
                    if is_hero_acting: hero_stack -= actual_raise_contribution; hero_invested_this_street += actual_raise_contribution
                    else: villain_stack -= actual_raise_contribution; villain_invested_this_street += actual_raise_contribution
                    pot_from_actions += actual_raise_contribution
                    current_bet_to_match_on_street = hero_invested_this_street if is_hero_acting else villain_invested_this_street
                hero_stack, villain_stack = max(0, hero_stack), max(0, villain_stack)
        return {"hero_stack": round(hero_stack, 2), "villain_stack": round(villain_stack, 2), "pot_from_actions": round(pot_from_actions, 2)}

INPUT_CSV_PATH = "/home/xuandong/mnt/poker/internal-search-distillation-postflop-solver/datasets/3chips_doubled_action_reset_ranges_postflop_500k_train_set_game_scenario_information.csv"
OUTPUT_CSV_PATH = "/home/xuandong/mnt/poker/internal-search-distillation-postflop-solver/datasets/4effective_stack_and_pot_chips_doubled_action_reset_ranges_postflop_500k_train_set_game_scenario_information.csv"

def _get_actions_before_street(
    full_postflop_action: Optional[str],
    evaluation_street: str
) -> Optional[str]:
    """
    Extracts the portion of the postflop_action string that occurred *before*
    the cards for the current evaluation_street were dealt.
    Returns None if no relevant prior postflop history exists.
    This logic is adapted from augment_csv_with_stacks.py's _extract_postflop_history_for_stack_calc.
    The key difference is ensuring we only take actions *before* the current street's deal.
    """
    if not full_postflop_action or pd.isna(full_postflop_action):
        return None

    actions_to_parse = str(full_postflop_action).strip()
    actions_before_current_street = None

    # Pattern looks for "/dealcards/" followed by one or more non-"/" characters (the card)
    # and then optionally a slash if more actions follow.
    deal_card_pattern = r"/dealcards/([^/]+)/?"
    matches = list(re.finditer(deal_card_pattern, actions_to_parse))

    if evaluation_street == "River":
        # We need all actions before the river card was dealt.
        # This means actions before the *second* "dealcards" segment.
        if len(matches) >= 2: # At least turn and river cards were dealt in the full history
            actions_before_current_street = actions_to_parse[:matches[1].start()]
        elif len(matches) == 1: # Only turn card dealt, means all actions are flop AND turn actions.
                                # This implies the evaluation is on the river, but river card might not be in string yet
                                # or the string ends after turn actions. For "start of river", this is fine.
            actions_before_current_street = actions_to_parse 
        elif not matches: # No dealcards, implies all actions are flop actions (if any postflop actions at all)
            actions_before_current_street = actions_to_parse # Should include all flop if evaluating river start and no turn/river cards dealt.

    elif evaluation_street == "Turn":
        # We need all actions before the turn card was dealt.
        # This means actions before the *first* "/dealcards/" segment.
        if matches: # If any dealcards segment exists
            actions_before_current_street = actions_to_parse[:matches[0].start()]
        elif "/dealcards/" not in actions_to_parse : # No dealcards implies these are all flop actions.
             actions_before_current_street = actions_to_parse # if any actions exist, they must be flop

    elif evaluation_street == "Flop":
        # For the start of the Flop, there are no *prior* postflop actions to consider.
        actions_before_current_street = None
    
    return actions_before_current_street.strip('/') if actions_before_current_street else None


def process_poker_data(input_path: str, output_path: str):
    """
    Reads the input CSV, calculates starting pot and effective stack for each row
    at the beginning of the evaluation_at street, and writes to output CSV.
    """
    try:
        df = pd.read_csv(input_path)
    except FileNotFoundError:
        print(f"Error: Input CSV file not found at '{input_path}'")
        # Try path relative to common workspace root as a fallback
        alt_input_path = os.path.join(os.getenv('HOME', '/home/xuandong'), "mnt/poker/internal-search-distillation-postflop-solver", input_path)
        try:
            df = pd.read_csv(alt_input_path)
            print(f"Successfully loaded from alternative path: {alt_input_path}")
            # Update output path to be relative to where input was found, or keep it as specified.
            # For simplicity, let's assume output_path is fine or should also be adjusted by user.
        except FileNotFoundError:
            print(f"Error: Also could not find input CSV at alternative path '{alt_input_path}'")
            return


    new_cols_data = {
        "starting_pot": [],
        "effective_stack": []
    }

    print(f"Processing {len(df)} rows from {input_path}...")

    for index, row in df.iterrows():
        hero_pos_str = str(row.get('hero_position', 'IP')).strip().upper() # Ensure it's a string
        hero_is_oop = (hero_pos_str == "OOP")
        
        preflop_action_str = str(row.get('preflop_action', '')) if pd.notna(row.get('preflop_action')) else None
        full_postflop_action_str = str(row.get('postflop_action', '')) if pd.notna(row.get('postflop_action')) else None
        evaluation_at_str = str(row.get('evaluation_at', 'Flop')).strip().capitalize() # Ensure string and capitalize

        # Get actions that happened *before* the current evaluation_at street's cards were dealt
        postflop_history_for_calc = _get_actions_before_street(
            full_postflop_action_str,
            evaluation_at_str
        )

        # Calculate stacks and pot based on preflop and prior postflop actions
        # The pot_from_actions here will be the total pot *before* any action on the current (evaluation_at) street
        stack_calc_results = calculate_stacks_after_actions(
            hero_is_oop=hero_is_oop,
            preflop_action_str=preflop_action_str,
            postflop_action_history_str=postflop_history_for_calc,
            initial_hero_stack=INITIAL_STACK_SIZE,
            initial_villain_stack=INITIAL_STACK_SIZE
        )
        
        current_starting_pot = stack_calc_results["pot_from_actions"]

        pf_action_str_for_check = preflop_action_str if preflop_action_str else ""
        
        # Split preflop actions to identify actors. Actors are at even indices.
        # e.g., "HJ/4.0bb/BB/call" -> tokens ["HJ", "4.0bb", "BB", "call"]
        # Actors are "HJ", "BB".
        action_tokens = pf_action_str_for_check.split('/')
        sb_is_active_player = False
        for i in range(0, len(action_tokens), 2): # Iterate through actor positions
            if action_tokens[i].strip() == "SB":
                sb_is_active_player = True
                break
        
        if not sb_is_active_player:
            current_starting_pot += 1.0 # Adding SB's 1bb contribution

        hero_stack_at_street_start = stack_calc_results["hero_stack"]
        villain_stack_at_street_start = stack_calc_results["villain_stack"]
        
        current_effective_stack = min(hero_stack_at_street_start, villain_stack_at_street_start)
        current_effective_stack = max(0, current_effective_stack) # Ensure non-negative

        new_cols_data["starting_pot"].append(current_starting_pot)
        new_cols_data["effective_stack"].append(current_effective_stack)

        if (index + 1) % 1000 == 0:
            print(f"  Processed {index + 1} rows...")

    df["starting_pot"] = new_cols_data["starting_pot"]
    df["effective_stack"] = new_cols_data["effective_stack"]

    try:
        df.to_csv(output_path, index=False)
        print(f"Successfully processed data and saved to '{output_path}'")
    except Exception as e:
        print(f"Error writing output CSV to '{output_path}': {e}")
        # Try alt path for output as well
        alt_output_path = os.path.join(os.getenv('HOME', '/home/xuandong'), "mnt/poker/internal-search-distillation-postflop-solver", output_path)
        try:
            df.to_csv(alt_output_path, index=False)
            print(f"Successfully processed data and saved to alternative path '{alt_output_path}'")
        except Exception as e_alt:
            print(f"Error writing output CSV to alternative path '{alt_output_path}': {e_alt}")


if __name__ == "__main__":
    # Make sure the script can find action_parser if not installed as a package
    # This is a common way to handle imports from sibling directories in scripts.
    import sys
    # Assuming the script is in data_processing and action_parser is in dataset_generator,
    # to import dataset_generator.action_parser, the parent of data_processing and dataset_generator
    # needs to be in sys.path.
    # Current file: /home/xuandong/mnt/poker/internal-search-distillation-postflop-solver/data_processing/starting_pot_effective_stack_counter.py
    # Workspace root: /home/xuandong/mnt/poker/internal-search-distillation-postflop-solver
    # So, the workspace root is already the parent of data_processing and dataset_generator.
    # Python should be able to find dataset_generator if the script is run from the workspace root.
    # If running `python data_processing/starting_pot_effective_stack_counter.py` from the workspace root, imports should work.
    # If running from within data_processing, then `../` is needed.
    # The `try-except ImportError` block for `calculate_stacks_after_actions` and `INITIAL_STACK_SIZE`
    # already provides a fallback by defining the functions directly if the import fails.
    
    # Check if action_parser essentials were imported or need to rely on fallback
    if 'calculate_stacks_after_actions' not in globals():
        print("Warning: Using fallback definitions for action_parser components. Ensure correct project structure for imports.")

    # Allow overriding paths via environment variables for flexibility, similar to augment_csv_with_stacks.py
    # but using the specific paths for this script.
    # The INPUT_CSV_PATH and OUTPUT_CSV_PATH are defined globally in this script.
    # We can add logic to check for env vars if needed, but for now, direct paths are used.
    
    # Ensure the input CSV exists before processing
    # The input_path is data_processing/gamestate_dataset_chips_doubled.csv
    # The script itself is in data_processing. So relative path should be just the filename.
    # However, the global INPUT_CSV_PATH is "data_processing/gamestate_dataset_chips_doubled.csv"
    # This implies the script should be run from the workspace root.
    # Let's adjust INPUT_CSV_PATH to be relative to the script's directory if run directly,
    # or assume it's run from project root.
    # For now, the paths are hardcoded assuming project root execution.

    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Correcting paths to be relative to the workspace root as per initial definition
    # The defined INPUT_CSV_PATH and OUTPUT_CSV_PATH are already relative to the workspace root.
    # input_csv = os.path.join(current_dir, "gamestate_dataset_chips_doubled.csv")
    # output_csv = os.path.join(current_dir, "gamestate_dataset_chips_doubled_augmented.csv")
    
    # Use the globally defined paths, assuming execution from project root.
    input_csv = INPUT_CSV_PATH
    output_csv = OUTPUT_CSV_PATH

    # Check if default input CSV exists where expected (relative to project root)
    # Construct absolute path for checking based on common workspace structure
    # This is a bit redundant given the try-except in process_poker_data, but good for a pre-check.
    workspace_root = os.path.abspath(os.path.join(current_dir, "..")) # up one level from data_processing
    abs_input_csv_path = os.path.join(workspace_root, input_csv)


    if not os.path.exists(abs_input_csv_path):
         print(f"Default input CSV '{abs_input_csv_path}' (derived from '{input_csv}') not found.")
         print("Please ensure the CSV exists or adjust INPUT_CSV_PATH.")
    else:
        process_poker_data(input_csv, output_csv) # Use the paths as defined


# to run: python -m data_processing.starting_pot_effective_stack_counter