'''
augment_csv_with_stacks.py
'''

import csv
import os
import re # For splitting action strings, if needed again for the helper
from typing import Optional, Dict # Added for type hints

# Assuming action_parser.py is in the same directory or accessible in Python path
from ..dataset_generator.action_parser import calculate_stacks_after_actions, INITIAL_STACK_SIZE

# Define input and output file paths
# These should be configured by the user or via arguments if the script is generalized
DEFAULT_ORIGINAL_CSV_PATH = "/home/xuandong/mnt/poker/internal-search-distillation-postflop-solver/datasets/ranges_postflop_500k_train_set_game_scenario_information.csv"
DEFAULT_AUGMENTED_CSV_PATH = "/home/xuandong/mnt/poker/internal-search-distillation-postflop-solver/datasets/augmented_poker_data_with_stacks.csv"

NEW_STACK_COLUMNS = [
    'calculated_hero_stack',
    'calculated_villain_stack',
    'calculated_eff_stack',
    'calculated_postflop_history_segment' # For debugging what was parsed
]

def _extract_postflop_history_for_stack_calc(
    full_postflop_action: Optional[str], 
    evaluation_street: str,
    turn_card_str: Optional[str], 
    river_card_str: Optional[str]
) -> Optional[str]:
    """
    Extracts the portion of the postflop_action string that occurred *before*
    the current evaluation_street's decision point for stack calculation.
    Returns None if no relevant prior postflop history exists.
    This logic is crucial and may need refinement based on action string consistency.
    """
    if not full_postflop_action:
        return None

    actions_to_parse = full_postflop_action.strip()
    history_segment = None

    # Use regex to find dealcards and split. This is more robust.
    # Pattern looks for "/dealcards/" followed by one or more non-"/" characters (the card)
    # and then optionally a slash if more actions follow.
    deal_card_pattern = r"/dealcards/([^/]+)/?"

    if evaluation_street == "River":
        # History includes Flop and Turn actions.
        # We need actions before the river card was dealt.
        # If river_card_str is known, we can be more specific.
        # Otherwise, we look for the *last* "dealcards" segment that isn't followed by another.
        # Or, more simply, if we find two "dealcards" segments, the second one is the river deal.
        
        # Find all dealcards occurrences
        matches = list(re.finditer(deal_card_pattern, actions_to_parse))
        
        if len(matches) >= 2: # At least turn and river cards dealt
            # The second match corresponds to the river card dealing
            # History is everything before this second match started
            history_segment = actions_to_parse[:matches[1].start()]
        elif len(matches) == 1: # Only turn card dealt, implies all actions are flop and turn
            history_segment = actions_to_parse 
        elif not matches: # No dealcards, implies all actions are flop actions (if any postflop actions at all)
            history_segment = actions_to_parse
        # else: ambiguous or error in string format, history_segment remains None or could be actions_to_parse

    elif evaluation_street == "Turn":
        # History includes Flop actions. We need actions before the turn card was dealt.
        # This means everything before the *first* "/dealcards/" segment.
        match = re.search(deal_card_pattern, actions_to_parse)
        if match:
            history_segment = actions_to_parse[:match.start()]
        elif "/dealcards/" not in actions_to_parse: # No dealcards implies these are all flop actions.
            history_segment = actions_to_parse
        # else: no prior postflop history relevant (e.g. preflop, or error)

    elif evaluation_street == "Flop":
        # For Flop evaluation, there are no *prior* postflop streets for stack calculation.
        # Any actions in `full_postflop_action_str` might be leading to the first flop decision
        # or are part of the decision itself. For calculating stacks *before* this decision, history is None.
        history_segment = None 
    
    return history_segment.strip('/') if history_segment else None

def augment_dataset_with_stacks(original_csv_path: str, augmented_csv_path: str):
    """
    Reads the original CSV, calculates stack sizes for each row, 
    and writes an augmented CSV with new stack columns.
    """
    print(f"Starting dataset augmentation: Reading from '{original_csv_path}'")
    processed_rows = 0
    written_rows = 0

    try:
        with open(original_csv_path, mode='r', newline='', encoding='utf-8') as infile, \
             open(augmented_csv_path, mode='w', newline='', encoding='utf-8') as outfile:
            
            reader = csv.DictReader(infile)
            if reader.fieldnames is None:
                print("Error: Could not read fieldnames from the input CSV.")
                return

            all_fieldnames = reader.fieldnames + NEW_STACK_COLUMNS
            writer = csv.DictWriter(outfile, fieldnames=all_fieldnames)
            writer.writeheader()

            for i, row in enumerate(reader):
                processed_rows += 1
                try:
                    # --- Pot Size Adjustment ---
                    original_pot_size_str = row.get('pot_size')
                    if original_pot_size_str is None:
                        print(f"Warning: Row {i+1} is missing 'pot_size'. Skipping pot adjustment for this row.")
                        # Decide how to handle this - skip row, use a default, or raise error?
                        # For now, we'll proceed, and stack calcs will use other data.
                        # The 'pot_size' field in the output row might remain empty or as is.
                        current_pot_float = 0.0 # Or handle as error
                    else:
                        try:
                            current_pot_float = float(original_pot_size_str)
                        except ValueError:
                            print(f"Warning: Row {i+1} has invalid 'pot_size': {original_pot_size_str}. Using 0.0 for pot adjustment.")
                            current_pot_float = 0.0

                    preflop_action_str = row.get('preflop_action')
                    
                    # Heuristic to add 0.5bb if SB folded preflop and pot looks like an integer
                    if preflop_action_str and \
                       "SB/" not in preflop_action_str and \
                       not preflop_action_str.startswith("SB") and \
                       not preflop_action_str.endswith("/SB") and \
                       "SB " not in preflop_action_str and \
                       current_pot_float % 1 == 0: # Check if it's a whole number
                        current_pot_float += 0.5
                    
                    # Update the row's pot_size with the potentially adjusted float value
                    # This will be written to the augmented CSV.
                    # The solver call in query_solver.py will use this (as float).
                    row['pot_size'] = str(current_pot_float) 

                    # --- End Pot Size Adjustment ---

                    hero_pos_str = row.get('hero_position', 'IP').strip().upper()
                    hero_is_oop = (hero_pos_str == "OOP")
                    
                    evaluation_at_str = row.get('evaluation_at', '').strip().capitalize()
                    full_postflop_action_str = row.get('postflop_action')
                    # turn_card = row.get('board_turn') # Not needed for stack calc if using full string directly
                    # river_card = row.get('board_river') # Not needed for stack calc if using full string directly

                    # The 'postflop_action' column from the CSV is assumed to be the
                    # full history of postflop actions leading up to the current decision point.
                    # We use this directly for stack calculation.
                    current_postflop_history_for_stacks = None
                    if full_postflop_action_str:
                        current_postflop_history_for_stacks = full_postflop_action_str.strip()
                    
                    # _extract_postflop_history_for_stack_calc is no longer called here
                    # postflop_history_segment = _extract_postflop_history_for_stack_calc(
                    # full_postflop_action_str,
                    # evaluation_at_str,
                    # turn_card, 
                    # river_card 
                    # )

                    # Calculate stacks based on actions from the start of the hand
                    stack_calc_results = calculate_stacks_after_actions(
                        hero_is_oop=hero_is_oop,
                        preflop_action_str=preflop_action_str, 
                        postflop_action_history_str=current_postflop_history_for_stacks, # Use the stripped full postflop string
                        initial_hero_stack=INITIAL_STACK_SIZE,
                        initial_villain_stack=INITIAL_STACK_SIZE
                    )
                    
                    calc_hero_stack = stack_calc_results["hero_stack"]
                    calc_villain_stack = stack_calc_results["villain_stack"]
                    # pot_from_parsed_actions = stack_calc_results["pot_from_actions"]
                    # The pot_size from CSV is the one at the point of decision.
                    # pot_from_parsed_actions can be a sanity check.

                    calc_eff_stack = min(calc_hero_stack, calc_villain_stack)
                    calc_eff_stack = max(0, calc_eff_stack) # Ensure non-negative

                    # Augment the row
                    augmented_row = row.copy() # Start with original data
                    augmented_row['calculated_hero_stack'] = calc_hero_stack
                    augmented_row['calculated_villain_stack'] = calc_villain_stack
                    augmented_row['calculated_eff_stack'] = calc_eff_stack
                    # Store the same postflop history that was used for stack calculation
                    augmented_row['calculated_postflop_history_segment'] = current_postflop_history_for_stacks if current_postflop_history_for_stacks else ""
                    
                    writer.writerow(augmented_row)
                    written_rows += 1

                    if (i + 1) % 5000 == 0:
                        print(f"  Processed {i+1} rows...")

                except Exception as e_row:
                    print(f"Error processing row {i+1}: {row.get('postflop_action', 'N/A')}")
                    print(f"Exception: {e_row}")
                    # Optionally, write the row anyway with blank new columns or skip
                    # For now, skipping rows with errors in augmentation.
            
        print(f"Augmentation complete. Processed {processed_rows} rows. Written {written_rows} rows to '{augmented_csv_path}'.")

    except FileNotFoundError:
        print(f"Error: Original CSV file not found at '{original_csv_path}'.")
    except Exception as e_file:
        print(f"An error occurred during file processing: {e_file}")

if __name__ == "__main__":
    # Allow overriding paths via environment variables for flexibility
    original_csv = os.getenv('ORIGINAL_POKER_CSV', DEFAULT_ORIGINAL_CSV_PATH)
    augmented_csv = os.getenv('AUGMENTED_POKER_CSV', DEFAULT_AUGMENTED_CSV_PATH)

    if original_csv == DEFAULT_ORIGINAL_CSV_PATH and not os.path.exists(original_csv):
         print(f"Default original CSV '{original_csv}' not found.")
         print("Please ensure the CSV exists or set the ORIGINAL_POKER_CSV environment variable.")
    else:
        augment_dataset_with_stacks(original_csv, augmented_csv) 