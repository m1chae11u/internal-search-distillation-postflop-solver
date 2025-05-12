import csv
import os
from dataset_generator.query_solver import run_solver_from_rust
from dataset_generator.trace_formatter import format_internal_search_trace

DEFAULT_INPUT_CSV_PATH = "/home/xuandong/mnt/poker/internal-search-distillation-postflop-solver/datasets/augmented_poker_data_with_stacks.csv"

def process_gamestate_row(row_data: dict):
    """Parses an augmented row, gets solver data, and formats the trace."""
    try:
        oop_range_str = row_data.get('oop_range_str', "")
        ip_range_str = row_data.get('ip_range_str', "")
        flop_str = row_data.get('board_flop', "")
        
        turn_card = row_data.get('board_turn')
        turn_card_opt_str = turn_card if turn_card and turn_card.strip() else None
        
        river_card = row_data.get('board_river')
        river_card_opt_str = river_card if river_card and river_card.strip() else None
        
        pot_at_decision = int(float(row_data.get('pot_size', 0))) # Ensure conversion from potential float string

        # Read pre-calculated stack values from the augmented CSV
        actual_hero_stack = float(row_data.get('calculated_hero_stack', 100.0))
        actual_villain_stack = float(row_data.get('calculated_villain_stack', 100.0))
        eff_stack_for_solver = float(row_data.get('calculated_eff_stack', 100.0))
        # calculated_postflop_history_segment = row_data.get('calculated_postflop_history_segment') # Available if needed for debug

        use_compression_flag = False
        max_iterations_val = int(float(row_data.get('max_iter', 10000)))
        target_exploit_percentage_val = float(row_data.get('exploit_pct', 0.01))
        should_print_progress = str(row_data.get('print_progress', 'false')).lower() == 'true'

        expected_node_type = "hero_decision"
        
        if should_print_progress:
            print(f"  Read from CSV: HeroStack={actual_hero_stack}, VillainStack={actual_villain_stack}, EffStack={eff_stack_for_solver}")
            print(f"  Pot for solver: {pot_at_decision}")
            # print(f"  Postflop history segment from CSV: '{calculated_postflop_history_segment}'")

        solver_output_model = run_solver_from_rust(
            expected_node_type=expected_node_type,
            oop_range_str=oop_range_str,
            ip_range_str=ip_range_str,
            flop_str=flop_str,
            turn_card_opt_str=turn_card_opt_str,
            river_card_opt_str=river_card_opt_str,
            initial_pot=pot_at_decision, 
            eff_stack=eff_stack_for_solver, 
            use_compression_flag=use_compression_flag,
            max_iterations_val=max_iterations_val,
            target_exploit_percentage_val=target_exploit_percentage_val,
            should_print_progress=should_print_progress,
        )
        
        if not solver_output_model:
            print(f"Warning: No solver output model received for row: {row_data.get('board_flop', 'N/A')}")
            return None

        trace_string = format_internal_search_trace(
            solver_data_model=solver_output_model,
            csv_row=row_data, 
            hero_actual_stack=actual_hero_stack,
            villain_actual_stack=actual_villain_stack
        )
        return trace_string

    except Exception as e:
        print(f"Error processing row: {row_data.get('board_flop', 'N/A')}")
        print(f"Exception: {e}")
        import traceback
        traceback.print_exc()
        return None

def main_create_trace_data(input_csv_path: str):
    """
    Main function to read the (augmented) CSV, process each row, and generate solver output traces.
    """
    if not os.path.exists(input_csv_path):
        print(f"Error: Input CSV file not found at {input_csv_path}")
        print(f"This script expects an augmented CSV with calculated stack columns.")
        print(f"Run augment_csv_with_stacks.py first or ensure '{input_csv_path}' exists.")
        
        # Create a dummy *augmented* CSV if the placeholder path is used and file not found
        if input_csv_path == DEFAULT_INPUT_CSV_PATH and DEFAULT_INPUT_CSV_PATH == "./augmented_poker_data_with_stacks.csv":
            print(f"Creating a dummy '{input_csv_path}' for demonstration.")
            dummy_fieldnames = [
                "preflop_action","board_flop","board_turn","board_river","aggressor_position",
                "postflop_action","evaluation_at","available_moves","pot_size","hero_position",
                "holding","correct_decision","oop_range_str","oop_range_type_selected",
                "ip_range_str","ip_range_type_selected", "eff_stack_orig", "compress", "max_iter", "exploit_pct", "print_progress",
                'calculated_hero_stack', 'calculated_villain_stack', 'calculated_eff_stack', 'calculated_postflop_history_segment'
            ]
            with open(input_csv_path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=dummy_fieldnames)
                writer.writeheader()
                writer.writerow({
                    "preflop_action": "HJ/2.0bb/BB/call", "board_flop": "JcJh4s", "board_turn": "4d", "board_river": "As",
                    "aggressor_position": "OOP", 
                    "postflop_action": "OOP_CHECK/IP_BET_1/OOP_CALL/dealcards/4d/OOP_CHECK/IP_BET_8/OOP_CALL/dealcards/As/OOP_CHECK",
                    "evaluation_at": "River", "available_moves": "['Check', 'Bet 17']", "pot_size": "21", "hero_position": "IP",
                    "holding": "AhKd", "correct_decision": "Check", 
                    "oop_range_str": "AA,AKo,AKs,A3s,KK,QQ,JJ", "oop_range_type_selected": "Balanced",
                    "ip_range_str": "AA,AKo,AJs,A5s,KK,QQ,J4o", "ip_range_type_selected": "Loose", 
                    "eff_stack_orig": "100", "compress": "False", "max_iter": "10000", "exploit_pct": "0.01", "print_progress": "True",
                    'calculated_hero_stack': 80.0, 'calculated_villain_stack': 80.0, 'calculated_eff_stack': 80.0, 
                    'calculated_postflop_history_segment': "OOP_CHECK/IP_BET_1/OOP_CALL/dealcards/4d/OOP_CHECK/IP_BET_8/OOP_CALL"
                })
        return

    processed_row_count = 0
    with open(input_csv_path, mode='r', newline='', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for i, row in enumerate(reader):
            print(f"--- Processing row {i+1} from CSV (Flop: {row.get('board_flop', 'N/A')}) ---")
            trace_output_string = process_gamestate_row(row) 
            if trace_output_string:
                print("Generated Internal Search Trace:")
                print(trace_output_string)
                processed_row_count += 1
            else:
                print(f"Skipping row {i+1} due to processing error or no solver output.")
            print(f"--- Finished row {i+1} ---\n")
            
            if processed_row_count >= 2: # Limiter for testing
                print("Reached test limit of 2 processed rows.")
                break 

    print(f"Finished processing. Total rows successfully processed: {processed_row_count}")

if __name__ == "__main__":
    # This script now expects the augmented CSV path
    csv_path = os.getenv('AUGMENTED_POKER_CSV', DEFAULT_INPUT_CSV_PATH) 
    if csv_path == DEFAULT_INPUT_CSV_PATH and not os.path.exists(csv_path):
         print(f"Default augmented CSV '{csv_path}' (expected at {DEFAULT_INPUT_CSV_PATH}) not found.")
         print(f"Please run augment_csv_with_stacks.py first, or set the AUGMENTED_POKER_CSV environment variable.")
    main_create_trace_data(csv_path) 