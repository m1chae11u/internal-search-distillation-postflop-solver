import csv

'''
to run: python -m data_processing.flop_splitter_turn_river

function that takes in a csv file, splits it into 2 csv files, separating all rows where the value in the "evaluation_at" column is "Flop" from the rest.
'''

def split_csv_by_evaluation(input_csv_path, flop_output_csv_path, rest_output_csv_path):
    """
    Splits the input CSV file into two files:
    - One containing rows where 'evaluation_at' == 'Flop'
    - One containing all other rows
    Args:
        input_csv_path (str): Path to the input CSV file.
        flop_output_csv_path (str): Path to output CSV for 'Flop' rows.
        rest_output_csv_path (str): Path to output CSV for all other rows.
    """
    with open(input_csv_path, 'r', newline='') as infile:
        reader = csv.DictReader(infile)
        fieldnames = reader.fieldnames
        flop_rows = []
        rest_rows = []
        for row in reader:
            if row.get('evaluation_at') == 'Flop':
                flop_rows.append(row)
            else:
                rest_rows.append(row)

    with open(flop_output_csv_path, 'w', newline='') as flopfile:
        writer = csv.DictWriter(flopfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(flop_rows)

    with open(rest_output_csv_path, 'w', newline='') as restfile:
        writer = csv.DictWriter(restfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rest_rows)

# Example usage (replace with actual file paths):
split_csv_by_evaluation('/home/xuandong/mnt/poker/internal-search-distillation-postflop-solver/datasets/4effective_stack_and_pot_chips_doubled_action_reset_ranges_postflop_500k_train_set_game_scenario_information.csv', '/home/xuandong/mnt/poker/internal-search-distillation-postflop-solver/datasets/1.1_flop_split/flop_rows.csv', '/home/xuandong/mnt/poker/internal-search-distillation-postflop-solver/datasets/1.2_turn_river_split/turn_river_rows.csv')
