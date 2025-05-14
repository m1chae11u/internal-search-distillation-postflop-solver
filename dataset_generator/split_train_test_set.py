import json
import argparse
import random
import os

def split_by_percentage(input_file, train_percentage, output_train_file, output_test_file):
    """
    Splits data from a JSON file into training and testing sets based on a percentage.

    Args:
        input_file (str): Path to the input JSON file.
        train_percentage (float): Percentage of data to be used for the training set (e.g., 0.8 for 80%).
        output_train_file (str): Path to save the training data JSON file.
        output_test_file (str): Path to save the test data JSON file.
    """
    try:
        with open(input_file, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: Input file '{input_file}' not found.")
        return
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from '{input_file}'.")
        return

    if not isinstance(data, list):
        print("Error: JSON data must be a list of records.")
        return

    random.shuffle(data)
    
    split_index = int(len(data) * train_percentage)
    
    train_data = data[:split_index]
    test_data = data[split_index:]
    
    try:
        os.makedirs(os.path.dirname(output_train_file), exist_ok=True)
        with open(output_train_file, 'w') as f:
            json.dump(train_data, f, indent=4)
        print(f"Training data saved to '{output_train_file}'")
    except IOError:
        print(f"Error: Could not write training data to '{output_train_file}'.")

    try:
        os.makedirs(os.path.dirname(output_test_file), exist_ok=True)
        with open(output_test_file, 'w') as f:
            json.dump(test_data, f, indent=4)
        print(f"Test data saved to '{output_test_file}'")
    except IOError:
        print(f"Error: Could not write test data to '{output_test_file}'.")


def split_by_row_count(input_file, test_rows, output_train_file, output_test_file):
    """
    Splits data from a JSON file into training and testing sets based on a fixed number of rows for the test set.

    Args:
        input_file (str): Path to the input JSON file.
        test_rows (int): Number of rows to be used for the test set.
        output_train_file (str): Path to save the training data JSON file.
        output_test_file (str): Path to save the test data JSON file.
    """
    try:
        with open(input_file, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: Input file '{input_file}' not found.")
        return
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from '{input_file}'.")
        return

    if not isinstance(data, list):
        print("Error: JSON data must be a list of records.")
        return
        
    if test_rows >= len(data):
        print(f"Error: Number of test rows ({test_rows}) cannot be greater than or equal to total rows ({len(data)}).")
        return
    if test_rows <= 0:
        print(f"Error: Number of test rows ({test_rows}) must be a positive integer.")
        return

    random.shuffle(data) # Shuffle to ensure random selection for test set
    
    test_data = data[:test_rows] # Could also do random.sample(data, test_rows) if order doesn't matter for test
    train_data = data[test_rows:]
    
    # Alternative if train_data should be the remainder of shuffled data NOT in test_data
    # This ensures that if we took test_data = random.sample(data, test_rows), train_data would be the rest
    # However, current approach of slicing after shuffle is simpler and achieves randomness for both sets.
    # If data was very large and test_rows small, random.sample might be more efficient for test_data
    # and then set difference for train_data, but for typical dataset sizes, shuffling the whole list is fine.

    try:
        os.makedirs(os.path.dirname(output_train_file), exist_ok=True)
        with open(output_train_file, 'w') as f:
            json.dump(train_data, f, indent=4)
        print(f"Training data saved to '{output_train_file}'")
    except IOError:
        print(f"Error: Could not write training data to '{output_train_file}'.")

    try:
        os.makedirs(os.path.dirname(output_test_file), exist_ok=True)
        with open(output_test_file, 'w') as f:
            json.dump(test_data, f, indent=4)
        print(f"Test data saved to '{output_test_file}'")
    except IOError:
        print(f"Error: Could not write test data to '{output_test_file}'.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split JSON data into training and test sets.")
    subparsers = parser.add_subparsers(dest="command", help="Sub-command help", required=True)

    # Sub-parser for splitting by percentage
    parser_percentage = subparsers.add_parser("percentage", help="Split data by percentage.")
    parser_percentage.add_argument("--input_file", type=str, required=True, help="Path to the input JSON file.")
    parser_percentage.add_argument("--train_percentage", type=float, required=True, help="Percentage for the training set (e.g., 0.8 for 80%).")
    parser_percentage.add_argument("--output_train_file", type=str, default="train_data.json", help="Path to save the training data. Defaults to train_data.json")
    parser_percentage.add_argument("--output_test_file", type=str, default="test_data.json", help="Path to save the test data. Defaults to test_data.json")

    # Sub-parser for splitting by row count
    parser_rows = subparsers.add_parser("rows", help="Split data by a fixed number of rows for the test set.")
    parser_rows.add_argument("--input_file", type=str, required=True, help="Path to the input JSON file.")
    parser_rows.add_argument("--test_rows", type=int, required=True, help="Number of rows for the test set.")
    parser_rows.add_argument("--output_train_file", type=str, default="train_data.json", help="Path to save the training data. Defaults to train_data.json")
    parser_rows.add_argument("--output_test_file", type=str, default="test_data.json", help="Path to save the test data. Defaults to test_data.json")
    
    args = parser.parse_args()

    if args.command == "percentage":
        if not (0 < args.train_percentage < 1):
            print("Error: Train percentage must be between 0 and 1 (exclusive).")
        else:
            split_by_percentage(args.input_file, args.train_percentage, args.output_train_file, args.output_test_file)
    elif args.command == "rows":
        split_by_row_count(args.input_file, args.test_rows, args.output_train_file, args.output_test_file)

'''
example usage:

python -m dataset_generator.split_train_test_set \
    rows \
    --input_file /home/xuandong/mnt/poker/internal-search-distillation-postflop-solver/datasets/1.2_turn_river_split/first_22k_turn_river_search_tree_datasubset.json \
    --test_rows 2000 \
    --output_train_file /home/xuandong/mnt/poker/internal-search-distillation-postflop-solver/datasets/test_turn_river_sets/first_20k_turn_river_search_tree_datasubset_train.json \
    --output_test_file /home/xuandong/mnt/poker/internal-search-distillation-postflop-solver/datasets/train_turn_river_sets/first_2k_test_turn_river_search_tree_datasubset_test.json
'''