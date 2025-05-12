import csv
import os
import math

'''
    To run: python -m data_processing.sample_from_csv
'''

def split_csv(input_csv_path, divide_by_N, output_directory):
    """
    Reads an input CSV file, divides it into N equal smaller CSV files,
    and saves them in the specified output directory.
    """
    try:
        with open(input_csv_path, 'r', newline='') as infile:
            reader = list(csv.reader(infile))
            header = reader[0]
            data_rows = reader[1:]
            total_num_rows = len(data_rows)

            if total_num_rows == 0:
                print("The input CSV file is empty or contains only a header.")
                return

            if divide_by_N <= 0:
                print("Error: divide_by_N must be a positive integer.")
                return

            rows_per_file = math.ceil(total_num_rows / divide_by_N)

            if not os.path.exists(output_directory):
                os.makedirs(output_directory)
                print(f"Created directory: {output_directory}")

            for i in range(divide_by_N):
                start_index = i * rows_per_file
                end_index = start_index + rows_per_file
                chunk = data_rows[start_index:end_index]

                if not chunk:  # Don't create empty files if divide_by_N is too large
                    continue

                output_filename = f"output_part_{i+1}_{input_csv_path.split('/')[-1]}"
                output_filepath = os.path.join(output_directory, output_filename)

                with open(output_filepath, 'w', newline='') as outfile:
                    writer = csv.writer(outfile)
                    writer.writerow(header)
                    writer.writerows(chunk)
                print(f"Saved: {output_filepath}")

    except FileNotFoundError:
        print(f"Error: Input CSV file not found at {input_csv_path}")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    # --- Configuration ---
    input_file = "/home/xuandong/mnt/poker/internal-search-distillation-postflop-solver/datasets/4effective_stack_and_pot_chips_doubled_action_reset_ranges_postflop_500k_train_set_game_scenario_information.csv"  # Replace with your input CSV file path
    num_splits = 5            # Replace with your desired N value
    output_dir = "/home/xuandong/mnt/poker/internal-search-distillation-postflop-solver/datasets/500k_split" # Hardcoded output directory path
    # ---------------------

    # Create a dummy input.csv for testing if it doesn't exist
    if not os.path.exists(input_file):
        print(f"Creating a dummy input file: {input_file}")
        with open(input_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["ID", "Name", "Value"]) # Header
            for i in range(1, 101): # 100 data rows
                writer.writerow([i, f"Name_{i}", i*10])
        print(f"Dummy {input_file} created with 100 data rows.")


    if os.path.exists(input_file):
        split_csv(input_file, num_splits, output_dir)
    else:
        print(f"Please create an '{input_file}' in the same directory as the script, or update the 'input_file' variable.")
        print("The script will not run without an input file.")
