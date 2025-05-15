import csv
import os

def remove_leading_csv_rows(input_csv_path: str, num_rows_to_remove: int, output_csv_path: str) -> None:
    """
    Removes a specified number of leading rows from an input CSV file
    and writes the remaining rows to an output CSV file.
    """
    if not os.path.exists(input_csv_path):
        print(f"Error: Input CSV file not found at {input_csv_path}")
        return

    if num_rows_to_remove < 0:
        print(f"Error: Number of rows to remove ({num_rows_to_remove}) cannot be negative.")
        return

    rows_to_keep = []
    try:
        with open(input_csv_path, 'r', newline='') as infile:
            reader = csv.reader(infile)
            # Attempt to skip the header rows
            for i in range(num_rows_to_remove):
                try:
                    next(reader)  # Skip a row
                except StopIteration:
                    # Reached end of file before skipping all desired rows
                    print(f"Warning: Tried to remove {num_rows_to_remove} rows, but file {input_csv_path} only has {i} rows. Output will be empty.")
                    break # No more rows to skip or keep
            
            # Read the rest of the rows
            for row in reader:
                rows_to_keep.append(row)

    except FileNotFoundError: # Should be caught by os.path.exists, but good practice
        print(f"Error: Input CSV file not found at {input_csv_path}")
        return
    except Exception as e:
        print(f"An error occurred while reading {input_csv_path}: {e}")
        return

    try:
        # Create the directory for the output file if it doesn't exist
        output_dir = os.path.dirname(output_csv_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        with open(output_csv_path, 'w', newline='') as outfile:
            writer = csv.writer(outfile)
            if rows_to_keep:
                writer.writerows(rows_to_keep)
            print(f"Successfully processed {input_csv_path}. Output written to {output_csv_path}.")
            if not rows_to_keep:
                print(f"Note: The output file {output_csv_path} is empty, either because all rows were removed or the input file was empty after the specified skips.")
        
    except IOError:
        print(f"Error: Could not write to output CSV file {output_csv_path}")
    except Exception as e:
        print(f"An unexpected error occurred while writing to {output_csv_path}: {e}")


if __name__ == '__main__':
    # --- USER MODIFIABLE VARIABLES ---
    # Please modify these paths and the number of rows to suit your needs.
    INPUT_CSV_FILE_PATH = "/home/xuandong/mnt/poker/internal-search-distillation-postflop-solver/datasets/1.2_turn_river_split/turn_river_rows.csv"
    NUMBER_OF_ROWS_TO_REMOVE = 32418   
    OUTPUT_CSV_FILE_PATH = "/home/xuandong/mnt/poker/internal-search-distillation-postflop-solver/datasets/1.2_turn_river_split/turn_river_rows.csv"
    # --- END OF USER MODIFIABLE VARIABLES ---

    print(f"Attempting to remove the first {NUMBER_OF_ROWS_TO_REMOVE} rows from {INPUT_CSV_FILE_PATH} and save to {OUTPUT_CSV_FILE_PATH}.")
    remove_leading_csv_rows(INPUT_CSV_FILE_PATH, NUMBER_OF_ROWS_TO_REMOVE, OUTPUT_CSV_FILE_PATH)
    print("Script finished.")

# to run: 
# python -m data_processing.row_remover