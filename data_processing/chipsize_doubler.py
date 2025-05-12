import pandas as pd
import re
import os

# Helper function to format the doubled number
def _format_doubled_number(num_str: str, doubled_value: float) -> str:
    """
    Formats the doubled number, trying to preserve int/float appearance.
    If original was int (no decimal) and doubled value is whole, format as int.
    Otherwise, format as float.
    """
    original_had_decimal = '.' in num_str
    if not original_had_decimal and doubled_value == int(doubled_value):
        return str(int(doubled_value))
    else:
        # Simple float to string, pandas/python handles precision.
        # For specific precision: f"{doubled_value:.1f}" or similar if needed.
        return str(doubled_value)

def _double_action_string_numbers(action_string: str) -> str:
    """
    Processes an action string, doubling numerical values found in specific patterns.
    Expected patterns: PREFIX_NUMBER (e.g., IP_BET_10) or NUMBERbb (e.g., 2.5bb).
    """
    if not isinstance(action_string, str) or action_string.lower() == "nan" or not action_string:
        # Handles actual NaNs if they bypassed astype(str), "nan" strings, or empty strings.
        return action_string if isinstance(action_string, str) else ""


    parts = action_string.split('/')
    new_parts = []
    for part in parts:
        processed_part = False
        # Pattern 1: PREFIX_NUMBER (e.g., "BET_10", "RAISE_2.5")
        # Group 1: Prefix (e.g., "BET_")
        # Group 2: Number (e.g., "10", "2.5")
        match_prefix_num = re.fullmatch(r"([A-Za-z_]+)_(\d+\.?\d*|\.\d+)", part)
        if match_prefix_num:
            prefix = match_prefix_num.group(1) + "_" # Add back the underscore
            num_str = match_prefix_num.group(2)
            try:
                num = float(num_str)
                doubled_num_val = num * 2
                formatted_doubled_num = _format_doubled_number(num_str, doubled_num_val)
                new_parts.append(f"{prefix}{formatted_doubled_num}")
                processed_part = True
            except ValueError:
                pass # Should not happen with regex, but safety
        
        if processed_part:
            continue

        # Pattern 2: NUMBERbb (e.g., "2.5bb", "10BB")
        # Group 1: Number (e.g., "2.5", "10")
        # Group 2: Suffix ("bb" or "BB")
        match_num_bb = re.fullmatch(r"(\d+\.?\d*|\.\d+)([Bb][Bb])", part)
        if match_num_bb:
            num_str = match_num_bb.group(1)
            suffix = match_num_bb.group(2)
            try:
                num = float(num_str)
                doubled_num_val = num * 2
                formatted_doubled_num = _format_doubled_number(num_str, doubled_num_val)
                new_parts.append(f"{formatted_doubled_num}{suffix}")
                processed_part = True
            except ValueError:
                pass
        
        if processed_part:
            continue

        # Pattern 3: The part is just a number (e.g., "10", "2.5")
        # This might be less common in action strings but good to cover.
        if re.fullmatch(r"(\d+\.?\d*|\.\d+)", part):
            num_str = part
            try:
                num = float(num_str)
                doubled_num_val = num * 2
                formatted_doubled_num = _format_doubled_number(num_str, doubled_num_val)
                new_parts.append(formatted_doubled_num)
                processed_part = True
            except ValueError:
                pass # Not a valid number string
        
        if not processed_part:
            new_parts.append(part) # Part does not match any numeric pattern or failed processing
            
    return "/".join(new_parts)

def process_chip_sizes_in_file(csv_filepath: str) -> pd.DataFrame:
    """
    Reads a CSV, doubles numeric chip sizes in action columns, and removes specified columns.
    """
    try:
        df = pd.read_csv(csv_filepath)
    except FileNotFoundError:
        print(f"Error: File not found at '{csv_filepath}'")
        return pd.DataFrame()
    except Exception as e:
        print(f"Error reading CSV file '{csv_filepath}': {e}")
        return pd.DataFrame()

    action_columns = ['preflop_action', 'postflop_action']
    for col in action_columns:
        if col not in df.columns:
            print(f"Warning: Action column '{col}' not found in CSV. Skipping its processing.")
        else:
            # Apply the doubling function. astype(str) handles NaNs by converting them to "nan" string.
            # The _double_action_string_numbers function is designed to pass "nan" through.
            df[col] = df[col].astype(str).apply(_double_action_string_numbers)

    columns_to_drop = ['available_moves', 'pot_size', 'correct_decision']
    existing_columns_to_drop = [col for col in columns_to_drop if col in df.columns]
    
    if len(existing_columns_to_drop) < len(columns_to_drop):
        missing_to_drop = set(columns_to_drop) - set(df.columns)
        if missing_to_drop:
             print(f"Warning: Columns to drop not found: {missing_to_drop}. They will be ignored.")
        
    df_modified = df.drop(columns=existing_columns_to_drop, errors='ignore')

    return df_modified

# Example usage:
if __name__ == '__main__':
    # --- Easily modifiable absolute paths ---
    input_csv_path = '/home/xuandong/mnt/poker/internal-search-distillation-postflop-solver/datasets/action_reset_ranges_postflop_500k_train_set_game_scenario_information.csv'
    output_csv_path = '/home/xuandong/mnt/poker/internal-search-distillation-postflop-solver/datasets/chips_doubled_action_reset_ranges_postflop_500k_train_set_game_scenario_information.csv'
    # --- End of modifiable paths ---
    # to run: python -m data_processing.chipsize_doubler

    # Ensure the output directory exists
    output_dir = os.path.dirname(output_csv_path)
    if output_dir: # Check if output_dir is not empty (i.e., not just a filename)
        os.makedirs(output_dir, exist_ok=True)
    else: # If output_csv_path is just a filename, output will be in current dir, which is fine.
        pass 

    print(f"Processing file: {input_csv_path}")
    modified_df = process_chip_sizes_in_file(input_csv_path)

    if not modified_df.empty:
        print("\\nModified DataFrame head (first 5 rows, relevant columns):")
        
        # Determine relevant columns for display more robustly
        action_columns_display = ['preflop_action', 'postflop_action']
        cols_to_show_in_head = [col for col in action_columns_display if col in modified_df.columns]
        
        # Attempt to add a couple of other non-action columns if they exist
        other_cols = [col for col in modified_df.columns if col not in action_columns_display]
        cols_to_show_in_head.extend(other_cols[:2])

        if cols_to_show_in_head:
            print(modified_df[cols_to_show_in_head].head())
        else:
            # This case would be rare unless the CSV was empty of these columns to begin with
            print(modified_df.head()) # Fallback to printing head of whatever is there
            print("(Could not find typical action columns to display; showing generic head)")
        
        try:
            modified_df.to_csv(output_csv_path, index=False)
            print(f"\\nModified data saved to {output_csv_path}")
        except Exception as e:
            print(f"Error saving modified data to '{output_csv_path}': {e}")
    else:
        print(f"Processing failed or resulted in an empty DataFrame. Ensure '{input_csv_path}' exists and is valid.")
