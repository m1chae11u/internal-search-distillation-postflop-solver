import pandas as pd
import os

def _clean_postflop_action(action_string: str, street: str) -> str:
    """
    Cleans a single postflop_action string to reset it to the beginning of the specified street.

    Args:
        action_string: The original postflop action string.
        street: The street at which evaluation occurs ("Flop", "Turn", "River").

    Returns:
        The modified postflop action string.
    """
    # Ensure action_string is a string, handle None/NaN from pandas
    if pd.isna(action_string):
        action_string = ""
    else:
        action_string = str(action_string)

    if street == "Flop":
        return ""
    elif street in ["Turn", "River"]:
        # If the action string is empty or doesn't contain "dealcards",
        # it implies it might be only Flop actions or already empty.
        # The rule is to "remove everything up until ... dealcards".
        # If no "dealcards", nothing to remove based on this rule.
        if not action_string or "dealcards" not in action_string:
            return action_string

        parts = action_string.split('/')
        
        # Find the indices of all "dealcards" tokens
        dealcards_indices = [i for i, part_name in enumerate(parts) if part_name == "dealcards"]

        if not dealcards_indices:
            # This case should ideally be caught by the '"dealcards" not in action_string'
            # check earlier, but as a fallback.
            return action_string
            
        last_dealcards_token_idx = dealcards_indices[-1]

        # We keep the "dealcards" token and the card token that follows it.
        # The slice should go up to `last_dealcards_token_idx + 2` (exclusive end for list slicing).
        # This means we keep parts from index 0 up to `last_dealcards_token_idx + 1` (inclusive).
        # Check if a card token actually exists after the "dealcards" token.
        if last_dealcards_token_idx + 1 < len(parts):
            return "/".join(parts[:last_dealcards_token_idx + 2])
        else:
            # Malformed: "dealcards" is the last token without a subsequent card.
            # In this scenario, we treat it as if this "dealcards" marker isn't complete,
            # so we cannot reliably determine the start of the street based on it.
            # Returning the string up to this malformed "dealcards" might be an option,
            # or returning the original. The current logic will return up to "dealcards"
            # if it's the very last element e.g. "A/B/dealcards" -> join parts[:idx+2] might error if idx+1 is last.
            # Let's be safer: if "dealcards" is last part, return original string.
            # However, `parts[:last_dealcards_token_idx + 2]` would correctly take up to the end if
            # `last_dealcards_token_idx + 1` is the last valid index.
            # Example: parts = ["A", "dealcards"], idx = 1. parts[:1+2] = parts[:3] which is ["A", "dealcards"].
            # This is acceptable, "A/dealcards" is the result.
            return "/".join(parts[:last_dealcards_token_idx + 2]) # Keeps "dealcards" if it's last, or "dealcards/card"
    else:
        # Unknown street or other unhandled case, return original string
        return action_string


def process_postflop_actions_in_file(csv_filepath: str) -> pd.DataFrame:
    """
    Reads a CSV file, modifies the 'postflop_action' column in each row
    to reset the game history to the beginning of the current street.

    The input CSV file is expected to have 'evaluation_at' and 'postflop_action' columns.
    'evaluation_at' can be "Flop", "Turn", or "River".
    'postflop_action' is a string of game actions separated by '/'.

    Modification logic:
    - If 'evaluation_at' is "Flop", 'postflop_action' becomes an empty string.
    - If 'evaluation_at' is "Turn" or "River", 'postflop_action' is trimmed.
      It keeps actions up to and including the dealing of the card for the current street.
      Effectively, it removes actions taken *on* the current street.
      This is done by finding the last occurrence of "dealcards/{card}" and keeping
      the string up to that point (including the card).

    Args:
        csv_filepath: The path to the .csv file.

    Returns:
        A pandas DataFrame with the modified 'postflop_action' column.
        Returns an empty DataFrame if the file is not found, there's a reading error,
        or if essential columns are missing.
    """
    try:
        df = pd.read_csv(csv_filepath)
    except FileNotFoundError:
        print(f"Error: File not found at '{csv_filepath}'")
        return pd.DataFrame()
    except Exception as e:
        print(f"Error reading CSV file '{csv_filepath}': {e}")
        return pd.DataFrame()

    required_columns = {'evaluation_at', 'postflop_action'}
    if not required_columns.issubset(df.columns):
        missing_cols = required_columns - set(df.columns)
        print(f"Error: CSV file must contain {required_columns} columns. Missing: {missing_cols}")
        # Return original df or empty df to signal error; returning copy of original is safer if it exists
        return df if not df.empty else pd.DataFrame()

    # Create a new list for the modified actions
    modified_actions_list = []
    for _, row in df.iterrows():
        # Use .get() for safer access in case a row somehow misses a column despite earlier check
        action = row.get('postflop_action', None) 
        street = row.get('evaluation_at', None)
        
        if street is None: 
            # If 'evaluation_at' is missing for this specific row, preserve original action
            current_action_str = str(action) if pd.notna(action) else ""
            modified_actions_list.append(current_action_str)
            continue
            
        modified_actions_list.append(_clean_postflop_action(action, street))
    
    # Assign the new list to a copy of the DataFrame to avoid SettingWithCopyWarning
    df_copy = df.copy()
    df_copy['postflop_action'] = modified_actions_list
    
    return df_copy

if __name__ == '__main__':
    # Make sure 'postflop_gamestates.csv' is in the same directory or provide correct path
    # For testing, you might want to use a copy of your actual CSV.
    test_csv_path = '/home/xuandong/mnt/poker/internal-search-distillation-postflop-solver/datasets/ranges_postflop_500k_train_set_game_scenario_information.csv' 
    # You might need to create a dummy 'postflop_gamestates.csv' or use the one provided in context
    # For example:
    # df_test_data = pd.DataFrame({
    #     'evaluation_at': ['Flop', 'Turn', 'River', 'Flop', 'Turn', 'River'],
    #     'postflop_action': [
    #         'OOP_CHECK/IP_BET_1', 
    #         'OOP_CHECK/IP_CHECK/dealcards/9s/OOP_BET_5/IP_RAISE_16', 
    #         'OOP_CHECK/IP_BET_1/OOP_CALL/dealcards/4d/OOP_CHECK/IP_BET_8/OOP_CALL/dealcards/As/OOP_CHECK',
    #         None, # Test None/NaN
    #         'FLOP_ACTION_ONLY', # Test Turn with no dealcards
    #         'A/B/dealcards/CardOnly' # Test Turn/River with dealcards but no actions after
    #         ],
    #     'other_column': [1,2,3,4,5,6]
    # })
    # df_test_data.to_csv(test_csv_path, index=False)


    print(f"Processing file: {test_csv_path}")
    modified_df = process_postflop_actions_in_file(test_csv_path)

    if not modified_df.empty:
        print("\\nModified DataFrame head:")
        print(modified_df[['evaluation_at', 'postflop_action']].head())
        
        # To save the output:
        output_dir = '/home/xuandong/mnt/poker/internal-search-distillation-postflop-solver/datasets/'
        output_filename = 'action_reset_ranges_postflop_500k_train_set_game_scenario_information.csv'
        output_path = os.path.join(output_dir, output_filename)
        
        # Create the output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        modified_df.to_csv(output_path, index=False)
        print(f"\\nModified data saved to {output_path}")
    else:
        print("Processing failed or resulted in an empty DataFrame.")
