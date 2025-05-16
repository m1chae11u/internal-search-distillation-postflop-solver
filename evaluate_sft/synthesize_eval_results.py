'''
to do: implement this script.

inputs:
- json file of the following format:

example:
[
  {
    "input": "<gamestate> eval_at=River starting_pot=26.50bb effective_stack=87.00bb flop_board=Jh4h5s turn_board=2d river_board=Qc ip_range=AA,AKo,AKs,AQo,AQs,AJo,ATo,ATs,A9o,A9s,A8o,A8s,A7o,A7s,A6o,A6s,A5o,A5s,A4o,A3o,A3s,A2o,A2s,KK,KQo,KQs,KJo,KJs,KTo,KTs,K9o,K8s,K7s,K6s,K5s,K4s,QQ,QJo,QJs,Q9o,Q9s,Q8s,Q5o,JJ,JTo,JTs,J9o,J9s,J8s,J7s,T9o,T9s,T8o,T8s,T7s,T6o,T4o,99,98s,97s,96s,88,86s,77,76o,75s,72s,66,65s,64s,55,54s,53s,44,42s,33,22 oop_range=AA,AKo,AKs,KK,K2o,QQ flop_bet_sizes=[OOP: { Bet: '[PotRelative(0.33), PotRelative(0.66), PotRelative(1.25), Geometric(0, inf), AllIn]', Raise: '[PrevBetRelative(2.5)]' }, IP: { Bet: '[PotRelative(0.33), PotRelative(0.66), PotRelative(1.25), Geometric(0, inf), AllIn]', Raise: '[PrevBetRelative(2.5)]' }] turn_bet_sizes=[OOP: { Bet: '[PotRelative(0.33), PotRelative(0.66), PotRelative(1.25), Geometric(0, inf), AllIn]', Raise: '[PrevBetRelative(2.5)]' }, IP: { Bet: '[PotRelative(0.33), PotRelative(0.66), PotRelative(1.25), Geometric(0, inf), AllIn]', Raise: '[PrevBetRelative(2.5)]' }] river_bet_sizes=[OOP: { Bet: '[PotRelative(0.33), PotRelative(0.66), PotRelative(1.25), Geometric(0, inf), AllIn]', Raise: '[PrevBetRelative(2.5)]' }, IP: { Bet: '[PotRelative(0.33), PotRelative(0.66), PotRelative(1.25), Geometric(0, inf), AllIn]', Raise: '[PrevBetRelative(2.5)]' }] river_donk_sizes={ Donk: '[PotRelative(0.5)]' } add_allin_threshold=1.5 force_allin_threshold=0.15 </gamestate>",
    "output": "<rangesummary> OOP_equity: 61.28%, OOP_ev: 15.05bb, IP_equity: 38.72%, IP_ev: 11.45bb </rangesummary>\n\n<oop>\nAction Check: Avg Frequency: 0.777, EV: 15.07bb\nAction Bet (33%): Avg Frequency: 0.221, EV: 19.17bb\nAction Bet (66%): Avg Frequency: 0.002, EV: 25.94bb\nAction Bet (125%): Avg Frequency: 0.000, EV: 21.49bb\nAction AllIn (328%): Avg Frequency: 0.000, EV: 9.13bb\n</oop>\n\n<oop action=\"check\">\n  <rangesummary> OOP_equity: 56.33%, OOP_ev: 13.36bb, IP_equity: 43.67%, IP_ev: 13.14bb </rangesummary>\n  <ip>\nAction Check: Avg Frequency: 0.640, EV: 6.73bb\nAction Bet (33%): Avg Frequency: 0.099, EV: 21.31bb\nAction Bet (66%): Avg Frequency: 0.135, EV: 29.89bb\nAction Bet (125%): Avg Frequency: 0.099, EV: 23.95bb\nAction AllIn (328%): Avg Frequency: 0.027, EV: 40.87bb\n  </ip>\n</oop>\n\n<ip_highest_freq_action>check</ip_highest_freq_action>",
    "predicted_output": "<rangesummary> OOP_equity: 62.99%, OOP_ev: 15.80bb, IP_equity: 37.01%, IP_ev: 10.70bb </rangesummary>\n\n<oop>\nAction Check: Avg Frequency: 0.998, EV: 15.80bb\nAction Bet (33%): Avg Frequency: 0.002, EV: 24.48bb\nAction Bet (66%): Avg Frequency: 0.000, EV: 0.00bb\nAction Bet (125%): Avg Frequency: 0.000, EV: 0.00bb\nAction AllIn (328%): Avg Frequency: 0.000, EV: 0.00bb\n</oop>\n\n<oop action=\"check\">\n  <rangesummary> OOP_equity: 62.98%, OOP_ev: 15.80bb, IP_equity: 37.02%, IP_ev: 10.70bb </rangesummary>\n  <ip>\nAction Check: Avg Frequency: 0.507, EV: 1.88bb\nAction Bet (33%): Avg Frequency: 0.260, EV: 15.20bb\nAction Bet (66%): Avg Frequency: 0.008, EV: 20.96bb\nAction Bet (125%): Avg Frequency: 0.225, EV: 32.30bb\nAction AllIn (328%): Avg Frequency: 0.000, EV: 0.00bb\n  </ip>\n</oop>\n\n<ip_highest_freq_action>check</ip_highest_freq_action>check</ip_highest_freq_action>check</ip_highest_freq_action>check</ip_highest_freq_action>check</ip_highest_freq_action>check</ip_highest_freq_action>check</ip_highest_freq_action>check</ip_highest_freq_action>check</ip_highest_freq_action>check</ip_highest_freq_action>check</ip_highest_freq_action>check</ip_highest_freq_action>check</ip_highest_freq_action>check</ip_highest_freq_action>check</ip_highest_freq_action>check</ip_highest_freq_action>check</ip_highest_freq_action>check</ip_highest_freq_action>check</ip_highest_freq_action>check</ip_highest_freq_action>check</ip",
    "results": {
      "formatting": {
        "has_outer_rangesummary": true,
        "has_main_oop_block": true,
        "has_expanded_oop_block": true,
        "has_expanded_oop_rangesummary": true,
        "has_ip_block_in_expansion": true,
        "has_ip_highest_freq_action_tag": true,
        "all_template_tags_present": true
      },
      "accuracy": {
        "oop_most_frequent_action": {
          "ground_truth": "Check",
          "predicted": "Check",
          "match": true
        },
        "ip_highest_freq_action": {
          "ground_truth": "check",
          "predicted": "check",
          "match": true
        },
        "oop_action_attr_consistency_pred": true
      },
      "parsing_errors_ground_truth": [],
      "parsing_errors_predicted": []
    }
  },
  ...
]

output:
- json file aggregating the results of the eval
'''

import json
import argparse
from collections import defaultdict

def aggregate_results(input_file_path, output_file_path):
    """
    Aggregates evaluation results from an input JSON file and saves
    the summary to an output JSON file.

    Args:
        input_file_path (str): Path to the input JSON file.
        output_file_path (str): Path to save the aggregated JSON results.
    """
    try:
        with open(input_file_path, 'r') as f:
            eval_data = json.load(f)
    except FileNotFoundError:
        print(f"Error: Input file not found at {input_file_path}")
        return
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {input_file_path}")
        return

    if not isinstance(eval_data, list):
        print("Error: Input JSON must be a list of evaluation objects.")
        return

    total_samples = len(eval_data)
    if total_samples == 0:
        print("Input data is empty. No results to aggregate.")
        # Create an empty summary or a summary indicating no data
        summary = {
            "total_samples": 0,
            "formatting_summary": {},
            "accuracy_summary": {},
            "parsing_errors_summary": {}
        }
        with open(output_file_path, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"Empty summary saved to {output_file_path}")
        return

    # Initialize counters
    formatting_counts = defaultdict(int)
    accuracy_counts = defaultdict(int)
    parsing_error_counts = {
        "ground_truth_errors": 0,
        "predicted_errors": 0
    }

    # Define keys to look for
    # These can be dynamically discovered, but explicit definition is safer
    # if the structure is fixed.
    # Example from user shows these keys:
    formatting_keys = [
        "has_outer_rangesummary", "has_main_oop_block", "has_expanded_oop_block",
        "has_expanded_oop_rangesummary", "has_ip_block_in_expansion",
        "has_ip_highest_freq_action_tag", "all_template_tags_present"
    ]
    accuracy_match_keys = [
        "oop_most_frequent_action", "ip_highest_freq_action"
    ]
    accuracy_direct_bool_keys = ["oop_action_attr_consistency_pred"]


    for item in eval_data:
        results = item.get("results", {})

        # Aggregate formatting
        formatting_section = results.get("formatting", {})
        for key in formatting_keys:
            if formatting_section.get(key) is True:
                formatting_counts[key] += 1

        # Aggregate accuracy
        accuracy_section = results.get("accuracy", {})
        for key in accuracy_match_keys:
            match_obj = accuracy_section.get(key, {})
            if match_obj.get("match") is True:
                accuracy_counts[f"{key}_match"] += 1
        
        for key in accuracy_direct_bool_keys:
            if accuracy_section.get(key) is True:
                accuracy_counts[key] += 1
        
        # Aggregate parsing errors
        if results.get("parsing_errors_ground_truth"): # Check if list is not empty
            parsing_error_counts["ground_truth_errors"] += 1
        if results.get("parsing_errors_predicted"): # Check if list is not empty
            parsing_error_counts["predicted_errors"] += 1

    # Prepare summary
    summary = {
        "total_samples": total_samples,
        "formatting_summary": {},
        "accuracy_summary": {},
        "parsing_errors_summary": {}
    }

    for key, count in formatting_counts.items():
        summary["formatting_summary"][key] = {
            "count": count,
            "percentage": round((count / total_samples) * 100, 2) if total_samples > 0 else 0
        }
    
    # Ensure all formatting keys are present in the summary, even if count is 0
    for key in formatting_keys:
        if key not in summary["formatting_summary"]:
            summary["formatting_summary"][key] = {
                "count": 0,
                "percentage": 0
            }


    for key, count in accuracy_counts.items():
        summary["accuracy_summary"][key] = {
            "count": count,
            "percentage": round((count / total_samples) * 100, 2) if total_samples > 0 else 0
        }
    
    # Ensure all accuracy keys are present in the summary
    for key_base in accuracy_match_keys:
        key = f"{key_base}_match"
        if key not in summary["accuracy_summary"]:
             summary["accuracy_summary"][key] = {
                "count": 0,
                "percentage": 0
            }
    for key in accuracy_direct_bool_keys:
        if key not in summary["accuracy_summary"]:
            summary["accuracy_summary"][key] = {
                "count": 0,
                "percentage": 0
            }
            

    for key, count in parsing_error_counts.items():
        summary["parsing_errors_summary"][key] = {
            "count": count,
            "percentage": round((count / total_samples) * 100, 2) if total_samples > 0 else 0
        }
    
    # Ensure all parsing error keys are present
    if "ground_truth_errors" not in summary["parsing_errors_summary"]:
        summary["parsing_errors_summary"]["ground_truth_errors"] = {"count": 0, "percentage": 0}
    if "predicted_errors" not in summary["parsing_errors_summary"]:
        summary["parsing_errors_summary"]["predicted_errors"] = {"count": 0, "percentage": 0}


    # Save the aggregated results
    try:
        with open(output_file_path, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"Aggregated results successfully saved to {output_file_path}")
    except IOError:
        print(f"Error: Could not write to output file at {output_file_path}")

def main():
    parser = argparse.ArgumentParser(description="Synthesize and aggregate evaluation results from a JSON file.")
    parser.add_argument("input_file", help="Path to the input JSON file containing evaluation results.")
    parser.add_argument("output_file", help="Path to save the aggregated JSON results.")
    
    args = parser.parse_args()
    
    aggregate_results(args.input_file, args.output_file)

if __name__ == "__main__":
    main()


    '''
    python -m evaluate_sft.synthesize_eval_results \
        input_file \
        output_file
    
    Example Usage:
    python -m evaluate_sft.synthesize_eval_results \
        /home/xuandong/mnt/poker/internal-search-distillation-postflop-solver/evaluate_sft/50k_peft_3.1-8b/test_accuracy_evals_50k_peft_3.1-8b.json \
        /home/xuandong/mnt/poker/internal-search-distillation-postflop-solver/evaluate_sft/50k_peft_3.1-8b/aggregated_test_accuracy_evals_50k_peft_3.1-8b.json
    '''