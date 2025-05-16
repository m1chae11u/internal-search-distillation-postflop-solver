import json
import re
from typing import List, Dict, Any, Optional, Tuple

# Regex patterns
ACTION_LINE_PATTERN = re.compile(
    r"Action\s+(?P<name>[^:]+):\s+Avg\s+Frequency:\s+(?P<freq>[0-9]+\.?[0-9]*),\s+EV:\s+(?P<ev>[0-9.-]+)bb"
)
TAG_PATTERN = r"<{tag_name}(?:\s+action=\"(?P<action_attr>[^\"]*)\")?>(?P<content>.*?)</{tag_name}>"

def extract_tag_content(text: str, tag_name: str) -> Optional[Tuple[str, Optional[str]]]:
    """
    Extracts content and optional 'action' attribute from a custom tag.
    Returns (content, action_attribute) or None if tag not found.
    Handles nested tags by being non-greedy.
    """
    pattern = TAG_PATTERN.format(tag_name=re.escape(tag_name))
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group("content").strip(), match.group("action_attr")
    return None

def parse_actions(action_block_content: str) -> List[Dict[str, Any]]:
    """Parses action lines from a block of text."""
    actions = []
    if not action_block_content:
        return actions
    for line in action_block_content.strip().split('\n'):
        match = ACTION_LINE_PATTERN.match(line.strip())
        if match:
            actions.append({
                "name": match.group("name").strip(),
                "frequency": float(match.group("freq")),
                "ev": float(match.group("ev"))
            })
    return actions

def get_most_frequent_action(actions: List[Dict[str, Any]]) -> Optional[str]:
    """Determines the most frequent action from a list of parsed actions."""
    if not actions:
        return None
    most_frequent = max(actions, key=lambda x: x["frequency"])
    return most_frequent["name"]

# Helper functions for new action comparison logic
def _extract_percentage_from_action_list_style(name_lower: str) -> Optional[str]:
    """Extracts number from 'Action Name (X%)' style, given lowercase name."""
    # E.g., "bet (125%)" -> "125"
    match = re.search(r'(\d+)%', name_lower)
    return match.group(1) if match else None

def _extract_percentage_from_attr_style(name_lower: str) -> Optional[str]:
    """
    Extracts number from 'keyword_X%pot' or 'keyword_X%' or '(X%)' style, given lowercase name.
    E.g. "bet_125%pot" -> "125", "raise_200%" -> "200", "bet (125%)" -> "125"
    """
    # Try to find keyword_NUMBER%... (e.g., bet_125%pot, raise_200%)
    match_keyword_style = re.search(r'(?:bet|raise|allin)_(\d+)%?', name_lower)
    if match_keyword_style:
        return match_keyword_style.group(1)
    
    # Try to find (X%) style (e.g., (125%))
    match_paren_style = re.search(r'(\d+)%', name_lower)
    return match_paren_style.group(1) if match_paren_style else None

def are_actions_equivalent(action_from_list: Optional[str], action_from_attr: Optional[str]) -> bool:
    """
    Compares action names based on user-defined rules.
    - "check", "allin", "fold": keyword presence.
    - "bet", "raise": keyword presence AND matching extracted percentages.
    """
    if action_from_list is None or action_from_attr is None:
        return False

    s_list = action_from_list.lower()
    s_attr = action_from_attr.lower()

    # Rule for "check"
    if "check" in s_list and "check" in s_attr:
        return True
    
    # Rule for "allin"
    if "allin" in s_list and "allin" in s_attr:
        return True

    # Rule for "fold"
    if "fold" in s_list and "fold" in s_attr:
        return True

    # Rule for "bet"
    if "bet" in s_list and "bet" in s_attr:
        num_list = _extract_percentage_from_action_list_style(s_list)
        num_attr = _extract_percentage_from_attr_style(s_attr)
        # If both have numbers and they match, it's true.
        if num_list and num_attr and num_list == num_attr:
            return True
        # If "bet" keyword is present, but numbers don't match or one/both are missing,
        # this rule evaluates to false as per user: "otherwise, false".
        return False

    # Rule for "raise"
    if "raise" in s_list and "raise" in s_attr:
        num_list = _extract_percentage_from_action_list_style(s_list)
        num_attr = _extract_percentage_from_attr_style(s_attr)
        if num_list and num_attr and num_list == num_attr:
            return True
        return False
    
    # If none of the keyword-specific rules led to a True/False decision for that keyword
    return False

def parse_model_output(output_str: str) -> Dict[str, Any]:
    """
    Parses the model output string to extract key information and check for tag presence.
    """
    parsed_data = {
        "tags_present": {},
        "overall_range_summary": None,
        "oop_actions": [],
        "oop_action_attr": None,
        "expanded_oop_range_summary": None,
        "ip_actions_in_oop_expansion": [],
        "ip_highest_freq_action_tag": None,
        "errors": []
    }

    # 1. <rangesummary>
    range_summary_match = extract_tag_content(output_str, "rangesummary")
    parsed_data["tags_present"]["outer_rangesummary"] = range_summary_match is not None
    if range_summary_match:
        parsed_data["overall_range_summary"] = range_summary_match[0]
    else:
        parsed_data["errors"].append("Outer <rangesummary> not found.")

    # 2. <oop> (main)
    oop_block_match = extract_tag_content(output_str, "oop")
    parsed_data["tags_present"]["main_oop_block"] = oop_block_match is not None
    if oop_block_match:
        oop_content, _ = oop_block_match
        parsed_data["oop_actions"] = parse_actions(oop_content)
        if not parsed_data["oop_actions"] and oop_content: # content was there but not parsable actions
             parsed_data["errors"].append(f"Main <oop> block content could not be parsed into actions: '{oop_content[:100]}...'")
    else:
        parsed_data["errors"].append("Main <oop> block not found.")

    # 3. <oop action="...">
    # To correctly find the <oop action="..."> tag, we need to be careful not to match the first <oop>
    # A more specific regex or sequential processing might be needed if structure is very complex.
    # For this template, we assume the expanded OOP block is distinct.
    
    # Let's try to find all <oop> tags first.
    all_oop_tags = list(re.finditer(TAG_PATTERN.format(tag_name="oop"), output_str, re.DOTALL))
    
    expanded_oop_match_info = None
    if len(all_oop_tags) > 1: # Assuming the second one is the expanded one
        match_obj = all_oop_tags[1]
        expanded_oop_match_info = (match_obj.group("content").strip(), match_obj.group("action_attr"))

    parsed_data["tags_present"]["expanded_oop_block"] = expanded_oop_match_info is not None
    if expanded_oop_match_info:
        expanded_oop_content, action_attr = expanded_oop_match_info
        parsed_data["oop_action_attr"] = action_attr

        # 3a. <rangesummary> inside <oop action="...">
        inner_range_summary_match = extract_tag_content(expanded_oop_content, "rangesummary")
        parsed_data["tags_present"]["expanded_oop_rangesummary"] = inner_range_summary_match is not None
        if inner_range_summary_match:
            parsed_data["expanded_oop_range_summary"] = inner_range_summary_match[0]
        else:
            parsed_data["errors"].append("<rangesummary> inside <oop action=...> not found.")

        # 3b. <ip> inside <oop action="...">
        ip_block_match = extract_tag_content(expanded_oop_content, "ip")
        parsed_data["tags_present"]["ip_block_in_expansion"] = ip_block_match is not None
        if ip_block_match:
            ip_content, _ = ip_block_match
            parsed_data["ip_actions_in_oop_expansion"] = parse_actions(ip_content)
            if not parsed_data["ip_actions_in_oop_expansion"] and ip_content:
                 parsed_data["errors"].append(f"<ip> block content in OOP expansion could not be parsed: '{ip_content[:100]}...'")

        else:
            parsed_data["errors"].append("<ip> block inside <oop action=...> not found.")
    elif len(all_oop_tags) <=1 and parsed_data["tags_present"]["main_oop_block"]: # Only one OOP tag found, means expanded is missing
         parsed_data["errors"].append("<oop action=...> block not found (only main <oop> was present).")
    elif not parsed_data["tags_present"]["main_oop_block"]: # No oop tags found at all
        parsed_data["errors"].append("<oop action=...> block not found (no <oop> tags were present).")


    # 4. <ip_highest_freq_action>
    ip_highest_action_match = extract_tag_content(output_str, "ip_highest_freq_action")
    parsed_data["tags_present"]["ip_highest_freq_action_tag"] = ip_highest_action_match is not None
    if ip_highest_action_match:
        parsed_data["ip_highest_freq_action_tag"] = ip_highest_action_match[0]
    else:
        parsed_data["errors"].append("<ip_highest_freq_action> tag not found.")
        
    return parsed_data


def evaluate_single_item(ground_truth_str: str, predicted_str: str) -> Dict[str, Any]:
    """
    Evaluates a single predicted output against its ground truth.
    """
    results = {
        "formatting": {},
        "accuracy": {},
        "parsing_errors_ground_truth": [],
        "parsing_errors_predicted": []
    }

    # Parse both ground truth and predicted output
    gt_parsed = parse_model_output(ground_truth_str)
    pred_parsed = parse_model_output(predicted_str)

    results["parsing_errors_ground_truth"] = gt_parsed["errors"]
    results["parsing_errors_predicted"] = pred_parsed["errors"]

    # 1. Formatting Evaluation
    # Check if all expected tags are present in the predicted output
    # Template tags:
    # <rangesummary> (outer)
    # <oop> (main)
    # <oop action="...">
    #   <rangesummary> (inner)
    #   <ip>
    # <ip_highest_freq_action>
    
    expected_tags_keys = [
        "outer_rangesummary", "main_oop_block", "expanded_oop_block",
        "expanded_oop_rangesummary", "ip_block_in_expansion", "ip_highest_freq_action_tag"
    ]
    
    formatting_summary = {}
    all_tags_present = True
    for tag_key in expected_tags_keys:
        present = pred_parsed["tags_present"].get(tag_key, False)
        formatting_summary[f"has_{tag_key}"] = present
        if not present:
            all_tags_present = False
    formatting_summary["all_template_tags_present"] = all_tags_present
    results["formatting"] = formatting_summary

    # 2. Accuracy Evaluation (assuming perfect formatting for simplicity, but use parsed data)

    # 2a. OOP most frequent action (from main <oop> block)
    gt_oop_actions = gt_parsed["oop_actions"]
    pred_oop_actions = pred_parsed["oop_actions"]

    gt_oop_most_freq_action = get_most_frequent_action(gt_oop_actions)
    pred_oop_most_freq_action = get_most_frequent_action(pred_oop_actions)

    results["accuracy"]["oop_most_frequent_action"] = {
        "ground_truth": gt_oop_most_freq_action,
        "predicted": pred_oop_most_freq_action,
        "match": gt_oop_most_freq_action == pred_oop_most_freq_action if gt_oop_most_freq_action is not None and pred_oop_most_freq_action is not None else False
    }
    if gt_oop_most_freq_action is None:
        results["accuracy"]["oop_most_frequent_action"]["error_gt"] = "Could not determine GT OOP most frequent action."
    if pred_oop_most_freq_action is None and pred_parsed["tags_present"].get("main_oop_block"): # only an error if block was there but unparsable
        results["accuracy"]["oop_most_frequent_action"]["error_pred"] = "Could not determine predicted OOP most frequent action from present block."


    # 2b. IP highest frequency action (from <ip_highest_freq_action> tag)
    gt_ip_action = gt_parsed["ip_highest_freq_action_tag"]
    pred_ip_action = pred_parsed["ip_highest_freq_action_tag"]

    results["accuracy"]["ip_highest_freq_action"] = {
        "ground_truth": gt_ip_action,
        "predicted": pred_ip_action,
        "match": gt_ip_action == pred_ip_action if gt_ip_action is not None and pred_ip_action is not None else False
    }
    if gt_ip_action is None:
         results["accuracy"]["ip_highest_freq_action"]["error_gt"] = "GT <ip_highest_freq_action> tag missing or empty."
    if pred_ip_action is None and pred_parsed["tags_present"].get("ip_highest_freq_action_tag"):
         results["accuracy"]["ip_highest_freq_action"]["error_pred"] = "Predicted <ip_highest_freq_action> tag missing or empty."


    # Bonus: Check if the action in <oop action="..."> matches the highest frequency OOP action
    # This is a check on the ground_truth template consistency, and predicted consistency
    
    # Initialize consistency check results
    results["accuracy"]["oop_action_attr_consistency_pred"] = None

    if pred_oop_most_freq_action and pred_parsed.get("oop_action_attr"):
         results["accuracy"]["oop_action_attr_consistency_pred"] = are_actions_equivalent(pred_oop_most_freq_action, pred_parsed["oop_action_attr"])
    elif pred_parsed.get("tags_present",{}).get("expanded_oop_block") and not pred_oop_most_freq_action : # if expanded oop block is there but main oop actions unparsable
        # This case is tricky, already covered by the line above if oop_action_attr is also None.
        # If pred_oop_most_freq_action is None, are_actions_equivalent will return False if oop_action_attr is not None.
        # If both are None, it also returns False. This seems okay.
        # Let's refine: if the expanded_oop_block is present, we expect to make a determination.
        # If we can't (e.g. main OOP actions are unparsable, or oop_action_attr is missing), it's effectively an inconsistency.
        pass # The logic above should handle it: if expanded_oop_block is present, but we can't get True from are_actions_equivalent, it becomes False.


    return results

def run_evaluation_on_file(input_filepath: str, output_filepath: str):
    """
    Reads a JSON file, performs evaluations, and writes results to a new JSON file.
    """
    try:
        with open(input_filepath, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: Input file not found at {input_filepath}")
        return
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {input_filepath}")
        return

    if not isinstance(data, list):
        print("Error: Input JSON must be a list of objects.")
        return

    results_data = []
    for i, item in enumerate(data):
        if not isinstance(item, dict):
            print(f"Warning: Item at index {i} is not a dictionary, skipping.")
            results_data.append(item) # Append as is or skip
            continue

        input_val = item.get("input")
        output_val = item.get("output")
        predicted_output_val = item.get("predicted_output")

        if output_val is None or predicted_output_val is None:
            print(f"Warning: Item at index {i} is missing 'output' or 'predicted_output', skipping evaluation for this item.")
            item["results"] = {"error": "Missing ground truth or predicted output."}
            results_data.append(item)
            continue
        
        print(f"Processing item {i+1}/{len(data)}...")
        evaluation_results = evaluate_single_item(output_val, predicted_output_val)
        item["results"] = evaluation_results
        results_data.append(item)

    try:
        with open(output_filepath, 'w') as f:
            json.dump(results_data, f, indent=2)
        print(f"Evaluation complete. Results saved to {output_filepath}")
    except IOError:
        print(f"Error: Could not write results to {output_filepath}")

if __name__ == '__main__':
    # ===== User Configuration =====
    # Please specify your input and output file paths here
    input_json_filepath = "/home/xuandong/mnt/poker/internal-search-distillation-postflop-solver/evaluate_sft/50k_peft_3.1-8b/test_predictions.json"
    output_json_filepath = "/home/xuandong/mnt/poker/internal-search-distillation-postflop-solver/evaluate_sft/50k_peft_3.1-8b/test_accuracy_evals_50k_peft_3.1-8b.json"
    # ============================
    print(f"Using input file: {input_json_filepath}")
    print(f"Using output file: {output_json_filepath}")
    run_evaluation_on_file(input_json_filepath, output_json_filepath)

    '''
    To run:
    python -m evaluate_sft.eval_pipeline
    '''
