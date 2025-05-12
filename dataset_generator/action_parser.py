from typing import Optional, Dict, List, Tuple
import re # Added for preflop parsing

# Initial player stack sizes
INITIAL_STACK_SIZE = 200.0

def parse_action_amount(action_part: str) -> float:
    """
    Parses an action like "2.0bb", "BET_10", "RAISE_TO_30", "RAISE_16" (assuming 16 is total amount).
    Returns the numeric amount. For simple "CALL", "CHECK", "FOLD", might return 0 
    or rely on context outside this specific helper.
    This will need to be robust.
    """
    action_part = action_part.strip() # Ensure robustness against leading/trailing whitespace
    action_upper = action_part.upper()
    # Handle direct bb amounts first if they exist and are simple
    # Updated to be more robust for formats like "2.0BB", "2BB"
    bb_match = re.search(r"(\d*\.?\d+)BB", action_upper)
    if bb_match:
        try:
            return float(bb_match.group(1))
        except ValueError:
            pass

    # Handle forms like BET_X, RAISE_X, RAISE_TO_X
    parts = action_upper.split('_') 
    if len(parts) > 1:
        try:
            # Try last part for BET_10, RAISE_16
            val = float(parts[-1])
            return val
        except ValueError:
            # Check if it was RAISE_TO_X, then parts[-1] is X
            if parts[0] == "RAISE" and parts[-2] == "TO":
                try:
                    return float(parts[-1])
                except ValueError:
                    pass 
            # Try to find number in the first part if no underscore but action verb, e.g. BET10
            # This is simplistic; better to ensure action_verb_and_amount from main parser is just the number string if possible
            elif parts[0] in ["BET", "RAISE"] and len(parts) == 1: # e.g. BET10 was split by _, so parts[0] = BET10
                numeric_part = "".join(filter(lambda c: c.isdigit() or c == '.', parts[0]))
                if numeric_part:
                    try: 
                        return float(numeric_part)
                    except ValueError:
                        pass
    
    # Fallback: if action_part itself is just a number string (e.g. after being processed by a more complex split)
    try:
        return float(action_part) 
    except ValueError:
        pass

    return 0.0 # Default for non-betting actions or unparsed

def _parse_preflop_commitment(preflop_action_str: Optional[str]) -> float:
    """
    Parses the preflop_action string to find the amount each of the two players
    committed to the pot if the betting ended in a call or a showdown.
    Returns the amount committed by each player (assuming they committed the same).
    Uses the heuristic: find the last bet/raise that was explicitly called.
    """
    if not preflop_action_str:
        return 0.0

    actions = preflop_action_str.strip().split('/')
    if not actions:
        return 0.0

    # If the last action is a call, the amount called is likely the total commitment.
    # The bet that was called is the action before the "call".
    if actions[-1].upper() == "CALL" and len(actions) > 1:
        # The action before the call is the bet/raise amount.
        potential_bet_str = actions[-2]
        # Check if this string contains "bb" which indicates a bet size.
        bb_match = re.search(r"(\d*\.?\d+)BB", potential_bet_str.upper())
        if bb_match:
            try:
                return float(bb_match.group(1))
            except ValueError:
                pass
        else: # If no explicit bb, try parsing it as a raw number if it was a simple bet (e.g. from a solver output not in Xbb format)
             # This part is less robust for typical human-readable strings.
             # For now, prioritize explicit Xbb formats for preflop commitments.
             pass # Fall back to general scan if simple call logic fails

    # General scan for the largest bet amount mentioned if the above fails
    # This is a less precise fallback.
    max_committed_bet = 0.0
    for action in actions:
        # Focus on parts that look like bets/raises (contain "bb")
        if "BB" in action.upper() and "CALL" not in action.upper() and "FOLD" not in action.upper():
            amount = parse_action_amount(action) # Use the existing parser
            if amount > max_committed_bet:
                max_committed_bet = amount
    
    # If the last action was a call, and we found a max_committed_bet, it implies this was the amount called.
    if actions[-1].upper() == "CALL" and max_committed_bet > 0:
        return max_committed_bet
    
    # If only one player bets and no call (e.g. blinds taken), this heuristic fails.
    # For 2 players to reach postflop, there must have been a call or all-in. 
    # If the string implies a walk (e.g., SB/call, BB/check), this is more complex.
    # For now, if the simple "last action is call" heuristic failed, and we only have max_committed_bet
    # and no explicit call, it's hard to say if it was matched. Let's assume for 2 players to postflop, it was.
    if max_committed_bet > 0: # Fallback if no clear call of a specific bet
        return max_committed_bet

    # Very simple cases like "SB/raises/3bb/BB/folds" - here preflop commitment is just blinds.
    # This heuristic is primarily for when betting completes and goes to postflop for two players.
    # Smallest preflop commitment if players see a flop is typically the BB amount (e.g. 1bb each if SB completes).
    # If the preflop string is just blinds, e.g. "SB posts 0.5, BB posts 1.0" and then flop. min 1bb from each.
    # This needs to be more sophisticated to handle walks and small blind completions.
    # For now, if no clear bet was called, assume at least BB was matched if we go to flop. This is a guess.
    # If preflop_action_str is "SB/call" (SB calls 0.5bb, BB checks), then 1bb is committed by each.
    if "CALL" in actions[-1].upper() and len(actions) ==1 and actions[0].upper() == "SB": # Special case for SB completing vs BB check
        return 1.0 # Assuming BB is 1.0
    if actions == ['SB', 'BB'] or actions == ['BTN','SB','BB'] and 'CALL' not in preflop_action_str.upper() and 'RAISE' not in preflop_action_str.upper(): # simplified check for just blinds posted and checked around (e.g. BB check) 
        return 1.0 # Everyone put in at least BB

    return 0.0 # Default if no clear commitment found


def calculate_stacks_after_actions(
    hero_is_oop: bool, 
    preflop_action_str: Optional[str],
    postflop_action_history_str: Optional[str], 
    initial_hero_stack: float = INITIAL_STACK_SIZE,
    initial_villain_stack: float = INITIAL_STACK_SIZE
) -> Dict[str, float]:
    """
    Calculates hero and villain stacks after parsing preflop and postflop actions.
    Returns a dictionary: {'hero_stack': X, 'villain_stack': Y, 'pot_from_actions': P}
    'pot_from_actions' is the amount added to the pot *by the actions parsed in this call*.
    Assumes initial_hero_stack and initial_villain_stack are stacks *before* any actions.
    """
    hero_stack = initial_hero_stack
    villain_stack = initial_villain_stack
    pot_from_actions = 0.0

    # --- Preflop Parsing --- 
    preflop_commitment_each = _parse_preflop_commitment(preflop_action_str)
    if preflop_commitment_each > 0:
        # Deduct from both stacks, add to pot.
        # Ensure they don't go below zero or commit more than they have.
        actual_hero_preflop_commit = min(preflop_commitment_each, hero_stack)
        actual_villain_preflop_commit = min(preflop_commitment_each, villain_stack)
        
        hero_stack -= actual_hero_preflop_commit
        villain_stack -= actual_villain_preflop_commit
        pot_from_actions += (actual_hero_preflop_commit + actual_villain_preflop_commit)
    
    # --- Postflop Parsing --- 
    hero_role_for_parser = "OOP" if hero_is_oop else "IP"
    # villain_role_for_parser = "IP" if hero_is_oop else "OOP" # Not directly used below, logic uses is_hero_acting

    if postflop_action_history_str:
        actions = postflop_action_history_str.split('/')
        current_bet_to_match_on_street = 0.0
        hero_invested_this_street = 0.0
        villain_invested_this_street = 0.0

        for action_part in actions:
            action_part = action_part.strip() # Clean whitespace from each part
            if not action_part: # Skip if part becomes empty after strip (e.g. from "//")
                continue

            if "dealcards" in action_part.lower(): 
                current_bet_to_match_on_street = 0.0
                hero_invested_this_street = 0.0
                villain_invested_this_street = 0.0
                continue

            actor_role_from_str = ""
            action_verb_and_amount = ""
            
            if action_part.startswith("OOP_"):
                actor_role_from_str = "OOP"
                action_verb_and_amount = action_part[len("OOP_"):].strip() # Strip here too
            elif action_part.startswith("IP_"):
                actor_role_from_str = "IP"
                action_verb_and_amount = action_part[len("IP_"):].strip() # Strip here too
            else:
                continue 

            is_hero_acting = (actor_role_from_str == hero_role_for_parser)
            
            action_amount_parsed = parse_action_amount(action_verb_and_amount)

            if "CHECK" in action_verb_and_amount.upper():
                pass
            elif "FOLD" in action_verb_and_amount.upper():
                pass 
            elif "CALL" in action_verb_and_amount.upper():
                actor_current_stack = hero_stack if is_hero_acting else villain_stack
                actor_invested_this_street_val = hero_invested_this_street if is_hero_acting else villain_invested_this_street
                
                amount_to_call = current_bet_to_match_on_street - actor_invested_this_street_val
                amount_to_call = max(0, amount_to_call) 
                actual_call_amount = min(amount_to_call, actor_current_stack)
                
                if is_hero_acting:
                    hero_stack -= actual_call_amount
                    hero_invested_this_street += actual_call_amount
                else:
                    villain_stack -= actual_call_amount
                    villain_invested_this_street += actual_call_amount
                pot_from_actions += actual_call_amount

            elif "BET" in action_verb_and_amount.upper():
                actor_current_stack = hero_stack if is_hero_acting else villain_stack
                bet_amount = min(action_amount_parsed, actor_current_stack)
                if is_hero_acting:
                    hero_stack -= bet_amount
                    hero_invested_this_street += bet_amount
                else:
                    villain_stack -= bet_amount
                    villain_invested_this_street += bet_amount
                pot_from_actions += bet_amount
                current_bet_to_match_on_street = hero_invested_this_street if is_hero_acting else villain_invested_this_street
            
            elif "RAISE" in action_verb_and_amount.upper():
                actor_current_stack = hero_stack if is_hero_acting else villain_stack
                actor_invested_this_street_val = hero_invested_this_street if is_hero_acting else villain_invested_this_street

                new_total_bet_for_street = action_amount_parsed
                amount_needed_for_raise = new_total_bet_for_street - actor_invested_this_street_val
                amount_needed_for_raise = max(0, amount_needed_for_raise)
                
                actual_raise_contribution = min(amount_needed_for_raise, actor_current_stack)

                if is_hero_acting:
                    hero_stack -= actual_raise_contribution
                    hero_invested_this_street += actual_raise_contribution
                else:
                    villain_stack -= actual_raise_contribution
                    villain_invested_this_street += actual_raise_contribution
                pot_from_actions += actual_raise_contribution
                current_bet_to_match_on_street = hero_invested_this_street if is_hero_acting else villain_invested_this_street
            
            hero_stack = max(0, hero_stack)
            villain_stack = max(0, villain_stack)

    return {
        "hero_stack": round(hero_stack, 2),
        "villain_stack": round(villain_stack, 2),
        "pot_from_actions": round(pot_from_actions, 2)
    }


if __name__ == '__main__':
    print("Action Parser - Stack Calculation Tests")

    # Preflop Test Cases using your examples:
    preflop_tests = [
        ("UTG/2.0bb/CO/6.5bb/UTG/call", 6.5),
        ("HJ/2.0bb/BB/call", 2.0),
        ("BTN/2.5bb/BB/13.0bb/BTN/call", 13.0),
        ("CO/2.3bb/BTN/7.5bb/CO/call", 7.5),
        ("SB/3.0bb/BB/call", 3.0),
        ("SB/call", 1.0), # SB completes to 1bb, BB checks
        ("BTN/2.5bb/SB/fold/BB/fold", 0.0) # No call, only blinds posted - this case is complex for current heuristic, should be 0 for *committed beyond blinds*
    ]
    for i, (pf_str, expected_commit) in enumerate(preflop_tests):
        commit = _parse_preflop_commitment(pf_str)
        print(f"Preflop Test {i+1}: '{pf_str}' -> Expected Commit: {expected_commit}, Got: {commit}")
        assert abs(commit - expected_commit) < 0.01, f"Failed Preflop Test {i+1}"

    print("\nPostflop Tests (assuming preflop commitments handled):")
    # Test Case 1: Flop action, Hero is IP
    stacks_t1 = calculate_stacks_after_actions(
        hero_is_oop=False, 
        preflop_action_str=None, # Preflop handled by initial stacks for this test setup
        postflop_action_history_str="OOP_CHECK/IP_BET_10/OOP_CALL",
        initial_hero_stack=100.0, # Stacks after preflop would be lower
        initial_villain_stack=100.0
    )
    print(f"Postflop Test 1 (Flop: OOP_CHECK/IP_BET_10/OOP_CALL, Hero IP, Initial Stacks 100): {stacks_t1}")
    # Expected: H:90, V:90, pot_from_actions:20
    assert abs(stacks_t1['hero_stack'] - 90.0) < 0.01 and abs(stacks_t1['villain_stack'] - 90.0) < 0.01 and abs(stacks_t1['pot_from_actions'] - 20.0) < 0.01

    # Test Case 2: Flop actions, Hero is OOP
    stacks_t2 = calculate_stacks_after_actions(
        hero_is_oop=True, 
        preflop_action_str=None,
        postflop_action_history_str="OOP_BET_5/IP_RAISE_TO_16", 
        initial_hero_stack=100.0,
        initial_villain_stack=100.0
    )
    print(f"Postflop Test 2 (Flop Hist: OOP_BET_5/IP_RAISE_TO_16, Hero OOP, Initial Stacks 100): {stacks_t2}")
    # Expected: H:95, V:84, pot_from_actions:21
    assert abs(stacks_t2['hero_stack'] - 95.0) < 0.01 and abs(stacks_t2['villain_stack'] - 84.0) < 0.01 and abs(stacks_t2['pot_from_actions'] - 21.0) < 0.01

    # Test Case 3: Full hand with preflop
    # "HJ/2.0bb/BB/call", then Flop+Turn Hist: "OOP_CHECK/IP_BET_1/OOP_CALL/dealcards/4d/OOP_CHECK/IP_BET_8/OOP_CALL"
    # Hero is IP.
    # Preflop: HJ (Hero or Villain) bets 2bb, BB (other player) calls. Each committed 2bb.
    # Stacks become 98 each. Pot from preflop = 4bb.
    # Postflop actions on top of this:
    # Flop: OOP_C. IP_B1 (Hero bets 1, H_pf=98 -> H_f=97). OOP_CALL (Villain calls 1, V_pf=98 -> V_f=97). Pot from flop_actions = 2.
    # Turn: dealcards/4d. OOP_C.
    # IP_B8 (Hero bets 8, H_f=97 -> H_t=89). OOP_CALL (Villain calls 8, V_f=97 -> V_t=89). Pot from turn_actions = 16.
    # Total pot_from_actions (preflop+postflop history) = 4 + 2 + 16 = 22.
    # Final stacks: Hero 89, Villain 89.
    stacks_t3_full = calculate_stacks_after_actions(
        hero_is_oop=False, # Hero is IP
        preflop_action_str="HJ/2.0bb/BB/call", 
        postflop_action_history_str="OOP_CHECK/IP_BET_1/OOP_CALL/dealcards/4d/OOP_CHECK/IP_BET_8/OOP_CALL",
        initial_hero_stack=100.0,
        initial_villain_stack=100.0
    )
    print(f"Full Hand Test 3 (PF: 'HJ/2.0bb/BB/call', Postflop Hist, Hero IP, Stacks 100): {stacks_t3_full}")
    assert abs(stacks_t3_full['hero_stack'] - 89.0) < 0.01, f"Hero stack mismatch: {stacks_t3_full['hero_stack']}"
    assert abs(stacks_t3_full['villain_stack'] - 89.0) < 0.01, f"Villain stack mismatch: {stacks_t3_full['villain_stack']}"
    assert abs(stacks_t3_full['pot_from_actions'] - 22.0) < 0.01, f"Pot from actions mismatch: {stacks_t3_full['pot_from_actions']}"

    print("\nAll assertions passed (if no errors above).") 