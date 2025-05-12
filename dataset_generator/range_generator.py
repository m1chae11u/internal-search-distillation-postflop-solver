import random

# --- Constants ---
RANKS = ['A', 'K', 'Q', 'J', 'T', '9', '8', '7', '6', '5', '4', '3', '2']
SUITS = ['s', 'h', 'd', 'c'] # For deck creation if ever needed, not directly for 169 combos
HAND_TYPES = ['s', 'o'] # Suited, Offsuit

ALL_169_HAND_COMBINATIONS = []

def get_rank_index(rank_char):
    """Returns the index of a rank character (A=0, K=1, ..., 2=12)."""
    if rank_char not in RANKS:
        raise ValueError(f"Invalid rank character: {rank_char}")
    return RANKS.index(rank_char)

def _initialize_169_hands():
    """
    Generates and stores all 169 unique starting hand combinations (e.g., 'AA', 'AKs', 'AQo').
    Populates the global ALL_169_HAND_COMBINATIONS list.
    """
    if ALL_169_HAND_COMBINATIONS: # Prevent re-initialization
        return

    # 1. Pocket pairs (13 hands)
    for r_idx, r_val in enumerate(RANKS):
        ALL_169_HAND_COMBINATIONS.append(f"{r_val}{r_val}")

    # 2. Suited and Offsuit hands (78 suited + 78 offsuit = 156 total non-pair combos)
    #    Each unique rank pairing (e.g., AK) has 1 suited and 1 offsuit version. 12C2 = 78 such pairings.
    for i in range(len(RANKS)):
        for j in range(i + 1, len(RANKS)):
            rank1 = RANKS[i]
            rank2 = RANKS[j]
            # Ensure canonical order (e.g., AKs, not KAs)
            ALL_169_HAND_COMBINATIONS.append(f"{rank1}{rank2}s")  # Suited
            ALL_169_HAND_COMBINATIONS.append(f"{rank1}{rank2}o")  # Offsuit
    # print(f"Initialized {len(ALL_169_HAND_COMBINATIONS)} hand combinations.")

_initialize_169_hands()

# Ensure ALL_169_HAND_COMBINATIONS is sorted in a canonical way that can represent strength.
# The _initialize_169_hands already sorts pairs first, then by rank, then suited before offsuit for same ranks.
# For HAND_STRENGTH_RANK, we want a single, definitive sorted list.
# The key used in expand_range_shorthand's final sort is good:
# key=lambda h: (get_rank_index(h[0]), get_rank_index(h[1]), h[2:] if len(h) > 2 else '')
# This effectively groups AA, then AKs, AKo, then AQs, AQo ... KK, KQs, KQo etc.
SORTED_MASTER_HAND_LIST = sorted(ALL_169_HAND_COMBINATIONS, key=lambda h: (
    get_rank_index(h[0]), 
    get_rank_index(h[1]),
    # Ensure pairs (e.g., 'AA', length 2) are treated consistently for sorting attribute
    # by giving them an empty string for the third sort key component (suit/type).
    # Suited ('s') will typically sort before offsuit ('o') with string comparison.
    (h[2:] if len(h) > 2 else '') 
))

HAND_STRENGTH_RANK = {hand_str: rank for rank, hand_str in enumerate(SORTED_MASTER_HAND_LIST)}

def expand_range_shorthand(shorthand_str):
    """
    Expands poker range shorthand into a list of specific hand combinations.
    Examples:
        "JJ+" -> ["JJ", "QQ", "KK", "AA"]
        "A9s+" -> ["A9s", "ATs", "AJs", "AQs", "AKs"]
        "KTo+" -> ["KTo", "KJo", "KQo"]
        "AQ+" -> ["AQs", "AQo", "AKs", "AKo"] (Covers both suited and offsuit if not specified)
        "77-99" -> ["77", "88", "99"]
        "A2s-A5s" -> ["A2s", "A3s", "A4s", "A5s"]
        "QTs-KJs" -> QTs, KJs - this interpretation is tricky. Current support: KTs-KQs -> KTs, KJs, KQs

    Handles individual hands like "AKs" or "77" correctly by returning them in a list.
    Note: Complex mixed ranges like "JJ+, AQs+, KQo" should be comma-separated
          and processed by splitting first, then calling this function on each part.
          This function handles one shorthand component at a time.
    """
    expanded_hands = set() # Use a set to avoid duplicates

    if not shorthand_str:
        return []

    # Check for direct match in all 169 hands (e.g. "AKs", "77")
    if shorthand_str in ALL_169_HAND_COMBINATIONS:
        return [shorthand_str]

    # Case 1: Pocket Pair Range (e.g., "JJ+", "77-99")
    if len(shorthand_str) == 3 and shorthand_str.endswith('+') and shorthand_str[0] == shorthand_str[1]: # e.g., "JJ+"
        pair_rank_char = shorthand_str[0]
        pair_rank_idx = get_rank_index(pair_rank_char)
        # Iterate upwards in rank (lower index means stronger rank)
        for i in range(pair_rank_idx, -1, -1):
            expanded_hands.add(f"{RANKS[i]}{RANKS[i]}")
    elif len(shorthand_str) == 5 and shorthand_str[2] == '-' and shorthand_str[0]==shorthand_str[1] and shorthand_str[3]==shorthand_str[4]: # e.g., "77-99"
        rank_char_1 = shorthand_str[0]
        rank_char_2 = shorthand_str[3]
        idx_1 = get_rank_index(rank_char_1)
        idx_2 = get_rank_index(rank_char_2)
        
        # Iterate from the lower rank (higher index) to the higher rank (lower index)
        # or vice-versa, min/max handles order e.g. "99-77" or "77-99"
        start_loop_idx = min(idx_1, idx_2)
        end_loop_idx = max(idx_1, idx_2)

        for i in range(start_loop_idx, end_loop_idx + 1):
            expanded_hands.add(f"{RANKS[i]}{RANKS[i]}")

    # Case 2: Ax+ type notation (e.g., "A9s+", "KTo+", "AQ+")
    # Covers XYo, XYs, XY+ (meaning both s and o)
    # This case assumes shorthand_str[0] is the higher ranked card of the two initial ones.
    elif shorthand_str.endswith('+') and len(shorthand_str) >= 2 and shorthand_str[0] != shorthand_str[1]: # Avoids re-processing pairs like "AA+"
        base = shorthand_str[:-1] # "A9s", "KTo", "AQ"
        
        primary_rank_char = base[0]
        base_kicker_char = base[1]
        primary_rank_idx = get_rank_index(primary_rank_char)

        stype = None
        if len(base) == 3: # "A9s" or "KTo"
            stype = base[2]
            if stype not in HAND_TYPES:
                # This could be an error, or we could try to infer if it's a typo for a rank.
                # For now, let's assume it's an invalid suit type.
                print(f"Warning: Invalid suit type '{stype}' in shorthand: {shorthand_str}")
                return [] # Or raise error
        elif len(base) != 2: # e.g. from "A+" or something too short
            print(f"Warning: Invalid base for '+' shorthand: {base} from {shorthand_str}")
            return [] # Or raise error
        
        base_kicker_idx = get_rank_index(base_kicker_char)

        if base_kicker_idx <= primary_rank_idx: # Kicker is stronger or same as primary card (e.g. "AAs+" or "KAs+")
            print(f"Warning: Kicker '{base_kicker_char}' not weaker than primary '{primary_rank_char}' in {shorthand_str}")
            # Potentially handle this as an error or specific case if e.g. KAs+ should mean AKs.
            # For now, returning empty as it's ambiguous or implies a pair, which is Case 1.
            if primary_rank_char == base_kicker_char and stype is None: # e.g. AA+ (no s/o) should be JJ+
                 # This should have been caught by Case 1 if it was e.g. "AA+" format. If it's "AAo+", it's invalid.
                 pass # Let it fall through to a general warning if nothing is added.
            else:
                return []

        # Iterate kicker upwards in strength (downwards in index) from base_kicker up to (but not including) primary_rank
        for k_idx in range(base_kicker_idx, primary_rank_idx, -1):
            current_kicker_char = RANKS[k_idx]
            
            # sorted() ensures canonical order (higher rank first)
            hr_char, lr_char = sorted([primary_rank_char, current_kicker_char], key=get_rank_index)

            if stype: # Specific suit type given e.g. "A9s+"
                expanded_hands.add(f"{hr_char}{lr_char}{stype}")
            else: # No suit type given e.g. "AQ+", add both suited and offsuit
                expanded_hands.add(f"{hr_char}{lr_char}s")
                expanded_hands.add(f"{hr_char}{lr_char}o")
                
    # Case 3: Range between two non-pair hands (e.g., "A2s-A5s", "KTs-KQs")
    # Assumes the primary card is fixed, and the kicker varies.
    elif '-' in shorthand_str and len(shorthand_str) >= 7 : # e.g. "A2s-A5s"
        parts = shorthand_str.split('-')
        if len(parts) == 2:
            start_hand_sh = parts[0].strip() # Clean input
            end_hand_sh = parts[1].strip()   # Clean input

            # Validate format: e.g., XNs (3 chars), primary card is X, kicker N, suit s
            if not (len(start_hand_sh) == 3 and len(end_hand_sh) == 3 and \
                    start_hand_sh[0].isalnum() and start_hand_sh[1].isalnum() and \
                    end_hand_sh[0].isalnum() and end_hand_sh[1].isalnum() and \
                    start_hand_sh[2] in HAND_TYPES and end_hand_sh[2] in HAND_TYPES):
                print(f"Warning: Invalid component format for '-' range: '{shorthand_str}'. Expected XNs-XZs.")
                return []
            
            # Further validation: primary card and suit type must be the same,
            # and primary card should not be the same as its kicker (not a pair like AAs)
            if not (start_hand_sh[0] == end_hand_sh[0] and 
                    start_hand_sh[2] == end_hand_sh[2] and 
                    start_hand_sh[0] != start_hand_sh[1] and # Start hand is not a pair e.g. AAs from AAs-A5s
                    end_hand_sh[0] != end_hand_sh[1]):   # End hand is not a pair e.g. AAs from A2s-AAs
                print(f"Warning: Range shorthand like '{shorthand_str}' expects fixed primary card, fixed suit type, and non-pair components (e.g. A2s-A5s)." )
                return []
            
            stype = start_hand_sh[2]
            # This stype check is technically redundant due to earlier check, but safe.
            # if stype not in HAND_TYPES:
            #     print(f"Warning: Invalid suit type '{stype}' in range: {shorthand_str}")
            #     return []

            fixed_primary_char = start_hand_sh[0]
            kicker1_char = start_hand_sh[1]
            kicker2_char = end_hand_sh[1]

            # Check if kickers are valid ranks
            if not (fixed_primary_char in RANKS and kicker1_char in RANKS and kicker2_char in RANKS):
                print(f"Warning: Invalid ranks in '-' range components: {shorthand_str}")
                return []

            fixed_primary_idx = get_rank_index(fixed_primary_char)
            kicker1_idx = get_rank_index(kicker1_char)
            kicker2_idx = get_rank_index(kicker2_char)

            # Kicker cannot be the same as primary (e.g. AAs from A2s-AAs where A is primary)
            # This is implicitly handled by sorted([fixed_primary_char, current_kicker_char], key=get_rank_index)
            # as it would form a pair, but we can be explicit if needed.
            # However, the check `start_hand_sh[0] != start_hand_sh[1]` already prevents initial pair format.

            loop_kicker_start_idx = min(kicker1_idx, kicker2_idx)
            loop_kicker_end_idx = max(kicker1_idx, kicker2_idx)

            for k_idx in range(loop_kicker_start_idx, loop_kicker_end_idx + 1):
                if k_idx == fixed_primary_idx: # Avoid forming a pair with the primary card e.g. AA from AAs-AKs (if A was kicker)
                    continue
                
                current_kicker_char = RANKS[k_idx]
                # Ensure canonical order (higher rank first)
                hr_char, lr_char = sorted([fixed_primary_char, current_kicker_char], key=get_rank_index)
                expanded_hands.add(f"{hr_char}{lr_char}{stype}")
        else:
             print(f"Warning: Invalid format for '-' range (expected one dash): {shorthand_str}")

    # Final check and warning if no hands were generated by patterns above
    # Strip trailing comma for single hand check like "AA,"
    cleaned_shorthand_str = shorthand_str.rstrip(',')
    if not expanded_hands and cleaned_shorthand_str in ALL_169_HAND_COMBINATIONS:
        expanded_hands.add(cleaned_shorthand_str)
    elif not expanded_hands and shorthand_str not in ALL_169_HAND_COMBINATIONS: # if still no match after trying cleaned
        print(f"Warning: Shorthand component '{shorthand_str}' not recognized or fully expanded.")

    return sorted(list(expanded_hands), key=lambda h: (
        get_rank_index(h[0]), 
        get_rank_index(h[1]),
        h[2:] # keeps 's' before 'o', and pairs like 'AA' distinct
    ))


# --- Reference Range Definitions ---

# These are example shorthand strings. You should tailor these to the specific
# preflop scenarios and player tendencies you want to model.
REFERENCE_RANGES_SHORTHAND = {
    'OOP': { # Out of Position Player
        'Tight': "QQ+,AKs,AKo",
        'Balanced': "77+,AJs+,KQs,ATo+,KQo",
        'Loose': "22+,A2s+,K7s+,Q8s+,J8s+,T7s+,97s+,87s+,A7o+,K9o+,Q9o+,J9o+,T9o"
    },
    'IP': { # In Position Player
        'Tight': "JJ+,AQs+,AQo+",
        'Balanced': "55+,ATs+,KJs+,QTs+,AJo+,KJo+",
        'Loose': "22+,A2s+,K5s+,Q8s+,J7s+,T7s+,96s+,86s+,75s+,64s+,53s+,A2o+,K9o+,Q9o+,J9o+,T8o+"
    }
}

PROCESSED_REFERENCE_RANGES = {}

def _process_reference_ranges():
    """
    Expands the shorthand strings in REFERENCE_RANGES_SHORTHAND
    and populates PROCESSED_REFERENCE_RANGES with lists of actual hand combinations.
    This should be called once when the module is initialized.
    """
    if PROCESSED_REFERENCE_RANGES: # Avoid reprocessing if called multiple times
        return

    for player_role, profiles in REFERENCE_RANGES_SHORTHAND.items():
        PROCESSED_REFERENCE_RANGES[player_role] = {}
        for range_type, shorthand_str in profiles.items():
            if not shorthand_str: # Handle empty shorthand string if any
                PROCESSED_REFERENCE_RANGES[player_role][range_type] = []
                continue
            
            expanded_hands_set = set()
            for part in shorthand_str.split(','):
                part_stripped = part.strip()
                if part_stripped: # Ensure part is not empty after strip
                    expanded_hands_set.update(expand_range_shorthand(part_stripped))
            
            PROCESSED_REFERENCE_RANGES[player_role][range_type] = sorted(list(expanded_hands_set), key=lambda h: (
                get_rank_index(h[0]),
                get_rank_index(h[1]),
                # Check length for pairs like 'AA' which don't have h[2:]
                h[2:] if len(h) > 2 else '' 
            ))
    # print("Debug: PROCESSED_REFERENCE_RANGES populated.")

_process_reference_ranges() # Populate at module load


# --- Utility for Range Analysis ---

def get_range_strength_bounds(range_list):
    """
    Finds the strength ranks of the strongest (lowest rank number) and 
    weakest (highest rank number) hands in a given list of hand strings.

    Args:
        range_list: A list of hand strings (e.g., ["AA", "KK", "AJs"]).

    Returns:
        A tuple (min_strength_rank, max_strength_rank) or (None, None) if range_list is empty.
    """
    if not range_list:
        return None, None

    min_rank = float('inf')
    max_rank = float('-inf')

    for hand_str in range_list:
        if hand_str not in HAND_STRENGTH_RANK:
            # This case should ideally not happen if range_list contains valid 169 hand strings
            print(f"Warning: Hand '{hand_str}' not found in HAND_STRENGTH_RANK. Skipping in bounds calculation.")
            continue
        
        rank = HAND_STRENGTH_RANK[hand_str]
        if rank < min_rank:
            min_rank = rank
        if rank > max_rank:
            max_rank = rank
            
    if min_rank == float('inf'): # Should only happen if all hands were invalid
        return None, None
        
    return min_rank, max_rank


# --- Adaptive Range Selection and Perturbation (Stubbed) ---

PLAYER_ROLES = ['OOP', 'IP']
# Order from tightest to loosest is important for adjustments
RANGE_TYPE_ORDER = ['Tight', 'Balanced', 'Loose'] 
DEFAULT_INITIAL_RANGE_TYPE = 'Balanced'

# How many strength ranks outside a category's bounds is acceptable for hero hand
ACCEPTABLE_WEAKNESS_OFFSET = 30 # e.g., hero hand can be up to X ranks weaker
ACCEPTABLE_STRENGTH_OFFSET = 15 # e.g., hero hand can be up to Y ranks stronger

def determine_hero_range_type_and_base_range(
    hero_hand_str,
    hero_player_role,
    initial_range_type_preference=DEFAULT_INITIAL_RANGE_TYPE,
    weakness_offset=ACCEPTABLE_WEAKNESS_OFFSET,
    strength_offset=ACCEPTABLE_STRENGTH_OFFSET
):
    """
    Determines the most appropriate range type (Tight, Balanced, Loose) for the hero
    based on their actual hand, and returns that type and its base range list.

    Args:
        hero_hand_str: The hero's specific hand (e.g., "AKo").
        hero_player_role: 'OOP' or 'IP'.
        initial_range_type_preference: Start with this type ('Tight', 'Balanced', 'Loose').
        weakness_offset: How much weaker (higher rank) hero hand can be than range's weakest.
        strength_offset: How much stronger (lower rank) hero hand can be than range's strongest.

    Returns:
        A tuple (final_hero_range_type_str, final_base_hero_range_list).
    """
    if hero_hand_str not in HAND_STRENGTH_RANK:
        raise ValueError(f"Hero hand '{hero_hand_str}' not found in HAND_STRENGTH_RANK.")
    hero_strength_rank = HAND_STRENGTH_RANK[hero_hand_str]

    if initial_range_type_preference not in RANGE_TYPE_ORDER:
        print(f"Warning: Invalid initial_range_type_preference '{initial_range_type_preference}'. Using default: '{DEFAULT_INITIAL_RANGE_TYPE}'")
        current_range_type = DEFAULT_INITIAL_RANGE_TYPE
    else:
        current_range_type = initial_range_type_preference
    
    # Iteratively adjust range type - at most a couple of steps
    for _ in range(len(RANGE_TYPE_ORDER)): # Max iterations to prevent infinite loops
        current_base_range_list = PROCESSED_REFERENCE_RANGES[hero_player_role][current_range_type]
        if not current_base_range_list: # Should not happen with current setup
            print(f"Warning: Empty base range for {hero_player_role} {current_range_type}. Cannot assess bounds.")
            break 

        strongest_rank_in_base, weakest_rank_in_base = get_range_strength_bounds(current_base_range_list)

        if strongest_rank_in_base is None: # Empty or invalid range
             print(f"Warning: Could not get bounds for {hero_player_role} {current_range_type}. Using current type.")
             break

        # Check if hero hand is too weak for the current range type
        if hero_strength_rank > weakest_rank_in_base + weakness_offset:
            current_type_idx = RANGE_TYPE_ORDER.index(current_range_type)
            if current_type_idx < len(RANGE_TYPE_ORDER) - 1: # Not already the loosest
                current_range_type = RANGE_TYPE_ORDER[current_type_idx + 1]
                # print(f"Debug: Hero hand {hero_hand_str} (rank {hero_strength_rank}) too weak for {RANGE_TYPE_ORDER[current_type_idx]}. Trying {current_range_type}.")
                continue # Re-evaluate with the new, looser type
            else:
                # print(f"Debug: Hero hand {hero_hand_str} too weak, already at loosest type {current_range_type}.")
                break # Already at the loosest, stop adjusting
        
        # Check if hero hand is too strong for the current range type
        elif hero_strength_rank < strongest_rank_in_base - strength_offset:
            current_type_idx = RANGE_TYPE_ORDER.index(current_range_type)
            if current_type_idx > 0: # Not already the tightest
                current_range_type = RANGE_TYPE_ORDER[current_type_idx - 1]
                # print(f"Debug: Hero hand {hero_hand_str} (rank {hero_strength_rank}) too strong for {RANGE_TYPE_ORDER[current_type_idx]}. Trying {current_range_type}.")
                continue # Re-evaluate with the new, tighter type
            else:
                # print(f"Debug: Hero hand {hero_hand_str} too strong, already at tightest type {current_range_type}.")
                break # Already at the tightest, stop adjusting
        else:
            # print(f"Debug: Hero hand {hero_hand_str} fits within {current_range_type} (bounds: {strongest_rank_in_base}-{weakest_rank_in_base}).")
            break # Hero hand fits, current_range_type is good
            
    final_base_list = PROCESSED_REFERENCE_RANGES[hero_player_role][current_range_type]
    return current_range_type, final_base_list


# --- Perturbation Configuration & Logic ---

# These are global for now, can be refined per range_type/role later
PERTURBATION_PARAMS = {
    # Probability of keeping a hand that's in the base range
    'prob_keep_core_hand': 0.90,  # e.g., 90% chance to keep a hand from the base
    
    # When considering adding neighbors:
    # Prob. of adding a neighbor stronger than the current base hand (but not in base)
    'prob_add_stronger_neighbor': 0.10, 
    # Prob. of adding a neighbor weaker than the current base hand (but not in base)
    'prob_add_weaker_neighbor': 0.08,
    
    # How many hands "around" a base hand in the SORTED_MASTER_HAND_LIST to consider as neighbors
    'neighbor_window_half_size': 7, # e.g., 7 stronger and 7 weaker (total window of 15 around hand)
    
    # To prevent range from drifting too much or becoming too small/large
    # Max percentage of original range size that can be added
    'max_added_hands_percentage': 0.20, # 20% of original size
    # Max percentage of original range size that can be removed (from original core hands)
    'max_removed_hands_percentage': 0.20, # 20% of original size
}

def _perform_perturbation(base_range_list, player_role, range_type):
    """
    Performs perturbation on a base range list.
    - Some core hands might be removed.
    - Some neighboring hands (stronger or weaker) might be added.
    - Total changes are capped to avoid distorting the range too much.
    """
    if not base_range_list:
        return []

    params = PERTURBATION_PARAMS # Could be specific if config is per role/type
    base_range_set = set(base_range_list)
    perturbed_hands_set = set()
    
    # --- Step 1: Decide which core hands to keep --- 
    removed_core_hands_count = 0
    max_removals_allowed = int(len(base_range_list) * params['max_removed_hands_percentage'])

    for hand in base_range_list:
        if random.random() < params['prob_keep_core_hand']:
            perturbed_hands_set.add(hand)
        else:
            if removed_core_hands_count < max_removals_allowed:
                removed_core_hands_count += 1
                # Hand is not added to perturbed_hands_set, effectively removed
            else:
                perturbed_hands_set.add(hand) # Cap on removals reached, keep it

    # --- Step 2: Identify candidate neighbors and probabilistically add them --- 
    candidate_neighbors_to_add = set()
    
    # Consider neighbors of ALL hands originally in the base range, 
    # even if some were tentatively removed in Step 1. This gives a broader pool of candidates.
    for core_hand_str in base_range_list: 
        if core_hand_str not in HAND_STRENGTH_RANK: continue # Should not happen
        core_hand_rank = HAND_STRENGTH_RANK[core_hand_str]

        # Define window for neighbors in the SORTED_MASTER_HAND_LIST
        start_idx = max(0, core_hand_rank - params['neighbor_window_half_size'])
        end_idx = min(len(SORTED_MASTER_HAND_LIST) -1, core_hand_rank + params['neighbor_window_half_size'])

        for i in range(start_idx, end_idx + 1):
            neighbor_hand = SORTED_MASTER_HAND_LIST[i]
            if neighbor_hand == core_hand_str: # Don't consider the core hand itself as a neighbor to add
                continue
            if neighbor_hand not in base_range_set: # Only consider adding if not originally in base
                candidate_neighbors_to_add.add(neighbor_hand)

    # Probabilistically add from candidates
    added_hands_count = 0
    max_additions_allowed = int(len(base_range_list) * params['max_added_hands_percentage'])
    
    # Shuffle candidates to avoid bias if max_additions_allowed is hit frequently
    shuffled_candidates = list(candidate_neighbors_to_add)
    random.shuffle(shuffled_candidates)

    for neighbor_hand in shuffled_candidates:
        if added_hands_count >= max_additions_allowed:
            break # Reached cap for additions

        if neighbor_hand in perturbed_hands_set: # Already decided to keep it (e.g. if it was also a core hand)
            continue

        # Determine if this neighbor is stronger or weaker than the *closest* original core hand
        # This is a simplification; true prob might depend on which core_hand it's a neighbor to.
        # For simplicity, just use a general probability based on its relation to *any* core hand.
        # A more precise way would be to check its rank relative to `core_hand_rank` in the loop above.
        # Here, we'll just use a blended approach or fixed probability for adding neighbors.
        
        # Let's use a generic prob_add_neighbor, or distinguish by its rank relative to the range bounds.
        # For now, using a simpler approach: if it's a neighbor, use a common probability.
        # The distinction `prob_add_stronger_neighbor` vs `prob_add_weaker_neighbor` is better applied
        # during the neighbor identification loop if we need that granularity.
        # For now, let's average them or pick one as a general `prob_add_any_valid_neighbor`.
        prob_to_add_this_neighbor = (params['prob_add_stronger_neighbor'] + params['prob_add_weaker_neighbor']) / 2.0

        if random.random() < prob_to_add_this_neighbor:
            perturbed_hands_set.add(neighbor_hand)
            added_hands_count += 1
            
    return sorted(list(perturbed_hands_set), key=lambda h: HAND_STRENGTH_RANK.get(h, float('inf')))

def generate_player_range_info(
    player_role,
    is_hero,
    hero_hand_str_if_any=None, # e.g., "AKo"
    range_type_preference=DEFAULT_INITIAL_RANGE_TYPE # For villain, or initial for hero
):
    """
    Generates final range information for a player, adapting for hero if specified.
    Includes (stubbed) perturbation and hero hand force-inclusion.
    """
    if player_role not in PLAYER_ROLES:
        raise ValueError(f"Invalid player_role: {player_role}")
    if range_type_preference not in RANGE_TYPE_ORDER:
        print(f"Warning: Invalid range_type_preference '{range_type_preference}'. Using default: '{DEFAULT_INITIAL_RANGE_TYPE}'")
        range_type_preference = DEFAULT_INITIAL_RANGE_TYPE

    actual_base_hands = []
    chosen_range_type = ""

    if is_hero:
        if not hero_hand_str_if_any:
            raise ValueError("hero_hand_str_if_any must be provided if is_hero is True.")
        
        chosen_range_type, actual_base_hands = determine_hero_range_type_and_base_range(
            hero_hand_str=hero_hand_str_if_any,
            hero_player_role=player_role,
            initial_range_type_preference=range_type_preference
        )
    else: # For Villain
        chosen_range_type = range_type_preference
        actual_base_hands = PROCESSED_REFERENCE_RANGES[player_role][chosen_range_type]

    # --- Perturbation Step (currently a stub) ---
    perturbed_hands_list = _perform_perturbation(actual_base_hands, player_role, chosen_range_type)

    # --- Hero Hand Force Inclusion (Safety Net) ---
    if is_hero and hero_hand_str_if_any:
        if hero_hand_str_if_any not in perturbed_hands_list:
            # Add and re-sort to maintain order if desired, though for solver string order may not matter
            temp_set = set(perturbed_hands_list)
            temp_set.add(hero_hand_str_if_any)
            perturbed_hands_list = sorted(list(temp_set), key=lambda h: HAND_STRENGTH_RANK.get(h, float('inf')))
            # print(f"Debug: Hero hand '{hero_hand_str_if_any}' force-added to perturbed list for {player_role}.")

    # --- Final comma-separated string for the solver ---
    # Solver might not care about the order, but consistency is good.
    final_range_str = ",".join(perturbed_hands_list) 

    return {
        'player_role': player_role,
        'is_hero': is_hero,
        'hero_hand_actual': hero_hand_str_if_any if is_hero else None,
        'range_type_selected': chosen_range_type,
        'base_hands_count': len(actual_base_hands),
        # 'base_hands_sample': actual_base_hands[:5], # For debugging
        'final_hands_count': len(perturbed_hands_list),
        'final_hands_sample': perturbed_hands_list[:10], # For debugging, show post-perturbation/inclusion
        'final_range_str': final_range_str
    }

# --- Gamestate Processing Functions (Re-inserting/Ensuring they are present) ---

def holding_to_hand_str(card1_repr, card2_repr):
    """
    Converts a hero's holding into the 169-hand string format (e.g., "AKs", "77").
    
    Args:
        card1_repr: Representation of the first card. 
                    Can be a tuple like ('A', 's') or a string like "As".
        card2_repr: Representation of the second card.

    Returns:
        The 169-hand string (e.g., "AKo", "77").
    """
    def parse_card_repr(card_r):
        if isinstance(card_r, str) and len(card_r) == 2:
            rank_char = card_r[0].upper()
            suit_char = card_r[1].lower()
            if rank_char not in RANKS or suit_char not in SUITS:
                raise ValueError(f"Invalid card string format: '{card_r}'")
            return rank_char, suit_char
        elif isinstance(card_r, tuple) and len(card_r) == 2:
            rank_char = str(card_r[0]).upper()
            suit_char = str(card_r[1]).lower()
            if rank_char not in RANKS or suit_char not in SUITS:
                raise ValueError(f"Invalid card tuple format: {card_r}")
            return rank_char, suit_char
        elif hasattr(card_r, 'rank') and hasattr(card_r, 'suit'): # For card objects
            rank_char = str(card_r.rank).upper()
            suit_char = str(card_r.suit).lower()
            if rank_char not in RANKS or suit_char not in SUITS:
                raise ValueError(f"Invalid card object properties: rank='{rank_char}', suit='{suit_char}'")
            return rank_char, suit_char
        else:
            raise ValueError(f"Unsupported card representation: {card_r}")

    r1_char, s1_char = parse_card_repr(card1_repr)
    r2_char, s2_char = parse_card_repr(card2_repr)

    idx1, idx2 = get_rank_index(r1_char), get_rank_index(r2_char)
    char1_sorted = RANKS[min(idx1, idx2)] 
    char2_sorted = RANKS[max(idx1, idx2)]

    if char1_sorted == char2_sorted:
        return f"{char1_sorted}{char2_sorted}"
    
    suited_char = 's' if s1_char == s2_char else 'o'
    return f"{char1_sorted}{char2_sorted}{suited_char}"


def augment_gamestate_with_ranges(gamestate_data, hero_is_oop_field='hero_is_oop', hero_holding_field='hero_holding'):
    """
    Augments a single gamestate dictionary with generated OOP and IP range strings.
    """
    if hero_is_oop_field not in gamestate_data:
        raise ValueError(f"Gamestate data missing '{hero_is_oop_field}' field.")
    if hero_holding_field not in gamestate_data:
        raise ValueError(f"Gamestate data missing '{hero_holding_field}' field.")

    hero_is_oop = gamestate_data[hero_is_oop_field]
    hero_holding_raw = gamestate_data[hero_holding_field]

    if not isinstance(hero_holding_raw, (list, tuple)) or len(hero_holding_raw) != 2:
        raise ValueError(f"Field '{hero_holding_field}' must be a list/tuple of two card representations.")

    hero_hand_str = holding_to_hand_str(hero_holding_raw[0], hero_holding_raw[1])
    
    oop_player_role_const = 'OOP'
    ip_player_role_const = 'IP'
    hero_actual_role = oop_player_role_const if hero_is_oop else ip_player_role_const
    
    oop_initial_pref = random.choice(RANGE_TYPE_ORDER)
    ip_initial_pref = random.choice(RANGE_TYPE_ORDER)

    oop_range_info = generate_player_range_info(
        player_role=oop_player_role_const,
        is_hero=(hero_actual_role == oop_player_role_const),
        hero_hand_str_if_any=hero_hand_str if (hero_actual_role == oop_player_role_const) else None,
        range_type_preference=oop_initial_pref
    )
    
    ip_range_info = generate_player_range_info(
        player_role=ip_player_role_const,
        is_hero=(hero_actual_role == ip_player_role_const),
        hero_hand_str_if_any=hero_hand_str if (hero_actual_role == ip_player_role_const) else None,
        range_type_preference=ip_initial_pref
    )

    augmented_gs = gamestate_data.copy()
    augmented_gs['oop_range_str'] = oop_range_info['final_range_str']
    augmented_gs['oop_range_type_selected'] = oop_range_info['range_type_selected']
    augmented_gs['ip_range_str'] = ip_range_info['final_range_str']
    augmented_gs['ip_range_type_selected'] = ip_range_info['range_type_selected']
    
    return augmented_gs

def process_gamestate_dataset(list_of_gamestate_dicts, hero_is_oop_field='hero_is_oop', hero_holding_field='hero_holding'):
    """
    Processes a list of gamestate dictionaries, augmenting each with range info.
    """
    augmented_dataset = []
    for i, gs_data in enumerate(list_of_gamestate_dicts):
        try:
            augmented_gs = augment_gamestate_with_ranges(gs_data, hero_is_oop_field, hero_holding_field)
            augmented_dataset.append(augmented_gs)
        except Exception as e:
            print(f"Error processing gamestate {i+1} (data: {gs_data}): {e}")
    return augmented_dataset
# --- End of Re-inserted Gamestate Processing Functions ---

if __name__ == "__main__":
    # Test holding_to_hand_str
    # Each item is a tuple: ( (card1_arg, card2_arg), expected_output_string )
    test_definitions = [
        ( ("As", "Ks"), "AKs" ), # Original: (["As", "Ks"], "AKs")
        ( ("Ad", "Kc"), "AKo" ), # Original: (["Ad", "Kc"], "AKo")
        ( ("As", "Ks"), "AKs" ), # Original: ("AsKs", "AKs") - now parsed to args
        ( ("Ad", "Kc"), "AKo" ), # Original: ("AdKc", "AKo") - now parsed to args
        ( ("2s", "2d"), "22"  ), # Original: ("2s2d", "22") - now parsed to args
        ( ("Th", "Jh"), "JTs" ), # Original: ("ThJh", "JTs") - now parsed to args (JTs if T,J are ranks)
        ( ("Jd", "Tc"), "JTo" ), # Original: ("JdTc", "JTo") - now parsed to args
        ( ("Qh", "Qd"), "QQ"  ), # Original: ("QhQd", "QQ") - now parsed to args
        ( ("5c", "5h"), "55"  ), # Original: ("5c5h", "55") - now parsed to args
    ]

    print("Running holding_to_hand_str tests...")
    all_tests_passed = True
    for i, (input_args_tuple, expected_value) in enumerate(test_definitions):
        try:
            result = holding_to_hand_str(*input_args_tuple) # Splat the arguments
            if result != expected_value:
                print(f"Test case {i+1} FAILED: Input={input_args_tuple}, Expected={expected_value}, Got={result}")
                all_tests_passed = False
            # else:
                # print(f"Test case {i+1} PASSED: Input={input_args_tuple}, Expected={expected_value}, Got={result}")
        except Exception as e:
            print(f"Test case {i+1} ERRORED with input {input_args_tuple}: {e}")
            all_tests_passed = False
            
    if all_tests_passed:
        print("All holding_to_hand_str tests passed successfully!")
    else:
        print("Some holding_to_hand_str tests FAILED or ERRORED.")

    # Cleaned up section - keeping test logic but removing verbose prints for brevity
    # during actual script runs. The test results (pass/fail) are still informative.
    pass
