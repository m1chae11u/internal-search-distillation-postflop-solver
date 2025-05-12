use postflop_solver_ffi::*;
use serde_json::{json, Value as JsonValue};
use std::fs::{File, self};
use csv::Reader;
use std::io::Write;
use std::path::Path;
use rayon::ThreadPoolBuilder;
use num_cpus;
// use rand_distr::{Distribution, WeightedIndex};

fn chips_to_bb(chips: f32, starting_stack_num: f32) -> f32 {
    if starting_stack_num == 0.0 {
        // Avoid division by zero; handle as an error or return a specific value.
        // For now, returning 0.0, but you might want to panic or return an Option/Result.
        return 0.0;
    }
    chips / (starting_stack_num / 100.0)
}

fn snap_percentage(p: f32, targets: &[f32], tolerance: f32) -> f32 {
    let mut best_snap = p;
    let mut min_diff_to_target_for_snap = f32::MAX;

    for &target_percent in targets {
        let diff = (p - target_percent).abs();
        if diff <= tolerance {
            if diff < min_diff_to_target_for_snap {
                min_diff_to_target_for_snap = diff;
                best_snap = target_percent;
            }
        }
    }
    best_snap
}

fn range_eval(game: &mut PostFlopGame, starting_stack_size: f32) -> (f32, f32, f32, f32) {
    game.cache_normalized_weights();
    let oop_equity = game.equity(0); // `0` means OOP player
    let oop_ev = game.expected_values(0);
    // // get equity and EV of a specific hand
    // println!("\nEquity of oop_hands[0]: {:.2}%", 100.0 * oop_equity[0]);
    // println!("EV of oop_hands[0]: {:.2}", oop_ev[0]);

    // get equity and EV of whole hand
    let weights = game.normalized_weights(0);
    let oop_average_equity = compute_average(&oop_equity, weights);
    let oop_average_ev = compute_average(&oop_ev, weights);
    println!("OOP_range_equity: {:.2}%", 100.0 * oop_average_equity);
    println!("OOP_range_EV: {:.2}bb", chips_to_bb(oop_average_ev, starting_stack_size));

    
    let ip_equity = game.equity(1); // `1` means IP player
    let ip_ev = game.expected_values(1);
    // // get equity and EV of a specific hand for IP player
    // println!("\nEquity of ip_hands[0]: {:.2}%", 100.0 * ip_equity[0]);
    // println!("EV of ip_hands[0]: {:.2}", ip_ev[0]);

    // get equity and EV of whole hand for IP player
    let ip_weights = game.normalized_weights(1);
    let ip_average_equity = compute_average(&ip_equity, ip_weights);
    let ip_average_ev = compute_average(&ip_ev, ip_weights);
    println!("IP_range_equity: {:.2}%", 100.0 * ip_average_equity);
    println!("IP_range_EV: {:.2}bb", chips_to_bb(ip_average_ev, starting_stack_size));

    (oop_average_equity, chips_to_bb(oop_average_ev, starting_stack_size), ip_average_equity, chips_to_bb(ip_average_ev, starting_stack_size))
}

fn action_eval(
    game: &mut PostFlopGame,
    player_index: usize,
    player_name: &str,
    _starting_stack_size: f32,
) -> (String, Vec<(Action, f32, f32, String)>) {
    let mut output_string = String::new();

    // Get available actions
    let actions = game.available_actions();

    // debugging
    println!("{}", &format!("\nAvailable {} actions: {:?}\n", player_name, actions)); 
    println!("{}", &format!("--- EV and Avg Strategy Prob for each available {} action ---\n", player_name));

    let original_history = game.history().to_vec();
    let initial_weights = game.normalized_weights(player_index).to_vec(); // Weights at the decision node
    let strategy_at_decision_node = game.strategy(); // Strategy for current player at this node
    let num_hands = game.num_private_hands(player_index);
    let mut action_evs_probs: Vec<(Action, f32, f32, String)> = Vec::new(); // MODIFIED: Added String for JSON action representation

    // Get pot size at the current decision node
    let total_bets_on_street_for_current_solve = game.total_bet_amount(); // Returns [i32; 2]
    let pot_at_decision_node = (game.tree_config().starting_pot
        + total_bets_on_street_for_current_solve[0]
        + total_bets_on_street_for_current_solve[1]) as f32;

    let snap_targets = [33.0, 66.0, 125.0, 250.0];
    let snap_tolerance = 2.0;

    for (action_index, action) in actions.iter().enumerate() {
        // Calculate average strategy probability for this action
        let start_index_slice = action_index * num_hands;
        let end_index_slice = (action_index + 1) * num_hands;
        if end_index_slice <= strategy_at_decision_node.len() {
            let action_strategy_slice = &strategy_at_decision_node[start_index_slice..end_index_slice];
            let average_strategy_prob = compute_average(action_strategy_slice, &initial_weights);

            game.play(action_index);
            game.cache_normalized_weights();

            let ev_after_action = game.expected_values(player_index);
            let average_ev_for_action = compute_average(&ev_after_action, &initial_weights);

            // String for general logging (uses snapping)
            let log_display_str = match *action {
                Action::Bet(chips) => {
                    let mut percentage = 0.0;
                    if pot_at_decision_node > 1e-6 {
                        percentage = (chips as f32 / pot_at_decision_node) * 100.0;
                    }
                    let snapped_percentage = snap_percentage(percentage, &snap_targets, snap_tolerance);
                    format!("Bet ({:.0}%)", snapped_percentage)
                }
                Action::Raise(chips) => {
                    let mut percentage = 0.0;
                    if pot_at_decision_node > 1e-6 {
                        percentage = (chips as f32 / pot_at_decision_node) * 100.0;
                    }
                    let snapped_percentage = snap_percentage(percentage, &snap_targets, snap_tolerance);
                    format!("Raise to ({:.0}%)", snapped_percentage)
                }
                Action::AllIn(chips) => {
                    let mut percentage = 0.0;
                    if pot_at_decision_node > 1e-6 {
                        percentage = (chips as f32 / pot_at_decision_node) * 100.0;
                    }
                    let snapped_percentage = snap_percentage(percentage, &snap_targets, snap_tolerance);
                    format!("AllIn ({:.0}%)", snapped_percentage)
                }
                _ => format!("{:?}", action),
            };

            // String for JSON output (specific format, no snapping for Bet/Raise %)
            let json_action_str = match *action {
                Action::Bet(chips) => {
                    let mut percentage = 0.0;
                    if pot_at_decision_node > 1e-6 {
                        percentage = (chips as f32 / pot_at_decision_node) * 100.0;
                    }
                    format!("bet_{:.0}%pot", percentage)
                }
                Action::Raise(chips) => {
                    let mut percentage = 0.0;
                    if pot_at_decision_node > 1e-6 {
                        percentage = (chips as f32 / pot_at_decision_node) * 100.0;
                    }
                    format!("raise_{:.0}%pot", percentage)
                }
                Action::Check => "check".to_string(),
                Action::Call => "call".to_string(),
                Action::Fold => "fold".to_string(),
                // Action::AllIn will use its Debug format, e.g., "AllIn(1234)"
                // Any other actions would also use their Debug format.
                _ => format!("{:?}", action),
            };

            output_string.push_str(&format!(
                "Action {}: Avg Frequency: {:.3}, EV: {:.2}bb\n",
                log_display_str, average_strategy_prob, chips_to_bb(average_ev_for_action, _starting_stack_size)
            ));
            action_evs_probs.push((*action, average_strategy_prob, average_ev_for_action, json_action_str)); // MODIFIED: added json_action_str

            game.back_to_root();
            game.apply_history(&original_history);
            game.cache_normalized_weights();
        } else {
            output_string.push_str(&format!(
                "Action {:?}: Could not calculate strategy (index out of bounds)\n",
                action
            ));
            action_evs_probs.push((*action, 0.0, 0.0, format!("{:?} (error)", action))); // MODIFIED: added error string for json_action_str
        }
    }
    (output_string, action_evs_probs)
}

fn run_solver_for_gamestate(
    oop_range: &str,
    ip_range: &str,
    flop_board_str: &str,
    turn_board_str: &str,
    river_board_str: &str,
    max_num_iterations: u32,
    exploitability_pct_pot_target: f32,
    pot_size: f32,
    eff_stack: f32,
    starting_stack_size: f32,
    evaluation_at: &str,
) -> JsonValue {

    let target_exploitability = pot_size * exploitability_pct_pot_target;
    let mut turn_board_str_mut = turn_board_str.to_string();
    let mut river_board_str_mut = river_board_str.to_string();

    if evaluation_at == "Flop" {
        turn_board_str_mut = String::new();
        river_board_str_mut = String::new();
    } else if evaluation_at == "Turn" {
        river_board_str_mut = String::new();
    }

    let card_config = CardConfig {
        range: [oop_range.parse().unwrap(), ip_range.parse().unwrap()],
        flop: flop_from_str(flop_board_str).unwrap(),
        turn: if turn_board_str_mut.is_empty() { NOT_DEALT } else { card_from_str(&turn_board_str_mut).unwrap() },
        river: if river_board_str_mut.is_empty() { NOT_DEALT } else { card_from_str(&river_board_str_mut).unwrap() },
    };

    // Determine the initial board state based on the provided cards
    let initial_board_state = if !river_board_str_mut.is_empty() {
        BoardState::River
    } else if !turn_board_str_mut.is_empty() {
        BoardState::Turn
    } else {
        BoardState::Flop
    };

    // we fix the bet sizes to be 33%, 66%, 125% of the pot, geometric size, and all-in
    // we fix the raise sizes to be 2.5x of the previous bet
    let bet_sizes = BetSizeOptions::try_from(("33%, 66%, 125%, e, a", "2.5x")).unwrap();

    let tree_config = TreeConfig {
        initial_state: initial_board_state, // must match `card_config`
        starting_pot: pot_size as i32,
        effective_stack: eff_stack as i32,
        rake_rate: 0.0,
        rake_cap: 0.0,
        flop_bet_sizes: [bet_sizes.clone(), bet_sizes.clone()], // [OOP, IP]
        turn_bet_sizes: [bet_sizes.clone(), bet_sizes.clone()],
        river_bet_sizes: [bet_sizes.clone(), bet_sizes.clone()], // Corrected, was `bet_sizes`
        turn_donk_sizes: None, // use default bet sizes
        river_donk_sizes: Some(DonkSizeOptions::try_from("50%").unwrap()),
        add_allin_threshold: 1.5, // add all-in if (maximum bet size) <= 1.5x pot
        force_allin_threshold: 0.15, // force all-in if (SPR after the opponent's call) <= 0.15
        merging_threshold: 0.1,
    };

    // SECTION 1: GAMESTATE
    println!("\n--- SECTION 1: GAMESTATE --- ✓");

    // Helper to format BetSizeOptions
    let format_bet_sizes = |bs: &BetSizeOptions| -> String {
        format!("Bet: '{:?}', Raise: '{:?}'", bs.bet, bs.raise)
    };
    
    // Helper to format DonkSizeOptions
    let format_donk_sizes = |ds: &Option<DonkSizeOptions>| -> String {
        match ds {
            Some(options) => format!("Donk: '{:?}'", options.donk),
            None => "None".to_string(),
        }
    };

    let game_context_str = format!("<GameContext eval_at={:?} starting_pot={:.2}bb effective_stack={:.2}bb flop_board={} turn_board={} river_board={} ip_range={} oop_range={} flop_bet_sizes=[OOP: {{ {} }}, IP: {{ {} }}] turn_bet_sizes=[OOP: {{ {} }}, IP: {{ {} }}] river_bet_sizes=[OOP: {{ {} }}, IP: {{ {} }}] river_donk_sizes={{ {} }} add_allin_threshold={} force_allin_threshold={} >",
        tree_config.initial_state,
        chips_to_bb(tree_config.starting_pot as f32, starting_stack_size),
        chips_to_bb(tree_config.effective_stack as f32, starting_stack_size),
        flop_board_str, 
        turn_board_str_mut,
        river_board_str_mut,
        ip_range, 
        oop_range, 
        format_bet_sizes(&tree_config.flop_bet_sizes[0]),
        format_bet_sizes(&tree_config.flop_bet_sizes[1]),
        format_bet_sizes(&tree_config.turn_bet_sizes[0]),
        format_bet_sizes(&tree_config.turn_bet_sizes[1]),
        format_bet_sizes(&tree_config.river_bet_sizes[0]),
        format_bet_sizes(&tree_config.river_bet_sizes[1]),
        format_donk_sizes(&tree_config.river_donk_sizes), 
        tree_config.add_allin_threshold,
        tree_config.force_allin_threshold,
    );
    println!("{}", game_context_str);

    // build the game tree
    // `ActionTree` can be edited manually after construction
    let action_tree = ActionTree::new(tree_config.clone()).unwrap();
    let mut game = PostFlopGame::with_config(card_config, action_tree).unwrap();

    // check memory usage
    let (mem_usage, _mem_usage_compressed) = game.memory_usage();
    println!(
        "\nMemory usage without compression (32-bit float): {:.2}GB\n",
        mem_usage as f64 / (1024.0 * 1024.0 * 1024.0)
    );

    // allocate memory without compression (use 32-bit float)
    game.allocate_memory(false);

    // solve the game
    let exploitability = solve(&mut game, max_num_iterations, target_exploitability, true);
    println!("Exploitability: {:.2}", exploitability);

    // SECTION 2: RANGE EVALUATION
    println!("\n--- SECTION 2: OOP RANGE EVALUATION --- ✓");
    let (oop_average_equity, oop_average_ev, ip_average_equity, ip_average_ev) = range_eval(&mut game, starting_stack_size);
    
    // SECTION 2.5: Sanity Check Nutted Hands (For Debugging)
    println!("\n--- SECTION 2.5: Sanity Check Nutted Hands --- ✓");

    // --- Display Top 3 Hands with Highest EV for OOP ---
    let oop_ev_at_decision_node = game.expected_values(0); // EV for each hand
    let oop_hands_str = holes_to_strings(game.private_cards(0)).unwrap_or_default();
    
    let mut indexed_evs_oop: Vec<(usize, f32)> = oop_ev_at_decision_node.iter().cloned().enumerate().collect();
    indexed_evs_oop.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    
    println!("\n Top 3 OOP Hands by EV at this node");
    for i in 0..std::cmp::min(3, indexed_evs_oop.len()) {
        let (hand_idx, ev_val) = indexed_evs_oop[i];
        if hand_idx < oop_hands_str.len() {
            println!("Hand {}: {}, EV: {:.2}bb", i + 1, oop_hands_str[hand_idx], chips_to_bb(ev_val, starting_stack_size));
        }
    }

    // --- Display Top 3 Hands with Highest EV for IP ---
    let ip_ev_at_decision_node = game.expected_values(1); // EV for each hand
    let ip_hands_str = holes_to_strings(game.private_cards(1)).unwrap_or_default();

    let mut indexed_evs_ip: Vec<(usize, f32)> = ip_ev_at_decision_node.iter().cloned().enumerate().collect();
    indexed_evs_ip.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    println!("\n Top 3 IP Hands by EV at this node");
    for i in 0..std::cmp::min(3, indexed_evs_ip.len()) {
        let (hand_idx, ev_val) = indexed_evs_ip[i];
        if hand_idx < ip_hands_str.len() {
            println!("Hand {}: {}, EV: {:.2}bb", i + 1, ip_hands_str[hand_idx], chips_to_bb(ev_val, starting_stack_size));
        }
    }

    // SECTION 3: ACTION EVALUATION
    println!("\n--- SECTION 3: ACTION EVALUATION --- ✓");

    // Evaluate OOP actions
    let (oop_action_evaluation_output, oop_action_data) = action_eval(&mut game, 0, "OOP", starting_stack_size);
    println!("{}", oop_action_evaluation_output);

    // SECTION 4: NODE EXPANSION (BY HIGHEST FREQUENCY)
    println!("\n--- SECTION 4: NODE EXPANSION (BY HIGHEST FREQUENCY) --- ✓");

    let chosen_oop_action_string_for_json = if oop_action_data.is_empty() {
        println!("OOP has no actions to choose from based on prior evaluation, or no actions were available.");
        "No OOP actions available/evaluated".to_string()
    } else {
        let mut best_action_original_index = 0;
        let mut max_freq = -1.0_f32; 
        let mut chosen_oop_action_details: Option<Action> = None;
        let mut temp_chosen_json_str: String = "Error: Action not chosen".to_string(); 

        for (i, (action, freq, _ev, json_str_for_this_action)) in oop_action_data.iter().enumerate() { 
            if *freq > max_freq {
                max_freq = *freq;
                best_action_original_index = i;
                chosen_oop_action_details = Some(*action);
                temp_chosen_json_str = json_str_for_this_action.clone(); 
            }
        }

        if let Some(action_to_play) = chosen_oop_action_details {
            let current_oop_actions = game.available_actions();
            if best_action_original_index < current_oop_actions.len() && current_oop_actions[best_action_original_index] == action_to_play {
                game.play(best_action_original_index);
                println!("OOP played action (highest frequency): {:?} with frequency {:.3}", action_to_play, max_freq);
                temp_chosen_json_str 
            } else {
                println!("Error: Mismatch or index out of bounds when trying to play OOP's highest frequency action. Action: {:?}, Index: {}", action_to_play, best_action_original_index);
                println!("Current available OOP actions: {:?}", current_oop_actions);
                format!("Error playing OOP action: {:?} (intended JSON: {})", action_to_play, temp_chosen_json_str) 
            }
        } else {
            println!("Could not determine OOP highest frequency action (e.g., all actions had zero or invalid frequency).");
            println!("OOP action data for context: {:?}", oop_action_data); 
            "Could not determine highest_freq OOP action (data invalid/empty)".to_string()
        }
    };
    
    println!("\nRange Evaluation");
    let (new_oop_average_equity, new_oop_average_ev, new_ip_average_equity, new_ip_average_ev) = range_eval(&mut game, starting_stack_size);
    let (ip_action_evaluation_output, ip_action_data) = action_eval(&mut game, 1, "IP", starting_stack_size);
    println!("{}", ip_action_evaluation_output);
    
    // SECTION 5: OUTPUT BEST IP ACTION (BY HIGHEST FREQUENCY)
    println!("\n--- SECTION 5: OUTPUT BEST IP ACTION (BY HIGHEST FREQUENCY) ---");

    let chosen_ip_action_string_for_json = if ip_action_data.is_empty() {
        println!("IP has no actions to choose from based on prior evaluation, or no actions were available.");
        "No IP actions available/evaluated".to_string()
    } else {
        let mut max_ip_freq = -1.0_f32;
        let mut temp_chosen_ip_json_str: String = "Error: IP Action not chosen".to_string();

        for (_action, freq, _ev, json_str_for_this_action) in ip_action_data.iter() {
            if *freq > max_ip_freq {
                max_ip_freq = *freq;
                temp_chosen_ip_json_str = json_str_for_this_action.clone();
            }
        }
        if max_ip_freq > -1.0 { 
            println!("Highest frequency IP action (for JSON output): {} with frequency {:.3}", temp_chosen_ip_json_str, max_ip_freq);
        } else {
            println!("Could not determine IP highest frequency action (e.g., all actions had zero or invalid frequency).");
            println!("IP action data for context: {:?}", ip_action_data);
        }
        temp_chosen_ip_json_str
    };

    let section_2_range_eval_string = format!(
        "OOP_equity: {:.2}%, OOP_ev: {:.2}bb, IP_equity: {:.2}%, IP_ev: {:.2}bb",
        oop_average_equity * 100.0, oop_average_ev, ip_average_equity * 100.0, ip_average_ev
    );

    let section_4_range_eval_string = format!(
        "OOP_equity: {:.2}%, OOP_ev: {:.2}bb, IP_equity: {:.2}%, IP_ev: {:.2}bb",
        new_oop_average_equity * 100.0, new_oop_average_ev, new_ip_average_equity * 100.0, new_ip_average_ev
    );

    let result_json = json!({
        "1. GameState": {
            "game_context_str": game_context_str
        },
        "2. RangeEval": section_2_range_eval_string,
        "3. OOP_Action_RangeEval": oop_action_evaluation_output,
        "4. Expand_OOP_Action": {
            "OOP high_freq_action": chosen_oop_action_string_for_json,
            "RangeEval": section_4_range_eval_string,
            "IP_Action_RangeEval": ip_action_evaluation_output
        },
        "5. IP_highest_freq_action": {
            "action": chosen_ip_action_string_for_json
        },
        "6. misc_metadata": {
            "exploitability_pct_pot_target": exploitability_pct_pot_target,
            "target_exploitability": target_exploitability,
            "actual_exploitability": exploitability,
            "max_num_iterations": max_num_iterations,
            "tree_config": format!("{:?}", tree_config)
        }
    });

    result_json
}

fn save_checkpoint(results: &Vec<JsonValue>, file_path: &str) {
    println!("Attempting to save checkpoint with {} results to {}...", results.len(), file_path);
    match serde_json::to_string_pretty(results) {
        Ok(json_output) => {
            // Ensure we write "[]" for an empty vector, not an empty string.
            // serde_json::to_string_pretty will correctly produce "[]" for an empty Vec.
            match File::create(file_path) {
                Ok(mut outfile) => {
                    if let Err(e) = outfile.write_all(json_output.as_bytes()) {
                        eprintln!("Failed to write JSON to checkpoint file {}: {}", file_path, e);
                    } else {
                        println!("Successfully wrote checkpoint with {} results to {}", results.len(), file_path);
                    }
                }
                Err(e) => {
                    eprintln!("Failed to create checkpoint file {}: {}", file_path, e);
                }
            }
        }
        Err(e) => {
            eprintln!("Failed to serialize results for checkpoint: {}", e);
        }
    }
}

fn main() {
    // Configure Rayon's global thread pool
    let num_cpus = num_cpus::get();
    let num_threads = (num_cpus as f64 * 0.5).ceil() as usize;
    println!(
        "Configuring Rayon to use {} threads (50% of {} available CPUs, rounded up).",
        num_threads, num_cpus
    );
    ThreadPoolBuilder::new()
        .num_threads(num_threads)
        .build_global()
        .unwrap();

    const CHECKPOINT_INTERVAL: usize = 100;
    let outpath_file_path = "/home/xuandong/mnt/poker/internal-search-distillation-postflop-solver/datasets/search_trace_prep/search_trace_prep_output.json";
    let csv_file_path = "/home/xuandong/mnt/poker/internal-search-distillation-postflop-solver/examples/test3.csv";

    let mut all_results: Vec<JsonValue> = Vec::new();
    let start_index: usize;

    if Path::new(outpath_file_path).exists() {
        println!("Checkpoint file found at {}. Attempting to load.", outpath_file_path);
        match fs::read_to_string(outpath_file_path) {
            Ok(contents) => {
                if contents.trim().is_empty() {
                    println!("Checkpoint file is empty. Starting fresh.");
                    all_results = Vec::new();
                } else {
                    match serde_json::from_str(&contents) {
                        Ok(parsed_results) => {
                            all_results = parsed_results;
                            println!("Successfully loaded {} results from checkpoint.", all_results.len());
                        }
                        Err(e) => {
                            eprintln!("Failed to parse JSON from checkpoint file {}: {}. Starting fresh.", outpath_file_path, e);
                            all_results = Vec::new(); 
                        }
                    }
                }
            }
            Err(e) => {
                eprintln!("Failed to read checkpoint file {}: {}. Starting fresh.", outpath_file_path, e);
                all_results = Vec::new();
            }
        }
    } else {
        println!("No checkpoint file found at {}. Starting fresh.", outpath_file_path);
    }
    start_index = all_results.len();
    if start_index > 0 {
        println!("Skipping first {} records from CSV as they are already processed.", start_index);
    }

    match File::open(csv_file_path) {
        Ok(file) => {
            let mut rdr = Reader::from_reader(file);
            for (loop_idx, result) in rdr.records().skip(start_index).enumerate() {
                let original_csv_row_num = loop_idx + start_index + 1;
                match result {
                    Ok(record) => {
                        println!("\n--- Processing row {} from CSV ---", original_csv_row_num);
                        let oop_range = record.get(10).expect("CSV must contain oop_range at column 10").to_string();
                        let ip_range = record.get(12).expect("CSV must contain ip_range at column 12").to_string();
                        let flop_board_str = record.get(2).expect("CSV must contain flop_board_str at column 2").to_string();
                        let turn_board_str_csv = record.get(3).expect("CSV must contain turn_board_str at column 3").to_string();
                        let river_board_str_csv = record.get(4).expect("CSV must contain river_board_str at column 4").to_string();
                        let max_num_iterations = 1000; 
                        let exploitability_pct_pot_target = 0.0025; 
                        let pot_size = record.get(14).expect("CSV must contain pot_size at column 14").parse::<f32>().unwrap();
                        let eff_stack = record.get(15).expect("CSV must contain eff_stack at column 15").parse::<f32>().unwrap();
                        let starting_stack_size = 200.0; 
                        let evaluation_at = record.get(7).expect("CSV must contain evaluation_at at column 7").to_string();
                        
                        let result_json = run_solver_for_gamestate(
                            &oop_range,
                            &ip_range,
                            &flop_board_str,
                            &turn_board_str_csv,
                            &river_board_str_csv,
                            max_num_iterations,
                            exploitability_pct_pot_target,
                            pot_size,
                            eff_stack,
                            starting_stack_size,
                            &evaluation_at
                        );
                        all_results.push(result_json); 
                        
                        let num_processed_in_this_run = all_results.len() - start_index;
                        if num_processed_in_this_run > 0 && num_processed_in_this_run % CHECKPOINT_INTERVAL == 0 {
                            save_checkpoint(&all_results, outpath_file_path);
                        }
                    }
                    Err(e) => {
                        eprintln!("Error processing CSV row {}: {}", original_csv_row_num, e);
                    }
                }
            }
            if !all_results.is_empty() {
                 println!("Processing complete. Performing final save...");
                 save_checkpoint(&all_results, outpath_file_path);
            } else if start_index == 0 {
                 println!("Processing complete. No new items processed. Ensuring empty output file or array.");
                 save_checkpoint(&all_results, outpath_file_path);
            } else {
                 println!("Processing complete. No new items processed since last checkpoint.");
            }

        }
        Err(e) => {
            eprintln!("Failed to open CSV file {}: {}", csv_file_path, e);
        }
    }
}
