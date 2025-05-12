use postflop_solver_ffi::{GameArgs, run_solver_for_gamestate};
use serde_json;
use std::env;
use std::process;
use rayon;
use num_cpus;

// The FFI types (PostFlopGame, Action, etc.) and other helpers 
// (compute_average, solve, etc.) are expected to be available via the postflop_solver crate (src/lib.rs).

fn main() {
    // Configure Rayon thread pool for this worker process
    let num_worker_threads = std::cmp::max(1, num_cpus::get() / 2);
    if let Err(e) = rayon::ThreadPoolBuilder::new().num_threads(num_worker_threads).build_global() {
        eprintln!(
            "Solver_worker: Failed to initialize Rayon global thread pool to {} thread(s): {}. \
            This worker might use more CPU cores than intended.",
            num_worker_threads,
            e
        );
        // Depending on requirements, you might choose to exit(1) here
        // if strict control over worker thread count is essential.
    }

    // Suppress println! output from within run_solver_for_gamestate and its children
    // for the worker, as we only want the final JSON on stdout.
    // This is a bit of a hack. A better way would be to pass a logger/config 
    // to run_solver_for_gamestate to control verbosity.
    // For now, this worker is meant to be quiet except for its JSON output or fatal errors.
    // However, the println! calls are deep inside and used for debugging/logging progress.
    // The parent process (trace_builder) will manage overall progress logs.
    // Individual worker logs might be useful if captured from stderr, but run_solver_for_gamestate
    // currently prints to stdout.
    // For now, we will let them print; the parent will only parse the *last line* or expect specific JSON.
    // This means run_solver_for_gamestate should ideally *only* print the final JSON to stdout
    // if it's to be used this way, or a flag should control its verbosity.

    // Let's assume for now that all `println!` in `run_solver_for_gamestate` are for debugging
    // and the actual result is what we care about from its return value.
    // The `trace_builder` will need to be robust in parsing stdout if it contains more than just JSON.
    // A better approach for the worker would be to ensure `run_solver_for_gamestate` returns Result
    // and logs to stderr, only printing final JSON to stdout.

    let cmd_args: Vec<String> = env::args().collect();
    if cmd_args.len() != 2 {
        eprintln!("Usage: solver_worker <json_game_args>");
        eprintln!("Received {} arguments: {:?}", cmd_args.len(), cmd_args);
        process::exit(1);
    }

    let game_args_json = &cmd_args[1];
    let game_args: GameArgs = match serde_json::from_str(game_args_json) {
        Ok(data) => data,
        Err(e) => {
            eprintln!("Failed to deserialize GameArgs JSON: {}. Input: {}", e, game_args_json);
            process::exit(1);
        }
    };

    // Call the function now located in lib.rs
    let result_json = run_solver_for_gamestate(
        &game_args.oop_range,
        &game_args.ip_range,
        &game_args.flop_board_str,
        &game_args.turn_board_str_csv, 
        &game_args.river_board_str_csv,
        game_args.max_num_iterations,
        game_args.exploitability_pct_pot_target,
        game_args.pot_size,
        game_args.eff_stack,
        game_args.starting_stack_size,
        &game_args.evaluation_at,
        game_args.original_index, // Pass original_index through
    );

    match serde_json::to_string(&result_json) {
        Ok(output_str) => {
            // Ensure only this JSON is printed to stdout by the worker for easy capture.
            println!("{}", output_str);
        }
        Err(e) => {
            eprintln!("Failed to serialize result to JSON: {}", e);
            process::exit(1);
        }
    }
} 