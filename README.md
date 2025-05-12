# Poker AI with Internal Search Distillation

This repository is dedicated to the development of a sophisticated poker-playing AI leveraging Large Language Models (LLMs). The core methodology, outlined in `internal_search/poker_internal_search_methodology.txt`, focuses on "Internal Search Distillation."

## Project Goal

The primary goal is to train an LLM to simulate the reasoning process of a game-theoretic poker solver. Instead of performing computationally intensive rollouts during inference, the LLM learns to predict the outcomes of deep strategic explorations by generating a structured, linearized textual "thought process." This trace is annotated with Expected Value (EV) estimates derived from solver data during its training phase.

The LLM aims to:
1.  Generate a linearized search trace representing its step-by-step reasoning.
2.  Utilize special tokens and keywords to delineate decision points, chance events, and actions in an interpretable format.
3.  Propose strategically relevant actions for "Hero" (the AI player).
4.  Estimate the game-theoretic EV for each proposed action, trained on solver data.
5.  Predict plausible opponent responses and their impact on EV.
6.  Handle stochastic elements like card dealing through "Outcome Abstraction," where the LLM reasons about strategically significant categories of card outcomes rather than individual cards.
7.  Learn to approximate Bellman-like value propagation to make decisions based on multi-step lookahead.

## Key Components

### 1. Internal Search Methodology
The LLM is trained to generate a textual trace that emulates the internal search process of a poker solver. This trace includes:
*   **Hero Decision Nodes:** Proposals of actions and their immediate EV estimates.
*   **Opponent Decision Nodes:** Predictions of opponent responses and state transitions.
*   **Chance Nodes:** Representation of card dealing using abstracted outcomes.
*   **Selective Expansion:** Strategies to focus the search on the most salient lines of play, guided by initial EV estimates and solver-derived opponent response frequencies.

### 2. Dataset Generation
To train the LLM, a comprehensive dataset of poker game scenarios is required. This involves:
*   Using a high-quality poker solver to generate optimal strategies and EV data for numerous game situations.
*   Formatting this solver data into the linearized search trace format that the LLM will learn to produce.
*   **Range Generation (`dataset_generator_design_doc.txt`):** Tools and scripts like `range_generator.py` and `create_augmented_dataset.py` are used to define, adapt, and perturb player hand ranges. These ranges are then used to augment gamestate datasets, providing richer context for training the LLM. The system can:
    *   Define base range archetypes (Tight, Balanced, Loose).
    *   Adaptively select appropriate ranges for a "hero" player based on their known hand.
    *   Perturb ranges to create diverse training examples.
    *   Process input CSVs of game scenarios and output augmented CSVs with generated range information for both In-Position (IP) and Out-of-Position (OOP) players.

## Inference
During gameplay, the LLM receives the current game state as a prompt and autoregressively generates its internal search trace. The final action taken by the AI is based on the initial move that leads to the highest overall backed-up EV through this multi-step simulated lookahead.

## Future Directions
Future work includes refining the trace format, enhancing selective expansion strategies, optimizing the training pipeline/testing training strategies, and evaluating the LLM's playing strength against established benchmarks.
