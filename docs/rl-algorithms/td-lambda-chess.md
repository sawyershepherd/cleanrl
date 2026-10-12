# TD(λ) Chess Strategy

## Overview

`td_lambda_chess.py` learns a chess evaluation through self-play with TD(λ), in the spirit of TD-Gammon and KnightCap. The evaluation is deliberately small and readable: a single linear layer over 21 hand-designed strategic features (material, mobility, center control, development, king safety, pawn structure, rook placement, endgame king activity, and hanging pieces), squashed by `tanh` into a value in $[-1, 1]$. Every learned weight can be read as "how many pawns this idea is worth", which makes the model a strategy you can inspect, question, and retune.

The script comes with an interactive board, [`cleanrl_utils/chess_strategy_board.html`](https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl_utils/chess_strategy_board.html). It is a single HTML file with no dependencies. It plays against you with an alpha-beta search over the learned weights, shows how each strategic term contributes to the current position, explains its candidate moves, and lets you move the weights with sliders or load the `strategy.json` written by a training run.

Original papers:

* [Learning to Predict by the Methods of Temporal Differences](https://link.springer.com/article/10.1007/BF00115009)
* [Temporal Difference Learning and TD-Gammon](https://dl.acm.org/doi/10.1145/203330.203343)

## Implemented Variants

| Variants Implemented      | Description |
| ----------- | ----------- |
| :material-github: [`td_lambda_chess.py`](https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/td_lambda_chess.py), :material-file-document: [docs](/rl-algorithms/td-lambda-chess/#td_lambda_chesspy) | Self-play chess with a linear strategy model over interpretable features. |

Below is our single-file implementation of TD(λ) chess:

## `td_lambda_chess.py`

The [td_lambda_chess.py](https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/td_lambda_chess.py) has the following features:

* Uses [python-chess](https://python-chess.readthedocs.io/) for the rules, so there is no gym environment to install
* Learns from self-play only; the material weights optionally start at textbook values (`--material-prior`)
* Exports the learned weights to `strategy.json` for the interactive board

### Usage

=== "uv"

    ```bash
    uv pip install chess==1.10.0
    uv run python cleanrl/td_lambda_chess.py --help
    uv run python cleanrl/td_lambda_chess.py --save-model
    ```

=== "pip"

    ```bash
    pip install chess==1.10.0
    python cleanrl/td_lambda_chess.py --help
    python cleanrl/td_lambda_chess.py --save-model
    ```

Then open `cleanrl_utils/chess_strategy_board.html` in a browser, press **Load strategy.json**, and pick `runs/{run_name}/strategy.json`.

### Explanation of the logged metrics

Running `python cleanrl/td_lambda_chess.py` will automatically record various metrics in Tensorboard. Below is the documentation for these metrics:

* `charts/game_length`: number of plies in the self-play game
* `charts/white_score`: result of the self-play game for White (1 win, 0.5 draw or truncation, 0 loss)
* `charts/decisive`: whether the self-play game ended in checkmate
* `charts/truncated`: whether the self-play game hit `--max-plies` and was bootstrapped
* `charts/epsilon`: the current exploration rate
* `charts/SPS`: number of self-play plies per second
* `losses/td_loss`: the mean squared error between the predicted values and the TD(λ) returns of a game
* `losses/values`: the mean predicted value over the game
* `strategy/<feature>`: the raw weight of each strategic feature, logged every 10 games
* `eval/score_vs_random`: score (win 1, draw 0.5) against a uniformly random player
* `eval/score_vs_material`: score against the same search using material-only weights

### Implementation details

1. **Features from the side to move.** `extract_features` computes each feature for both colors and returns `us - them`, so the value $V(s)$ always belongs to the player about to move (negamax convention). Two features are asymmetric on purpose: `threats` is the opponent's material that is en prise (we can take it now) and `own_en_prise` is ours (we still have a move to save it).
1. **Model.** $V(s) = \tanh(w^\top \phi(s) + b)$, implemented as `nn.Linear(21, 1)` followed by `tanh`. The bias acts as a tempo bonus.
1. **Move selection.** Self-play uses an ε-greedy negamax search of depth `--search-depth` (1 by default, with leaf evaluations batched through numpy). The first `--random-opening-plies` plies are random for diversity. Checkmate, stalemate and insufficient material end a game with an exact value; repetition draws are not claimed during self-play so that games do not stall in early shuffles.
1. **TD(λ) returns with alternating perspective.** Because successive values belong to opposite players, the next value is negated when it is backed up:
    $$G_t^\lambda = -\gamma \left[(1 - \lambda) V(s_{t+1}) + \lambda G_{t+1}^\lambda\right],$$
    with $G_T = V(s_T)$ equal to $-1$ when the side to move at $s_T$ is checkmated, $0$ for a draw, and the model's own estimate when the game is truncated at `--max-plies`.
1. **Offline update.** After each game, one Adam step minimizes $\frac{1}{T}\sum_t (V(s_t) - G_t^\lambda)^2$ over all positions of that game (the forward view of TD(λ)).
1. **Feature parity with the board.** The JavaScript engine in `chess_strategy_board.html` reimplements `extract_features` exactly, so loaded weights mean the same thing in the browser as in training.

### Experiment results

A run with the default settings (`seed=1`, 2,000 games) took about 9.5 minutes on 4 CPU cores. It scored 1.00 against both the random player and the material-only baseline (20 games each) at every evaluation from game 250 onward. Its learned weights, in pawns per unit of each feature, are the "Self-play TD(λ)" preset on the board:

| Feature | Weight | Feature | Weight |
| ----------- | ----------- | ----------- | ----------- |
| pawn | +1.00 | king_shelter | −0.18 |
| knight | +2.69 | king_pressure | +0.15 |
| bishop | +2.78 | doubled_pawns | −0.13 |
| rook | +3.71 | isolated_pawns | −0.01 |
| queen | +8.47 | passed_pawns | +0.52 |
| bishop_pair | −0.09 | passed_advance | +3.62 |
| mobility | +0.83 | rook_open_file | −0.07 |
| center_control | −0.08 | rook_seventh | +0.08 |
| center_pawns | +0.02 | king_activity | +0.17 |
| development | +0.50 | threats | +2.80 |
| | | own_en_prise | −0.24 |

Self-play learned to reward mobility, development, threats and advanced passed pawns, and to penalize doubled pawns and loose pieces. Some weights are still noisy after 2,000 games of shallow self-play (the rook is undervalued and king shelter is slightly negative). Longer runs, a deeper `--search-depth`, or the "Textbook" preset on the board make useful comparisons.
