# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/td-lambda-chess/#td_lambda_chesspy
import json
import os
import random
import time
from dataclasses import dataclass

import chess
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tyro
from torch.utils.tensorboard import SummaryWriter


@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "cleanRL"
    """the wandb's project name"""
    wandb_entity: str = None
    """the entity (team) of wandb's project"""
    save_model: bool = False
    """whether to save model into the `runs/{run_name}` folder"""

    # Algorithm specific arguments
    env_id: str = "Chess"
    """the id of the environment"""
    total_games: int = 2000
    """total number of self-play games"""
    learning_rate: float = 1e-3
    """the learning rate of the optimizer"""
    gamma: float = 1.0
    """the discount factor gamma"""
    td_lambda: float = 0.7
    """the lambda for TD(lambda) returns"""
    max_plies: int = 200
    """the number of plies after which a self-play game is truncated (and bootstrapped)"""
    search_depth: int = 1
    """the alpha-beta search depth used to pick moves during self-play"""
    random_opening_plies: int = 4
    """the number of uniformly random plies played at the start of each game for diversity"""
    start_e: float = 0.3
    """the starting epsilon for exploration"""
    end_e: float = 0.05
    """the ending epsilon for exploration"""
    exploration_fraction: float = 0.5
    """the fraction of `total-games` it takes from start-e to go end-e"""
    material_prior: bool = True
    """if toggled, the material weights start at textbook piece values instead of zero"""
    eval_frequency: int = 100
    """the number of games between evaluations against fixed opponents"""
    eval_games: int = 20
    """the number of games played against each fixed opponent per evaluation"""


PIECE_VALUES = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 100}
FEATURE_NAMES = [
    "pawn",
    "knight",
    "bishop",
    "rook",
    "queen",
    "bishop_pair",
    "mobility",
    "center_control",
    "center_pawns",
    "development",
    "king_shelter",
    "king_pressure",
    "doubled_pawns",
    "isolated_pawns",
    "passed_pawns",
    "passed_advance",
    "rook_open_file",
    "rook_seventh",
    "king_activity",
    "threats",
    "own_en_prise",
]
NUM_FEATURES = len(FEATURE_NAMES)
CENTER = [chess.D4, chess.E4, chess.D5, chess.E5]
BB_CENTER = chess.BB_D4 | chess.BB_E4 | chess.BB_D5 | chess.BB_E5
ADJACENT_FILES = [(chess.BB_FILES[f - 1] if f > 0 else 0) | (chess.BB_FILES[f + 1] if f < 7 else 0) for f in range(8)]
# squares in front of a pawn (same and adjacent files) that must be free of enemy pawns for it to be passed
PASSED_MASKS = {color: [0] * 64 for color in chess.COLORS}
for _sq in chess.SQUARES:
    _f, _r = chess.square_file(_sq), chess.square_rank(_sq)
    _files = chess.BB_FILES[_f] | ADJACENT_FILES[_f]
    for _rank in range(8):
        if _rank > _r:
            PASSED_MASKS[chess.WHITE][_sq] |= _files & chess.BB_RANKS[_rank]
        if _rank < _r:
            PASSED_MASKS[chess.BLACK][_sq] |= _files & chess.BB_RANKS[_rank]


def shelter_mask(king_sq: int, color: bool, ahead: int) -> int:
    rank = chess.square_rank(king_sq) + (ahead if color == chess.WHITE else -ahead)
    if not 0 <= rank <= 7:
        return 0
    f = chess.square_file(king_sq)
    files = chess.BB_FILES[f] | ADJACENT_FILES[f]
    return files & chess.BB_RANKS[rank]


def centralization(sq: int) -> float:
    return (3.5 - max(abs(chess.square_file(sq) - 3.5), abs(chess.square_rank(sq) - 3.5))) / 3.0


def side_features(board: chess.Board, color: bool, phase: float) -> np.ndarray:
    """Strategic features of `color`'s position; the model input is `side_features(us) - side_features(them)`."""
    f = np.zeros(NUM_FEATURES, dtype=np.float32)
    own = board.occupied_co[color]
    pawns = board.pawns & own
    bishops = board.bishops & own
    rooks = board.rooks & own
    king_sq = board.king(color)
    enemy_king_sq = board.king(not color)

    # material
    f[0] = chess.popcount(pawns)
    f[1] = chess.popcount(board.knights & own)
    f[2] = chess.popcount(bishops)
    f[3] = chess.popcount(rooks)
    f[4] = chess.popcount(board.queens & own)
    f[5] = float(f[2] >= 2)

    # piece activity
    mobility = 0
    for sq in chess.scan_forward((board.knights | board.bishops | board.rooks | board.queens) & own):
        mobility += chess.popcount(board.attacks_mask(sq) & ~own)
    f[6] = mobility / 10.0
    f[7] = sum(chess.popcount(board.attackers_mask(color, sq)) for sq in CENTER) / 4.0
    f[8] = chess.popcount(pawns & BB_CENTER)
    home_rank = chess.BB_RANK_1 if color == chess.WHITE else chess.BB_RANK_8
    f[9] = chess.popcount((board.knights | board.bishops) & own & ~home_rank) * phase

    # king safety
    f[10] = (
        chess.popcount(pawns & shelter_mask(king_sq, color, 1)) + 0.5 * chess.popcount(pawns & shelter_mask(king_sq, color, 2))
    ) * phase
    zone = chess.BB_KING_ATTACKS[enemy_king_sq] | chess.BB_SQUARES[enemy_king_sq]
    f[11] = sum(chess.popcount(board.attackers_mask(color, sq) & ~board.kings) for sq in chess.scan_forward(zone)) / 4.0

    # pawn structure
    enemy_pawns = board.pawns & board.occupied_co[not color]
    for file in range(8):
        n = chess.popcount(pawns & chess.BB_FILES[file])
        if n:
            f[12] += n - 1
            if not pawns & ADJACENT_FILES[file]:
                f[13] += n
    for sq in chess.scan_forward(pawns):
        if not enemy_pawns & PASSED_MASKS[color][sq]:
            f[14] += 1
            rel_rank = chess.square_rank(sq) if color == chess.WHITE else 7 - chess.square_rank(sq)
            f[15] += ((rel_rank - 1) / 5.0) ** 2

    # rooks
    seventh = chess.BB_RANK_7 if color == chess.WHITE else chess.BB_RANK_2
    for sq in chess.scan_forward(rooks):
        file_mask = chess.BB_FILES[chess.square_file(sq)]
        if not board.pawns & file_mask:
            f[16] += 1.0
        elif not pawns & file_mask:
            f[16] += 0.5
        if chess.BB_SQUARES[sq] & seventh:
            f[17] += 1

    # endgame king activity
    f[18] = centralization(king_sq) * (1.0 - phase)

    # pieces of `color` that are en prise: attacked and either undefended or attacked by a cheaper piece
    en_prise = 0
    for sq in chess.scan_forward(own & ~board.kings):
        attackers = board.attackers_mask(not color, sq)
        if not attackers:
            continue
        value = PIECE_VALUES[board.piece_type_at(sq)]
        cheapest = min(PIECE_VALUES[board.piece_type_at(a)] for a in chess.scan_forward(attackers))
        if cheapest < value or not board.attackers_mask(color, sq):
            en_prise += value
    f[20] = en_prise / 3.0
    return f


def extract_features(board: chess.Board) -> np.ndarray:
    """Features from the perspective of the side to move (negamax convention)."""
    phase = min(
        1.0,
        (chess.popcount(board.knights | board.bishops) + 2 * chess.popcount(board.rooks) + 4 * chess.popcount(board.queens))
        / 24.0,
    )
    us = side_features(board, board.turn, phase)
    them = side_features(board, not board.turn, phase)
    diff = us - them
    # we are to move, so the opponent's en-prise material is a threat we can cash in, while ours can still be saved
    diff[19] = them[20]
    diff[20] = us[20]
    return diff


def terminal_value(board: chess.Board, claim_draw: bool = False):
    """Exact value for the side to move if the game is over, else None."""
    if board.is_checkmate():
        return -1.0
    if board.is_stalemate() or board.is_insufficient_material():
        return 0.0
    if claim_draw and (board.can_claim_fifty_moves() or board.is_repetition(3)):
        return 0.0
    return None


# ALGO LOGIC: initialize agent here:
class StrategyNetwork(nn.Module):
    """A linear evaluation over interpretable strategic features, squashed into a value in [-1, 1]."""

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(NUM_FEATURES, 1)

    def forward(self, x):
        return torch.tanh(self.linear(x)).squeeze(-1)


class Searcher:
    """Negamax alpha-beta search with a numpy copy of the strategy weights."""

    def __init__(self, weights: np.ndarray, bias: float):
        self.weights = weights
        self.bias = bias

    def evaluate(self, board: chess.Board) -> float:
        return float(np.tanh(extract_features(board) @ self.weights + self.bias))

    def negamax(self, board: chess.Board, depth: int, alpha: float, beta: float) -> float:
        value = terminal_value(board)
        if value is not None:
            return value
        if depth == 0:
            return self.evaluate(board)
        for move in board.legal_moves:
            board.push(move)
            score = -self.negamax(board, depth - 1, -beta, -alpha)
            board.pop()
            if score >= beta:
                return score
            alpha = max(alpha, score)
        return alpha

    def best_move(self, board: chess.Board, depth: int) -> chess.Move:
        moves = list(board.legal_moves)
        random.shuffle(moves)  # break ties randomly
        if depth == 1:
            # batch the leaf evaluations of a one-ply search
            scores = np.empty(len(moves), dtype=np.float32)
            leaves, leaf_idx = [], []
            for i, move in enumerate(moves):
                board.push(move)
                value = terminal_value(board)
                if value is None:
                    leaves.append(extract_features(board))
                    leaf_idx.append(i)
                else:
                    scores[i] = -value
                board.pop()
            if leaves:
                scores[leaf_idx] = -np.tanh(np.stack(leaves) @ self.weights + self.bias)
            return moves[int(np.argmax(scores))]
        best_move, alpha = moves[0], -2.0
        for move in moves:
            board.push(move)
            score = -self.negamax(board, depth - 1, -2.0, -alpha)
            board.pop()
            if score > alpha:
                best_move, alpha = move, score
        return best_move


def linear_schedule(start_e: float, end_e: float, duration: int, t: int):
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)


def lambda_returns(values: np.ndarray, final_value: float, gamma: float, td_lambda: float) -> np.ndarray:
    """TD(lambda) targets for alternating-perspective values: every value belongs to the side to move,
    so the next state's value is negated when it is backed up."""
    returns = np.empty_like(values)
    next_return, next_value = final_value, final_value
    for t in reversed(range(len(values))):
        returns[t] = -gamma * ((1 - td_lambda) * next_value + td_lambda * next_return)
        next_return, next_value = returns[t], values[t]
    return returns


def play_match(searcher: Searcher, opponent, games: int, max_plies: int, depth: int) -> float:
    """Average score (win=1, draw=0.5, loss=0) of `searcher` against `opponent`, alternating colors."""
    score = 0.0
    for game in range(games):
        board = chess.Board()
        agent_color = chess.WHITE if game % 2 == 0 else chess.BLACK
        while not board.is_game_over(claim_draw=True) and board.ply() < max_plies:
            if board.turn == agent_color:
                board.push(searcher.best_move(board, depth))
            else:
                board.push(opponent(board))
        if board.is_checkmate():
            score += float(board.turn != agent_color)
        else:
            score += 0.5
    return score / games


def export_strategy(model: StrategyNetwork, path: str, games: int):
    weights = model.linear.weight.detach().cpu().numpy()[0]
    strategy = {
        "algorithm": "td_lambda_chess",
        "games": games,
        "features": FEATURE_NAMES,
        "weights": [round(float(w), 5) for w in weights],
        "bias": round(float(model.linear.bias.item()), 5),
    }
    with open(path, "w") as f:
        json.dump(strategy, f, indent=2)


if __name__ == "__main__":
    args = tyro.cli(Args)
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            save_code=True,
        )
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    model = StrategyNetwork()
    with torch.no_grad():
        model.linear.weight.zero_()
        model.linear.bias.zero_()
        if args.material_prior:
            for i, piece in enumerate([chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN]):
                model.linear.weight[0, i] = 0.1 * PIECE_VALUES[piece]
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    material_weights = np.zeros(NUM_FEATURES, dtype=np.float32)
    material_weights[:5] = [0.1, 0.3, 0.3, 0.5, 0.9]
    material_searcher = Searcher(material_weights, 0.0)

    def random_opponent(board):
        return random.choice(list(board.legal_moves))

    def material_opponent(board):
        return material_searcher.best_move(board, args.search_depth)

    start_time = time.time()
    total_plies = 0
    for game in range(1, args.total_games + 1):
        epsilon = linear_schedule(args.start_e, args.end_e, args.exploration_fraction * args.total_games, game)
        searcher = Searcher(model.linear.weight.detach().numpy()[0].copy(), model.linear.bias.item())

        # ALGO LOGIC: play one self-play game
        board = chess.Board()
        features = []
        while True:
            final_value = terminal_value(board)
            if final_value is not None or board.ply() >= args.max_plies:
                break
            features.append(extract_features(board))
            if board.ply() < args.random_opening_plies or random.random() < epsilon:
                move = random.choice(list(board.legal_moves))
            else:
                move = searcher.best_move(board, args.search_depth)
            board.push(move)
        truncated = final_value is None
        if truncated:
            final_value = searcher.evaluate(board)
        total_plies += len(features)

        # ALGO LOGIC: training.
        obs = torch.tensor(np.stack(features))
        values = model(obs)
        returns = torch.tensor(lambda_returns(values.detach().numpy(), final_value, args.gamma, args.td_lambda))
        loss = F.mse_loss(values, returns)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # TRY NOT TO MODIFY: record results for plotting purposes
        outcome = board.outcome()
        white_score = 0.5 if outcome is None or outcome.winner is None else float(outcome.winner)
        writer.add_scalar("charts/game_length", board.ply(), game)
        writer.add_scalar("charts/white_score", white_score, game)
        writer.add_scalar("charts/decisive", float(outcome is not None and outcome.winner is not None), game)
        writer.add_scalar("charts/truncated", float(truncated), game)
        writer.add_scalar("charts/epsilon", epsilon, game)
        writer.add_scalar("losses/td_loss", loss.item(), game)
        writer.add_scalar("losses/values", values.mean().item(), game)
        writer.add_scalar("charts/SPS", int(total_plies / (time.time() - start_time)), game)
        if game % 10 == 0:
            print(f"game={game}, plies={board.ply()}, result={board.result()}, td_loss={loss.item():.4f}")
            for name, weight in zip(FEATURE_NAMES, model.linear.weight.detach().numpy()[0]):
                writer.add_scalar(f"strategy/{name}", weight, game)

        if game % args.eval_frequency == 0 or game == args.total_games:
            searcher = Searcher(model.linear.weight.detach().numpy()[0].copy(), model.linear.bias.item())
            score_vs_random = play_match(searcher, random_opponent, args.eval_games, args.max_plies, args.search_depth)
            score_vs_material = play_match(searcher, material_opponent, args.eval_games, args.max_plies, args.search_depth)
            print(f"game={game}, eval/score_vs_random={score_vs_random:.2f}, eval/score_vs_material={score_vs_material:.2f}")
            writer.add_scalar("eval/score_vs_random", score_vs_random, game)
            writer.add_scalar("eval/score_vs_material", score_vs_material, game)

    pawn_weight = model.linear.weight[0, 0].item()
    print("learned strategy weights (in pawns):")
    for name, weight in zip(FEATURE_NAMES, model.linear.weight.detach().numpy()[0]):
        print(f"  {name:>15}: {weight / pawn_weight:+.2f}")

    if args.save_model:
        model_path = f"runs/{run_name}/{args.exp_name}.cleanrl_model"
        torch.save(model.state_dict(), model_path)
        print(f"model saved to {model_path}")
        strategy_path = f"runs/{run_name}/strategy.json"
        export_strategy(model, strategy_path, args.total_games)
        print(f"strategy weights exported to {strategy_path} (load them in cleanrl_utils/chess_strategy_board.html)")

    writer.close()
