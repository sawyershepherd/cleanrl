import subprocess


def test_td_lambda_chess():
    subprocess.run(
        "python cleanrl/td_lambda_chess.py --total-games 4 --max-plies 40 --eval-frequency 2 --eval-games 2 --save-model",
        shell=True,
        check=True,
    )
