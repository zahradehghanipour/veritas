"""
RL Verification Evaluation Phases
	1.	Training - Expert Imitation (FSB for now)
	2.	Training - DQN

	3.	Testing - DQN
	4.	Testing - Expert (FSB)
	5.	Testing - Comparison

Usage:
Run the script with a YAML configuration:
    e.g. : python rl_experiments/src/rl-experiment.py --config rl_experiments/configs/acasxu_RL_full.yaml
"""
import argparse
from base import *

def main():
    # Argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)

    args_cli = parser.parse_args()
    initialization(args_cli)

    # Phase 1: Expert training
    train(expert=True)

    # Phase 2: RL DQN training
    train(expert=False)

    # Phase 3: Testing the test set
    test()

    # Phase 4: Testing with expert
    testother('fsb')

    # # Phase 5: Compare with kfsb
    # testother('kfsb')

    # Plots
    plots()


if __name__ == "__main__":
    main()