#!/usr/bin/env python
# Run All Experiments
# Reproduces all figures and tables from the paper:
# "Intrinsic Smoothness Barriers for Optimization via the Matrix Exponential on Lie Algebras"
# Usage:
#     python experiments/run_all.py [--quick]
# Options:
#     --quick     Run with reduced samples for testing (~10 min instead of ~1 hours)

import argparse
import subprocess
import sys
from pathlib import Path


EXPERIMENTS = [
    ("adversarial_sampling.py", "Figure 1, Table 1: Adversarial vs Random Sampling"),
    ("hard_direction.py", "Figure 2, Table 2: Lower Bound Verification"),
    ("local_smoothness.py", "Figure 3: Local vs Global Smoothness"),
    ("convergence_burnin.py", "Figure SM1: Convergence Analysis"),
    ("dimension_scaling.py", "Table SM1: Dimension Scaling"),
    ("cayley_comparison.py", "Figure SM2, Table SM2: Cayley vs Exponential"),
]


def main():
    parser = argparse.ArgumentParser(description="Run all experiments")
    parser.add_argument("--quick", action="store_true", 
                        help="Quick run with reduced samples")
    args = parser.parse_args()
    
    # Get experiment directory
    exp_dir = Path(__file__).parent
    
    print("=" * 70)
    print("Running All Experiments")
    print("=" * 70)
    
    if args.quick:
        print("MODE: Quick (reduced samples)")
    else:
        print("MODE: Full (this may take 2-4 hours)")
    print()
    
    for script, description in EXPERIMENTS:
        print("-" * 70)
        print(f"Running: {description}")
        print(f"Script:  {script}")
        print("-" * 70)
        
        cmd = [sys.executable, str(exp_dir / script)]
        if args.quick:
            cmd.append("--quick")
        
        try:
            result = subprocess.run(cmd, check=True)
            print(f"✓ Completed: {script}\n")
        except subprocess.CalledProcessError as e:
            print(f"✗ Failed: {script} (exit code {e.returncode})\n")
            continue
        except FileNotFoundError:
            print(f"✗ Not found: {script}\n")
            continue
    
    print("=" * 70)
    print("All experiments complete!")
    print("Figures saved to: figures/")
    print("Results saved to: results/")
    print("=" * 70)


if __name__ == "__main__":
    main()
