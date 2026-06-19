"""
Quick comparison between two experiment versions.
Usage: python compare_experiments.py results/VERSION1.json results/VERSION2.json
"""

import json
import sys
import statistics
from pathlib import Path

def load_results(filepath):
    """Load results from JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)

def compare_versions(v1, v2):
    """Compare two versions and print analysis."""
    print(f"\n{'='*100}")
    print(f"EXPERIMENT COMPARISON")
    print(f"{'='*100}\n")

    print(f"Version 1: {Path(v1).name}")
    print(f"Version 2: {Path(v2).name}\n")

    v1_data = load_results(v1)
    v2_data = load_results(v2)

    print(f"Model 1: {v1_data.get('description', 'N/A')}")
    print(f"Model 2: {v2_data.get('description', 'N/A')}\n")

    # Collect metrics
    comparisons = []
    v1_wins = 0
    v2_wins = 0
    ties = 0

    print(f"{'Dataset':<30} | {'V1 Acc':<10} | {'V2 Acc':<10} | {'Improvement':<15} | {'Winner':<10}")
    print("-" * 100)

    for dataset in sorted(v1_data['datasets'].keys()):
        if dataset not in v2_data['datasets']:
            print(f"{dataset:<30} | {'SKIP':<10} | {'-':<10} | {'-':<15} | {'(missing)':<10}")
            continue

        v1_acc = v1_data['datasets'][dataset]['comparison'].get('pdgnet_acc')
        v2_acc = v2_data['datasets'][dataset]['comparison'].get('pdgnet_acc')

        if v1_acc is None or v2_acc is None:
            status = "FAIL" if v1_acc is None else "OK"
            print(f"{dataset:<30} | {str(v1_acc):<10} | {str(v2_acc):<10} | {'N/A':<15} | {status:<10}")
            continue

        improvement = v2_acc - v1_acc
        improvement_pct = improvement * 100

        comparisons.append({
            'dataset': dataset,
            'v1': v1_acc,
            'v2': v2_acc,
            'improvement': improvement_pct
        })

        if improvement > 0.001:
            winner = "V2 ✓"
            v2_wins += 1
        elif improvement < -0.001:
            winner = "V1 ✓"
            v1_wins += 1
        else:
            winner = "TIE"
            ties += 1

        print(f"{dataset:<30} | {v1_acc:<10.4f} | {v2_acc:<10.4f} | {improvement_pct:>+13.2f}% | {winner:<10}")

    print("-" * 100)

    # Summary statistics
    if comparisons:
        improvements = [c['improvement'] for c in comparisons]
        avg_improvement = statistics.mean(improvements)
        std_improvement = statistics.stdev(improvements) if len(improvements) > 1 else 0

        print(f"\n{'SUMMARY':<30} | {'':<10} | {'':<10} | {'':<15} | {'':<10}")
        print(f"Number of datasets tested: {len(comparisons)}")
        print(f"Version 2 wins: {v2_wins}")
        print(f"Version 1 wins: {v1_wins}")
        print(f"Ties: {ties}")
        print(f"\nAverage improvement: {avg_improvement:+.2f}%")
        print(f"Std deviation: {std_improvement:.2f}%")
        print(f"Max improvement: {max(improvements):+.2f}% ({comparisons[improvements.index(max(improvements))]['dataset']})")
        print(f"Min improvement: {min(improvements):+.2f}% ({comparisons[improvements.index(min(improvements))]['dataset']})")

        # Verdict
        print(f"\n{'='*100}")
        if avg_improvement > 0.5:
            print(f"VERDICT: Version 2 is BETTER overall (+{avg_improvement:.2f}%)")
        elif avg_improvement > 0:
            print(f"VERDICT: Version 2 is SLIGHTLY BETTER (+{avg_improvement:.2f}%)")
        elif avg_improvement > -0.5:
            print(f"VERDICT: Version 1 is SLIGHTLY BETTER ({avg_improvement:.2f}%)")
        else:
            print(f"VERDICT: Version 1 is BETTER overall ({avg_improvement:.2f}%)")
        print(f"{'='*100}\n")

        # Detailed recommendations
        print("RECOMMENDATIONS:")
        print("-" * 100)

        # Find best and worst changes
        improvements_sorted = sorted(comparisons, key=lambda x: x['improvement'], reverse=True)

        print(f"\nTop improvements:")
        for comp in improvements_sorted[:3]:
            print(f"  {comp['dataset']:30s}: {comp['v1']:.4f} -> {comp['v2']:.4f} ({comp['improvement']:+.2f}%)")

        print(f"\nTop regressions:")
        for comp in improvements_sorted[-3:]:
            if comp['improvement'] < 0:
                print(f"  {comp['dataset']:30s}: {comp['v1']:.4f} -> {comp['v2']:.4f} ({comp['improvement']:+.2f}%)")

        # Which datasets favor which version
        v2_favored = [c['dataset'] for c in comparisons if c['improvement'] > 2]
        v1_favored = [c['dataset'] for c in comparisons if c['improvement'] < -2]

        if v2_favored:
            print(f"\nDatasets where V2 excels (>+2%):")
            for ds in v2_favored:
                print(f"  - {ds}")

        if v1_favored:
            print(f"\nDatasets where V1 is preferred (<-2%):")
            for ds in v1_favored:
                print(f"  - {ds}")

    print("\n" + "="*100 + "\n")

def main():
    if len(sys.argv) < 3:
        print("Usage: python compare_experiments.py <version1.json> <version2.json>")
        print("\nExample:")
        print("  python compare_experiments.py results/20260619_100000_comparison.json results/20260619_110000_comparison.json")
        sys.exit(1)

    v1_path = sys.argv[1]
    v2_path = sys.argv[2]

    if not Path(v1_path).exists():
        print(f"Error: {v1_path} not found")
        sys.exit(1)

    if not Path(v2_path).exists():
        print(f"Error: {v2_path} not found")
        sys.exit(1)

    compare_versions(v1_path, v2_path)

if __name__ == '__main__':
    main()
