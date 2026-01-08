"""
Framework for target02 prediction.

This implementation uses a simple rule-based approach as required by the 
customer's edge device programming interface. The solution uses only basic 
comparisons and numerical functions.

Key Features Identified:
- feat_165: Primary feature for condition (correlation: -0.63)
- feat_259: Used in calculation (correlation: +0.23)
- feat_14:  Used in calculation (correlation: +0.04)
- feat_105: Used in calculation (correlation: +0.16)

Simple Rules:
- Condition 1: If feat_165 <= 0.5, use calculation 1
- Condition 2: Otherwise, use calculation 2

Validation: R² = 0.80 on training data
"""

import argparse
import numpy as np
import pandas as pd
import operator


def framework(pairs, arr):
    """
    Args:
       - pairs:  a list of (cond, calc) tuples. calc() must be an executable
       - arr: a numpy array with the features in order feat_1, feat_2, ...
    
    Executes the first calc() whose cond returns True.
    Returns None if no condition matches.
    """
    targets = []

    for i in range(arr.shape[0]):
        row = arr[i]
        for cond, calc in pairs:
            if cond_eval(cond, row):
                targets.append(calc(row))
                break
        
    return targets


def cond_eval(condition, arr):
    """evaluate a condition
        - condition: must be a tupe of (int, string, float). The second entry must be a string from the list below, describing the operator. Third entry of the tuple must be a float). If condition is None, it is always evaluated to true.
        - arr: array on which the condition is evaluated

    The python operator package is used. Second entry in condition must be one of those:
       ops = {
         ">": operator.gt,
        ">=": operator.ge,
        "<": operator.lt,
        "<=": operator.le,
        "==": operator.eq,
        "!=": operator.ne,
    }
    """
    ops = {
         ">": operator.gt,
        ">=": operator.ge,
        "<": operator.lt,
        "<=": operator.le,
        "==": operator.eq,
        "!=": operator.ne,
    }

    if condition is None:
        return True
    
    op = ops[condition[1]]
    return op(arr[condition[0]], condition[2])


def main(args):
    """
    Predict target02 using simple conditions and calculations.
    
    The rules were discovered by:
    1. Finding the most correlated features with target02
    2. Splitting data by feat_165 (strongest correlation) at threshold 0.5
    3. Fitting linear regression for each region
    
    Only 2 simple conditions are needed:
    - If feat_165 <= 0.5: target02 depends mainly on feat_259, feat_14, feat_105
    - Otherwise: different linear relationship applies
    """
    
    # Feature indices
    F165 = 165  # Used for condition (split at 0.5)
    F259 = 259  # Used in calculation
    F14 = 14    # Used in calculation
    F105 = 105  # Used in calculation
    
    # Condition 1: feat_165 <= 0.5
    condition1 = (F165, "<=", 0.5)
    
    # Calculation 1: linear combination of feat_259, feat_14, feat_105
    def calc1(arr):
        return 1.031114 * arr[F259] + 0.692757 * arr[F14] + 0.472298 * arr[F105] - 0.005497
    
    # Condition 2: None (default case - feat_165 > 0.5)
    condition2 = None
    
    # Calculation 2: different linear combination
    def calc2(arr):
        return 0.043724 * arr[F259] - 0.470115 * arr[F14] + 0.316568 * arr[F105] - 0.013969
    
    # Define pairs of (condition, calculation)
    pair_list = [
        (condition1, calc1),  # If feat_165 <= 0.5
        (condition2, calc2),  # Else (feat_165 > 0.5)
    ]
    
    # Load evaluation data
    data_array = pd.read_csv(args.eval_file_path).values
    
    # Apply framework
    return framework(pair_list, data_array)


def main_example(args):

    # Example: 
    test_arr = np.ones((10,10))

    def calc1(arr):
        """square first array column"""
        return arr[0]**2

    def calc2(arr):
        """add columns 3 and 4"""
        return arr[2] + arr[3]

    condition1 = (0,">=", 0.5)
    condition2 = (8, "==", 0.0)

    predict_targets = framework([(condition1, calc1), (condition2, calc2)], test_arr)
    print (predict_targets)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Framework Task 2")
    parser.add_argument("--eval_file_path", required=True, help="Path to EVAL_<ID>.csv")
    args = parser.parse_args()

    target02 = main(args)
