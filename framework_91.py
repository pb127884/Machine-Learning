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
    Rule-Based Prediction for target02
    ID: 91
    
    Features used:
    - feat_165 (index 165): Primary predictor, correlation = -0.6290, DT importance = 81.6%
    - feat_259 (index 259): Secondary predictor, correlation = +0.2301, DT importance = 9.1%
    - feat_14 (index 14): Tertiary predictor, DT importance = 6.5%
    - feat_105 (index 105): Quaternary predictor, DT importance = 2.9%
    
    Decision tree rules (depth=4, 16 leaf nodes) derived from sklearn DecisionTreeRegressor
    Performance: R² = 0.8757, RMSE = 0.2567
    """
    
    # =========================================================================
    # DECISION TREE CALCULATION (handles all branching logic)
    # Since framework only supports single conditions, we implement the full
    # decision tree logic inside the calculation function
    # =========================================================================
    
    def calc_decision_tree(arr):
        """
        Full decision tree implementation for target02 prediction.
        Uses features: feat_165 (idx 165), feat_259 (idx 259), feat_14 (idx 14), feat_105 (idx 105)
        """
        # Feature values
        f165 = arr[165]
        f259 = arr[259]
        f14 = arr[14]
        f105 = arr[105]
        
        # Decision tree logic
        if f165 <= 0.499972:
            if f259 <= 0.539155:
                if f14 <= 0.488796:
                    if f165 <= 0.201313:
                        return 0.428199   # Leaf 1
                    else:
                        return 0.821636   # Leaf 2
                else:  # f14 > 0.488796
                    if f259 <= 0.282283:
                        return 0.895559   # Leaf 3
                    else:
                        return 1.158007   # Leaf 4
            else:  # f259 > 0.539155
                if f14 <= 0.537704:
                    if f259 <= 0.857818:
                        return 1.126337   # Leaf 5
                    else:
                        return 1.396311   # Leaf 6
                else:  # f14 > 0.537704
                    if f165 <= 0.200357:
                        return 1.869131   # Leaf 7
                    else:
                        return 1.346590   # Leaf 8
        else:  # f165 > 0.499972
            if f165 <= 0.700854:
                if f259 <= 0.462765:
                    if f105 <= 0.502041:
                        return -0.143440  # Leaf 9
                    else:
                        return -0.364609  # Leaf 10
                else:  # f259 > 0.462765
                    if f105 <= 0.495375:
                        return -0.368569  # Leaf 11
                    else:
                        return -0.584090  # Leaf 12
            else:  # f165 > 0.700854
                if f14 <= 0.454465:
                    if f105 <= 0.488271:
                        return 0.169210   # Leaf 13
                    else:
                        return 0.550530   # Leaf 14
                else:  # f14 > 0.454465
                    if f105 <= 0.557603:
                        return -0.241965  # Leaf 15
                    else:
                        return 0.145418   # Leaf 16
    
    # =========================================================================
    # PAIR LIST
    # Using None condition (always True) with decision tree calculation
    # This way the framework will apply our decision tree to every row
    # =========================================================================
    
    pair_list = [
        (None, calc_decision_tree),  # None condition always evaluates to True
    ]
    
    data_array = pd.read_csv(args.eval_file_path).values
    
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
    
    # Save predictions to CSV
    output_df = pd.DataFrame({'target02': target02})
    output_df.to_csv('EVAL_target02_91.csv', index=False)
    print(f"Predictions saved to EVAL_target02_91.csv ({len(target02)} rows)")
