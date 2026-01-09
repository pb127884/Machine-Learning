# My Data Analysis Methodology

This folder documents **how I analyzed the data** - including the code I used, the methods I tried, and my reasoning at each step.

## My Analysis Steps

| Step | What I Did | Key Finding |
|------|------------|-------------|
| [01_data_understanding.md](01_data_understanding.md) | Loaded data, checked shapes and target | 10K samples, 273 features, right-skewed target |
| [02_data_quality_analysis.md](02_data_quality_analysis.md) | Checked missing, duplicates, types | Clean data (0% missing, 0 duplicates) |
| [03_feature_analysis.md](03_feature_analysis.md) | Analyzed variance, correlations, outliers | Max correlation only 0.16 → need non-linear models |
| [04_model_selection_rationale.md](04_model_selection_rationale.md) | Tested 5 models, compared results | HGB won with 60% R² |
| [05_hyperparameter_decisions.md](05_hyperparameter_decisions.md) | Tuned 2592 combinations | 24% improvement achieved |
| [06_final_conclusions.md](06_final_conclusions.md) | Summary and lessons learned | Final R² = 0.786 |

## What Each Document Contains

- **My actual code** - the exact code I used for analysis
- **My observations** - what I noticed from the results
- **Methods I tried but didn't use** - and why
- **My reasoning** - why I made each decision

## How to Read This

1. Start with `01_data_understanding.md` to see how I first explored the data
2. Follow the steps in order to understand my thought process
3. Each document explains **what code I wrote** and **what I learned from it**
