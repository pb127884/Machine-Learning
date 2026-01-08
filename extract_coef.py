import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

y_df = pd.read_csv('target_91.csv')
X = pd.read_csv('dataset_91.csv')
y = y_df['target02'].values
f165, f259, f14, f105 = X['feat_165'].values, X['feat_259'].values, X['feat_14'].values, X['feat_105'].values

# Get exact coefficients for all 4 regions
coeffs = {}
for label, low, high in [('Q1', 0.0, 0.25), ('Q2', 0.25, 0.5), ('Q3', 0.5, 0.75), ('Q4', 0.75, 1.01)]:
    if low == 0.0:
        mask = f165 <= high
    elif high > 1.0:
        mask = f165 > low
    else:
        mask = (f165 > low) & (f165 <= high)
    
    Xm = np.column_stack([f259[mask], f14[mask], f105[mask]])
    lr = LinearRegression()
    lr.fit(Xm, y[mask])
    coeffs[label] = [lr.coef_[0], lr.coef_[1], lr.coef_[2], lr.intercept_]
    print(f'{label}: [{lr.coef_[0]:.6f}, {lr.coef_[1]:.6f}, {lr.coef_[2]:.6f}, {lr.intercept_:.6f}]')

# Validate
def predict(i):
    if f165[i] <= 0.25:
        c = coeffs['Q1']
    elif f165[i] <= 0.50:
        c = coeffs['Q2']
    elif f165[i] <= 0.75:
        c = coeffs['Q3']
    else:
        c = coeffs['Q4']
    return c[0] * f259[i] + c[1] * f14[i] + c[2] * f105[i] + c[3]

y_pred = [predict(i) for i in range(len(y))]
r2 = r2_score(y, y_pred)
print(f'\nR2 = {r2:.6f}')
