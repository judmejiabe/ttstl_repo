import numpy as np
import statsmodels.api as sm
from statsmodels.regression.linear_model import OLS

rng = np.random.default_rng(123)

def simulate(n=1500, p=3):
    X = rng.normal(size=(n, p))
    X = sm.add_constant(X)
    beta = np.array([1.0, 0.8, 0.0, -0.5][:p+1])
    sigma = 0.5 + 0.7 * np.abs(X[:, 2] if p >= 2 else X[:, 1])
    u = sigma * rng.normal(size=n)
    y = X @ beta + u
    return y, X

def selector_matrix(k, idx):
    R = np.zeros((len(idx), k))
    for r, j in enumerate(idx):
        R[r, j] = 1.0
    return R

def main():
    y, X = simulate()
    res = OLS(y, X).fit(cov_type="HC0")

    beta_hat = res.params
    V = res.cov_params()
    k = beta_hat.size

    # Joint test: H0: x2 = 0 and x3 = 0
    J = [2, 3]
    R = selector_matrix(k, J)
    r = np.zeros(len(J))
    q = len(J)

    # Manual Wald
    diff = R @ beta_hat - r
    W_manual = float(diff.T @ np.linalg.inv(R @ V @ R.T) @ diff)

    # statsmodels Wald (χ²) and F with scalar outputs (no FutureWarning)
    wald_chi = res.wald_test((R, r), use_f=False, scalar=True)
    wald_F   = res.wald_test((R, r), use_f=True,  scalar=True)

    # Now these are true Python floats (no NumPy scalar coercion warnings)
    W_sm   = wald_chi.statistic         # float
    F_sm   = wald_F.statistic           # float (F-stat)
    W_name = res.wald_test("x2 = 0, x3 = 0", use_f=False, scalar=True).statistic

    # Checks
    assert np.allclose(W_manual, W_sm,   rtol=1e-10, atol=1e-10)
    assert np.allclose(W_manual / q, F_sm, rtol=1e-12, atol=1e-12)
    assert np.allclose(W_manual, W_name, rtol=1e-12, atol=1e-12)

    # Single restriction: H0: x2 = 0  => W = t^2
    j = 2
    a = np.zeros(k); a[j] = 1.0
    W_single = float((a @ beta_hat)**2 / (a @ V @ a))
    t_j = res.tvalues[j]
    assert np.allclose(W_single, t_j**2, rtol=1e-12, atol=1e-12)

    print("All Wald (χ²) and F tests match manual computations under HC0, no warnings.")

if __name__ == "__main__":
    main()