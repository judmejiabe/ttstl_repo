import numpy as np
import statsmodels.api as sm
from statsmodels.stats.sandwich_covariance import cov_hac

rng = np.random.default_rng(123)

def simulate(T=800, k=3, rho=0.6):
    X = rng.normal(size=(T, k))
    X = sm.add_constant(X)  # include intercept
    beta = np.linspace(0.5, 1.5, X.shape[1])
    eps = np.zeros(T)
    eta = rng.normal(size=T)
    for t in range(1, T):
        eps[t] = rho*eps[t-1] + eta[t]  # AR(1) errors
    y = X @ beta + eps
    return y, X

def hac_manual(res, L=1, use_correction=True):
    """
    Manual Newey-West (Bartlett) with bandwidth L.
    Matches statsmodels' S_hac_simple and cov_hac_simple.
    """
    X = res.model.exog
    u = res.resid
    Z = X * u[:, None]  # z_t = x_t * u_t

    # Bartlett weights w_l = 1 - l/(L+1), including w_0 = 1
    w = 1.0 - np.arange(L+1) / (L+1.0)

    # Build S_hat
    S = w[0] * (Z.T @ Z)
    for lag in range(1, L + 1):
        s = Z[lag:].T @ Z[:-lag]
        S += w[lag] * (s + s.T)

    # Bread = (X'X)^(-1)
    bread = np.linalg.inv(X.T @ X)
    V = bread @ S @ bread

    if use_correction:
        nobs = X.shape[0]
        k_params = X.shape[1]
        V *= nobs / float(nobs - k_params)

    return V

def main():
    y, X = simulate()
    ols = sm.OLS(y, X).fit()

    # 1) statsmodels "fit(cov_type='HAC')" result
    ols_hac = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": 0}, use_correction=False)
    V1 = ols_hac.cov_params()

    # 2) statsmodels direct cov_hac utility
    V2 = cov_hac(ols, nlags=0, use_correction=False)  # default Bartlett, default small-sample correction

    # 3) Manual formula
    V3 = hac_manual(ols, L=0, use_correction=False)

    ols_hc0 = sm.OLS(y, X).fit(cov_type="HC0")
    V4 = ols_hc0.cov_params()
    print(ols_hc0.summary())
    print(ols_hc0.pvalues)

    def check(A, B, nameA, nameB, atol=1e-4, rtol=1e-6):
        if not np.allclose(A, B, atol=atol, rtol=rtol):
            diff = np.abs(A - B).max()
            raise AssertionError(f"{nameA} != {nameB}; max abs diff = {diff:e}")
        
    print(V1)
    print(V2)
    print(V3)
    print(V4)

    check(V1, V2, "fit(cov_type='HAC')", "cov_hac(nlags=1)")
    check(V1, V3, "fit(cov_type='HAC')", "manual HAC L=1")
    check(V1, V4, "fit(cov_type='HAC')", "fit(cov_type='HC0')")

    print("All HC0 tests passed.")

if __name__ == "__main__":
    main()