library(glmnet)
library(rugarch)

# 4) Define adaptive LASSO wrapper
run_adalasso <- function(y, X, alpha_enet, lambda_enet,
                         tau_adalasso, lambda_adalasso,
                         garchOrder = c(1,1),
                         distribution.model = 'sstd') {
  # 4.1 Elastic‑Net initial fit
  enet_fit <- glmnet(x = X, y = y,
                     alpha = alpha_enet,
                     lambda = lambda_enet,
                     intercept = TRUE,
                     standardize = FALSE)
  # 4.2 Form penalty weights
  coef_enet <- as.numeric(coef(enet_fit))
  beta_init <- coef_enet[-1]
  eps       <- 1e-6
  w         <- 1 / (abs(beta_init) + eps)^tau_adalasso
  
  # 4.3 Adaptive LASSO fit
  ada_fit <- glmnet(x = X, y = y,
                    alpha = 1,
                    lambda = lambda_adalasso,
                    penalty.factor = w,
                    intercept = TRUE,
                    standardize = FALSE)
  # 4.4 Residuals
  y_hat <- as.numeric(predict(ada_fit, newx = X, s = lambda_adalasso))
  resid  <- y - y_hat
  
  # 4.5 Fit GARCH on residuals
  spec_resid <- ugarchspec(
    variance.model     = list(model = "sGARCH", garchOrder = garchOrder),
    mean.model         = list(armaOrder = c(0,0), include.mean = FALSE),
    distribution.model = distribution.model
  )
  fit_resid <- ugarchfit(spec = spec_resid, data = resid, solver = "hybrid")
  
  # 4.6 Compute AIC/BIC
  llh        <- fit_resid@fit$LLH
  k_adalasso <- sum(abs(coef(ada_fit, s = lambda_adalasso)[-1,1]) > eps)
  k_garch    <- length(coef(fit_resid))
  n          <- length(y)
  k_total    <- k_adalasso + k_garch
  AIC <- 2 * k_total - 2 * llh
  BIC <- k_total * log(n) - 2 * llh
  
  list(ada_fit = ada_fit,
       garch_fit = fit_resid,
       enet_coef = as.numeric(coef_enet),
       penalty_weights = w,
       ada_coef = as.numeric(coef(ada_fit, s = lambda_adalasso)),
       AIC = AIC,
       BIC = BIC)
}