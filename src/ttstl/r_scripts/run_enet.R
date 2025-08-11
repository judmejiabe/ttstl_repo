library(glmnet)

run_enet <- function(X, y, alpha_enet, lambda_enet) {
  
  # Fit elastic net model
  fit <- glmnet(X, 
                y, 
                alpha = alpha_enet, 
                lambda = lambda_enet,
                intercept = TRUE,
                standardize = FALSE)
  
  list(fit = fit,
       coefs = as.numeric(coef(fit)))
}