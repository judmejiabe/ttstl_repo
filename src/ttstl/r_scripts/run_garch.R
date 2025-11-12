library(rugarch)
library(WeightedPortTest)
library(ggplot2)
# library(forecast)  # For ggAcf and ggPacf functions - not available
# library(gridExtra) # For grid.arrange function - not available

run_garch <- 
    function(y, X = NULL, Z = NULL, model = 'sGARCH',
             p = 1, q = 1, m = 1, s = 1, 
             archm = FALSE, dist = 'norm',
             out_sample = 0,
             start_pars = list(),
             display = TRUE,
             plot_label = '') {
    # Specify model with exogenous variables
    spec <- ugarchspec(variance.model = list(model = model, 
                                            garchOrder = c(m, s),
                                            external.regressors = Z),
                        mean.model = list(armaOrder = c(p, q), 
                                        include.mean = TRUE, 
                                        external.regressors = X,
                                        archm = archm),
                        distribution.model = dist,
                        start.pars = start_pars)

    # Fit the model
    fit <- ugarchfit(spec, y, solver = 'hybrid', out.sample = out_sample)

    if(is.nan(fit@fit$condH)) {
        print("Hybrid solver converged to a saddlepoint. Refitting with nloptr.")
        fit <- ugarchfit(spec, y, solver = 'nloptr', 
                         solver.control = list(trace = 1), out.sample = out_sample)
    }
    
    # if(is.nan(fit@fit$condH)) {
    #     print("Saddlepoint condition failed to converge. Refitting with nlminb solver.")
    #     fit <- ugarchfit(spec, y, solver = 'nlminb', 
    #                      solver.control = list(trace = 1), out.sample = out_sample)
    # }

    std_res = residuals(fit, standardize = TRUE)
    std_res_sq = std_res^2

    if(display) {
        # Open a PNG device to save the plots
        png(paste0(plot_label, "_garch_fit_plots.png"), width = 800, height = 600)

        # Plot and save the fit object
        plot(fit, which = 8)

        # Close the device to save the image
        dev.off()

        png(paste0(plot_label, "_garch_fit_residuals_(p)acf.png"), width = 800, height = 600)
        # Comment out ggAcf/ggPacf functions as forecast package is not available
        # acf_std_res <- ggAcf(std_res) + ggtitle("ACF of Standardized Residuals")
        # pacf_std_res <- ggPacf(std_res) + ggtitle("PACF of Standardized Residuals")
        # acf_std_res_sq <- ggAcf(std_res_sq) + ggtitle("ACF of Squared Standardized Residuals")
        # pacf_std_res_sq <- ggPacf(std_res_sq) + ggtitle("PACF of Squared Standardized Residuals")
        
        # Use base R ACF/PACF plots instead
        par(mfrow = c(2, 2))
        acf(std_res, main = "ACF of Standardized Residuals")
        pacf(std_res, main = "PACF of Standardized Residuals")
        acf(std_res_sq, main = "ACF of Squared Standardized Residuals")
        pacf(std_res_sq, main = "PACF of Squared Standardized Residuals")

        # grid.arrange(acf_std_res, pacf_std_res, acf_std_res_sq, pacf_std_res_sq, ncol = 2)

        # Close the device to save the image
        dev.off()
    }
    
    stdresid <- residuals(fit, standardize = TRUE)

    # Use the weighted Box test function from rugarch-tests.R

    lb_1 <- Weighted.Box.test(stdresid, lag = 1, type = "Ljung-Box")
    lb_sq_1 <- Weighted.Box.test(stdresid, lag = 1, type = "Ljung-Box", 
                                 sqrd.res = TRUE)

    # Handle the case when p + q = 0 (no ARMA terms)
    if (p + q == 0) {
        lb_2 <- list(p.value = 1.0)
        lb_3 <- list(p.value = 1.0)
    } else {
        lb_2 <- Weighted.Box.test(stdresid, lag = 3 * (p + q) - 1, 
                                  fitdf = p + q, type = "Ljung-Box")
        lb_3 <- Weighted.Box.test(stdresid, lag = 5 * (p + q) - 1, 
                                  fitdf = p + q, type = "Ljung-Box")
    }
    
    # Handle the case when m + s = 0 (no GARCH terms) - though this is unlikely
    if (m + s == 0) {
        lb_sq_2 <- list(p.value = 1.0)
        lb_sq_3 <- list(p.value = 1.0)
    } else {
        lb_sq_2 <- Weighted.Box.test(stdresid, lag = 3 * (m + s) - 1, 
                                     fitdf = m + s, type = "Ljung-Box", 
                                     sqrd.res = TRUE)
        lb_sq_3 <- Weighted.Box.test(stdresid, lag = 5 * (m + s) - 1, 
                                     fitdf = m + s, type = "Ljung-Box", 
                                     sqrd.res = TRUE)
    }

    return(list(fit = fit, 
                coef = fit@fit$coef, 
                se_coef = fit@fit$se.coef,
                pvals = fit@fit$matcoef[, 4],
                gof = gof(fit, groups = c(20, 30, 40, 50))[, 3],
                lb = c(lb_1$p.value, lb_2$p.value, lb_3$p.value),
                lb_sq = c(lb_sq_1$p.value, lb_sq_2$p.value, lb_sq_3$p.value),
                coef_names = names(fit@fit$coef),
                stdresid = stdresid,
                saddle = ifelse(is.nan(fit@fit$condH), 1, 0),
                spec = spec))
}

set_fixed_parameters <- function(spec, fixed_params) {
  setfixed(spec) <- fixed_params
  return(spec)
}