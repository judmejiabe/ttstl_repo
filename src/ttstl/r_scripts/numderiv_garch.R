library(rugarch)
library(numDeriv)

numderiv_garch <- function(spec, y, theta, rec.init = "all", numderiv.control = list()) {

  theta_vec <- if (is.list(theta)) unlist(theta) else theta
  stopifnot(is.numeric(theta_vec), !is.null(names(theta_vec)))
  
  # Helper to pull a control value or a default
  pick <- function(x, default) if (is.null(x)) default else x
  
  # Build Richardson method args for numDeriv (align with rugarch defaults)
  grad_args <- list(
    eps      = pick(numderiv.control$grad.eps,      1e-4),
    d        = pick(numderiv.control$grad.d,        1e-4),
    zero.tol = pick(numderiv.control$grad.zero.tol, sqrt(.Machine$double.eps/7e-7)),
    r        = pick(numderiv.control$r,             4),
    v        = pick(numderiv.control$v,             2)
  )
  hess_args <- list(
    eps      = pick(numderiv.control$hess.eps,      1e-4),
    d        = pick(numderiv.control$hess.d,        0.1),
    zero.tol = pick(numderiv.control$hess.zero.tol, sqrt(.Machine$double.eps/7e-7)),
    r        = pick(numderiv.control$r,             4),
    v        = pick(numderiv.control$v,             2)
  )
  
  # 1) Fit once (no parameter movement because all are fixed) so we can reuse
  #    rugarch’s Hessian/SE machinery if you want; stationarity=0 avoids clipping
  sp <- spec
  setfixed(sp) <- theta_vec
  fit_theta <- ugarchfit(
    sp, data = y, solver = "hybrid",
    fit.control      = list(rec.init = rec.init, fixed.se = 1, stationarity = 0),
    numderiv.control = numderiv.control
  )
  
  # 2) Define NEGATIVE log-likelihood at arbitrary theta (same objective we care about)
  llk_neg <- function(par_vec, spec, data, rec.init) {
    sp2 <- spec
    setfixed(sp2) <- as.list(par_vec)
    f <- ugarchfilter(sp2, data = data, rec.init = rec.init)
    -likelihood(f)
  }
  
  # 3) Gradient via numDeriv (Richardson) at 'theta' for NEGATIVE log-likelihood
  grad_num <- numDeriv::grad(
    func = llk_neg,
    x    = theta_vec,
    spec = spec, data = y, rec.init = rec.init,
    method = "Richardson",
    method.args = grad_args
  )
  # make sure names are preserved
  names(grad_num) <- names(theta_vec)


  S <- fit_theta@fit$scores           # T x p, rows are g_t for -logLik
  Tn <- nrow(S)
  
  # (1) OPG (uncentered, denom T)
  B_opg <- crossprod(S) / Tn          # = t(S) %*% S / T
  #B_opg <- 0
  
  list(
    grad         = - grad_num,                  # ∇ℓ from numDeriv (Richardson)
    hessian      = - fit_theta@fit$hessian,     # ∇^2(ℓ) from rugarch (Richardson)
    I            = B_opg,                       # Approximate Fischer Info. Matrix
    fit_object   = fit_theta                    # rugarch fit object
  )
}