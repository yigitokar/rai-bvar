#!/usr/bin/env Rscript
# =================================================================================================
# make_r_fixtures.R: numerical-oracle fixtures from the CRAN package BVAR (Kuschnig & Vashold)
# =================================================================================================
#
# Writes the JSON fixtures read by tests/parity/test_r_bvar.py (SPEC §10.4):
#
#   tests/parity/fixtures/r_bvar_logml.json      (a) log marginal likelihood at fixed hyperparameters
#   tests/parity/fixtures/r_bvar_posterior.json  (b) NIW posterior moments at fixed hyperparameters,
#                                                    plus forecast quantiles from iid posterior draws
#   tests/parity/fixtures/r_bvar_eb.json         (c) posterior mode of the hyperparameters
#   tests/parity/fixtures/r_bvar_mh.json         (d) Metropolis-Hastings posterior quantiles
#
# Every convention behind these numbers (regressor layout, Ω, Ψ, dummy observations, ȳ0, the
# marginal-likelihood formula, hyperpriors, MH proposal, forecast simulation) is documented in
# tests/parity/BVAR_PARAMETRIZATION.md. Read it before changing anything here.
#
# Not run in CI. Reproducible: every random number comes from a fixed seed.
#
# Usage, from anywhere:
#   Rscript tools/make_r_fixtures.R              # everything (a few minutes)
#   Rscript tools/make_r_fixtures.R --no-mh      # skip (d), the long MH chains
#   Rscript tools/make_r_fixtures.R --quick      # short MH chains (smoke test; do not commit)
#
# Requirements: R >= 4.1 with BVAR 1.0.5, mvtnorm, jsonlite and coda, e.g. in the user library:
#   lib <- Sys.getenv("R_LIBS_USER"); dir.create(lib, recursive = TRUE, showWarnings = FALSE)
#   install.packages(c("BVAR", "jsonlite", "coda"), lib = lib, repos = "https://cloud.r-project.org")
# (On macOS without gfortran, install the CRAN *binary* of mvtnorm first.)
#
# Licensing (SPEC §2.1): BVAR is GPL-3. This script *runs* BVAR, including a few internal
# functions through `:::`, as a numerical oracle; its outputs are data. Nothing in this file is
# part of the rai-bvar library, and no library code may be derived from BVAR's source.
#
# Two kinds of numbers are produced and both are exported:
#   * "bvar":  computed by BVAR itself (bvar(), BVAR:::bv_ml, BVAR:::draw_post, predict()).
#   * "exact": computed by the independent, QR-based implementation of the textbook conjugate
#     formulas in the section "Reference formulas" below. It agrees with BVAR to ~1e-11 wherever
#     BVAR is exact, and it is the comparison target where BVAR is numerically less accurate
#     (posterior mean via normal equations, condition number ~1e10 on the levels data), deviates
#     from the exact marginal likelihood (dummy observations with a prior mean that is not a
#     random walk), or has a bug (forecast noise scaled by Σ instead of chol(Σ)).
# =================================================================================================

suppressPackageStartupMessages({
  lib <- path.expand(Sys.getenv("R_LIBS_USER"))
  if (nzchar(lib) && dir.exists(lib)) .libPaths(c(lib, .libPaths()))
  library(BVAR)
  library(jsonlite)
  library(coda)
})

args <- commandArgs(trailingOnly = TRUE)
RUN_MH <- !("--no-mh" %in% args)
QUICK <- "--quick" %in% args

if (as.character(packageVersion("BVAR")) != "1.0.5") {
  warning("fixtures were designed against BVAR 1.0.5; found ", packageVersion("BVAR"))
}

script_path <- function() {
  file_arg <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
  if (length(file_arg) == 1) normalizePath(sub("^--file=", "", file_arg))
  else normalizePath(file.path("tools", "make_r_fixtures.R"), mustWork = TRUE)
}
ROOT <- dirname(dirname(script_path()))
FIXTURES <- file.path(ROOT, "tests", "parity", "fixtures")
stopifnot(dir.exists(FIXTURES))

# ------------------------------------------------------------------------------------------------
# Settings shared by all fixtures
# ------------------------------------------------------------------------------------------------

LAGS <- 4L
ALPHA <- 2
VAR_CONST <- 1e7
LAMBDAS <- c(0.1, 0.2, 0.5)
MU <- 1      # SOC hyperparameter at fixed hyperparameters
DELTA <- 1   # SUR hyperparameter at fixed hyperparameters
PROBS <- c(0.05, 0.5, 0.95)
HORIZON <- 8L

# Hyperpriors: GLP (2015) Gamma(mode, sd) priors with BVAR's default bounds.
HYPERPRIORS <- list(
  lambda = list(mode = 0.2, sd = 0.4, lower = 1e-4, upper = 5),
  soc = list(mode = 1, sd = 1, lower = 1e-4, upper = 50),
  sur = list(mode = 1, sd = 1, lower = 1e-4, upper = 50)
)
HYPER_ORDER <- c("lambda", "soc", "sur")

MH_CHAINS <- 4L
MH_DRAW <- if (QUICK) 3000L else 25000L   # total iterations per chain (bvar()'s n_draw)
MH_BURN <- if (QUICK) 1000L else 5000L    # burn-in per chain (bvar()'s n_burn)
MH_SEEDS <- 20261021L + seq_len(MH_CHAINS) - 1L
MCSE_BATCHES <- 25L                       # batches per MH chain for the MCSE estimates
MCSE_BATCHES_IID <- 100L                  # batches for the single stream of iid draws

# ------------------------------------------------------------------------------------------------
# Data
# ------------------------------------------------------------------------------------------------

read_dataset <- function(id, file) {
  path <- file.path(FIXTURES, file)
  frame <- read.csv(path, check.names = FALSE, stringsAsFactors = FALSE)
  Y <- as.matrix(frame[, -1])
  storage.mode(Y) <- "double"
  list(id = id, file = file, md5 = unname(tools::md5sum(path)), dates = frame[[1]],
       variables = colnames(frame)[-1], Yraw = Y)
}

DATA <- list(
  qd = read_dataset("qd", "qd_small_levels.csv"),
  sim = read_dataset("sim", "sim_var3.csv")
)

dataset_meta <- function(ds) {
  list(file = ds$file, md5 = ds$md5, variables = I(ds$variables), T = nrow(ds$Yraw),
       first = ds$dates[1], last = ds$dates[length(ds$dates)])
}

# Regressors x_t = (1, y_{t-1}', ..., y_{t-p}')': constant first, then lag blocks in variable
# order. This is BVAR's layout (cbind(1, lag_var(Y, lags))) and rai-bvar's.
make_design <- function(Yraw, p) {
  rows <- (p + 1):nrow(Yraw)
  X <- do.call(cbind, c(list(rep(1, length(rows))),
                        lapply(seq_len(p), function(l) Yraw[rows - l, , drop = FALSE])))
  list(Y = Yraw[rows, , drop = FALSE], X = X)
}

regressor_names <- function(variables, p) {
  c("const", unlist(lapply(seq_len(p), function(l) paste0(variables, ".L", l))))
}

prior_mean_matrix <- function(b, M, p) {
  B0 <- matrix(0, 1 + M * p, M)
  B0[1 + seq_len(M), ] <- diag(b, M)
  B0
}

# ȳ0 used by the SOC/SUR dummies.
#   "glp":  mean of the first p observations of the raw sample (the initial conditions y_{1..p});
#           GLP (2015) and SPEC §4.6.
#   "bvar": BVAR's built-in bv_soc()/bv_sur(): mean of the first p rows of the *trimmed* Y,
#           i.e. raw observations p+1..2p (for p = 1: the second raw observation).
ybar0_of <- function(Yraw, p, convention) {
  rows <- switch(convention, glp = seq_len(p), bvar = p + seq_len(p))
  colMeans(Yraw[rows, , drop = FALSE])
}

# ------------------------------------------------------------------------------------------------
# Reference formulas (independent of BVAR's code; QR-based)
# ------------------------------------------------------------------------------------------------

omega_diag <- function(lambda, alpha, psi, p, var_const = VAR_CONST) {
  c(var_const, unlist(lapply(seq_len(p), function(l) lambda^2 / l^alpha / psi)))
}

dummy_obs <- function(y0, p, soc = NULL, sur = NULL) {
  M <- length(y0)
  Yd <- NULL
  Xd <- NULL
  if (!is.null(soc)) {   # sum of coefficients: n rows
    Ys <- diag(y0, M) / soc
    Yd <- rbind(Yd, Ys)
    Xd <- rbind(Xd, cbind(0, do.call(cbind, rep(list(Ys), p))))
  }
  if (!is.null(sur)) {   # single unit root (dummy initial observation): 1 row
    ys <- y0 / sur
    Yd <- rbind(Yd, ys)
    Xd <- rbind(Xd, c(1 / sur, rep(ys, p)))
  }
  if (!is.null(Yd)) dimnames(Yd) <- dimnames(Xd) <- NULL
  list(Y = Yd, X = Xd)
}

# log p(Y | X) under B | Σ ~ MN(B0, Ω, Σ), Σ ~ IW(diag(psi), d = n + 2), plus the NIW posterior.
niw_core <- function(Y, X, B0, omega, psi) {
  N <- nrow(Y)
  M <- ncol(Y)
  K <- ncol(X)
  d <- M + 2
  Xa <- rbind(X, diag(1 / sqrt(omega), K))
  Ya <- rbind(Y, diag(1 / sqrt(omega), K) %*% B0)
  q <- qr(Xa, tol = 1e-14)   # Householder QR (LINPACK dqrdc2); no pivoting at this tolerance
  if (q$rank < K) stop("rank-deficient augmented design")
  R <- qr.R(q)
  stopifnot(all(q$pivot == seq_len(K)))
  B <- backsolve(R, qr.qty(q, Ya)[seq_len(K), , drop = FALSE])
  E <- Ya - Xa %*% B
  S <- diag(psi, M) + crossprod(E)          # S̄ = Ψ + Ê'Ê + (B̂ − B0)'Ω⁻¹(B̂ − B0)
  S <- (S + t(S)) / 2
  logdet_xx <- 2 * sum(log(abs(diag(R))))    # log|X'X + Ω⁻¹|
  logdet_s <- 2 * sum(log(diag(chol(S))))
  j <- seq_len(M) - 1
  log_ml <- -N * M / 2 * log(pi) + sum(lgamma((N + d - j) / 2) - lgamma((d - j) / 2)) -
    M / 2 * (sum(log(omega)) + logdet_xx) + d / 2 * sum(log(psi)) - (N + d) / 2 * logdet_s
  list(log_ml = log_ml, mean = B, scale = S, df = N + d, R = R, N = N)
}

# log p(Y | h) = log p(Y, Y_d) − log p(Y_d) with the exact dummy-only marginal likelihood.
exact_eval <- function(st, values) {
  h <- st$hyper_defaults
  h[names(values)] <- values
  omega <- omega_diag(h[["lambda"]], ALPHA, st$psi, st$p)
  dmy <- dummy_obs(st$y0, st$p,
                   if ("soc" %in% st$dummies) h[["soc"]],
                   if ("sur" %in% st$dummies) h[["sur"]])
  if (is.null(dmy$Y)) {
    full <- niw_core(st$Y, st$X, st$B0, omega, st$psi)
    return(list(log_ml = full$log_ml, full = full$log_ml, dummy = 0, post = full, omega = omega))
  }
  full <- niw_core(rbind(dmy$Y, st$Y), rbind(dmy$X, st$X), st$B0, omega, st$psi)
  dd <- niw_core(dmy$Y, dmy$X, st$B0, omega, st$psi)
  list(log_ml = full$log_ml - dd$log_ml, full = full$log_ml, dummy = dd$log_ml, post = full,
       omega = omega)
}

gamma_shape_scale <- function(mode, sd) {
  r <- mode^2 / sd^2
  k <- (2 + r + sqrt((4 + r) * r)) / 2
  list(shape = k, scale = sqrt(sd^2 / k))
}

log_hyperprior_terms <- function(values) {
  out <- list()
  for (name in names(values)) {
    hp <- HYPERPRIORS[[name]]
    g <- gamma_shape_scale(hp$mode, hp$sd)
    out[[name]] <- dgamma(values[[name]], shape = g$shape, scale = g$scale, log = TRUE)
  }
  out
}

in_bounds <- function(values) {
  all(vapply(names(values), function(nm) {
    values[[nm]] >= HYPERPRIORS[[nm]]$lower && values[[nm]] <= HYPERPRIORS[[nm]]$upper
  }, logical(1)))
}

exact_log_posterior <- function(st, values) {
  if (!in_bounds(values)) return(-1e18)
  exact_eval(st, values)$log_ml + sum(unlist(log_hyperprior_terms(values)))
}

# ------------------------------------------------------------------------------------------------
# BVAR set-up (mirrors the preparation inside bvar() so that BVAR:::bv_ml can be called directly)
# ------------------------------------------------------------------------------------------------

bvar_dummy_priors <- function(dummies, convention, Yraw, p) {
  hp <- HYPERPRIORS
  out <- list()
  if (convention == "bvar") {
    if ("soc" %in% dummies) {
      out$soc <- bv_soc(mode = hp$soc$mode, sd = hp$soc$sd, min = hp$soc$lower, max = hp$soc$upper)
    }
    if ("sur" %in% dummies) {
      out$sur <- bv_sur(mode = hp$sur$mode, sd = hp$sur$sd, min = hp$sur$lower, max = hp$sur$upper)
    }
    return(out)
  }
  # GLP convention through BVAR's documented custom-dummy interface: the function receives the
  # trimmed Y, so the initial-observation mean is captured from the raw data instead.
  y0 <- ybar0_of(Yraw, p, "glp")
  soc_fun <- function(Y, lags, par) {
    Ys <- diag(y0, ncol(Y)) / par
    list(Y = Ys, X = cbind(0, matrix(rep(Ys, lags), nrow = ncol(Y))))
  }
  sur_fun <- function(Y, lags, par) {
    ys <- y0 / par
    list(Y = ys, X = c(1 / par, rep(ys, lags)))
  }
  if ("soc" %in% dummies) {
    out$soc <- bv_dummy(mode = hp$soc$mode, sd = hp$soc$sd, min = hp$soc$lower,
                        max = hp$soc$upper, fun = soc_fun)
  }
  if ("sur" %in% dummies) {
    out$sur <- bv_dummy(mode = hp$sur$mode, sd = hp$sur$sd, min = hp$sur$lower,
                        max = hp$sur$upper, fun = sur_fun)
  }
  out
}

make_setup <- function(ds, p, b, psi, dummies = character(0), convention = "glp",
                       hyper = "lambda") {
  Yraw <- ds$Yraw
  des <- make_design(Yraw, p)
  M <- ncol(Yraw)
  K <- 1 + M * p
  hp <- HYPERPRIORS$lambda
  mn <- bv_mn(lambda = bv_lambda(mode = hp$mode, sd = hp$sd, min = hp$lower, max = hp$upper),
              alpha = bv_alpha(mode = ALPHA), psi = bv_psi(mode = psi), var = VAR_CONST, b = b)
  priors <- do.call(bv_priors, c(list(hyper = hyper, mn = mn),
                                 bvar_dummy_priors(dummies, convention, Yraw, p)))
  # The same processing bvar() applies before optimisation:
  processed <- priors
  processed$b <- prior_mean_matrix(b, M, p)
  for (i in seq_len(M)) {
    processed[[paste0("psi", i)]] <- c(mode = processed$psi$mode[i], min = processed$psi$min[i],
                                       max = processed$psi$max[i])
  }
  par_names <- names(processed)[!grepl("^hyper$|^var$|^b$|^psi[0-9]+$|^dummy$", names(processed))]
  pars <- do.call(c, lapply(par_names, function(x) processed[[x]][["mode"]]))
  names(pars) <- BVAR:::name_pars(par_names, M)
  list(ds = ds, p = p, b = b, psi = psi, dummies = dummies, convention = convention,
       y0 = ybar0_of(Yraw, p, convention), Y = des$Y, X = des$X, M = M, K = K, N = nrow(des$Y),
       B0 = prior_mean_matrix(b, M, p), priors = priors, processed = processed, pars = pars,
       hyper_defaults = c(lambda = hp$mode, soc = MU, sur = DELTA))
}

# BVAR's own log ML (hyperprior = FALSE) or log posterior (hyperprior = TRUE: adds the Gamma
# log densities of the hyperparameters named in `values`), evaluated by BVAR:::bv_ml.
bvar_eval <- function(st, values, hyperprior = FALSE, lower = -Inf, upper = Inf) {
  pri <- st$processed
  pri$hyper <- if (hyperprior) names(values) else character(0)
  BVAR:::bv_ml(hyper = values, hyper_min = lower, hyper_max = upper, pars = st$pars,
               priors = pri, Y = st$Y, X = st$X, XX = crossprod(st$X), K = st$K, M = st$M,
               N = st$N, lags = st$p)
}

# BVAR's computation on the stacked data [Y_d; Y] without the dummy-only denominator.
bvar_full_part <- function(st, values) {
  pri <- st$processed
  h <- st$hyper_defaults
  h[names(values)] <- values
  dm <- lapply(pri$dummy, function(x) pri[[x]]$fun(Y = st$Y, lags = st$p, par = h[[x]]))
  Yd <- do.call(rbind, lapply(dm, function(z) matrix(z$Y, ncol = st$M)))
  Xd <- do.call(rbind, lapply(dm, function(z) matrix(z$X, ncol = st$K)))
  pri$dummy <- character(0)
  pri$hyper <- character(0)
  Ys <- rbind(Yd, st$Y)
  Xs <- rbind(Xd, st$X)
  val <- BVAR:::bv_ml(hyper = values["lambda"], pars = st$pars, priors = pri, Y = Ys, X = Xs,
                      XX = crossprod(Xs), K = st$K, M = st$M, N = nrow(Ys), lags = st$p)$log_ml
  list(value = val, Yd = Yd, Xd = Xd)
}

hyper_values <- function(lambda, dummies) {
  v <- c(lambda = lambda)
  if ("soc" %in% dummies) v["soc"] <- MU
  if ("sur" %in% dummies) v["sur"] <- DELTA
  v
}

as_named_list <- function(v) as.list(v)

rel_diff <- function(a, b) abs(a - b) / max(abs(b), 1e-300)

# ------------------------------------------------------------------------------------------------
# ψ conventions
# ------------------------------------------------------------------------------------------------

psi_conventions <- function(ds, p) {
  des <- make_design(ds$Yraw, p)
  Y <- des$Y
  bvar_auto <- suppressMessages(BVAR:::auto_psi(Y, p))$mode
  arima_var <- vapply(seq_len(ncol(Y)), function(j) {
    suppressWarnings(arima(Y[, j], order = c(p, 0, 0)))$sigma2
  }, numeric(1))
  ols <- vapply(seq_len(ncol(ds$Yraw)), function(j) {
    d <- make_design(ds$Yraw[, j, drop = FALSE], p)
    fit <- lm.fit(d$X, d$Y[, 1])
    c(ssr = sum(fit$residuals^2), n = nrow(d$X), k = ncol(d$X))
  }, numeric(3))
  stopifnot(isTRUE(all.equal(bvar_auto^2, arima_var, tolerance = 1e-10)))
  list(
    bvar_auto_sd = bvar_auto,
    arima_ml_variance = arima_var,
    ols_ar_variance_dof = ols["ssr", ] / (ols["n", ] - ols["k", ]),
    ols_ar_variance_mle = ols["ssr", ] / ols["n", ]
  )
}

PSI <- lapply(DATA, psi_conventions, p = LAGS)

PSI_DEFINITIONS <- list(
  bvar_auto_sd = paste(
    "BVAR's psi = 'auto': sqrt(sigma2) of stats::arima(y_j, order = c(p, 0, 0)) (method",
    "'CSS-ML', with mean) fitted to rows p+1..T of the raw data (the trimmed Y); a STANDARD",
    "DEVIATION that BVAR nevertheless uses as the IW scale diagonal and in Omega."),
  arima_ml_variance = "sigma2 of the same arima fit (= bvar_auto_sd^2), variance units.",
  ols_ar_variance_dof = paste(
    "OLS AR(p) with constant on the full raw series (T - p equations, p + 1 regressors):",
    "SSR / (T - p - (p + 1))."),
  ols_ar_variance_mle = "Same OLS AR(p) fit: SSR / (T - p)."
)

# ------------------------------------------------------------------------------------------------
# Output helpers
# ------------------------------------------------------------------------------------------------

META <- list(
  generator = "tools/make_r_fixtures.R",
  r_version = R.version.string,
  platform = R.version$platform,
  bvar_version = as.character(packageVersion("BVAR")),
  mvtnorm_version = as.character(packageVersion("mvtnorm")),
  jsonlite_version = as.character(packageVersion("jsonlite")),
  coda_version = as.character(packageVersion("coda")),
  blas = unname(extSoftVersion()["BLAS"]),
  lapack = La_version(),
  created = format(Sys.time(), "%Y-%m-%dT%H:%M:%S%z"),
  documentation = "tests/parity/BVAR_PARAMETRIZATION.md",
  layout = paste(
    "Coefficient matrices are k x n (rows: const, then lag 1 block in variable order, ...,",
    "lag p block; columns: equations), y_t = B'x_t + u_t. Matrices are row-major nested lists.")
)

write_fixture <- function(obj, name) {
  path <- file.path(FIXTURES, name)
  write_json(obj, path, auto_unbox = TRUE, digits = NA, pretty = TRUE, null = "null",
             na = "null")
  message("wrote ", path)
}

common_case_fields <- function(st, case_id, psi_key) {
  list(
    id = case_id,
    dataset = st$ds$id,
    lags = st$p,
    mean = st$b,
    alpha = ALPHA,
    var_const = VAR_CONST,
    psi = I(st$psi),
    psi_source = psi_key,
    N = st$N, K = st$K, M = st$M,
    random_walk_mean = isTRUE(all(st$b == 1)),
    ybar0 = list(glp = I(ybar0_of(st$ds$Yraw, st$p, "glp")),
                 bvar = I(ybar0_of(st$ds$Yraw, st$p, "bvar")))
  )
}

checks <- list()
note_check <- function(name, value, limit) {
  ok <- is.finite(value) && value <= limit
  checks[[length(checks) + 1]] <<- list(check = name, value = value, limit = limit, ok = ok)
  if (!ok) warning(sprintf("check failed: %s = %.3e > %.1e", name, value, limit))
}

# ================================================================================================
# (a) Log marginal likelihood at fixed hyperparameters
# ================================================================================================

message("(a) log marginal likelihood at fixed hyperparameters")

CASES_A <- list(
  list(id = "qd_p4_rw", data = "qd", b = 1, psi = "arima_ml_variance", lambdas = LAMBDAS),
  list(id = "sim_p4_zero", data = "sim", b = 0, psi = "arima_ml_variance", lambdas = LAMBDAS),
  list(id = "sim_p4_rw", data = "sim", b = 1, psi = "arima_ml_variance", lambdas = LAMBDAS),
  list(id = "qd_p4_rw_bvarpsi", data = "qd", b = 1, psi = "bvar_auto_sd", lambdas = 0.2)
)
DUMMY_SETS <- list(character(0), "soc", "sur", c("soc", "sur"))

logml_cases <- lapply(CASES_A, function(cs) {
  ds <- DATA[[cs$data]]
  psi <- PSI[[cs$data]][[cs$psi]]
  grid <- list()
  base <- NULL
  for (dummies in DUMMY_SETS) {
    conventions <- if (length(dummies)) c("glp", "bvar") else "glp"
    for (conv in conventions) {
      st <- make_setup(ds, LAGS, cs$b, psi, dummies, conv)
      if (is.null(base)) base <- st
      for (lambda in cs$lambdas) {
        values <- hyper_values(lambda, dummies)
        ex <- exact_eval(st, values)
        lml_bvar <- bvar_eval(st, values)$log_ml
        lpost_bvar <- bvar_eval(st, values, hyperprior = TRUE)$log_ml
        terms <- log_hyperprior_terms(values)
        if (length(dummies)) {
          fp <- bvar_full_part(st, values)
          own <- dummy_obs(st$y0, st$p, if ("soc" %in% dummies) MU, if ("sur" %in% dummies) DELTA)
          note_check(sprintf("%s dummy rows identical (%s, %s)", cs$id,
                             paste(dummies, collapse = "+"), conv),
                     max(abs(fp$Yd - own$Y), abs(fp$Xd - own$X)), 1e-12)
          full_bvar <- fp$value
        } else {
          full_bvar <- lml_bvar
        }
        bvar_is_exact <- length(dummies) == 0 || isTRUE(all(cs$b == 1))
        note_check(sprintf("%s lambda=%g %s: |sum(hyperprior) - (bvar post - bvar ml)|", cs$id,
                           lambda, paste(dummies, collapse = "+")),
                   abs(sum(unlist(terms)) - (lpost_bvar - lml_bvar)), 1e-9)
        note_check(sprintf("%s lambda=%g %s %s: rel diff full part bvar vs exact", cs$id, lambda,
                           paste(dummies, collapse = "+"), conv),
                   rel_diff(full_bvar, ex$full), 1e-9)
        if (bvar_is_exact) {
          note_check(sprintf("%s lambda=%g %s %s: rel diff log ML bvar vs exact", cs$id, lambda,
                             paste(dummies, collapse = "+"), conv),
                     rel_diff(lml_bvar, ex$log_ml), 1e-9)
        }
        grid[[length(grid) + 1]] <- list(
          lambda = lambda,
          soc = if ("soc" %in% dummies) MU else NULL,
          sur = if ("sur" %in% dummies) DELTA else NULL,
          ybar0 = if (length(dummies)) conv else NULL,
          log_ml = ex$log_ml,
          log_ml_full = ex$full,
          log_ml_dummy = ex$dummy,
          log_ml_bvar = lml_bvar,
          log_ml_full_bvar = full_bvar,
          log_ml_dummy_bvar = full_bvar - lml_bvar,
          bvar_is_exact = bvar_is_exact,
          rel_diff_bvar_exact = rel_diff(lml_bvar, ex$log_ml),
          log_hyperprior = terms,
          log_posterior_bvar = lpost_bvar
        )
      }
    }
  }
  c(common_case_fields(base, cs$id, cs$psi), list(grid = grid))
})

write_fixture(list(
  meta = META,
  description = paste(
    "(a) log p(Y | h) at fixed hyperparameters, alpha = 2, var_const = 1e7, explicit psi.",
    "log_ml/log_ml_full/log_ml_dummy: exact QR-based values, log_ml = log_ml_full -",
    "log_ml_dummy (data + dummies minus dummies only). *_bvar: BVAR 1.0.5 (BVAR:::bv_ml).",
    "BVAR's dummy-only term plugs in the prior mean b instead of the dummy posterior mean, which",
    "is exact only when Y_d = X_d b (random-walk prior mean): bvar_is_exact flags this.",
    "log_posterior_bvar = log_ml_bvar + sum(log_hyperprior) (Gamma densities, natural scale)."),
  hyperpriors = HYPERPRIORS,
  hyperprior_gamma = lapply(HYPERPRIORS, function(h) gamma_shape_scale(h$mode, h$sd)),
  datasets = lapply(DATA, dataset_meta),
  psi_conventions = list(definitions = PSI_DEFINITIONS,
                         values = lapply(PSI, function(x) lapply(x, I))),
  cases = logml_cases
), "r_bvar_logml.json")

# ================================================================================================
# (b) NIW posterior at fixed hyperparameters (+ forecasts from iid posterior draws)
# ================================================================================================

message("(b) posterior moments at fixed hyperparameters")

bvar_posterior <- function(st, values) {
  out <- bvar_eval(st, values)
  B0 <- st$B0
  dev <- out$beta_hat - B0
  scale <- out$psi + out$sse + t(dev) %*% out$omega_inv %*% dev   # as in BVAR:::draw_post
  df <- out$N + st$M + 2
  list(mean = out$beta_hat, scale = scale, df = df,
       omega = chol2inv(chol(out$XX + out$omega_inv)), ml = out)
}

max_scaled_diff <- function(a, b) max(abs(a - b)) / max(abs(b))

draw_bvar_posterior <- function(ml_out, st, n) {
  beta <- array(NA_real_, c(n, st$K, st$M))
  sigma <- array(NA_real_, c(n, st$M, st$M))
  for (i in seq_len(n)) {
    dr <- BVAR:::draw_post(XX = ml_out$XX, N = ml_out$N, M = st$M, lags = st$p, b = st$B0,
                           psi = ml_out$psi, sse = ml_out$sse, beta_hat = ml_out$beta_hat,
                           omega_inv = ml_out$omega_inv)
    beta[i, , ] <- dr$beta_draw
    sigma[i, , ] <- dr$sigma_draw
  }
  list(beta = beta, sigma = sigma)
}

# One forecast path per posterior draw: y_{T+h} = B'x_{T+h} + t(chol(Σ)) z, z ~ N(0, I).
# (BVAR's predict() adds Σ z instead; see BVAR_PARAMETRIZATION.md.)
simulate_paths <- function(beta, sigma, Yraw, p, horizon) {
  n <- dim(beta)[1]
  M <- ncol(Yraw)
  paths <- array(NA_real_, c(n, horizon, M))
  hist0 <- Yraw[nrow(Yraw):(nrow(Yraw) - p + 1), , drop = FALSE]   # most recent first
  for (i in seq_len(n)) {
    B <- beta[i, , ]
    L <- t(chol(sigma[i, , ]))
    hist <- hist0
    for (h in seq_len(horizon)) {
      x <- c(1, as.vector(t(hist)))
      y <- drop(crossprod(B, x)) + drop(L %*% rnorm(M))
      paths[i, h, ] <- y
      hist <- rbind(y, hist[-p, , drop = FALSE])
    }
  }
  paths
}

batch_index <- function(n, n_batch) split(seq_len(n), cut(seq_len(n), n_batch, labels = FALSE))

# Delta-method MCSE of the p-quantile: sparsity dQ/dp (central difference of the empirical
# quantile function, ±0.01) times the batch-means standard error of the indicator 1{x <= q_p}
# (robust to autocorrelation; for iid draws ≈ sqrt(p (1 - p) / n)).
delta_mcse <- function(chains, probs, q, n_batch) {
  pooled <- unlist(chains)
  eps <- 0.01
  sparsity <- (quantile(pooled, pmin(probs + eps, 1), names = FALSE) -
                 quantile(pooled, pmax(probs - eps, 0), names = FALSE)) /
    (pmin(probs + eps, 1) - pmax(probs - eps, 0))
  vapply(seq_along(probs), function(k) {
    means <- unlist(lapply(chains, function(x) {
      vapply(batch_index(length(x), n_batch), function(i) mean(x[i] <= q[k]), numeric(1))
    }))
    size <- length(pooled) / length(means)
    sparsity[k] * sqrt(var(means) * size / length(pooled))
  }, numeric(1))
}

# Quantiles of the pooled draws and their Monte Carlo standard errors. `chains` is a list of
# vectors in iteration order. MCSE = max(batch-quantile, delta-method, between-chain MCSE).
quantile_summary <- function(chains, probs = PROBS, n_batch = MCSE_BATCHES) {
  pooled <- unlist(chains)
  q <- quantile(pooled, probs, names = FALSE, type = 7)
  bq <- do.call(rbind, lapply(chains, function(x) {
    t(vapply(batch_index(length(x), n_batch), function(i) {
      quantile(x[i], probs, names = FALSE, type = 7)
    }, numeric(length(probs))))
  }))
  mcse_batch <- apply(bq, 2, sd) / sqrt(nrow(bq))
  mcse_delta <- delta_mcse(chains, probs, q, n_batch)
  if (length(chains) > 1) {
    cq <- t(vapply(chains, quantile, numeric(length(probs)), probs = probs, names = FALSE))
    mcse_chain <- apply(cq, 2, sd) / sqrt(length(chains))
  } else {
    mcse_chain <- rep(0, length(probs))
  }
  list(q = I(q), mcse = I(pmax(mcse_batch, mcse_delta, mcse_chain)), mcse_batch = I(mcse_batch),
       mcse_delta = I(mcse_delta), mcse_between_chain = I(mcse_chain), mean = mean(pooled),
       sd = sd(pooled))
}

forecast_summary <- function(path_chains, variables, n_batch = MCSE_BATCHES) {
  H <- dim(path_chains[[1]])[2]
  M <- dim(path_chains[[1]])[3]
  qarr <- array(NA_real_, c(length(PROBS), H, M))
  marr <- array(NA_real_, c(length(PROBS), H, M))
  for (h in seq_len(H)) for (m in seq_len(M)) {
    s <- quantile_summary(lapply(path_chains, function(pc) pc[, h, m]), n_batch = n_batch)
    qarr[, h, m] <- s$q
    marr[, h, m] <- s$mcse
  }
  to_nested <- function(a) lapply(seq_len(dim(a)[1]), function(i) a[i, , , drop = TRUE])
  list(horizon = H, variables = I(variables), probs = I(PROBS),
       dims = I(c("quantile", "horizon", "variable")), q = to_nested(qarr),
       mcse = to_nested(marr))
}

CASES_B <- list(
  list(id = "qd_p4_rw", data = "qd", b = 1),
  list(id = "sim_p4_zero", data = "sim", b = 0)
)
FORECAST_B <- list(
  qd_p4_rw = list(lambda = 0.2, dummies = c("soc", "sur"), convention = "glp", seed = 101L),
  sim_p4_zero = list(lambda = 0.2, dummies = character(0), convention = "glp", seed = 102L)
)
N_IID <- if (QUICK) 4000L else 100000L

posterior_cases <- lapply(CASES_B, function(cs) {
  ds <- DATA[[cs$data]]
  psi <- PSI[[cs$data]]$arima_ml_variance
  configs <- list(
    list(dummies = character(0), conv = "glp", lambdas = LAMBDAS),
    list(dummies = c("soc", "sur"), conv = "glp", lambdas = LAMBDAS),
    list(dummies = c("soc", "sur"), conv = "bvar", lambdas = 0.2)
  )
  grid <- list()
  base <- NULL
  for (cfg in configs) {
    st <- make_setup(ds, LAGS, cs$b, psi, cfg$dummies, cfg$conv)
    if (is.null(base)) base <- st
    for (lambda in cfg$lambdas) {
      values <- hyper_values(lambda, cfg$dummies)
      ex <- exact_eval(st, values)$post
      bv <- bvar_posterior(st, values)
      omega_exact <- chol2inv(ex$R)
      diffs <- list(mean = max_scaled_diff(bv$mean, ex$mean),
                    scale = max_scaled_diff(bv$scale, ex$scale),
                    omega_diag = max(abs(diag(bv$omega) - diag(omega_exact)) / diag(omega_exact)))
      note_check(sprintf("%s lambda=%g %s %s: posterior mean bvar vs exact (scaled)", cs$id,
                         lambda, paste(cfg$dummies, collapse = "+"), cfg$conv),
                 diffs$mean, 1e-7)
      note_check(sprintf("%s lambda=%g %s %s: posterior scale bvar vs exact (scaled)", cs$id,
                         lambda, paste(cfg$dummies, collapse = "+"), cfg$conv),
                 diffs$scale, 1e-7)
      note_check(sprintf("%s df", cs$id), abs(bv$df - ex$df), 0)
      entry <- list(
        lambda = lambda,
        soc = if ("soc" %in% cfg$dummies) MU else NULL,
        sur = if ("sur" %in% cfg$dummies) DELTA else NULL,
        ybar0 = if (length(cfg$dummies)) cfg$conv else NULL,
        n_rows = ex$N,
        mean = ex$mean,
        scale = ex$scale,
        df = ex$df,
        omega = omega_exact,
        sigma_mean = ex$scale / (ex$df - st$M - 1),
        mean_bvar = bv$mean,
        scale_bvar = bv$scale,
        omega_bvar = bv$omega,
        bvar_vs_exact = diffs
      )
      if (lambda == 0.2 && cfg$conv == "glp") {
        set.seed(7L)
        n_mc <- if (QUICK) 2000L else 10000L
        dr <- draw_bvar_posterior(bv$ml, st, n_mc)
        mb <- apply(dr$beta, c(2, 3), mean)
        sb <- apply(dr$beta, c(2, 3), sd)
        ms <- apply(dr$sigma, c(2, 3), mean)
        ss <- apply(dr$sigma, c(2, 3), sd)
        entry$mc_check <- list(
          draws = n_mc,
          max_abs_z_mean = max(abs(mb - ex$mean) / (sb / sqrt(n_mc))),
          max_abs_z_sigma = max(abs(ms - entry$sigma_mean) / (ss / sqrt(n_mc)))
        )
        note_check(sprintf("%s %s: BVAR draw_post MC z-score (mean B)", cs$id,
                           paste(cfg$dummies, collapse = "+")),
                   entry$mc_check$max_abs_z_mean, 5)
        note_check(sprintf("%s %s: BVAR draw_post MC z-score (mean Sigma)", cs$id,
                           paste(cfg$dummies, collapse = "+")),
                   entry$mc_check$max_abs_z_sigma, 5)
      }
      grid[[length(grid) + 1]] <- entry
    }
  }
  # Forecasts from iid draws of BVAR:::draw_post at fixed hyperparameters.
  fc <- FORECAST_B[[cs$id]]
  st <- make_setup(ds, LAGS, cs$b, psi, fc$dummies, fc$convention)
  values <- hyper_values(fc$lambda, fc$dummies)
  bv <- bvar_posterior(st, values)
  set.seed(fc$seed)
  dr <- draw_bvar_posterior(bv$ml, st, N_IID)
  paths <- simulate_paths(dr$beta, dr$sigma, ds$Yraw, st$p, HORIZON)
  forecast <- c(
    list(lambda = fc$lambda,
         soc = if ("soc" %in% fc$dummies) MU else NULL,
         sur = if ("sur" %in% fc$dummies) DELTA else NULL,
         ybar0 = if (length(fc$dummies)) fc$convention else NULL,
         draws = N_IID, seed = fc$seed,
         noise = "chol(Sigma) z, one path per iid posterior draw (BVAR:::draw_post)"),
    forecast_summary(list(paths), ds$variables, n_batch = MCSE_BATCHES_IID)
  )
  c(common_case_fields(base, cs$id, "arima_ml_variance"),
    list(grid = grid, forecast = forecast))
})

write_fixture(list(
  meta = META,
  description = paste(
    "(b) Conjugate NIW posterior at fixed hyperparameters: Sigma | Y ~ IW(scale, df),",
    "B | Sigma, Y ~ MN(mean, omega, Sigma); sigma_mean = scale / (df - n - 1).",
    "mean/scale/omega: exact QR-based values (target). *_bvar: BVAR 1.0.5 (normal equations;",
    "BVAR:::bv_ml + the S_post formula of BVAR:::draw_post), less accurate on ill-conditioned",
    "levels data: see bvar_vs_exact (max |diff| / max |value|). forecast: quantiles of",
    "one-path-per-draw simulations from iid posterior draws, with batch-quantile MCSE."),
  datasets = lapply(DATA, dataset_meta),
  cases = posterior_cases
), "r_bvar_posterior.json")

# ================================================================================================
# (c) Empirical Bayes: posterior mode of the hyperparameters
# ================================================================================================

message("(c) posterior mode of the hyperparameters")

num_grad <- function(g, u, h = 1e-5) {
  vapply(seq_along(u), function(i) {
    e <- replace(numeric(length(u)), i, h)
    (g(u + e) - g(u - e)) / (2 * h)
  }, numeric(1))
}

num_hess <- function(g, u, h = 1e-4) {
  m <- length(u)
  H <- matrix(NA_real_, m, m)
  for (i in seq_len(m)) for (j in i:m) {
    ei <- replace(numeric(m), i, h)
    ej <- replace(numeric(m), j, h)
    H[i, j] <- H[j, i] <- (g(u + ei + ej) - g(u + ei - ej) - g(u - ei + ej) + g(u - ei - ej)) /
      (4 * h^2)
  }
  H
}

# Maximise a natural-scale log posterior f(h) over u = log h (no Jacobian: the maximiser is the
# mode of the natural-scale density), then polish with Newton steps when the mode is interior.
refine_mode <- function(f, start, names) {
  lower <- vapply(names, function(nm) HYPERPRIORS[[nm]]$lower, numeric(1))
  upper <- vapply(names, function(nm) HYPERPRIORS[[nm]]$upper, numeric(1))
  lo <- log(lower)
  hi <- log(upper)
  g <- function(u) {
    v <- f(setNames(exp(pmin(pmax(u, lo), hi)), names))
    if (is.finite(v)) v else -1e18
  }
  u <- pmin(pmax(log(start), lo), hi)
  if (length(u) == 1) {
    o <- optimize(g, interval = c(max(lo, u - 3), min(hi, u + 3)), maximum = TRUE, tol = 1e-12)
    u <- o$maximum
  } else {
    o <- optim(u, g, gr = function(z) num_grad(g, z), method = "L-BFGS-B", lower = lo,
               upper = hi, control = list(fnscale = -1, factr = 1, pgtol = 0, maxit = 10000))
    u <- o$par
  }
  at_bound <- (u - lo < 1e-6) | (hi - u < 1e-6)
  if (!any(at_bound)) {
    for (it in 1:20) {
      step <- solve(num_hess(g, u), num_grad(g, u))
      cand <- u - step
      if (any(cand < lo | cand > hi) || g(cand) < g(u) - 1e-12) break
      u <- cand
      if (max(abs(step)) < 1e-10) break
    }
  }
  mode <- setNames(exp(u), names)
  grad <- if (any(at_bound)) rep(NA_real_, length(u)) else num_grad(g, u)
  hess_nat <- if (any(at_bound)) NULL else {
    fn <- function(x) f(setNames(x, names))
    num_hess(fn, unname(mode), h = 1e-5 * max(1, min(mode)))
  }
  sd_nat <- if (is.null(hess_nat)) NULL else sqrt(diag(solve(-hess_nat)))
  list(mode = mode, value = f(mode), grad_log = grad, at_bound = setNames(at_bound, names),
       laplace_sd = if (is.null(sd_nat)) NULL else as.list(setNames(sd_nat, names)))
}

CASES_C <- list(
  list(id = "qd_p4_rw_lambda", data = "qd", b = 1, dummies = character(0), conv = "glp"),
  list(id = "qd_p4_rw_lambda_soc_sur_glp", data = "qd", b = 1, dummies = c("soc", "sur"),
       conv = "glp"),
  list(id = "qd_p4_rw_lambda_soc_sur_bvar", data = "qd", b = 1, dummies = c("soc", "sur"),
       conv = "bvar"),
  list(id = "sim_p4_zero_lambda", data = "sim", b = 0, dummies = character(0), conv = "glp"),
  list(id = "sim_p4_rw_lambda_soc_sur_glp", data = "sim", b = 1, dummies = c("soc", "sur"),
       conv = "glp"),
  list(id = "sim_p4_zero_lambda_soc_sur_glp", data = "sim", b = 0, dummies = c("soc", "sur"),
       conv = "glp")
)

eb_runs <- lapply(CASES_C, function(cs) {
  ds <- DATA[[cs$data]]
  psi <- PSI[[cs$data]]$arima_ml_variance
  hyper <- c("lambda", cs$dummies)
  st <- make_setup(ds, LAGS, cs$b, psi, cs$dummies, cs$conv, hyper = hyper)
  set.seed(1L)
  fit <- bvar(ds$Yraw, lags = LAGS, n_draw = 20L, n_burn = 10L, priors = st$priors,
              verbose = FALSE)
  note_check(sprintf("%s: bvar() design equals the reference design", cs$id),
             max(abs(fit$meta$X - st$X), abs(fit$meta$Y - st$Y)), 0)
  opt <- fit$optim
  lower <- vapply(hyper, function(nm) HYPERPRIORS[[nm]]$lower, numeric(1))
  upper <- vapply(hyper, function(nm) HYPERPRIORS[[nm]]$upper, numeric(1))
  f_bvar <- function(v) bvar_eval(st, v, hyperprior = TRUE, lower = lower, upper = upper)$log_ml
  note_check(sprintf("%s: bvar_eval reproduces bvar()'s optim value", cs$id),
             abs(f_bvar(setNames(opt$par, hyper)) - opt$value), 1e-8)
  f_exact <- function(v) exact_log_posterior(st, v)
  start <- setNames(vapply(hyper, function(nm) HYPERPRIORS[[nm]]$mode, numeric(1)), hyper)
  r_bvar <- refine_mode(f_bvar, opt$par, hyper)
  r_exact <- refine_mode(f_exact, start, hyper)
  bvar_is_exact <- length(cs$dummies) == 0 || isTRUE(all(cs$b == 1))
  if (bvar_is_exact) {
    # The two objectives agree to ~1e-11 relative, but BVAR's eigenvalue-based one is noisier
    # (~1e-9 absolute), which limits how precisely its maximiser can be located (~1e-5).
    note_check(sprintf("%s: refined modes (bvar vs exact objective), max rel diff", cs$id),
               max(abs(r_bvar$mode - r_exact$mode) / r_exact$mode), 1e-4)
  }
  ex_mode <- exact_eval(st, r_exact$mode)
  list(
    id = cs$id,
    dataset = cs$data,
    lags = LAGS,
    mean = cs$b,
    alpha = ALPHA,
    var_const = VAR_CONST,
    psi = I(psi),
    psi_source = "arima_ml_variance",
    soc = if ("soc" %in% cs$dummies) MU else NULL,
    sur = if ("sur" %in% cs$dummies) DELTA else NULL,
    ybar0 = if (length(cs$dummies)) cs$conv else NULL,
    hyper = I(hyper),
    hyperpriors = HYPERPRIORS[hyper],
    bvar_is_exact = bvar_is_exact,
    bvar_optim = list(
      par = as_named_list(setNames(opt$par, hyper)),
      value = opt$value,
      convergence = opt$convergence,
      message = opt$message,
      counts = as.list(opt$counts),
      method = "stats::optim L-BFGS-B on the natural scale, numerical gradient (ndeps = 1e-3)"
    ),
    mode_bvar_objective = as_named_list(r_bvar$mode),
    log_posterior_bvar_objective = r_bvar$value,
    at_bound_bvar_objective = as.list(r_bvar$at_bound),
    mode = as_named_list(r_exact$mode),
    log_posterior = r_exact$value,
    log_ml = ex_mode$log_ml,
    log_hyperprior = log_hyperprior_terms(r_exact$mode),
    grad_log_scale = I(r_exact$grad_log),
    at_bound = as.list(r_exact$at_bound),
    laplace_sd = r_exact$laplace_sd
  )
})

write_fixture(list(
  meta = META,
  description = paste(
    "(c) Posterior mode of h maximising log p(Y | h) + log p(h) (Gamma hyperpriors, natural",
    "scale, no Jacobian), within the hyperprior bounds; alpha and psi fixed. bvar_optim: BVAR's",
    "own optim() result from bvar(). mode_bvar_objective: BVAR's objective (BVAR:::bv_ml)",
    "maximised to high precision from there. mode/log_posterior: the exact objective (QR-based,",
    "exact dummy-only marginal likelihood) maximised to high precision; identical to the BVAR",
    "objective when bvar_is_exact. For sim_p4_zero_lambda_soc_sur_glp BVAR's objective is",
    "degenerate (mode at the lower bounds) because its dummy term is not exact for b = 0."),
  datasets = lapply(DATA, dataset_meta),
  runs = eb_runs
), "r_bvar_eb.json")

# ================================================================================================
# (d) Metropolis-Hastings over the hyperparameters
# ================================================================================================

coef_selection <- function(M, variables, p) {
  sel <- rbind(
    cbind(0, seq_len(M) - 1),                 # constants
    cbind(seq_len(M), seq_len(M) - 1),         # own first lags
    c(2, 0),                                   # variable 2, lag 1, in equation 1
    c(1 + M, 0),                               # variable 1, lag 2, in equation 1
    c(3, 1)                                    # variable 3, lag 1, in equation 2
  )
  names <- regressor_names(variables, p)
  lapply(seq_len(nrow(sel)), function(i) {
    list(row = sel[i, 1], col = sel[i, 2],
         name = paste0(names[sel[i, 1] + 1], " -> ", variables[sel[i, 2] + 1]))
  })
}
SIGMA_SELECTION <- list(c(0, 0), c(1, 1), c(2, 2), c(0, 2))

CASES_D <- list(
  list(id = "qd_p4_rw_lambda_soc_sur_glp", data = "qd", b = 1, dummies = c("soc", "sur"),
       conv = "glp", scale_hess = c(0.012, 0.0016, 0.007)),
  list(id = "qd_p4_rw_lambda", data = "qd", b = 1, dummies = character(0), conv = "glp",
       scale_hess = 0.1),
  list(id = "sim_p4_zero_lambda", data = "sim", b = 0, dummies = character(0), conv = "glp",
       scale_hess = 0.06)
)

if (RUN_MH) {
  message("(d) Metropolis-Hastings (", MH_CHAINS, " chains x ", MH_DRAW - MH_BURN, " draws)")
  mh_runs <- lapply(CASES_D, function(cs) {
    ds <- DATA[[cs$data]]
    psi <- PSI[[cs$data]]$arima_ml_variance
    hyper <- c("lambda", cs$dummies)
    st <- make_setup(ds, LAGS, cs$b, psi, cs$dummies, cs$conv, hyper = hyper)
    mh <- bv_mh(scale_hess = cs$scale_hess, adjust_acc = TRUE)
    fits <- lapply(MH_SEEDS, function(seed) {
      set.seed(seed)
      t0 <- Sys.time()
      fit <- bvar(ds$Yraw, lags = LAGS, n_draw = MH_DRAW, n_burn = MH_BURN, priors = st$priors,
                  mh = mh, verbose = FALSE)
      message(sprintf("  %s seed %d: acceptance %.3f (%.1fs)", cs$id, seed,
                      fit$meta$accepted / fit$meta$n_save,
                      as.numeric(Sys.time() - t0, units = "secs")))
      fit
    })
    n_save <- fits[[1]]$meta$n_save
    hyper_chains <- lapply(hyper, function(nm) lapply(fits, function(f) f$hyper[, nm]))
    names(hyper_chains) <- hyper
    mc <- mcmc.list(lapply(fits, function(f) mcmc(f$hyper[, hyper, drop = FALSE])))
    ess <- effectiveSize(mc)
    rhat <- if (MH_CHAINS > 1) gelman.diag(mc, autoburnin = FALSE)$psrf[, 1] else NA
    coefs <- lapply(coef_selection(st$M, ds$variables, LAGS), function(cf) {
      c(cf, quantile_summary(lapply(fits, function(f) f$beta[, cf$row + 1, cf$col + 1])))
    })
    sigmas <- lapply(SIGMA_SELECTION, function(ij) {
      c(list(row = ij[1], col = ij[2]),
        quantile_summary(lapply(fits, function(f) f$sigma[, ij[1] + 1, ij[2] + 1])))
    })
    paths <- lapply(seq_along(fits), function(i) {
      set.seed(MH_SEEDS[i] + 100000L)
      simulate_paths(fits[[i]]$beta, fits[[i]]$sigma, ds$Yraw, LAGS, HORIZON)
    })
    native <- lapply(seq_along(fits), function(i) {
      set.seed(MH_SEEDS[i] + 200000L)
      predict(fits[[i]], horizon = HORIZON)$fcast
    })
    fc <- forecast_summary(paths, ds$variables)
    fc$noise <- "chol(Sigma) z, one path per stored MH draw"
    fc_native <- forecast_summary(native, ds$variables)
    fc_native$noise <- paste(
      "BVAR predict(): Sigma %*% z (covariance Sigma^2 instead of Sigma; BVAR bug).",
      "Reference only; do not use as a target.")
    list(
      id = cs$id,
      dataset = cs$data,
      lags = LAGS,
      mean = cs$b,
      alpha = ALPHA,
      var_const = VAR_CONST,
      psi = I(psi),
      psi_source = "arima_ml_variance",
      soc = if ("soc" %in% cs$dummies) MU else NULL,
      sur = if ("sur" %in% cs$dummies) DELTA else NULL,
      ybar0 = if (length(cs$dummies)) cs$conv else NULL,
      hyper = I(hyper),
      hyperpriors = HYPERPRIORS[hyper],
      settings = list(chains = MH_CHAINS, n_draw = MH_DRAW, n_burn = MH_BURN,
                      n_save_per_chain = n_save, seeds = I(MH_SEEDS),
                      scale_hess = I(cs$scale_hess), adjust_acc = TRUE,
                      mcse = paste("max(batch-quantile, delta-method (sparsity x batch-means",
                                   "SE of the indicator), between-chain); batches per chain:",
                                   MCSE_BATCHES)),
      optim_mode = as_named_list(setNames(fits[[1]]$optim$par, hyper)),
      acceptance = I(vapply(fits, function(f) f$meta$accepted / f$meta$n_save, numeric(1))),
      ess = as.list(ess),
      rhat = as.list(rhat),
      probs = I(PROBS),
      hyper_quantiles = lapply(hyper_chains, quantile_summary),
      coefficients = coefs,
      sigma = sigmas,
      forecast = fc,
      forecast_bvar_native = fc_native
    )
  })

  write_fixture(list(
    meta = META,
    description = paste(
      "(d) Hierarchical posterior by BVAR's random-walk Metropolis-Hastings on the natural",
      "scale of h (target log p(Y | h) + log p(h), truncated to the bounds), one (B, Sigma)",
      "draw from the conjugate posterior per stored iteration. Quantiles (R type 7 = numpy",
      "'linear') of the pooled chains with Monte Carlo standard errors. forecast: one path per",
      "stored draw with chol(Sigma) noise; forecast_bvar_native: BVAR's predict() (buggy noise)."),
    datasets = lapply(DATA, dataset_meta),
    runs = mh_runs
  ), "r_bvar_mh.json")
}

# ------------------------------------------------------------------------------------------------

failed <- Filter(function(x) !x$ok, checks)
message(sprintf("%d consistency checks, %d failed", length(checks), length(failed)))
for (x in failed) message("  FAILED: ", x$check, " = ", format(x$value, digits = 3))
worst <- function(pattern) {
  v <- vapply(Filter(function(x) grepl(pattern, x$check), checks), `[[`, numeric(1), "value")
  if (length(v)) max(v) else NA
}
message(sprintf("max rel diff log ML BVAR vs exact (where exact): %.2e",
                worst("rel diff log ML bvar vs exact")))
message(sprintf("max scaled diff posterior mean BVAR vs exact: %.2e",
                worst("posterior mean bvar vs exact")))
if (length(failed)) quit(status = 1)
