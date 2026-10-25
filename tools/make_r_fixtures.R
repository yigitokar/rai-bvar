#!/usr/bin/env Rscript
# =================================================================================================
# make_r_fixtures.R: numerical-oracle fixtures from R packages (BVAR; stochvol; bayesianVARs; bvars)
# =================================================================================================
#
# Four sections, selected with --sections (default: all four). No test job runs it: the Phase 3
# fixtures come from the dispatch-only `r-fixtures` job of .github/workflows/ci.yml (Phase 3
# DESIGN D-54), whose artifact is downloaded and committed. Reproducible: every random number
# comes from a fixed seed.
#
# bvar (Phase 1, SPEC §10.4): JSON fixtures read by tests/parity/test_r_bvar.py, from the CRAN
# package BVAR (Kuschnig & Vashold):
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
# stochvol and bvars (Phase 2, docs/phase2/DESIGN.md §9.9, work package WP8): see the section
# "Phase 2" at the end of this file. They read the NumPy datasets tests/parity/fixtures/data/*.csv
# (tools/make_sv_parity_data.py) and stage JSON in --staging; `python -I
# tools/make_sv_parity_data.py pack <staging>` turns it into tests/parity/fixtures/*.nc (no NetCDF
# writer for R is installed here).
#
#   stochvol  (e) stochvol runs (svsample; exact runs: its sampler, importance-weighted) on the
#                 datasets (i)–(v) of DESIGN §9.9
#                 -> tests/parity/fixtures/stochvol_<dataset>_<run>.nc
#   bvars     (f) bayesianVARs 0.1.8 (Phase 3 DESIGN §9.8, work package WP29): the convention
#                 probes P1–P10 and the parity runs R1–R9 when bayesianVARs is installed;
#                 otherwise P1–P5 are recorded as unresolved, with the evidence. The exact
#                 system-wide Gibbs oracle (N04 §9.5) written here, with stochvol's single-step SV
#                 update, replaced bayesianVARs in Phase 2 (skip it with --no-swa)
#                 -> tests/parity/fixtures/probe_bayesianvars.nc (P1–P5),
#                    probe_bayesianvars_p3.nc (P6–P10), probe_bvars_p8_*.nc (P8 data),
#                    bvars_bayesianvars_*.nc (R1, R2), bvars_gl_*.nc (R3–R9), bvars_swa_*.nc
#   wbvars    (h) bvars 1.1 (Woźniak; Phase 3 DESIGN §9.8): probes P11–P16 and the conjugate
#                 CommonSV parity runs -> tests/parity/fixtures/probe_wbvars.nc,
#                 probe_wbvars_p16.nc, wbvars_csv.nc, wbvars_ncsv.nc
#
# Usage, from anywhere:
#   Rscript tools/make_r_fixtures.R                       # everything (Phase 1 + Phase 2)
#   Rscript tools/make_r_fixtures.R --sections=bvar       # Phase 1 only
#   Rscript tools/make_r_fixtures.R --no-mh               # skip (d), the long MH chains
#   Rscript tools/make_r_fixtures.R --quick               # short chains (smoke test; do not commit;
#                                                         # mixing checks only warn)
#   Rscript tools/make_r_fixtures.R --sections=stochvol,bvars --staging=DIR
#   Rscript tools/make_r_fixtures.R --sections=stochvol --sv-only=stochvol_i_exact,stochvol_iv_exact
#                                                         # selected stochvol fixtures (same seeds)
#   Rscript tools/make_r_fixtures.R --sections=stochvol,bvars,wbvars --sv-only=stochvol_v_t5 \
#     --no-swa --staging=DIR                              # the Phase 3 job (ci.yml r-fixtures)
#   ... --only='bvars_gl_*,probe_*'                       # stage only these Phase 3 fixtures
#   python -I tools/make_sv_parity_data.py all            # data, the Phase 2 sections, pack
#
# Requirements: R >= 4.1 with BVAR 1.0.5, mvtnorm, jsonlite and coda (bvar section), stochvol
# (stochvol and bvars sections), bayesianVARs (bvars) and bvars with bsvars (wbvars). The Phase 3
# fixtures were made with R 4.5.3 and the packages of one Posit Package Manager snapshot
# (ci.yml job r-fixtures records it; every fixture's `packages` attribute lists the versions).
# Phase 1/2 example, in the user library:
#   lib <- Sys.getenv("R_LIBS_USER"); dir.create(lib, recursive = TRUE, showWarnings = FALSE)
#   install.packages(c("BVAR", "jsonlite", "coda", "stochvol"), lib = lib,
#                    repos = "https://cloud.r-project.org")
# (On macOS without gfortran, install the CRAN *binary* of mvtnorm first. bayesianVARs has no
# CRAN binary for R 4.1 and does not build without gfortran; see section (f).)
#
# Licensing (SPEC §2.1): BVAR and stochvol are GPL (BVAR GPL-3, stochvol GPL >= 2), bayesianVARs is
# GPL-3, bvars and bsvars are GPL (>= 3). This script *runs* them, including a few internal BVAR
# functions through `:::`, as numerical oracles; their outputs are data. It never prints, deparses
# or stores a function, closure, R6 method or call of theirs (as_data() drops them; errors and
# warnings are reported by their message only). Nothing in this file is part of the rai-bvar
# library, and no library code may be derived from their sources.
#
# Two kinds of numbers are produced by the bvar section and both are exported:
#   * "bvar":  computed by BVAR itself (bvar(), BVAR:::bv_ml, BVAR:::draw_post, predict()).
#   * "exact": computed by the independent, QR-based implementation of the textbook conjugate
#     formulas in the section "Reference formulas" below. It agrees with BVAR to ~1e-11 wherever
#     BVAR is exact, and it is the comparison target where BVAR is numerically less accurate
#     (posterior mean via normal equations, condition number ~1e10 on the levels data), deviates
#     from the exact marginal likelihood (dummy observations with a prior mean that is not a
#     random walk), or has a bug (forecast noise scaled by Σ instead of chol(Σ)).
# =================================================================================================

options(showErrorCalls = FALSE, showWarnCalls = FALSE, warn = 1)
args <- commandArgs(trailingOnly = TRUE)
RUN_MH <- !("--no-mh" %in% args)
NO_SWA <- "--no-swa" %in% args
QUICK <- "--quick" %in% args
arg_value <- function(name, default) {
  hit <- grep(paste0("^--", name, "="), args, value = TRUE)
  if (length(hit)) sub(paste0("^--", name, "="), "", hit[length(hit)]) else default
}
ALL_SECTIONS <- c("bvar", "stochvol", "bvars", "wbvars")
SECTIONS <- strsplit(arg_value("sections", paste(ALL_SECTIONS, collapse = ",")), ",")[[1]]
if (!length(SECTIONS) || !all(SECTIONS %in% ALL_SECTIONS)) {
  stop("--sections must be a comma-separated subset of: ", paste(ALL_SECTIONS, collapse = ", "))
}

suppressPackageStartupMessages({
  lib <- path.expand(Sys.getenv("R_LIBS_USER"))
  if (nzchar(lib) && dir.exists(lib)) .libPaths(c(lib, .libPaths()))
  library(jsonlite)
  if ("bvar" %in% SECTIONS) {
    library(BVAR)
    library(coda)
  }
  if (any(c("stochvol", "bvars") %in% SECTIONS)) library(stochvol)
})

script_path <- function() {
  file_arg <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
  if (length(file_arg) == 1) normalizePath(sub("^--file=", "", file_arg))
  else normalizePath(file.path("tools", "make_r_fixtures.R"), mustWork = TRUE)
}
ROOT <- dirname(dirname(script_path()))
FIXTURES <- file.path(ROOT, "tests", "parity", "fixtures")
stopifnot(dir.exists(FIXTURES))
STAGING <- path.expand(arg_value("staging", file.path(ROOT, "build", "r_fixtures_staging")))

# Consistency checks of every section; a failed hard check makes the script exit with status 1.
# A soft check (hard = FALSE: the mixing checks of the short --quick chains) only warns.
checks <- list()
note_check <- function(name, value, limit, hard = TRUE) {
  ok <- is.finite(value) && value <= limit
  checks[[length(checks) + 1]] <<- list(check = name, value = value, limit = limit, ok = ok,
                                        hard = hard)
  if (!ok) {
    warning(sprintf("%s: %s = %.3e > %.1e", if (hard) "check failed" else "soft check (warning)",
                    name, value, limit))
  }
}

# =================================================================================================
# Section "bvar" (Phase 1). The body is not indented, to keep it identical to the Phase 1 script.
# =================================================================================================

if ("bvar" %in% SECTIONS) {

if (as.character(packageVersion("BVAR")) != "1.0.5") {
  warning("fixtures were designed against BVAR 1.0.5; found ", packageVersion("BVAR"))
}

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

}  # end of section "bvar"

# =================================================================================================
# Phase 2 (docs/phase2/DESIGN.md §9.9; work package WP8)
# =================================================================================================
#
# Data: tests/parity/fixtures/data/*.csv, written by tools/make_sv_parity_data.py with NumPy (the
# data-generating processes live in our code; N02 §13.6, N03 §7.4). Output: one JSON object per
# fixture in STAGING, {name, attrs, coords, vars}, each array flat in C (row-major) order with its
# dims and shape; `python -I tools/make_sv_parity_data.py pack STAGING` writes the NetCDF files.
#
# Time conventions: data rows are t = 1..T; the log-variance h_t belongs to y_t, and t = 0 is the
# initial state h_0 (stochvol's `latent0`, stationary prior h_0 ~ N(μ, σ²/(1 − φ²))).
#
# Monte Carlo standard errors (N06 §10): Vehtari, Gelman, Simpson, Carpenter & Bürkner (2021,
# Bayesian Analysis 16(2), 667–718). Multi-chain ESS with Geyer's initial monotone sequence (§3.2);
# quantile MCSE from the ESS S of the indicator 1{θ ≤ θ̂_α}, mapping the Beta(Sα + 1, S(1 − α) + 1)
# quantiles at Φ(∓1) through the empirical quantile function (§4.3). A batch-quantile MCSE
# (P2_BATCHES batches per chain) is computed as a cross-check; `*_mcse` is the larger of the two.
# Quantiles are R type 7 (= numpy "linear").

P2_PROBS <- c(0.05, 0.25, 0.5, 0.75, 0.95)
P2_BATCHES <- 20L
P2_DATA <- file.path(FIXTURES, "data")
MCSE_METHOD <- paste(
  "q_mcse = max(Vehtari et al. 2021 sec. 4.3 quantile MCSE from the multi-chain ESS of the",
  "indicator 1{theta <= q}, batch-quantile MCSE with", P2_BATCHES, "batches per chain);",
  "mcse_mean = sd / sqrt(ess); ess = multi-chain ESS (Geyer initial monotone sequence, sec. 3.2);",
  "rhat = split R-hat; quantiles are R type 7 (numpy 'linear')")

p2_meta <- function(packages) {
  out <- list(
    generator = "tools/make_r_fixtures.R",
    r_version = R.version.string,
    platform = R.version$platform,
    blas = unname(extSoftVersion()["BLAS"]),
    lapack = La_version(),
    rng_kind = paste(RNGkind(), collapse = " / "),
    created = format(Sys.time(), "%Y-%m-%dT%H:%M:%S%z"),
    design = "docs/phase2/DESIGN.md sec. 9.9 (work package WP8)",
    mcse_method = MCSE_METHOD,
    quick = QUICK
  )
  for (pkg in packages) out[[paste0(pkg, "_version")]] <- as.character(packageVersion(pkg))
  out
}

read_p2_data <- function(file) {
  path <- file.path(P2_DATA, file)
  if (!file.exists(path)) {
    stop(path, " is missing: run `python -I tools/make_sv_parity_data.py data` first")
  }
  frame <- read.csv(path, check.names = FALSE, stringsAsFactors = FALSE)
  list(file = file, md5 = unname(tools::md5sum(path)), frame = frame)
}

# One array for the staged JSON: dims (outermost first), shape, data flat in C order. Doubles
# travel as "%.17g" strings (dtype "float64"): jsonlite prints at most 15 significant digits.
nc_array <- function(x, dims, description = NULL) {
  shape <- if (is.null(dim(x))) length(x) else dim(x)
  stopifnot(length(shape) == length(dims))
  flat <- if (length(shape) > 1) as.vector(aperm(x, rev(seq_along(shape)))) else as.vector(x)
  dtype <- if (is.integer(x)) "int32" else if (is.double(x)) "float64" else NULL
  if (identical(dtype, "float64")) flat <- sprintf("%.17g", flat)
  out <- list(dims = I(dims), shape = I(as.integer(shape)), data = I(flat))
  if (!is.null(dtype)) out$dtype <- dtype
  if (!is.null(description)) out$attrs <- list(description = description)
  out
}

stage <- function(name, attrs, coords, vars) {
  dir.create(STAGING, recursive = TRUE, showWarnings = FALSE)
  path <- file.path(STAGING, paste0(name, ".json"))
  write_json(list(name = name, attrs = attrs, coords = coords, vars = vars), path,
             auto_unbox = TRUE, digits = NA, null = "null", na = "null")
  message("staged ", path)
}

# ------------------------------------------------------------------------------------------------
# Monte Carlo summaries (Vehtari et al. 2021)
# ------------------------------------------------------------------------------------------------

autocov <- function(x) {   # biased autocovariances γ_0..γ_{N−1} by FFT (double arithmetic)
  N <- as.numeric(length(x))
  m <- as.numeric(nextn(2L * length(x)))
  f <- fft(c(x - mean(x), numeric(m - N)))
  Re(fft(f * Conj(f), inverse = TRUE))[seq_len(N)] / (m * N)
}

# Multi-chain ESS: ρ̂_t = 1 − (W − mean_m γ̂_{t,m}) / var⁺ (eq. 10), Geyer's initial positive
# sequence of pair sums P̂_k = ρ̂_{2k} + ρ̂_{2k+1}, made monotone, τ̂ = −1 + 2 Σ_k P̂_k (eq. 11).
ess_multi <- function(chains) {
  N <- min(lengths(chains))
  M <- as.numeric(length(chains))
  X <- matrix(vapply(chains, function(x) x[seq_len(N)], numeric(N)), N, M)
  if (all(apply(X, 2, function(x) max(x) == min(x)))) return(NA_real_)
  acov <- matrix(apply(X, 2, autocov), N, M)
  W <- mean(acov[1, ] * N / (N - 1))
  var_plus <- W * (N - 1) / N + (if (M > 1) var(colMeans(X)) else 0)
  rho <- 1 - (W - rowMeans(acov)) / var_plus
  rho[1] <- 1
  K <- N %/% 2L
  P <- rho[2L * seq_len(K) - 1L] + rho[2L * seq_len(K)]
  stop_at <- which(P[-1] <= 0)[1]
  if (!is.na(stop_at)) P <- P[seq_len(stop_at)]
  tau <- -1 + 2 * sum(cummin(P))
  NM <- as.numeric(N) * M
  min(NM / tau, NM * log10(NM))
}

mcse_quantile_vehtari <- function(chains, pooled, prob) {
  q <- quantile(pooled, prob, names = FALSE, type = 7)
  S <- ess_multi(lapply(chains, function(x) as.numeric(x <= q)))
  if (!is.finite(S)) return(NA_real_)
  ab <- qbeta(pnorm(c(-1, 1)), S * prob + 1, S * (1 - prob) + 1)
  th <- quantile(pooled, ab, names = FALSE, type = 7)
  (th[2] - th[1]) / 2
}

mcse_quantile_batch <- function(chains, probs) {
  bq <- do.call(rbind, lapply(chains, function(x) {
    idx <- split(seq_along(x), cut(seq_along(x), P2_BATCHES, labels = FALSE))
    t(vapply(idx, function(i) quantile(x[i], probs, names = FALSE, type = 7),
             numeric(length(probs))))
  }))
  apply(bq, 2, sd) / sqrt(nrow(bq))
}

rhat_split <- function(chains) {
  h <- min(lengths(chains)) %/% 2L
  X <- do.call(cbind, lapply(chains, function(x) cbind(x[seq_len(h)], x[h + seq_len(h)])))
  W <- mean(apply(X, 2, var))
  if (!(W > 0)) return(NA_real_)
  B <- h * var(colMeans(X))
  sqrt(((h - 1) / h * W + B / h) / W)
}

mc_summary <- function(chains, probs = P2_PROBS, weights = NULL) {
  if (!is.null(weights)) return(mc_summary_weighted(chains, weights, probs))
  pooled <- unlist(chains, use.names = FALSE)
  q <- quantile(pooled, probs, names = FALSE, type = 7)
  mv <- vapply(probs, function(p) mcse_quantile_vehtari(chains, pooled, p), numeric(1))
  mb <- mcse_quantile_batch(chains, probs)
  ess <- ess_multi(chains)
  s <- sd(pooled)
  list(q = q, mcse = pmax(mv, mb, na.rm = TRUE), mcse_vehtari = mv, mcse_batch = mb,
       mean = mean(pooled), sd = s, ess = ess, mcse_mean = s / sqrt(ess),
       rhat = rhat_split(chains))
}

# --- Importance-weighted summaries (the stochvol "exact" runs) ----------------------------------
#
# stochvol's correction for the mixture approximation (correct_model_misspecification = TRUE)
# leaves the chain unchanged and attaches a weight w_i ∝ p(y | h_i) / p_mix(y | h_i) to every
# draw, normalised to sum 1 within each chain. The exact posterior is estimated by self-normalised
# importance sampling of the raw (time-ordered) chain instead of svsample's resampling, so the
# chain's autocorrelation stays visible to the ESS and MCSE. Within each chain the weights are
# scaled to mean 1 (w̃), so the pooled estimators average the per-chain SNIS estimators:
#   F̂(x) = mean_i w̃_i 1{θ_i ≤ x},   mean_w = mean_i w̃_i θ_i,   sd_w² = mean_i w̃_i (θ_i − mean_w)².
# Linearisation (ratio estimator): F̂(q) − α ≈ mean_i z_i with z_i = w̃_i (1{θ_i ≤ q} − α), so
# s_α = sd(z) / sqrt(ESS(z)) is the MCSE of F̂(q_α), with ESS(z) the multi-chain ESS of the z
# chains; it contains both the autocorrelation and the weight variability. The quantile MCSE maps
# α ± s_α through the weighted quantile function (the weighted form of Vehtari et al. 2021 §4.3:
# with w̃ ≡ 1, s_α = sqrt(α(1 − α)/S) is the normal approximation of their Beta interval). The
# batch MCSE uses the weighted quantile of each batch. The mean's MCSE is sd(z) / sqrt(ESS(z))
# with z_i = w̃_i (θ_i − mean_w), and `ess` = sd_w² / mcse_mean² is the effective sample size
# of the weighted mean (= the multi-chain ESS when w̃ ≡ 1). R-hat is the split R-hat of the raw
# chains (a diagnostic of the Markov chain itself).

# Weighted quantile: linear interpolation of the sorted draws x_(k) at the positions
# p_k = C_{k−1} / (1 − w_(N)), C the cumulative normalised weight; equal weights give
# p_k = (k − 1)/(N − 1), R's type 7.
wquantile <- function(x, w, probs) {
  o <- order(x)
  xs <- x[o]
  ws <- w[o] / sum(w)
  p <- (cumsum(ws) - ws) / (1 - ws[length(ws)])
  approx(p, xs, xout = probs, rule = 2, ties = list("ordered", mean))$y
}

mc_summary_weighted <- function(chains, weights, probs = P2_PROBS) {
  stopifnot(identical(lengths(chains), lengths(weights)))
  w <- lapply(weights, function(wc) wc / mean(wc))   # w̃: mean 1 within each chain
  pooled <- unlist(chains, use.names = FALSE)
  pw <- unlist(w, use.names = FALSE)
  q <- wquantile(pooled, pw, probs)
  mv <- vapply(seq_along(probs), function(k) {
    z <- Map(function(x, wc) wc * (as.numeric(x <= q[k]) - probs[k]), chains, w)
    s <- sd(unlist(z, use.names = FALSE)) / sqrt(ess_multi(z))
    th <- wquantile(pooled, pw, pmin(pmax(probs[k] + c(-1, 1) * s, 0), 1))
    (th[2] - th[1]) / 2
  }, numeric(1))
  bq <- do.call(rbind, Map(function(x, wc) {
    idx <- split(seq_along(x), cut(seq_along(x), P2_BATCHES, labels = FALSE))
    t(vapply(idx, function(i) wquantile(x[i], wc[i], probs), numeric(length(probs))))
  }, chains, w))
  mb <- apply(bq, 2, sd) / sqrt(nrow(bq))
  m <- sum(pw * pooled) / sum(pw)
  s <- sqrt(sum(pw * (pooled - m)^2) / sum(pw))
  zc <- Map(function(x, wc) wc * (x - m), chains, w)
  mcse_mean <- sd(unlist(zc, use.names = FALSE)) / sqrt(ess_multi(zc))
  list(q = q, mcse = pmax(mv, mb, na.rm = TRUE), mcse_vehtari = mv, mcse_batch = mb,
       mean = m, sd = s, ess = (s / mcse_mean)^2, mcse_mean = mcse_mean,
       rhat = rhat_split(chains))
}

# Staged variables <prefix>_{q, mcse, mcse_vehtari, mcse_batch} (dims + quantile) and
# <prefix>_{mean, sd, ess, mcse_mean, rhat} (dims) for an array of scalar quantities.
# `chains_of(index)` returns the list of chains of the quantity at the (1-based) array index;
# `weights` (optional) is the list of per-chain importance weights shared by all quantities.
summary_vars <- function(prefix, chains_of, shape, dims, description, weights = NULL) {
  grid <- as.matrix(expand.grid(lapply(shape, seq_len)))   # column-major, like array()
  S <- lapply(seq_len(nrow(grid)), function(r) mc_summary(chains_of(grid[r, ]), weights = weights))
  per_q <- function(field) {
    array(t(vapply(S, `[[`, numeric(length(P2_PROBS)), field)), c(shape, length(P2_PROBS)))
  }
  scalar <- function(field) array(vapply(S, `[[`, numeric(1), field), shape)
  dq <- c(dims, "quantile")
  wtd <- if (is.null(weights)) "" else " (importance-weighted raw chains, see mc_summary_weighted)"
  vars <- list(
    nc_array(per_q("q"), dq, paste0(description, " - posterior quantiles", wtd)),
    nc_array(per_q("mcse"), dq, "MCSE of the quantiles: max(mcse_vehtari, mcse_batch)"),
    nc_array(per_q("mcse_vehtari"), dq, paste0("quantile MCSE, Vehtari et al. (2021) sec. 4.3",
                                               wtd)),
    nc_array(per_q("mcse_batch"), dq, paste0("quantile MCSE from batch quantiles", wtd)),
    nc_array(scalar("mean"), dims, paste0("posterior mean", wtd)),
    nc_array(scalar("sd"), dims, paste0("posterior standard deviation", wtd)),
    nc_array(scalar("ess"), dims, if (is.null(weights)) {
      "multi-chain effective sample size (all chains)"
    } else {
      "effective sample size of the weighted mean, sd^2 / mcse_mean^2 (autocorrelation and weights)"
    }),
    nc_array(scalar("mcse_mean"), dims, paste0("MCSE of the posterior mean", wtd)),
    nc_array(scalar("rhat"), dims, paste0("split R-hat", if (is.null(weights)) "" else
                                            " of the raw (unweighted) chains"))
  )
  names(vars) <- paste0(prefix, c("_q", "_mcse", "_mcse_vehtari", "_mcse_batch", "_mean", "_sd",
                                  "_ess", "_mcse_mean", "_rhat"))
  list(vars = vars, rhat = scalar("rhat"), ess = scalar("ess"))
}

# Mixing requirements of the committed runs (split R-hat < 1.01, ESS > 400). The short --quick
# chains are a smoke test: there the (looser) limits only warn, so `all --quick` reaches `pack`.
check_mixing <- function(label, sv) {
  note_check(sprintf("%s: max split R-hat - 1", label), max(sv$rhat, na.rm = TRUE) - 1,
             if (QUICK) 0.1 else 0.01, hard = !QUICK)
  note_check(sprintf("%s: 400 / min ESS", label), 400 / min(sv$ess, na.rm = TRUE),
             if (QUICK) 10 else 1, hard = !QUICK)
}

path_quantiles <- function(M) t(apply(M, 2, quantile, probs = P2_PROBS, names = FALSE, type = 7))

mcmc_columns <- function(mcl, cols) {   # list (per column) of lists (per chain) of numeric vectors
  lapply(cols, function(col) lapply(mcl, function(m) as.numeric(m[, col])))
}
mcmc_matrix <- function(mcl) do.call(rbind, lapply(mcl, function(m) {
  matrix(as.numeric(m), nrow = nrow(m))
}))

# =================================================================================================
# Phase 3 helpers (docs/phase3/DESIGN.md §9.8; work package WP29), shared by sections (e)–(h)
# =================================================================================================
#
# Licensing (SPEC §2.1, DESIGN Conventions "Clean room"): bayesianVARs, bvars, bsvars and stochvol
# are GPL. They are run as black boxes and only the DATA they return is recorded: numbers,
# dimensions, dimnames, field names. `as_data()` drops every function, closure, R6 method, call or
# formula before anything is described or staged, so no line of their code is ever printed,
# deparsed or stored. Warnings and errors are reported by their message only
# (options(showErrorCalls = FALSE, showWarnCalls = FALSE) at the top of this file).

P3_DESIGN <- "docs/phase3/DESIGN.md sec. 9.8 (work package WP29)"
P3_PACKAGES <- c("bayesianVARs", "bvars", "bsvars", "stochvol", "factorstochvol", "GIGrvg",
                 "RcppArmadillo", "Rcpp", "RcppProgress", "RcppTN", "R6", "mvtnorm", "jsonlite",
                 "coda")
`%||%` <- function(a, b) if (is.null(a) || (length(a) == 1L && is.na(a))) b else a

# --only=<glob,...> restricts the Phase 3 fixtures that are staged (e.g. --only='wbvars_*');
# the probes a staged fixture depends on are computed either way.
P3_ONLY <- arg_value("only", "")
P3_ONLY <- if (nzchar(P3_ONLY)) strsplit(P3_ONLY, ",", fixed = TRUE)[[1]] else character()
want <- function(name) {
  !length(P3_ONLY) || any(vapply(P3_ONLY, function(g) grepl(utils::glob2rx(g), name), logical(1)))
}
LAYOUT_DIR <- file.path(STAGING, "layout")   # not packed: evidence for the probes and the PR

package_record <- function(pkgs = P3_PACKAGES) {
  vapply(pkgs, function(p) {
    d <- suppressWarnings(tryCatch(utils::packageDescription(p), error = function(e) NULL))
    if (!inherits(d, "packageDescription")) return(sprintf("%s: not installed", p))
    sprintf("%s %s (Built: %s; Repository: %s)", p, d$Version, d$Built %||% "?",
            d$Repository %||% "?")
  }, character(1), USE.NAMES = FALSE)
}

os_release <- function() {
  f <- "/etc/os-release"
  if (!file.exists(f)) return(paste(Sys.info()[c("sysname", "release")], collapse = " "))
  v <- sub('^PRETTY_NAME="?([^"]*)"?$', "\\1", grep("^PRETTY_NAME=", readLines(f), value = TRUE))
  if (length(v)) v[1] else "unknown"
}

p3_meta <- function(packages, kind) {
  out <- p2_meta(unique(c("stochvol", packages)))
  out$design <- P3_DESIGN
  out$kind <- kind
  out$repos <- paste(getOption("repos"), collapse = " ")
  out$ppm_snapshot <- Sys.getenv("RAI_BVAR_PPM_SNAPSHOT", "not set")
  out$os <- os_release()
  out$packages <- paste(package_record(), collapse = "\n")
  out$ci_run <- Sys.getenv("RAI_BVAR_CI_RUN", "not set")
  out
}

is_code <- function(x) is.function(x) || is.language(x)

# Data content of a fit, prior or R6 object: lists and environments (R6 objects) are walked,
# functions, methods, calls and formulas are dropped without being looked at.
as_data <- function(x, depth = 4L) {
  if (is_code(x)) return(NULL)
  if (is.environment(x)) {
    if (depth <= 0L) return(NULL)
    out <- list()
    for (k in sort(ls(x))) {
      v <- tryCatch(get(k, envir = x, inherits = FALSE), error = function(e) NULL)
      if (is.null(v) || is_code(v)) next
      dv <- as_data(v, depth - 1L)
      if (!is.null(dv)) out[[k]] <- dv
    }
    return(out)
  }
  if (is.list(x) && !is.data.frame(x)) {
    x <- unclass(x)
    if (!length(x)) return(list())
    keep <- !vapply(x, is_code, logical(1))
    x <- x[keep]
    out <- lapply(x, as_data, depth = depth - 1L)
    return(out[!vapply(out, is.null, logical(1))])
  }
  x
}

# One line per data element: path, class, type, shape, (dim)names, and the values of short ones.
describe <- function(x, path = "x") {
  walk <- function(v, p) {
    if (is.null(v)) return(character())
    if (is.list(v) && !is.data.frame(v)) {
      nm <- names(v)
      if (is.null(nm)) nm <- rep("", length(v))
      nm[!nzchar(nm)] <- paste0("[[", which(!nzchar(nm)), "]]")
      return(unlist(lapply(seq_along(v), function(i) walk(v[[i]], paste0(p, "$", nm[i]))),
                    use.names = FALSE))
    }
    d <- dim(v)
    shape <- if (is.null(d)) paste("length", length(v)) else paste("dim", paste(d, collapse = " x "))
    dn <- dimnames(v)
    nms <- if (!is.null(dn)) {
      paste0("; dimnames: ", paste(vapply(dn, function(z) {
        if (is.null(z)) "-" else paste(utils::head(z, 10), collapse = ",")
      }, character(1)), collapse = " | "))
    } else if (!is.null(names(v))) paste0("; names: ", paste(utils::head(names(v), 12), collapse = ","))
    else ""
    vals <- if ((is.numeric(v) || is.logical(v) || is.character(v)) && length(v) <= 40) {
      paste0("; values: ", paste(if (is.numeric(v)) signif(as.vector(v), 8) else as.vector(v),
                                 collapse = ", "))
    } else if (is.numeric(v) && length(v)) {
      f <- v[is.finite(v)]
      sprintf("; range %.6g .. %.6g, %d non-finite", if (length(f)) min(f) else NA,
              if (length(f)) max(f) else NA, sum(!is.finite(v)))
    } else ""
    sprintf("%s: class %s, %s, %s%s%s", p, paste(class(v), collapse = "/"), typeof(v), shape, nms,
            vals)
  }
  walk(as_data(x), path)
}

record_layout <- function(file, lines) {
  dir.create(LAYOUT_DIR, recursive = TRUE, showWarnings = FALSE)
  writeLines(lines, file.path(LAYOUT_DIR, file))
  message("  layout record ", file.path("layout", file), " (", length(lines), " lines)")
}

# An array staged with dimensions named by their sizes (n<size>): for recorded data fields of
# arbitrary shape (P7 grouping matrices, P11 defaults).
nc_auto <- function(x, description = NULL) {
  d <- dim(x)
  attributes(x) <- NULL
  if (!is.null(d)) dim(x) <- d
  if (is.logical(x) || (is.integer(x) && anyNA(x))) storage.mode(x) <- "double"
  if (is.numeric(x) && !is.integer(x)) storage.mode(x) <- "double"
  shape <- if (is.null(dim(x))) length(x) else dim(x)
  if (length(shape) == 1L) x <- as.vector(x)
  nc_array(x, if (length(shape) == 1L) paste0("n", shape) else
    paste0("n", shape, "_", seq_along(shape)), description)
}

# Move the axis of size D (the draws) last; the last such axis if several have size D.
draws_last <- function(x, D) {
  x <- unclass(x)
  d <- dim(x)
  if (is.null(d)) {
    if (length(x) == D) return(array(x, c(1L, D)))
    stop(sprintf("no draws axis of size %d (a vector of length %d)", D, length(x)))
  }
  ax <- which(d == D)
  if (!length(ax)) stop(sprintf("no draws axis of size %d in an array of dim %s", D,
                                paste(d, collapse = " x ")))
  ax <- ax[length(ax)]
  if (ax == length(d)) return(x)
  aperm(x, c(setdiff(seq_along(d), ax), ax))
}

# Warnings are reported by their message only (no call: it could quote a line of a GPL package).
quiet_calls <- function(expr) {
  withCallingHandlers(expr, warning = function(w) {
    message("Warning: ", conditionMessage(w))
    invokeRestart("muffleWarning")
  })
}

attempt_probe <- function(id, expr) {
  t0 <- proc.time()[["elapsed"]]
  out <- tryCatch(expr, error = function(e) list(
    status = "unresolved", answer = "(probe failed)",
    evidence = paste("probe failed with the error:", conditionMessage(e))))
  out$seconds <- proc.time()[["elapsed"]] - t0
  message(sprintf("  %s: %s - %s (%.1fs)", id, out$status, out$answer, out$seconds))
  out
}

# The Phase 2 parity rule (DESIGN Conventions): m quantile comparisons pass when every
# |z| <= max(4, z_{1 - 0.01/(2m)}), z = difference / combined quantile MCSE.
parity_bound <- function(m) max(4, qnorm(1 - 0.01 / (2 * m)))
parity_z <- function(a, b) {   # two mc_summary lists (or q / mcse pairs)
  (a$q - b$q) / sqrt(a$mcse^2 + b$mcse^2)
}
parity_compare <- function(pairs) {   # list of list(a, b) mc_summary pairs
  z <- unlist(lapply(pairs, function(pr) parity_z(pr$a, pr$b)))
  m <- length(z)
  list(max_abs_z = max(abs(z)), m = m, bound = parity_bound(m), pass = max(abs(z)) <= parity_bound(m))
}

# Per-chain summaries of a list of chains of a scalar quantity (or with an exact reference).
chains_summary <- function(chains) mc_summary(lapply(chains, as.numeric))

check_mixing_p3 <- function(label, sv, hard = TRUE) {
  note_check(sprintf("%s: max split R-hat - 1", label), max(sv$rhat, na.rm = TRUE) - 1,
             if (QUICK) 0.1 else 0.01, hard = hard && !QUICK)
  note_check(sprintf("%s: 400 / min ESS", label), 400 / min(sv$ess, na.rm = TRUE),
             if (QUICK) 10 else 1, hard = hard && !QUICK)
}

mixing_summary <- function(blocks) {
  rh <- unlist(lapply(blocks, `[[`, "rhat"))
  es <- unlist(lapply(blocks, `[[`, "ess"))
  list(max_rhat = max(rh, na.rm = TRUE), min_ess = min(es, na.rm = TRUE),
       ok = as.integer(max(rh, na.rm = TRUE) < 1.01 && min(es, na.rm = TRUE) > 400))
}

# Thinned draws of a list of per-chain (draws x m) matrices as a (chain, draw, <dim>) array.
thinned_draws <- function(mats, keep_n) {
  D <- nrow(mats[[1]])
  keep <- unique(round(seq(1, D, length.out = min(keep_n, D))))
  out <- array(NA_real_, c(length(mats), length(keep), ncol(mats[[1]])))
  for (ch in seq_along(mats)) out[ch, , ] <- mats[[ch]][keep, , drop = FALSE]
  list(draws = out, keep = as.integer(keep))
}

# Probability of each support point of a discrete quantity, with the MCSE from the multi-chain
# ESS of its indicator (binomial MCSE with the effective sample size).
categorical_summary <- function(chains, support) {
  t(vapply(support, function(s) {
    ind <- lapply(chains, function(x) as.numeric(abs(x - s) <= 1e-12 * max(1, abs(s))))
    pr <- mean(unlist(ind))
    ess <- if (pr > 0 && pr < 1) ess_multi(ind) else NA_real_
    c(prob = pr, mcse = if (is.finite(ess)) sqrt(pr * (1 - pr) / ess) else 0, ess = ess)
  }, numeric(3)))
}

stage_probe_fixture <- function(name, attrs, probes, ids, vars = list(), coords = list()) {
  for (id in ids) {
    pr <- probes[[id]]
    for (field in c("question", "documented", "dataset", "design", "status", "answer",
                    "evidence")) {
      attrs[[paste0(id, "_", field)]] <- paste(pr[[field]] %||% "", collapse = "\n")
    }
    attrs[[paste0(id, "_seconds")]] <- pr$seconds %||% NA_real_
  }
  status <- vapply(ids, function(id) as.integer(identical(probes[[id]]$status, "answered")),
                   integer(1))
  coords$probe <- nc_array(ids, "probe")
  vars$answered <- nc_array(unname(status), "probe", "1 if answered, 0 if unresolved")
  stage(name, attrs, coords, vars)
}

# =================================================================================================
# (e) stochvol (DESIGN §9.9; N02 §13.6; N03 §7.4)
# =================================================================================================
#
# Block level: rai-bvar runs the production runner with ZeroMean, n = 1, on the same y. Each run
# derives both sides from one parameter set (DESIGN §6.3: μ ~ N(b_μ, B_μ) with priormu =
# c(b_μ, sqrt(B_μ)); (φ+1)/2 ~ Beta(a_φ, b_φ); σ² ~ B_σ χ²₁ with priorsigma = B_σ, i.e.
# sv_gamma(0.5, 0.5/B_σ)); the matching rai-bvar SV(...) is stored in the `rai_bvar_spec` attribute.
# All runs use offset 0 (the data contain no zeros, so stochvol adds none) and exact = FALSE,
# except "exact" (DESIGN §9.9, optional), which corrects the mixture approximation ex post.
# With correct_model_misspecification = TRUE stochvol's sampler leaves the chain unchanged and
# returns importance weights w_i ∝ p(y | h_i) / p_mix(y | h_i) for the parameter draws and for
# the (thinned) latent draws. svsample then resamples every chain with sample.int, which destroys
# the time order (the ESS, R-hat and MCSE of a resampled chain are meaningless) and does not touch
# latent0 (h_0 would stay uncorrected). The exact runs therefore call stochvol's sampler
# svsample_fast_cpp per chain, with the prior, the arguments and the start values svsample uses
# (sv_exact_code; a self-check confirms identical draws), keep the raw chains and their weights,
# and summarise by self-normalised importance sampling (mc_summary_weighted), h_0 included.
# Known oracle deviation (D-7): stochvol's non-centred step takes |σ| without flipping h̃.

SV_DRAWS <- if (QUICK) 2000L else 50000L
SV_BURN <- if (QUICK) 500L else 5000L
SV_CHAINS <- 4L
SV_THIN_LATENT <- 5L        # storage only: h (and τ) are kept every 5th draw
SV_KEEP_PER_CHAIN <- 500L   # thinned parameter draws stored per chain
SV_PARAMS <- c("mu", "phi", "sigma")
SV_MAIN_ARGS <- list(priormu = c(0, 10), priorphi = c(20, 1.5), priorsigma = 0.1,
                     priorlatent0 = "stationary")

SV_RUNS <- list(
  main = list(
    args = SV_MAIN_ARGS,
    rai_bvar = paste0("SV(mu_prior=Normal(0.0, 100.0), phi_prior=Beta(20, 1.5), ",
                      "sigma2_prior=ScaledChi2(0.1), h0=\"stationary\", offset=0.0, exact=False)")),
  defaults = list(
    args = list(),
    rai_bvar = paste0("SV(mu_prior=Normal(0.0, 1e4), phi_prior=Beta(5, 1.5), ",
                      "sigma2_prior=ScaledChi2(1.0), h0=\"stationary\", offset=0.0, exact=False)")),
  t5 = list(
    args = list(priorspec = quote(specify_priors(mu = sv_normal(0, 10), phi = sv_beta(20, 1.5),
                                                 sigma2 = sv_gamma(0.5, 0.5 / 0.1),
                                                 nu = sv_constant(5)))),
    rai_bvar = paste0("SV(mu_prior=Normal(0.0, 100.0), phi_prior=Beta(20, 1.5), ",
                      "sigma2_prior=ScaledChi2(0.1), h0=\"stationary\", tails=\"t\", ",
                      "nu_prior=DiscreteUniform(5, 5), t_scale=\"standardized\", offset=0.0, ",
                      "exact=False)")),
  exact = list(
    exact = TRUE,   # run by sv_exact_code; `args` gives the equivalent svsample call
    args = c(SV_MAIN_ARGS,
             list(expert = quote(list(correct_model_misspecification = TRUE)))),
    rai_bvar = paste0("SV(mu_prior=Normal(0.0, 100.0), phi_prior=Beta(20, 1.5), ",
                      "sigma2_prior=ScaledChi2(0.1), h0=\"stationary\", offset=0.0, exact=True)"))
)

SV_FIXTURES <- list(
  list(data = "i", runs = c("main", "defaults", "exact")),
  list(data = "ii", runs = c("main", "defaults")),
  list(data = "iii", runs = c("main", "defaults")),
  list(data = "iv", runs = c("main", "defaults", "exact")),
  list(data = "v", runs = "t5")
)
SV_SEED0 <- 20261100L

weight_ess <- function(w) sum(w)^2 / sum(w^2)

# R code of an exact run, evaluated as written (with `y` in scope) and stored as the `call`
# attribute: the prior svsample builds from SV_MAIN_ARGS, svsample's default start values, and
# one svsample_fast_cpp call per chain in sequence after one set.seed, as svsample does without
# the resampling step (stochvol 3.2.1). The value is the list of per-chain raw results.
sv_exact_code <- function(seed, draws, burnin, chains, correct = TRUE) {
  paste0(
    "priorspec <- specify_priors(mu = sv_normal(0, 10), phi = sv_beta(20, 1.5), ",
    "sigma2 = sv_gamma(0.5, 0.5/0.1), latent0_variance = \"stationary\"); ",
    "startpara <- list(mu = stochvol:::init_mu(y, priorspec), phi = 2 * mean(priorspec$phi) - 1, ",
    "sigma = sqrt(mean(priorspec$sigma2)), nu = Inf, rho = 0, beta = 0, latent0 = -10); ",
    sprintf("set.seed(%dL); ", seed),
    sprintf("lapply(seq_len(%dL), function(chain) stochvol::svsample_fast_cpp(y, ", chains),
    sprintf("draws = %dL, burnin = %dL, designmatrix = matrix(NA_real_), ", draws, burnin),
    sprintf("priorspec = priorspec, thinpara = 1L, thinlatent = %dL, ", SV_THIN_LATENT),
    "keeptime = \"all\", startpara = startpara, startlatent = rep(startpara$mu, length(y)), ",
    sprintf("keeptau = FALSE, print_settings = list(quiet = TRUE, chain = chain, n_chains = %dL), ",
            chains),
    sprintf("correct_model_misspecification = %s, interweave = TRUE, myoffset = 0, ",
            if (correct) "TRUE" else "FALSE"),
    "fast_sv = get_default_fast_sv()))")
}

eval_sv_code <- function(code, y) {
  env <- new.env(parent = globalenv())
  env$y <- y
  list(chains = eval(parse(text = code), env), priorspec = env$priorspec)
}

# Self-check: the direct sampler call (weights on) reproduces svsample's draws (weights off; no
# resampling) for two chains, i.e. the same prior, start values and random-number stream, and
# computing the weights does not change the chain.
sv_direct_call_check <- function() {
  y <- read_p2_data("sv_i.csv")$frame$y
  raw <- eval_sv_code(sv_exact_code(1L, 300L, 100L, 2L), y)$chains
  set.seed(1L)
  ref <- svsample(y, draws = 300L, burnin = 100L, n_chains = 2L, thinlatent = SV_THIN_LATENT,
                  quiet = TRUE, priormu = c(0, 10), priorphi = c(20, 1.5), priorsigma = 0.1,
                  priorlatent0 = "stationary")
  diff <- max(vapply(1:2, function(ch) {
    max(abs(raw[[ch]]$para[, SV_PARAMS] - as.matrix(ref$para[[ch]])[, SV_PARAMS]),
        abs(raw[[ch]]$latent - as.matrix(ref$latent[[ch]])),
        abs(raw[[ch]]$latent0 - as.matrix(ref$latent0[[ch]])))
  }, numeric(1)))
  note_check("stochvol: direct sampler call reproduces svsample, max |diff|", diff, 0)
  diff
}

run_stochvol_fixture <- function(ds_id, run_name, seed, direct_check = NA_real_) {
  data <- read_p2_data(sprintf("sv_%s.csv", ds_id))
  y <- data$frame$y
  stopifnot(all(is.finite(y)), all(y != 0))
  Tn <- length(y)
  run <- SV_RUNS[[run_name]]
  weighted <- isTRUE(run$exact)
  cl <- as.call(c(list(quote(svsample), quote(y), draws = SV_DRAWS, burnin = SV_BURN,
                       n_chains = SV_CHAINS, thinlatent = SV_THIN_LATENT, quiet = TRUE),
                  run$args))
  svsample_string <- paste(deparse(cl, width.cutoff = 500L), collapse = " ")
  name <- sprintf("stochvol_%s_%s", ds_id, run_name)
  t0 <- proc.time()[["elapsed"]]
  if (weighted) {
    call_string <- sv_exact_code(seed, SV_DRAWS, SV_BURN, SV_CHAINS)
    ev <- eval_sv_code(call_string, y)
    para <- lapply(ev$chains, `[[`, "para")
    latent <- lapply(ev$chains, `[[`, "latent")
    latent0 <- lapply(ev$chains, `[[`, "latent0")
    tau <- NULL
    priorspec <- ev$priorspec
    w_para <- lapply(ev$chains, function(r) as.numeric(r$correction_weight_para))
    w_latent <- lapply(ev$chains, function(r) as.numeric(r$correction_weight_latent))
    stopifnot(lengths(w_para) == SV_DRAWS, lengths(w_latent) == vapply(latent, nrow, 1L),
              vapply(latent0, nrow, 1L) == vapply(latent, nrow, 1L))
  } else {
    call_string <- svsample_string
    set.seed(seed)
    res <- eval(cl, list(y = y), globalenv())
    para <- res$para
    latent <- res$latent
    latent0 <- res$latent0
    tau <- res$tau
    priorspec <- res$priors
    w_para <- w_latent <- NULL
  }
  elapsed <- proc.time()[["elapsed"]] - t0
  message(sprintf("  %s: %.1fs", name, elapsed))

  infl_para <- infl_latent <- 1
  wess_para <- wess_latent <- NULL
  if (weighted) {
    wess_para <- vapply(w_para, weight_ess, numeric(1))
    wess_latent <- vapply(w_latent, weight_ess, numeric(1))
    infl_para <- sqrt(sum(lengths(w_para)) / sum(wess_para))
    infl_latent <- sqrt(sum(lengths(w_latent)) / sum(wess_latent))
  }

  pc <- mcmc_columns(para, SV_PARAMS)
  par_sum <- summary_vars("param", function(ix) pc[[ix[1]]], length(SV_PARAMS), "param",
                          "SV parameters (mu, phi, sigma; sigma is the s.d. of the h innovation)",
                          weights = w_para)
  t_sel <- as.integer(c(0L, 1L, Tn %/% 4L, Tn %/% 2L, Tn))
  h_chains <- function(t) {
    if (t == 0L) lapply(latent0, function(m) as.numeric(m[, 1]))
    else lapply(latent, function(m) as.numeric(m[, t]))
  }
  hc <- lapply(t_sel, h_chains)
  h_sum <- summary_vars("h", function(ix) hc[[ix[1]]], length(t_sel), "t_sel",
                        "log-variance h_t at t_sel (thinned latent draws)", weights = w_latent)
  check_mixing(name, list(rhat = c(par_sum$rhat, h_sum$rhat), ess = c(par_sum$ess, h_sum$ess)))
  H <- cbind(mcmc_matrix(latent0), mcmc_matrix(latent))   # draws x (T + 1)
  if (weighted) {
    wl <- unlist(lapply(w_latent, function(w) w / mean(w)), use.names = FALSE)
    h_path_q <- t(apply(H, 2, wquantile, w = wl, probs = P2_PROBS))
    h_path_mean <- colSums(H * wl) / sum(wl)
    h_path_sd <- sqrt(colSums(wl * sweep(H, 2, h_path_mean)^2) / sum(wl))
  } else {
    h_path_q <- path_quantiles(H)
    h_path_mean <- colMeans(H)
    h_path_sd <- apply(H, 2, sd)
  }

  keep <- round(seq(1, SV_DRAWS, length.out = SV_KEEP_PER_CHAIN))
  draws <- array(NA_real_, c(SV_CHAINS, SV_KEEP_PER_CHAIN, length(SV_PARAMS)))
  for (ch in seq_len(SV_CHAINS)) for (k in seq_along(SV_PARAMS)) {
    draws[ch, , k] <- pc[[k]][[ch]][keep]
  }

  coords <- list(
    quantile = nc_array(P2_PROBS, "quantile"),
    param = nc_array(SV_PARAMS, "param"),
    t_sel = nc_array(t_sel, "t_sel"),
    t = nc_array(0:Tn, "t"),
    obs = nc_array(seq_len(Tn), "obs"),
    chain = nc_array(seq_len(SV_CHAINS), "chain"),
    draw = nc_array(as.integer(keep), "draw")
  )
  wtd <- if (weighted) ", importance-weighted" else ""
  vars <- c(
    list(y = nc_array(y, "obs", "the data y_t, t = 1..T (copied from the CSV)")),
    par_sum$vars, h_sum$vars,
    list(
      h_path_q = nc_array(h_path_q, c("t", "quantile"), paste0(
        "quantiles of h_t for t = 0..T (thinned latent draws", wtd, ", no MCSE)")),
      h_path_mean = nc_array(h_path_mean, "t", paste0("posterior mean of h_t, t = 0..T", wtd)),
      h_path_sd = nc_array(h_path_sd, "t", paste0("posterior s.d. of h_t, t = 0..T", wtd)),
      param_draws = nc_array(draws, c("chain", "draw", "param"), paste0(
        "evenly thinned parameter draws (iteration index in `draw`)",
        if (weighted) "; raw chain of the approximate sampler, weight with param_draws_weight"))
    )
  )
  if (weighted) {
    pw <- t(vapply(w_para, function(w) (w / mean(w))[keep], numeric(SV_KEEP_PER_CHAIN)))
    vars$param_draws_weight <- nc_array(pw, c("chain", "draw"), paste(
      "importance weight of each param_draws row (stochvol correction_weight_para scaled to",
      "mean 1 over all draws of its chain)"))
  }
  if (!is.null(tau)) {
    tau_t <- as.integer(sort(order(abs(y), decreasing = TRUE)[1:3]))
    tc <- lapply(tau_t, function(t) lapply(tau, function(m) as.numeric(m[, t])))
    tau_sum <- summary_vars("tau", function(ix) tc[[ix[1]]], length(tau_t), "tau_t",
                            "standardized-t mixing variable tau_t at the 3 largest |y_t|")
    check_mixing(paste(name, "tau"), tau_sum)
    coords$tau_t <- nc_array(tau_t, "tau_t")
    TAU <- mcmc_matrix(tau)
    vars <- c(vars, tau_sum$vars,
              list(tau_path_q = nc_array(path_quantiles(TAU), c("obs", "quantile"),
                                         "quantiles of tau_t, t = 1..T (thinned, no MCSE)")))
  }

  attrs <- c(p2_meta("stochvol"), list(
    kind = "stochvol",
    dataset = ds_id,
    run = run_name,
    data_file = file.path("data", data$file),
    data_md5 = data$md5,
    T = Tn,
    call = call_string,
    seed = seed,
    draws = SV_DRAWS, burnin = SV_BURN, chains = SV_CHAINS, thinlatent = SV_THIN_LATENT,
    rai_bvar_spec = run$rai_bvar,
    rai_bvar_setup = paste(
      "block level: production runner with ZeroMean, n = 1, on y (DESIGN sec. 9.9); compare",
      "param_q and h_q within 4 * sqrt(mcse_R^2 + mcse_ours^2); exclude the fixture if our",
      "non-centred fold fraction exceeds 0.5% (D-7)"),
    priors = paste(capture.output(print(priorspec)), collapse = "\n"),
    time_convention = paste(
      "obs = t = 1..T indexes y; h_t belongs to y_t; t = 0 is h_0 (stochvol latent0,",
      "h_0 ~ N(mu, sigma^2/(1 - phi^2)) under priorlatent0 = 'stationary'); t_sel = {0, 1,",
      "T %/% 4, T %/% 2, T}"),
    oracle_deviation = paste(
      "stochvol's non-centred step takes |sigma| without flipping h-tilde (DESIGN D-7)"),
    resampled = FALSE,
    importance_weighted = weighted,
    weight_ess_para = if (weighted) I(wess_para) else NULL,
    weight_ess_latent = if (weighted) I(wess_latent) else NULL,
    mcse_inflation_para = infl_para,
    mcse_inflation_latent = infl_latent,
    runtime_seconds = elapsed
  ))
  if (weighted) {
    attrs$correction <- paste(
      "exact posterior by self-normalised importance sampling of the raw (time-ordered) chains:",
      "stochvol's correction weights (correct_model_misspecification = TRUE; the chain itself is",
      "that of the approximate sampler) scaled to mean 1 within each chain, para weights for mu,",
      "phi, sigma and latent weights for h_0..h_T (latent0 is thinned like latent, so h_0 is",
      "corrected too). No resampling: ESS, R-hat and MCSE are those of the raw chains, and the",
      "MCSE and ESS of the weighted estimators include the weight variability (see mcse_method)")
    attrs$mcse_method <- paste(MCSE_METHOD, "| importance-weighted runs: weighted quantiles",
                               "(positions C_{k-1} / (1 - w_(N)), type 7 for equal weights);",
                               "quantile MCSE = max(weighted Vehtari analogue: alpha +- sd(z) /",
                               "sqrt(ess(z)), z = w (1{theta <= q} - alpha), mapped through the",
                               "weighted quantile function; weighted batch quantiles);",
                               "mcse_mean = sd(z)/sqrt(ess(z)), z = w (theta - mean);",
                               "ess = sd^2 / mcse_mean^2; rhat = split R-hat of the raw chains")
    attrs$mcse_inflation_note <- paste(
      "sqrt(draws / sum of the per-chain weight ESS), for information only: the weighted MCSE",
      "already contain the weight variability and are not multiplied by it")
    attrs$equivalent_svsample_call <- paste(
      svsample_string, "-- the same sampler, prior, start values and random-number stream;",
      "svsample would then resample each chain with sample.int, which this run omits")
    attrs$check_direct_call_max_abs_diff <- direct_check
  }
  if (as.character(packageVersion("stochvol")) != "3.2.9") {
    attrs$stochvol_version_note <- paste(
      "DESIGN sec. 9.9 names stochvol 3.2.9; these fixtures were generated with",
      as.character(packageVersion("stochvol")), "(the version installed here)")
  }
  if (run_name == "t5") attrs <- c(attrs, sv_t5_four_obs(), list(design_p3 = paste(
    "regenerated for Phase 3 (DESIGN sec. 9.8, 0.2 row 19; P2 sec. 2.10 Stage 4 amendment 1)",
    "with the same seed and dataset (v) under", R.version.string, "and stochvol",
    as.character(packageVersion("stochvol")))))
  stage(name, attrs, coords, vars)
}

# SV-t on the four-observation problem of tests/parity/SV_PARAMETRIZATION.md §3.2 (y = (0.4, −2.5,
# 0.3, 0.6), μ ~ N(−1, 0.25), (φ+1)/2 ~ Beta(20, 1.5), σ² ~ 0.1·χ²₁, ν = 5 fixed): stochvol's
# quantiles of μ, h_2 and τ_2, stored with the SV-t fixture so that the stochvol version that made
# it can be judged against the exact importance-sampling posterior of that section without R.
SV_T5_EXACT <- list(mu = c(-1.495, -1.042, -0.732, -0.425, 0.017),
                    h2 = c(-1.291, -0.759, -0.353, 0.124, 1.022),
                    tau2 = c(0.681, 1.361, 2.263, 3.888, 9.171))

sv_t5_four_obs <- function(seed = 20261199L) {
  draws <- if (QUICK) 20000L else 200000L
  code <- sprintf(paste0(
    "set.seed(%dL); svsample(c(0.4, -2.5, 0.3, 0.6), draws = %dL, burnin = 10000L, ",
    "n_chains = 2L, priorspec = specify_priors(mu = sv_normal(-1, 0.5), phi = sv_beta(20, 1.5), ",
    "sigma2 = sv_gamma(0.5, 0.5/0.1), nu = sv_constant(5), latent0_variance = \"stationary\"), ",
    "quiet = TRUE)"), seed, draws)
  tryCatch({
    res <- eval(parse(text = code), new.env(parent = globalenv()))
    if (is.null(res$tau)) stop("svsample returned no tau draws")
    col <- function(mcl, j) unlist(lapply(mcl, function(m) as.numeric(as.matrix(m)[, j])),
                                   use.names = FALSE)
    q <- function(x) quantile(x, P2_PROBS, names = FALSE, type = 7)
    mu <- q(col(res$para, "mu"))
    h2 <- q(col(res$latent, 2L))
    tau2 <- q(col(res$tau, 2L))
    list(check_4obs_mu_q = I(mu), check_4obs_h2_q = I(h2), check_4obs_tau2_q = I(tau2),
         check_4obs_exact_mu_q = I(SV_T5_EXACT$mu), check_4obs_exact_h2_q = I(SV_T5_EXACT$h2),
         check_4obs_exact_tau2_q = I(SV_T5_EXACT$tau2),
         check_4obs_max_abs_diff = max(abs(c(mu - SV_T5_EXACT$mu, h2 - SV_T5_EXACT$h2,
                                             tau2 - SV_T5_EXACT$tau2))),
         check_4obs_call = code,
         check_4obs_note = paste(
           "quantiles 5/25/50/75/95% of stochvol's posterior of mu, h_2 and tau_2 on the",
           "four-observation SV-t problem of SV_PARAMETRIZATION.md sec. 3.2, next to the exact",
           "importance-sampling quantiles of that section (check_4obs_exact_*); stochvol 3.2.1",
           "gave mu -1.549..0.029, h_2 -1.630..1.018, tau_2 0.697..7.681 (ASIS)"))
  }, error = function(e) list(check_4obs_error = conditionMessage(e)))
}

if ("stochvol" %in% SECTIONS) {
  message("(e) stochvol (", SV_CHAINS, " chains x ", SV_DRAWS, " draws)")
  if (as.character(packageVersion("stochvol")) != "3.2.9") {
    message("  note: DESIGN sec. 9.9 names stochvol 3.2.9; using ", packageVersion("stochvol"))
  }
  SV_ONLY <- arg_value("sv-only", "")   # e.g. --sv-only=stochvol_i_exact,stochvol_iv_exact
  SV_ONLY <- if (nzchar(SV_ONLY)) strsplit(SV_ONLY, ",", fixed = TRUE)[[1]] else character()
  selected <- unlist(lapply(SV_FIXTURES, function(fx) sprintf("stochvol_%s_%s", fx$data, fx$runs)))
  if (length(SV_ONLY)) selected <- intersect(selected, SV_ONLY)
  # the direct-call check calls stochvol's sampler with 3.2.1's arguments: only for exact runs
  direct_check <- if (any(grepl("_exact$", selected))) sv_direct_call_check() else NA_real_
  i <- 0L
  for (fx in SV_FIXTURES) for (run_name in fx$runs) {
    if (!length(SV_ONLY) || sprintf("stochvol_%s_%s", fx$data, run_name) %in% SV_ONLY) {
      quiet_calls(run_stochvol_fixture(fx$data, run_name, SV_SEED0 + i, direct_check))
    }
    i <- i + 1L
  }
}

# =================================================================================================
# (f) bayesianVARs (DESIGN §9.9; N04 §3.4, §9.5, §9.8, §10) and the system-wide Gibbs oracle
# =================================================================================================
#
# bayesianVARs 0.1.8 is the oracle DESIGN §9.9 names for the reduced-form sampler. It cannot be
# installed in the environment that generated the committed fixtures: it has no CRAN binary for
# R 4.1, and building it from source needs gfortran (R's FLIBS), which is missing. Therefore:
#
# * The convention probes P1–P5 are recorded in probe_bayesianvars.nc as unresolved, with the
#   environment evidence, the documented conventions they test and the probe design (datasets
#   tests/parity/fixtures/data/probe_*.csv). When bayesianVARs is installed, run_bvars_probes()
#   fits the probes and records the answers it can decide, plus the layout of every array it
#   returns (str()), as evidence. That path has never been executed: review it on first use.
# * The bayesianVARs parity runs are replaced by an exact system-wide Gibbs oracle (N04 §9.5):
#   vec(Π) | A, Λ is drawn jointly from its Gaussian full conditional (precision
#   Q = P₀ + Σ_t Ω_t ⊗ x_t x_t', Ω_t = A'Λ_t⁻¹A), the rows of A by Bayesian regression, and each
#   equation's SV state by one call of stochvol's single-step fast sampler (svsample_fast_cpp with
#   draws = 1, burnin = 0, started at the current state: indicators | h, then h, then (μ, φ, σ)
#   with ASIS, as bayesianVARs does through stochvol's update_fast_sv) on e_jt = (A u_t)_j with
#   offset 0. Π and A are exact conditionals, so the chain targets the same posterior as
#   rai-bvar's CCCM sampler with SV(exact=False, offset=0.0); it is written here, in R, sharing no
#   code with the JAX engine. The fixtures use one parameter set, stored as arrays, from which the
#   rai-bvar arrays are built directly (priors.reduced.explicit_reduced), so P1–P5 do not arise.

SWA_CHAINS <- 4L
SWA_DRAWS <- if (QUICK) 600L else 20000L
SWA_BURN <- if (QUICK) 300L else 2000L
SWA_PATH_THIN <- 10L
SWA_HORIZON <- 4L

# The parameter set (DESIGN §9.9): π_j ~ N(m_j, diag v_j) with m = 0.5 on the own first lag and 0
# elsewhere, v = (0.4/l)² (own lag l), (0.2/l)² (cross lag l), s_c² = 10² (intercept);
# a_ij ~ N(0, 1); SV(mu_prior=Normal(0, 1e4), phi_prior=Beta(20, 1.5), sigma2_prior=ScaledChi2(1),
# h0="stationary", offset=0, exact=False); Constant: λ_i ~ IG(shape 3, scale 2).
SWA_PRIOR <- list(m_own = 0.5, own_sd = 0.4, cross_sd = 0.2, intercept_sd = 10, a_var = 1,
                  b_mu = 0, B_mu = 1e4, a_phi = 20, b_phi = 1.5, B_sigma = 1,
                  ig_shape = 3, ig_scale = 2)
SWA_FIXTURES <- list(
  list(name = "bvars_swa_n3p2_sv", data = "var_sv_n3.csv", lags = 2L, vol = "sv",
       seed = 20261201L),
  list(name = "bvars_swa_n3p2_const", data = "var_sv_n3.csv", lags = 2L, vol = "constant",
       seed = 20261202L),
  list(name = "bvars_swa_n7p4_sv", data = "var_sv_n7.csv", lags = 4L, vol = "sv",
       seed = 20261203L)
)

swa_priorspec <- function() {
  specify_priors(mu = sv_normal(SWA_PRIOR$b_mu, sqrt(SWA_PRIOR$B_mu)),
                 phi = sv_beta(SWA_PRIOR$a_phi, SWA_PRIOR$b_phi),
                 sigma2 = sv_gamma(0.5, 0.5 / SWA_PRIOR$B_sigma),
                 latent0_variance = "stationary")
}

swa_design <- function(Yraw, p) {   # x_t = (1, y_{t−1}', …, y_{t−p}')' (rai-bvar layout)
  rows <- (p + 1):nrow(Yraw)
  X <- do.call(cbind, c(list(rep(1, length(rows))),
                        lapply(seq_len(p), function(l) Yraw[rows - l, , drop = FALSE])))
  list(Y = Yraw[rows, , drop = FALSE], X = unname(X))
}

swa_regressor_names <- function(variables, p) {
  c("const", unlist(lapply(seq_len(p), function(l) paste0(variables, ".L", l))))
}

swa_prior_arrays <- function(n, p) {
  k <- 1L + n * p
  mean <- matrix(0, k, n)
  var <- matrix(SWA_PRIOR$intercept_sd^2, k, n)
  for (l in seq_len(p)) for (i in seq_len(n)) {
    row <- 1L + (l - 1L) * n + i              # lag l of variable i, in every equation j
    var[row, ] <- (SWA_PRIOR$cross_sd / l)^2
    var[row, i] <- (SWA_PRIOR$own_sd / l)^2
  }
  mean[1L + seq_len(n), ] <- diag(SWA_PRIOR$m_own, n)
  free <- lower.tri(diag(n))
  list(mean = mean, var = var,
       a_mean = matrix(0, n, n), a_var = ifelse(free, SWA_PRIOR$a_var, 1),
       ig_shape = rep(SWA_PRIOR$ig_shape, n), ig_scale = rep(SWA_PRIOR$ig_scale, n))
}

# Q = diag(P₀) + Σ_t Ω_t ⊗ x_t x_t' and b = P₀ m + Σ_t (Ω_t y_t) ⊗ x_t for
# vec(Π) = (π_1', …, π_n')', built blockwise: block (j, l) = X' diag(ω_{jl,·}) X with ω_{jl,t} = Σ_i a_ij a_il / λ_it.
swa_pi_system <- function(Y, X, A, Lam, P0, b0) {
  n <- ncol(Y)
  k <- ncol(X)
  iL <- 1 / Lam
  Q <- diag(P0, n * k)
  blk <- function(j) (j - 1L) * k + seq_len(k)
  for (j in seq_len(n)) for (l in j:n) {
    G <- crossprod(X * drop(iL %*% (A[, j] * A[, l])), X)
    Q[blk(j), blk(l)] <- Q[blk(j), blk(l)] + G
    if (l > j) Q[blk(l), blk(j)] <- G
  }
  Oy <- ((Y %*% t(A)) * iL) %*% A          # row t: (A'Λ_t⁻¹A y_t)'
  list(Q = Q, b = b0 + as.vector(crossprod(X, Oy)))
}

gaussian_draw <- function(Q, b) {   # θ ~ N(Q⁻¹b, Q⁻¹): two triangular solves, no inverse
  R <- chol(Q)
  mean <- backsolve(R, backsolve(R, b, transpose = TRUE))
  drop(mean + backsolve(R, rnorm(length(b))))
}

swa_draw_pi <- function(Y, X, A, Lam, P0, b0) {
  s <- swa_pi_system(Y, X, A, Lam, P0, b0)
  matrix(gaussian_draw(s$Q, s$b), ncol(X), ncol(Y))
}

# Row i of A (e_it = u_it + Σ_{j<i} a_ij u_jt ~ N(0, λ_it)): regression of −u_i on u_{<i}.
swa_a_system <- function(U, Lam, a_mean, a_var, i) {
  js <- seq_len(i - 1L)
  Z <- U[, js, drop = FALSE]
  Zw <- Z / Lam[, i]
  list(Q = crossprod(Zw, Z) + diag(1 / a_var[i, js], i - 1L),
       b = a_mean[i, js] / a_var[i, js] - drop(crossprod(Zw, U[, i])))
}

swa_draw_a <- function(U, Lam, a_mean, a_var) {
  n <- ncol(U)
  A <- diag(n)
  for (i in seq_len(n)[-1]) {
    s <- swa_a_system(U, Lam, a_mean, a_var, i)
    A[i, seq_len(i - 1L)] <- gaussian_draw(s$Q, s$b)
  }
  A
}

# One stochvol fast-sampler sweep (indicators, h_{0:T}, then μ, φ, σ with ASIS) at offset 0.
sv_single_step <- function(e, st, priorspec, fast_sv) {
  r <- svsample_fast_cpp(e, draws = 1L, burnin = 0L, priorspec = priorspec,
                         startpara = list(mu = st$mu, phi = st$phi, sigma = st$sigma, nu = Inf,
                                          rho = 0, beta = 0, latent0 = st$h0),
                         startlatent = st$h, myoffset = 0, fast_sv = fast_sv)
  list(mu = r$para[1, "mu"], phi = r$para[1, "phi"], sigma = r$para[1, "sigma"],
       h0 = r$latent0[1, 1], h = as.numeric(r$latent[1, ]))
}

swa_chain <- function(Y, X, prior, vol, t_sel, priorspec, fast_sv) {
  n <- ncol(Y)
  k <- ncol(X)
  Tn <- nrow(Y)
  P0 <- 1 / as.vector(prior$var)
  b0 <- P0 * as.vector(prior$mean)
  Pi <- qr.coef(qr(X), Y)
  A <- diag(n)
  E <- Y - X %*% Pi
  sv <- lam <- NULL
  if (vol == "sv") {
    sv <- lapply(seq_len(n), function(j) {
      m <- log(mean(E[, j]^2))
      list(mu = m, phi = 0.9, sigma = 0.3, h0 = m, h = rep(m, Tn))
    })
  } else {
    lam <- colMeans(E^2)
  }
  D <- SWA_DRAWS
  out <- list(Pi = array(NA_real_, c(D, k, n)), A = array(NA_real_, c(D, n, n)))
  if (vol == "sv") {
    out$mu <- out$phi <- out$sigma <- matrix(NA_real_, D, n)
    out$h_sel <- array(NA_real_, c(D, length(t_sel), n))
    out$h_path <- array(NA_real_, c(D %/% SWA_PATH_THIN, Tn + 1L, n))
  } else {
    out$lam <- matrix(NA_real_, D, n)
  }
  for (it in seq_len(SWA_BURN + D)) {
    Lam <- if (vol == "sv") exp(vapply(sv, `[[`, numeric(Tn), "h")) else
      matrix(lam, Tn, n, byrow = TRUE)
    Pi <- swa_draw_pi(Y, X, A, Lam, P0, b0)
    U <- Y - X %*% Pi
    A <- swa_draw_a(U, Lam, prior$a_mean, prior$a_var)
    E <- U %*% t(A)
    if (vol == "sv") {
      for (j in seq_len(n)) sv[[j]] <- sv_single_step(E[, j], sv[[j]], priorspec, fast_sv)
    } else {
      lam <- (prior$ig_scale + 0.5 * colSums(E^2)) / rgamma(n, prior$ig_shape + Tn / 2)
    }
    d <- it - SWA_BURN
    if (d < 1L) next
    out$Pi[d, , ] <- Pi
    out$A[d, , ] <- A
    if (vol == "sv") {
      hfull <- vapply(sv, function(s) c(s$h0, s$h), numeric(Tn + 1L))   # (T + 1) x n
      out$mu[d, ] <- vapply(sv, `[[`, numeric(1), "mu")
      out$phi[d, ] <- vapply(sv, `[[`, numeric(1), "phi")
      out$sigma[d, ] <- vapply(sv, `[[`, numeric(1), "sigma")
      out$h_sel[d, , ] <- hfull[t_sel + 1L, , drop = FALSE]
      if (d %% SWA_PATH_THIN == 0L) out$h_path[d %/% SWA_PATH_THIN, , ] <- hfull
    } else {
      out$lam[d, ] <- lam
    }
  }
  out
}

# One forecast path per stored draw: h_{T+s} from the AR(1) (SV) or λ (Constant),
# y_{T+s} = Π'x_{T+s} + A⁻¹ diag(√λ_{T+s}) ε.
swa_forecast <- function(out, Yraw, p, vol, H) {
  D <- dim(out$Pi)[1]
  n <- dim(out$Pi)[3]
  last <- as.vector(t(Yraw[nrow(Yraw):(nrow(Yraw) - p + 1L), , drop = FALSE]))
  lags <- matrix(rep(last, each = D), D)
  h <- if (vol == "sv") out$h_sel[, dim(out$h_sel)[2], ] else NULL   # h_T (t_sel ends at T)
  paths <- array(NA_real_, c(D, H, n))
  for (s in seq_len(H)) {
    x <- cbind(1, lags)
    m <- vapply(seq_len(n), function(j) rowSums(x * out$Pi[, , j]), numeric(D))
    if (vol == "sv") {
      h <- out$mu + out$phi * (h - out$mu) + out$sigma * matrix(rnorm(D * n), D)
      sd <- exp(h / 2)
    } else {
      sd <- sqrt(out$lam)
    }
    u <- sd * matrix(rnorm(D * n), D)
    for (i in seq_len(n)[-1]) for (j in seq_len(i - 1L)) u[, i] <- u[, i] - out$A[, i, j] * u[, j]
    y <- m + u
    paths[, s, ] <- y
    lags <- cbind(y, lags[, seq_len(n * (p - 1L)), drop = FALSE])
  }
  paths
}

# Deterministic self-checks of the oracle's two exact steps against dense formulas.
swa_self_checks <- function() {
  set.seed(99L)
  n <- 3L
  k <- 4L
  Tn <- 12L
  X <- cbind(1, matrix(rnorm(Tn * (k - 1L)), Tn))
  Y <- matrix(rnorm(Tn * n), Tn)
  A <- diag(n)
  A[lower.tri(A)] <- rnorm(n * (n - 1) / 2)
  Lam <- matrix(exp(rnorm(Tn * n)), Tn)
  P0 <- exp(rnorm(n * k))
  b0 <- rnorm(n * k)
  s <- swa_pi_system(Y, X, A, Lam, P0, b0)
  Qd <- diag(P0)
  bd <- b0
  for (t in seq_len(Tn)) {
    Om <- crossprod(A, diag(1 / Lam[t, ]) %*% A)
    Qd <- Qd + kronecker(Om, tcrossprod(X[t, ]))
    bd <- bd + kronecker(drop(Om %*% Y[t, ]), X[t, ])
  }
  note_check("SWA: blockwise precision vs dense Kronecker sum (scaled)",
             max(abs(s$Q - Qd)) / max(abs(Qd)), 1e-13)
  note_check("SWA: blockwise b vs dense Kronecker sum (scaled)",
             max(abs(s$b - bd)) / max(abs(bd)), 1e-13)
  # A rows: the gradient of the row-i log density at the conditional mean is zero.
  U <- Y
  a_mean <- matrix(rnorm(n * n), n)
  a_var <- matrix(exp(rnorm(n * n)), n)
  worst <- 0
  for (i in 2:n) {
    sa <- swa_a_system(U, Lam, a_mean, a_var, i)
    m <- drop(solve(sa$Q, sa$b))
    js <- seq_len(i - 1L)
    logd <- function(a) {
      e <- U[, i] + drop(U[, js, drop = FALSE] %*% a)
      -0.5 * sum(e^2 / Lam[, i]) - 0.5 * sum((a - a_mean[i, js])^2 / a_var[i, js])
    }
    g <- vapply(js, function(j) {
      d <- replace(numeric(i - 1L), j, 1e-5)
      (logd(m + d) - logd(m - d)) / 2e-5
    }, numeric(1))
    worst <- max(worst, max(abs(g)) / max(1, max(abs(sa$b))))
  }
  note_check("SWA: A-row gradient at the conditional mean (finite differences)", worst, 1e-6)
  list(pi_system = max(abs(s$Q - Qd)) / max(abs(Qd)), a_gradient = worst)
}

# stochvol single-step chain vs svsample on dataset (i) with the main priors: the single-step use
# inside the oracle must target the same posterior (max |z| over quantiles of μ, φ, σ, h_T).
sv_single_step_check <- function() {
  y <- read_p2_data("sv_i.csv")$frame$y
  Tn <- length(y)
  draws <- if (QUICK) 1000L else 20000L
  pri <- specify_priors(mu = sv_normal(0, 10), phi = sv_beta(20, 1.5),
                        sigma2 = sv_gamma(0.5, 0.5 / 0.1), latent0_variance = "stationary")
  fs <- get_default_fast_sv()
  single <- lapply(seq_len(4L), function(ch) {
    set.seed(20261300L + ch)
    st <- list(mu = -1, phi = 0.9, sigma = 0.3, h0 = -1, h = rep(-1, Tn))
    out <- matrix(NA_real_, draws, 4L)
    for (it in seq_len(1000L + draws)) {
      st <- sv_single_step(y, st, pri, fs)
      if (it > 1000L) out[it - 1000L, ] <- c(st$mu, st$phi, st$sigma, st$h[Tn])
    }
    out
  })
  set.seed(20261305L)
  ref <- svsample(y, draws = draws, burnin = 1000L, n_chains = 4L, thinlatent = 5L, quiet = TRUE,
                  priormu = c(0, 10), priorphi = c(20, 1.5), priorsigma = 0.1,
                  priorlatent0 = "stationary")
  rc <- c(mcmc_columns(ref$para, SV_PARAMS), mcmc_columns(ref$latent, Tn))
  z <- unlist(lapply(seq_len(4L), function(k) {
    a <- mc_summary(lapply(single, function(m) m[, k]))
    b <- mc_summary(rc[[k]])
    abs(a$q - b$q) / sqrt(a$mcse^2 + b$mcse^2)
  }))
  note_check("SWA: stochvol single-step chain vs svsample, max |z| (20 quantiles)", max(z), 4.5)
  max(z)
}

run_swa_fixture <- function(fx, self_checks, single_step_z) {
  data <- read_p2_data(fx$data)
  variables <- setdiff(names(data$frame), "date")
  Yraw <- as.matrix(data$frame[, variables])
  storage.mode(Yraw) <- "double"
  p <- fx$lags
  n <- ncol(Yraw)
  des <- swa_design(Yraw, p)
  Tn <- nrow(des$Y)
  k <- ncol(des$X)
  prior <- swa_prior_arrays(n, p)
  t_sel <- as.integer(c(0L, 1L, Tn %/% 4L, Tn %/% 2L, (3L * Tn) %/% 4L, Tn))
  priorspec <- swa_priorspec()
  fs <- get_default_fast_sv()
  t0 <- proc.time()[["elapsed"]]
  chains <- lapply(seq_len(SWA_CHAINS), function(ch) {
    set.seed(fx$seed + 1000L * ch)
    out <- swa_chain(des$Y, des$X, prior, fx$vol, t_sel, priorspec, fs)
    out$fc <- swa_forecast(out, Yraw, p, fx$vol, SWA_HORIZON)
    out
  })
  elapsed <- proc.time()[["elapsed"]] - t0
  message(sprintf("  %s: %.1fs", fx$name, elapsed))

  free <- which(lower.tri(diag(n)), arr.ind = TRUE)
  free <- free[order(free[, 1], free[, 2]), , drop = FALSE]
  chains_at <- function(field, idx) lapply(chains, function(ch) {
    x <- ch[[field]]
    switch(length(idx), x[, idx[1]], x[, idx[1], idx[2]])
  })
  blocks <- list(
    summary_vars("Pi", function(ix) chains_at("Pi", ix), c(k, n), c("regressor", "equation"),
                 "reduced-form coefficients Pi[regressor, equation] (y_t = Pi'x_t + u_t)"),
    summary_vars("A", function(ix) chains_at("A", free[ix[1], ]), nrow(free), "a_free",
                 "free entries a_ij (i > j) of the unit lower-triangular A (A u_t = e_t)"),
    summary_vars("forecast", function(ix) lapply(chains, function(ch) ch$fc[, ix[1], ix[2]]),
                 c(SWA_HORIZON, n), c("horizon", "equation"),
                 "predictive y_{T+s}, one path per stored draw")
  )
  coords <- list(
    quantile = nc_array(P2_PROBS, "quantile"),
    regressor = nc_array(swa_regressor_names(variables, p), "regressor"),
    equation = nc_array(variables, "equation"),
    a_free = nc_array(sprintf("a%d%d", free[, 1], free[, 2]), "a_free"),
    a_row = nc_array(as.integer(free[, 1]), "a_free"),
    a_col = nc_array(as.integer(free[, 2]), "a_free"),
    horizon = nc_array(seq_len(SWA_HORIZON), "horizon")
  )
  if (fx$vol == "sv") {
    sv_names <- c("mu", "phi", "sigma")
    blocks <- c(blocks, list(
      summary_vars("sv", function(ix) chains_at(sv_names[ix[1]], ix[2]), c(3L, n),
                   c("sv_param", "equation"), "SV parameters (mu, phi, sigma) per equation"),
      summary_vars("h", function(ix) chains_at("h_sel", ix), c(length(t_sel), n),
                   c("t_sel", "equation"), "log-variance h_it at t_sel")
    ))
    coords$sv_param <- nc_array(sv_names, "sv_param")
    coords$t_sel <- nc_array(t_sel, "t_sel")
    coords$t <- nc_array(0:Tn, "t")
    HP <- do.call(abind_first, lapply(chains, `[[`, "h_path"))
    hq <- array(NA_real_, c(Tn + 1L, n, length(P2_PROBS)))
    for (j in seq_len(n)) hq[, j, ] <- path_quantiles(HP[, , j])
    extra <- list(h_path_q = nc_array(hq, c("t", "equation", "quantile"),
                                      "quantiles of h_it, t = 0..T (every 10th draw, no MCSE)"))
  } else {
    blocks <- c(blocks, list(
      summary_vars("lambda", function(ix) chains_at("lam", ix), n, "equation",
                   "constant variances lambda_i of e_it")))
    extra <- list()
  }
  for (b in blocks) check_mixing(fx$name, b)
  vars <- c(
    unlist(lapply(blocks, `[[`, "vars"), recursive = FALSE),
    extra,
    list(
      prior_mean = nc_array(prior$mean, c("regressor", "equation"), "prior mean of Pi"),
      prior_var = nc_array(prior$var, c("regressor", "equation"), "prior variance of Pi"),
      a_prior_mean = nc_array(prior$a_mean[free], "a_free", "prior mean of a_ij"),
      a_prior_var = nc_array(prior$a_var[free], "a_free", "prior variance of a_ij")
    )
  )
  if (fx$vol == "constant") {
    vars$ig_shape <- nc_array(prior$ig_shape, "equation", "lambda_i ~ IG(shape, scale)")
    vars$ig_scale <- nc_array(prior$ig_scale, "equation", "lambda_i ~ IG(shape, scale)")
  }
  attrs <- c(p2_meta("stochvol"), list(
    kind = "bvars_swa",
    oracle = paste(
      "exact system-wide Gibbs (N04 sec. 9.5), written in tools/make_r_fixtures.R: joint draw of",
      "vec(Pi) | A, Lambda; rows of A | Pi, Lambda; per equation one stochvol fast-sampler sweep",
      "(svsample_fast_cpp, draws = 1, burnin = 0, myoffset = 0) on e_t = A u_t, or",
      "lambda_i | e ~ IG(shape + T/2, scale + sum(e^2)/2). Replaces bayesianVARs (unavailable)."),
    replaces = paste("bayesianVARs 0.1.8 parity runs of DESIGN sec. 9.9 (blocked; see",
                     "probe_bayesianvars.nc)"),
    model = paste(
      "y_t = Pi'x_t + u_t, x_t = (1, y_{t-1}', ..., y_{t-p}')', A u_t = e_t with A unit lower",
      "triangular (A[i, j] = a_ij for i > j), e_it ~ N(0, lambda_it); SV: lambda_it = exp(h_it),",
      "h_it = mu_i + phi_i (h_{i,t-1} - mu_i) + sigma_i eta_it, h_i0 stationary"),
    vol = fx$vol,
    lags = p, n = n, T = Tn, k = k,
    data_file = file.path("data", data$file),
    data_md5 = data$md5,
    seed = fx$seed,
    chain_seeds = I(fx$seed + 1000L * seq_len(SWA_CHAINS)),
    draws = SWA_DRAWS, burnin = SWA_BURN, chains = SWA_CHAINS, h_path_thin = SWA_PATH_THIN,
    gibbs_order = "Pi | A, Lambda; A | Pi, Lambda; volatility | Pi, A",
    prior_coef = paste(
      "pi_j ~ N(m_j, diag v_j), independent over equations j: m = 0.5 on the own first lag, 0",
      "elsewhere; v = (0.4/l)^2 own lag l, (0.2/l)^2 cross lag l, 10^2 intercept (prior_mean,",
      "prior_var)"),
    prior_a = "a_ij ~ N(0, 1) independently (a_prior_mean, a_prior_var)",
    prior_vol = if (fx$vol == "sv") paste(
      "mu_i ~ N(0, 1e4), (phi_i + 1)/2 ~ Beta(20, 1.5), sigma_i^2 ~ 1 * chi2_1, h_i0 stationary;",
      "stochvol specify_priors(mu = sv_normal(0, 100), phi = sv_beta(20, 1.5),",
      "sigma2 = sv_gamma(0.5, 0.5), latent0_variance = 'stationary')") else
      "lambda_i ~ IG(shape = 3, scale = 2), density prop. to lambda^(-shape-1) exp(-scale/lambda)",
    rai_bvar_spec = if (fx$vol == "sv") paste(
      "BVAR(Linear(Minnesota(form='reduced')), SV(mu_prior=Normal(0.0, 1e4),",
      "phi_prior=Beta(20, 1.5), sigma2_prior=ScaledChi2(1.0), h0='stationary', offset=0.0,",
      "exact=False)) through fit_mcmc with explicit arrays") else paste(
      "BVAR(Linear(Minnesota(form='reduced')), Constant('ig')) through fit_mcmc with explicit",
      "arrays and ConstantPrior(shape=ig_shape, scale=ig_scale)"),
    rai_bvar_arrays = paste(
      "priors.reduced.explicit_reduced(mean0 = prior_mean.T, var0 = prior_var.T, a_mean0, a_var0):",
      "rai-bvar CoefPriorArrays are equation-major (n, k), the fixture arrays are",
      "(regressor, equation); a_free lists a_ij row-major over i > j"),
    layout = paste(
      "regressor order: const, lag-1 block (variables in data order), ..., lag-p block;",
      "t_sel = {0, 1, T %/% 4, T %/% 2, 3T %/% 4, T} with t = 0 the initial state h_0;",
      "forecast horizon s = 1..4 after the last data row"),
    compare = paste(
      "quantiles of Pi, A, h (t >= 1), mu, phi, sigma (or lambda) and forecasts within",
      "4 * sqrt(mcse_R^2 + mcse_ours^2)"),
    check_pi_system_rel_err = self_checks$pi_system,
    check_a_gradient = self_checks$a_gradient,
    check_stochvol_single_step_max_abs_z = single_step_z,
    runtime_seconds = elapsed
  ))
  stage(fx$name, attrs, coords, vars)
}

abind_first <- function(...) {   # stack arrays along the first axis
  xs <- list(...)
  d <- dim(xs[[1]])
  out <- array(NA_real_, c(sum(vapply(xs, function(x) dim(x)[1], numeric(1))), d[-1]))
  at <- 0L
  for (x in xs) {
    rows <- at + seq_len(dim(x)[1])
    out[rows, , ] <- x
    at <- at + dim(x)[1]
  }
  out
}

# ------------------------------------------------------------------------------------------------
# bayesianVARs environment, probes P1–P5 and the derived parity calls
# ------------------------------------------------------------------------------------------------

r_config <- function(var) {
  out <- tryCatch(system2(file.path(R.home("bin"), "R"), c("CMD", "config", var), stdout = TRUE,
                          stderr = TRUE), error = function(e) conditionMessage(e))
  paste(out, collapse = " ")
}

bvars_environment <- function() {
  flibs <- r_config("FLIBS")
  lib_dirs <- sub("^-L", "", regmatches(flibs, gregexpr("-L[^ ]+", flibs))[[1]])
  list(
    bayesianvars_available = requireNamespace("bayesianVARs", quietly = TRUE),
    bayesianvars_target_version = "0.1.8",
    gfortran_on_path = unname(Sys.which("gfortran")),
    r_config_fc = r_config("FC"),
    r_config_flibs = flibs,
    flibs_dirs_exist = paste(sprintf("%s: %s", lib_dirs, dir.exists(lib_dirs)), collapse = "; "),
    cran_binary_note = paste(
      "bayesianVARs (first CRAN release 2023) has no CRAN macOS binary for R 4.1; a source",
      "build links RcppArmadillo code against R's FLIBS (gfortran), which is not installed")
  )
}

# The bayesianVARs call derived from the bvars_swa parameter set (DESIGN §9.9, E11). The two
# readings of P1 give two candidate calls; record both until P1 is answered.
bvars_parity_calls <- function() {
  vapply(SWA_FIXTURES, function(fx) {
    vol <- if (fx$vol == "sv") paste0(
      "cholesky_heteroscedastic = TRUE, cholesky_priormu = c(0, 100), cholesky_priorphi = ",
      "c(20, 1.5), cholesky_priorsigma2 = c(0.5, 0.5), cholesky_priorh0 = \"stationary\"") else
      "cholesky_heteroscedastic = FALSE, cholesky_priorhomoscedastic = cbind(rep(3, M), rep(2, M))"
    sprintf(paste0(
      "%s: bvar(data, lags = %d, draws = D, burnin = B, thin = 1, sv_keep = \"all\", ",
      "prior_intercept = 10, prior_phi = specify_prior_phi(data = data, lags = %d, ",
      "prior = \"normal\", normal_sds = <prior_var lag rows, column-major (P1: variance) or ",
      "their square roots (P1: s.d.)>, priormean = 0.5), prior_sigma = specify_prior_sigma(",
      "data = data, type = \"cholesky\", cholesky_U_prior = \"normal\", cholesky_normal_sds = 1, ",
      "%s)); map A = t(U) (P2), drop the intercept row of PHI (P3), read sv_para by P4"),
      fx$name, fx$lags, fx$lags, vol)
  }, character(1))
}

# ------------------------------------------------------------------------------------------------
# bayesianVARs: running it, and reading its output by the layout the probes settle
# ------------------------------------------------------------------------------------------------
#
# Every bayesianVARs call is built as R code text from one parameter set, evaluated as written and
# stored verbatim in the fixture's `call` attribute (E11). The layout of the returned arrays is
# not documented in full; the probes below settle it from the package's own output (the returned
# design matrix X, the documented vcov(), posterior moments on data with known answers) and
# every parity run re-derives its regressor map from its own X.

BV_PROBE_DRAWS <- if (QUICK) 2000L else 20000L
BV_PROBE_BURN <- if (QUICK) 500L else 2000L
BV_PROBE_SEED <- 20261400L
BV_DRAWS <- if (QUICK) 2000L else 50000L
BV_BURN <- if (QUICK) 500L else 10000L
BV_RUNS_N <- 4L
BV_HORIZON <- 4L
BV_PATH_THIN <- 10L
BV_HYPER_KEEP <- 500L
BV_SV_ARGS <- paste0("cholesky_heteroscedastic = TRUE, cholesky_priormu = c(0, 100), ",
                     "cholesky_priorphi = c(20, 1.5), cholesky_priorsigma2 = c(0.5, 0.5), ",
                     "cholesky_priorh0 = \"stationary\"")
# The simulated n = 3 DGP of var_sv_n3.csv (tools/make_sv_parity_data.py var_parameters):
# B_l[i, j] is the coefficient of equation i on lag l of variable j.
BV_N3_DGP <- list(c = c(0.5, -0.3, 0.2),
                  B = list(rbind(c(0.5, 0.1, 0.0), c(0.05, 0.4, 0.1), c(0.0, -0.1, 0.6)),
                           rbind(c(0.15, 0.0, 0.0), c(0.0, 0.1, 0.0), c(0.05, 0.0, 0.1))))

bv_data <- function(file) {
  d <- read_p2_data(file)
  vars <- setdiff(names(d$frame), "date")
  Y <- as.matrix(d$frame[, vars, drop = FALSE])
  storage.mode(Y) <- "double"
  list(file = file, md5 = d$md5, variables = vars, Y = Y)
}

bv_call <- function(lags, draws, burnin, phi_args, sigma_args, intercept = "10") {
  sprintf(paste0(
    "bvar(data, lags = %dL, draws = %dL, burnin = %dL, thin = 1L, sv_keep = \"all\", ",
    "quiet = TRUE, prior_intercept = %s, prior_phi = specify_prior_phi(data = data, lags = %dL, ",
    "%s), prior_sigma = specify_prior_sigma(data = data, type = \"cholesky\", %s, quiet = TRUE))"),
    lags, draws, burnin, intercept, lags, phi_args, sigma_args)
}

num_text <- function(x) if (x == 0) "0" else sprintf("%.17g", x)

bv_sigma_args <- function(vol, homo = "cbind(rep(3, M), rep(2, M))", normal_sds = "1",
                          u_tol = 0) {
  paste0("cholesky_U_prior = \"normal\", cholesky_normal_sds = ", normal_sds,
         ", cholesky_U_tol = ", num_text(u_tol), ", ",
         if (vol == "sv") BV_SV_ARGS else
           paste0("cholesky_heteroscedastic = FALSE, cholesky_priorhomoscedastic = ", homo))
}

bv_eval <- function(call, vars, seed) {
  env <- new.env(parent = globalenv())
  for (k in names(vars)) assign(k, vars[[k]], envir = env)
  set.seed(seed)
  t0 <- proc.time()[["elapsed"]]
  fit <- eval(parse(text = call), env)
  list(fit = fit, seconds = proc.time()[["elapsed"]] - t0, call = call)
}

bv_probe_fit <- function(file, lags, phi_args, sigma_args, intercept, draws = BV_PROBE_DRAWS,
                         burnin = BV_PROBE_BURN, seed = BV_PROBE_SEED, Y = NULL, vars = list()) {
  ds <- bv_data(file)
  data <- if (is.null(Y)) ds$Y else Y
  call <- bv_call(lags, draws, burnin, phi_args, sigma_args, intercept)
  r <- bv_eval(call, c(list(data = data, M = ncol(data)), vars), seed)
  c(r, list(data = data, ds = ds, lags = lags, D = draws))
}

# PHI as (K, M, D) in bayesianVARs' row order.
bv_phi <- function(f, M, K, D) {
  P <- draws_last(f$PHI, D)
  if (length(dim(P)) != 3L) stop("PHI is not a 3-D array")
  if (dim(P)[1] != K && dim(P)[2] == K) P <- aperm(P, c(2, 1, 3))
  if (dim(P)[1] != K || dim(P)[2] != M) stop("PHI has dim ", paste(dim(P), collapse = " x "))
  P
}

# logvar as (Tlv, M, D); Tlv is 1 (constant variances), T or T + 1.
bv_logvar <- function(f, M, D) {
  L <- draws_last(f$logvar, D)
  if (length(dim(L)) == 2L) L <- array(L, c(1L, dim(L)))
  if (dim(L)[2] != M && dim(L)[1] == M) L <- aperm(L, c(2, 1, 3))
  if (dim(L)[2] != M) stop("logvar has dim ", paste(dim(L), collapse = " x "))
  L
}
lv_row <- function(t, Tlv, Tn) if (Tlv == 1L) 1L else if (Tlv == Tn + 1L) t + 1L else t
lv_at <- function(lv, row, idx) t(matrix(lv[row, , idx], nrow = dim(lv)[2]))   # length(idx) x M

u_draw <- function(U, d) if (length(dim(U)) == 3L) U[, , d] else U[, d]

tri_index <- function(M, which) {
  switch(which,
         upper_col = do.call(rbind, lapply(2:M, function(j) cbind(seq_len(j - 1L), j))),
         upper_row = do.call(rbind, lapply(1:(M - 1L), function(i) cbind(i, (i + 1L):M))),
         lower_col = do.call(rbind, lapply(1:(M - 1L), function(j) cbind((j + 1L):M, j))),
         lower_row = do.call(rbind, lapply(2:M, function(i) cbind(i, seq_len(i - 1L)))))
}

a_from_u <- function(u, cand, M) {
  B <- if (cand$rep == "matrix") matrix(u, M, M) else {
    Z <- diag(M)
    Z[tri_index(M, cand$fill)] <- u
    Z
  }
  switch(cand$op, t = t(B), flip_t = 2 * diag(M) - t(B), inv_t = solve(t(B)),
         id = B, flip = 2 * diag(M) - B, inv = solve(B))
}

u_candidates <- function(U, M) {
  ops_upper <- c(t = "A = t(U)", flip_t = "A = 2I - t(U)", inv_t = "A = solve(t(U))")
  ops_lower <- c(id = "A = U", flip = "A = 2I - U", inv = "A = solve(U)")
  out <- list()
  if (length(dim(U)) == 3L) {
    for (op in names(c(ops_upper, ops_lower))) {
      out[[paste0("matrix_", op)]] <- list(rep = "matrix", fill = NA, op = op, label = paste0(
        "U stored as an M x M matrix per draw; ", c(ops_upper, ops_lower)[[op]]))
    }
  } else {
    for (fill in c("upper_col", "upper_row", "lower_col", "lower_row")) {
      ops <- if (startsWith(fill, "upper")) ops_upper else ops_lower
      for (op in names(ops)) {
        out[[paste0(fill, "_", op)]] <- list(rep = "vector", fill = fill, op = op, label = paste0(
          "U stored as the vector of its free elements (", sub("_", " triangle, ", fill),
          "-major order); ", ops[[op]]))
      }
    }
  }
  out
}

# Σ_t of draw d as (M, M) from the documented vcov() output.
bv_vcov <- function(fit, Tn, M, D) {
  V <- draws_last(stats::vcov(fit), D)
  d <- dim(V)
  if (length(d) != 4L) stop("vcov() returned dim ", paste(d, collapse = " x "))
  tax <- which(d[1:3] == Tn)
  if (length(tax) != 1L) stop("vcov(): no unique time axis in dim ", paste(d, collapse = " x "))
  if (tax != 1L) V <- aperm(V, c(tax, setdiff(1:3, tax), 4L))
  V
}

sigma_from_a <- function(A, lam) {
  Ai <- solve(A)
  Ai %*% (lam * t(Ai))
}

# The row map of one fit: bv_to_ours[r] is the rai-bvar regressor (1 = const, then lag blocks)
# of PHI row r, from exact column matches of the returned X with the design built from the data.
bv_row_map <- function(fit, Y, p, intercept_rule = NA_character_) {
  M <- ncol(Y)
  K <- 1L + M * p
  des <- swa_design(Y, p)$X
  X <- unclass(unclass(fit)$X)
  X <- as.matrix(X)
  storage.mode(X) <- "double"
  if (nrow(X) != nrow(des) && ncol(X) == nrow(des)) X <- t(X)
  if (nrow(X) != nrow(des)) stop(sprintf("X has %d rows, the design %d", nrow(X), nrow(des)))
  m <- apply(X, 2, function(col) {
    w <- which(apply(des, 2, function(o) max(abs(o - col))) < 1e-10)
    if (length(w) == 1L) w else NA_integer_
  })
  if (length(m) == K) {
    bv_to_ours <- as.integer(m)
  } else if (length(m) == K - 1L && intercept_rule %in% c("first", "last")) {
    bv_to_ours <- if (intercept_rule == "first") c(1L, as.integer(m)) else c(as.integer(m), 1L)
  } else {
    stop(sprintf("X has %d columns (K = %d) and the intercept rule is %s", length(m), K,
                 intercept_rule))
  }
  if (anyNA(bv_to_ours) || !setequal(bv_to_ours, seq_len(K))) {
    stop("the columns of X do not match the design: ", paste(bv_to_ours, collapse = ","))
  }
  list(bv_to_ours = bv_to_ours, phi_rows = match(seq_len(K), bv_to_ours),
       x_names = colnames(X) %||% rep("", ncol(X)), x_cols = ncol(X),
       intercept_bv = which(bv_to_ours == 1L))
}

# ------------------------------------------------------------------------------------------------
# Probes P1–P5 (P2 §9.9), each decided by a fixed rule from bayesianVARs' output
# ------------------------------------------------------------------------------------------------

PROBES <- list(
  P1 = list(
    question = paste(
      "Are specify_prior_phi(normal_sds) and specify_prior_sigma(cholesky_normal_sds) prior",
      "variances or standard deviations, and in which order does normal_sds fill the",
      "coefficients?"),
    documented = paste(
      "bayesianVARs 0.1.8 manual: normal_sds 'prior standard deviations' of the lags*M^2",
      "coefficients (intercept excluded); cholesky_normal_sds described as prior variances of",
      "the free off-diagonal elements of U although named sds (N04 sec. 3.1-3.2, sec. 10 item 4)"),
    dataset = "data/probe_tiny.csv (n = 2, p = 1, T_eff = 4)",
    design = paste(
      "prior-dominated fit with distinct slot values normal_sds = 1e-3 * (1, 2, 3, 4),",
      "cholesky_normal_sds = 1e-3, prior_intercept = 1e3, PHI_tol = cholesky_U_tol = 0: the",
      "posterior variance of the lag coefficient in slot i is ~1e-3 * i if the argument is a",
      "variance and ~1e-6 * i^2 if it is a standard deviation (rounded slot numbers must form",
      "a permutation within 10%); the slot numbers give the fill order (column- or row-major",
      "over the PHI lag rows and the equations)")),
  P2 = list(
    question = "Is U stored as U or U^-1, in which layout, and with which sign?",
    documented = paste(
      "Sigma_t = U'^-1 D_t U^-1 with U upper unitriangular (manual; GK eq. 2), hence A = U';",
      "GK App. B prints the U step without the minus sign (N04 sec. 3.4); vcov() documents",
      "Sigma_t draws as T x M x M x draws"),
    dataset = paste(
      "data/var_sv_n7.csv (n = 7, p = 1, SV) and data/probe_corr.csv (n = 2, p = 1,",
      "corr(u1, u2) = +0.8)"),
    design = paste(
      "Sigma_t from the documented vcov() against A^-1 D_t A^-T, D_t = diag(exp(logvar)), for",
      "every candidate reading of the U draws (M x M matrix or vector of free elements; upper",
      "or lower triangle, column- or row-major; A = t(U), 2I - t(U), solve(t(U)), ...) at",
      "three (t, draw) pairs: a candidate within 1e-8 (relative) answers; guard: the posterior",
      "mean correlation of u1, u2 on probe_corr is +0.8 +- 0.1 by vcov() and by the selected",
      "mapping")),
  P3 = list(
    question = "Which row of the returned PHI draws holds the intercept, and in which order are the lags?",
    documented = "undocumented (N04 sec. 3.4, sec. 10 item 4)",
    dataset = "data/probe_corr.csv (intercepts 5 and -3, own lag 0.3) and data/var_sv_n3.csv",
    design = paste(
      "the columns of the returned X are matched exactly (max abs difference < 1e-10) to the",
      "rai-bvar design (const, lag-1 block, ..., lag-p block) built from the CSV, and PHI row r",
      "is taken to belong to X column r; checks: the PHI row of the constant has posterior mean",
      "(5, -3) +- 0.5 on probe_corr, and on var_sv_n3 (p = 2, loose prior) the mapped posterior",
      "mean of every coefficient is within 0.3 of the simulated (c, B)")),
  P4 = list(
    question = "What is the layout of the returned sv_para draws?",
    documented = "undocumented (N04 sec. 9.8, sec. 10 item 4)",
    dataset = paste(
      "data/probe_sv.csv (n = 3, p = 1, T = 400, mu = (-3, 0, 2), phi = (0.8, 0.9, 0.97),",
      "sigma = (0.4, 0.3, 0.15))"),
    design = paste(
      "the posterior means of the sv_para slices are matched to the distinct true",
      "(mu, phi, sigma) of the three equations, both orientations of the array, sigma against",
      "the s.d. and the variance (tolerances 0.6, 0.1, 0.12); among slices near mu, the one most",
      "correlated with h_1 is h_0")),
  P5 = list(
    question = paste(
      "Is cholesky_priorhomoscedastic = cbind(shape, scale) the inverse-gamma scale beta0 of",
      "p(lambda) ~ lambda^(-alpha0-1) exp(-beta0/lambda)? What is used when it is NA?"),
    documented = paste(
      "manual: an M x 2 matrix, column 1 the inverse-gamma shape, column 2 the scale; default",
      "NA with an undocumented substitute (N04 sec. 3.3)"),
    dataset = "data/probe_tiny.csv (n = 2, p = 1, T_eff = 4)",
    design = paste(
      "cholesky_heteroscedastic = FALSE, cholesky_priorhomoscedastic = cbind(50, 100):",
      "posterior mean of lambda ~ 2 (IG scale) or ~ 2e-4 (scale of a Gamma on 1/lambda);",
      "a second fit with the default NA records the substituted value (prior object field)"))
)

BV_LOOSE_PHI <- 'prior = "normal", normal_sds = 100, priormean = 0, PHI_tol = 0'
BV_HOMO_LOOSE <- bv_sigma_args("constant", normal_sds = "100")

# OLS coefficients (rai-bvar regressor order) of a VAR(p) with intercept: the posterior mean under
# a loose Normal prior with constant variances (identical regressors: GLS = OLS).
ols_coef <- function(Y, p) {
  des <- swa_design(Y, p)
  solve(crossprod(des$X), crossprod(des$X, des$Y))
}

probe_p3 <- function() {
  check_fit <- function(file, lags) {
    r <- bv_probe_fit(file, lags, BV_LOOSE_PHI, BV_HOMO_LOOSE, "100",
                      draws = if (QUICK) 1000L else 5000L)
    Y <- r$data
    M <- ncol(Y)
    K <- 1L + M * lags
    rm <- bv_row_map(r$fit, Y, lags, NA_character_)
    pm <- apply(bv_phi(unclass(r$fit), M, K, r$D), c(1, 2), mean)
    err <- max(abs(pm[rm$phi_rows, , drop = FALSE] - ols_coef(Y, lags)))
    list(fit = r$fit, rm = rm, pm = pm, err = err, K = K)
  }
  a <- check_fit("probe_corr.csv", 1L)
  b <- check_fit("var_sv_n3.csv", 2L)
  rule <- if (all(c(a$rm$intercept_bv == 1L, b$rm$intercept_bv == 1L))) "first" else
    if (a$rm$intercept_bv == a$K && b$rm$intercept_bv == b$K) "last" else "other"
  lag_order <- if (identical(b$rm$bv_to_ours[-b$rm$intercept_bv], 2:b$K)) "rai-bvar" else "other"
  ok <- a$err < 0.05 && b$err < 0.05 && rule != "other" && lag_order == "rai-bvar"
  list(
    status = if (ok) "answered" else "unresolved",
    answer = if (ok) sprintf(paste(
      "PHI row r belongs to column r of the returned X; the intercept is the %s row and the lag",
      "rows follow rai-bvar's order (lag 1 of y1..yM, ..., lag p of y1..yM)"), rule)
      else "(undecided)",
    evidence = sprintf(paste(
      "X columns (var_sv_n3, p = 2): %s, matched exactly to the rai-bvar regressors (%s);",
      "probe_corr: X columns -> rai-bvar regressors (%s); max |mapped posterior mean - OLS|",
      "(loose prior, constant variances): probe_corr %.4f, var_sv_n3 %.4f (limit 0.05);",
      "probe_corr posterior means (PHI rows) %s"), paste(b$rm$x_names, collapse = ", "),
      paste(b$rm$bv_to_ours, collapse = ","), paste(a$rm$bv_to_ours, collapse = ","), a$err,
      b$err, paste(apply(signif(a$pm, 4), 1, paste, collapse = "/"), collapse = "; ")),
    layout = list(intercept_rule = rule, lag_order = lag_order, rowmap_n3p2 = b$rm),
    describe = c(describe(a$fit, "probe_corr_fit"), describe(b$fit, "var_sv_n3_fit")))
}

probe_p2 <- function() {
  D <- if (QUICK) 300L else 1000L
  r <- bv_probe_fit("var_sv_n7.csv", 1L, BV_LOOSE_PHI,
                    bv_sigma_args("sv", normal_sds = "100"), "100", draws = D, burnin = 300L)
  f <- unclass(r$fit)
  M <- ncol(r$data)
  Tn <- nrow(r$data) - 1L
  V <- bv_vcov(r$fit, Tn, M, D)
  U <- draws_last(f$U, D)
  lv <- bv_logvar(f, M, D)
  cands <- u_candidates(U, M)
  pts <- rbind(c(1L, 1L), c(Tn, D), c((Tn + 1L) %/% 2L, (D + 1L) %/% 2L))
  err <- vapply(cands, function(cd) {
    tryCatch(max(apply(pts, 1, function(pt) {
      A <- a_from_u(u_draw(U, pt[2]), cd, M)
      S <- sigma_from_a(A, exp(lv[lv_row(pt[1], dim(lv)[1], Tn), , pt[2]]))
      Sv <- V[pt[1], , , pt[2]]
      max(abs(S - Sv)) / max(abs(Sv))
    })), error = function(e) Inf)
  }, numeric(1))
  hit <- names(err)[err < 1e-8]
  unit_lower <- function(A) max(abs(A[upper.tri(A)])) < 1e-12 && max(abs(diag(A) - 1)) < 1e-12
  hit <- hit[vapply(hit, function(h) unit_lower(a_from_u(u_draw(U, 1L), cands[[h]], M)),
                    logical(1))]
  # guard on probe_corr (+0.8)
  rc <- bv_probe_fit("probe_corr.csv", 1L, BV_LOOSE_PHI, BV_HOMO_LOOSE, "100", draws = D,
                     burnin = 300L)
  fc <- unclass(rc$fit)
  Tc <- nrow(rc$data) - 1L
  Vc <- bv_vcov(rc$fit, Tc, 2L, D)
  corr_v <- mean(Vc[Tc, 1, 2, ] / sqrt(Vc[Tc, 1, 1, ] * Vc[Tc, 2, 2, ]))
  corr_m <- NA_real_
  if (length(hit)) {
    Uc <- draws_last(fc$U, D)
    lvc <- bv_logvar(fc, 2L, D)
    corr_m <- mean(vapply(seq_len(D), function(d) {
      S <- sigma_from_a(a_from_u(u_draw(Uc, d), u_candidates(Uc, 2L)[[hit[1]]], 2L),
                        exp(lvc[lv_row(Tc, dim(lvc)[1], Tc), , d]))
      S[1, 2] / sqrt(S[1, 1] * S[2, 2])
    }, numeric(1)))
  }
  guard <- abs(corr_v - 0.8) < 0.1 && is.finite(corr_m) && abs(corr_m - 0.8) < 0.1
  status <- if (length(hit) && guard) "answered" else "unresolved"
  list(
    status = status,
    answer = if (status == "answered") paste0(cands[[hit[1]]]$label,
                                              " (A unit lower triangular, A u_t = e_t)")
      else "(undecided)",
    evidence = paste0(
      sprintf("U draws: dim %s; logvar: dim %s; vcov(): dim %s; ",
              paste(dim(U), collapse = " x "), paste(dim(lv), collapse = " x "),
              paste(dim(V), collapse = " x ")),
      "relative error of Sigma_t per candidate: ",
      paste(sprintf("%s %.2e", names(err), err), collapse = "; "),
      sprintf("; probe_corr posterior mean corr(u1, u2): vcov() %.3f, selected mapping %.3f",
              corr_v, corr_m)),
    layout = list(u_cand = if (length(hit)) cands[[hit[1]]] else NULL,
                  u_rep = if (length(dim(U)) == 3L) "matrix" else "vector"),
    describe = describe(r$fit, "var_sv_n7_fit"))
}

bv_svpara <- function(f, M, D, sv) {
  S <- draws_last(f$sv_para, D)
  if (sv$orient == "cols_params") S <- aperm(S, c(2, 1, 3))
  get <- function(row) t(matrix(S[row, , ], nrow = M))      # D x M
  out <- list(mu = get(sv$rows[["mu"]]), phi = get(sv$rows[["phi"]]),
              sigma = get(sv$rows[["sigma"]]))
  if (sv$sigma_is == "variance") out$sigma <- sqrt(out$sigma)
  out$h0 <- if (!is.na(sv$rows[["h0"]])) get(sv$rows[["h0"]]) else NULL
  out
}

probe_p4 <- function() {
  r <- bv_probe_fit("probe_sv.csv", 1L, BV_LOOSE_PHI, bv_sigma_args("sv", normal_sds = "100"),
                    "100", draws = if (QUICK) 1000L else 5000L)
  f <- unclass(r$fit)
  D <- r$D
  M <- 3L
  S <- draws_last(f$sv_para, D)
  lv <- bv_logvar(f, M, D)
  truth <- rbind(mu = c(-3, 0, 2), phi = c(0.8, 0.9, 0.97), sigma = c(0.4, 0.3, 0.15),
                 sigma2 = c(0.4, 0.3, 0.15)^2)
  # rule: every slice is nearest to exactly one target row, at less than half its distance to
  # the second-nearest target; mu, phi and one of sigma / sigma^2 must be matched
  try_orient <- function(orient) {
    So <- if (orient == "cols_params") aperm(S, c(2, 1, 3)) else S
    if (dim(So)[2] != M) return(NULL)
    pm <- apply(So, c(1, 2), mean)
    dist <- t(apply(pm, 1, function(row) apply(truth, 1, function(tr) max(abs(row - tr)))))
    near <- apply(dist, 1, which.min)
    clear <- apply(dist, 1, function(d) {
      o <- sort(d)
      o[1] < 0.5 * o[2]
    })
    tn <- rownames(truth)[near]
    rows <- c(mu = NA_integer_, phi = NA_integer_, sigma = NA_integer_, h0 = NA_integer_)
    for (k in c("mu", "phi")) if (sum(tn == k) == 1L) rows[[k]] <- which(tn == k)
    sig <- which(tn %in% c("sigma", "sigma2"))
    if (length(sig) == 1L) rows[["sigma"]] <- sig
    if (sum(tn == "mu") == 2L) {     # h_0 next to mu: the slice most correlated with h_1
      two <- which(tn == "mu")
      cr <- vapply(two, function(rw) mean(vapply(seq_len(M), function(j) {
        stats::cor(So[rw, j, ], lv[lv_row(1L, dim(lv)[1], nrow(r$data) - 1L), j, ])
      }, numeric(1))), numeric(1))
      rows[["h0"]] <- two[which.max(cr)]
      rows[["mu"]] <- two[which.min(cr)]
    }
    ok <- all(clear) && !anyNA(rows[c("mu", "phi", "sigma")]) &&
      length(unique(rows[!is.na(rows)])) == sum(!is.na(rows)) &&
      nrow(pm) == sum(!is.na(rows))
    list(orient = orient, rows = rows,
         sigma_is = if (length(sig) == 1L && tn[sig] == "sigma2") "variance" else "sd",
         ok = ok, pm = pm, dist = dist)
  }
  res <- Filter(Negate(is.null), lapply(c("rows_params", "cols_params"), try_orient))
  good <- Filter(function(x) x$ok, res)
  dn <- dimnames(S)
  status <- if (length(good) == 1L) "answered" else "unresolved"
  g <- if (length(good) == 1L) good[[1]] else NULL
  list(
    status = status,
    answer = if (status == "answered") sprintf(
      "sv_para is (parameter, equation, draw): slice %d = mu, %d = phi, %d = sigma (%s)%s",
      g$rows[["mu"]], g$rows[["phi"]], g$rows[["sigma"]], if (g$sigma_is == "sd")
        "the s.d. of the h innovation" else "a variance",
      if (is.na(g$rows[["h0"]])) "; no h_0 slice" else sprintf("; h_0 = slice %d", g$rows[["h0"]]))
      else "(undecided)",
    evidence = paste0(
      sprintf("sv_para: dim %s; dimnames %s; ", paste(dim(S), collapse = " x "),
              if (is.null(dn)) "none" else paste(vapply(dn, function(z) paste(z, collapse = ","),
                                                         ""), collapse = " | ")),
      paste(vapply(res, function(x) sprintf("%s: posterior means %s; distance to (mu, phi, sigma, sigma^2) %s (ok = %s)",
                                            x$orient, paste(apply(signif(x$pm, 4), 1, paste, collapse = "/"), collapse = "; "),
                                            paste(apply(signif(x$dist, 3), 1, paste, collapse = "/"), collapse = "; "),
                                            x$ok), ""), collapse = " || ")),
    layout = list(sv = if (!is.null(g)) g[c("orient", "rows", "sigma_is")] else NULL),
    describe = describe(r$fit, "probe_sv_fit"))
}

probe_p1 <- function() {
  r <- bv_probe_fit("probe_tiny.csv", 1L,
                    'prior = "normal", normal_sds = 1e-3 * (1:4), priormean = 0, PHI_tol = 0',
                    bv_sigma_args("constant", normal_sds = "1e-3"), "1e3")
  f <- unclass(r$fit)
  M <- 2L
  K <- 3L
  P <- bv_phi(f, M, K, r$D)
  v <- apply(P, c(1, 2), stats::var)
  int_row <- which.max(rowMeans(v))
  lag <- v[-int_row, , drop = FALSE]                      # (lag rows, equations)
  reading <- function(idx_raw) {
    idx <- round(idx_raw)
    ok <- setequal(as.vector(idx), 1:4) && max(abs(idx_raw / idx - 1)) < 0.1
    list(idx = idx, ok = ok)
  }
  rv <- reading(lag / 1e-3)
  rs <- reading(sqrt(lag) / 1e-3)
  U <- draws_last(f$U, r$D)
  uv <- stats::var(as.vector(if (length(dim(U)) == 3L) U[1, 2, ] + U[2, 1, ] else U[1, ]))
  ureading <- if (abs(log10(uv) + 3) < 0.5) "variance" else if (abs(log10(uv) + 6) < 0.5)
    "standard deviation" else "undecided"
  which_ok <- c(variance = rv$ok, `standard deviation` = rs$ok)
  normal_reading <- if (sum(which_ok) == 1L) names(which_ok)[which_ok] else "undecided"
  idx <- if (normal_reading == "variance") rv$idx else rs$idx
  colmajor <- matrix(1:4, 2, 2)
  rowmajor <- matrix(1:4, 2, 2, byrow = TRUE)
  slot_rule <- if (normal_reading == "undecided") "undecided" else
    if (identical(unname(idx) + 0, colmajor + 0)) "column-major" else
      if (identical(unname(idx) + 0, rowmajor + 0)) "row-major" else "other"
  vp <- if (!is.null(f$V_prior)) apply(draws_last(f$V_prior, r$D), c(1, 2), mean) else NULL
  vp_txt <- if (is.null(vp)) "V_prior not returned" else sprintf(
    "V_prior (bayesianVARs' prior variances of PHI, rows as PHI): %s",
    paste(signif(vp, 4), collapse = ", "))
  status <- if (normal_reading != "undecided" && ureading != "undecided" &&
                slot_rule %in% c("column-major", "row-major")) "answered" else "unresolved"
  list(
    status = status,
    answer = if (status == "answered") sprintf(
      "normal_sds: %s, filled %s over (PHI lag rows, equations); cholesky_normal_sds: %s",
      if (normal_reading == "variance") "prior variances" else "prior standard deviations",
      slot_rule, if (ureading == "variance") "prior variances" else "prior standard deviations")
      else "(undecided)",
    evidence = sprintf(paste(
      "posterior variance of the lag coefficients (PHI lag rows x equations, intercept row %d",
      "excluded): %s; slot numbers as variance %s, as s.d. %s; posterior variance of the free U",
      "element: %.3e; %s"), int_row, paste(signif(lag, 4), collapse = ", "),
      paste(signif(lag / 1e-3, 4), collapse = ", "), paste(signif(sqrt(lag) / 1e-3, 4),
                                                           collapse = ", "), uv, vp_txt),
    layout = list(normal_reading = normal_reading, chol_reading = ureading,
                  slot_rule = slot_rule))
}

probe_p5 <- function(L) {
  hom <- function(mat) {
    bv_probe_fit("probe_tiny.csv", 1L, 'prior = "normal", normal_sds = 1, priormean = 0, PHI_tol = 0',
                 bv_sigma_args("constant", homo = mat), "100")
  }
  r <- hom("cbind(rep(50, M), rep(100, M))")
  lam <- apply(exp(bv_logvar(unclass(r$fit), 2L, r$D)), 2, mean)
  m <- stats::median(lam)
  # IG(50, 100) and 4 observations: E(lambda | y) ~ 2. A Gamma(50, scale 100) on 1/lambda gives
  # E(lambda | y) < 0.1 for any residuals of this size.
  answer <- if (abs(log10(m) - log10(100 / 49)) < 0.2) "IG scale" else
    if (m < 0.2) "scale of a Gamma on 1/lambda" else "undecided"
  rn <- tryCatch(hom("as.numeric(NA)"), error = function(e) e)
  na_txt <- if (inherits(rn, "error")) paste("the NA default failed:", conditionMessage(rn)) else {
    pri <- bayesianVARs::specify_prior_sigma(data = rn$data, type = "cholesky",
                                             cholesky_U_prior = "normal",
                                             cholesky_heteroscedastic = FALSE, quiet = TRUE)
    sub <- describe(pri, "prior_sigma")
    sub <- grep("homo", sub, value = TRUE, ignore.case = TRUE)
    sprintf("with the NA default: posterior mean of lambda %s; prior fields %s",
            paste(signif(apply(exp(bv_logvar(unclass(rn$fit), 2L, rn$D)), 2, mean), 4),
                  collapse = ", "), paste(sub, collapse = " / "))
  }
  list(
    status = if (answer == "undecided") "unresolved" else "answered",
    answer = if (answer == "IG scale") paste(
      "column 2 is the inverse-gamma scale beta0 (lambda ~ IG(shape, scale), density",
      "lambda^(-shape-1) exp(-scale/lambda))") else if (answer == "undecided") "(undecided)"
      else "column 2 is the scale of a Gamma prior on 1/lambda",
    evidence = sprintf("cbind(50, 100): posterior mean of lambda %s (IG scale predicts %.3g); %s",
                       paste(signif(lam, 4), collapse = ", "), 100 / 49, na_txt),
    layout = list(homo_reading = answer))
}

run_bvars_probes <- function() {
  res <- list()
  res$P3 <- attempt_probe("P3", probe_p3())
  res$P2 <- attempt_probe("P2", probe_p2())
  res$P4 <- attempt_probe("P4", probe_p4())
  res$P1 <- attempt_probe("P1", probe_p1())
  res$P5 <- attempt_probe("P5", probe_p5())
  res <- res[names(PROBES)]
  for (id in names(PROBES)) res[[id]] <- c(PROBES[[id]], res[[id]])
  res
}

bv_layout <- function(probes) {
  L <- list()
  for (id in names(probes)) if (identical(probes[[id]]$status, "answered")) {
    L <- c(L, probes[[id]]$layout)
  }
  L
}

# Value vector for normal_sds (P1 order and reading) from a (K x M) rai-bvar array of variances.
bv_slot_vector <- function(var_ours, L, rowmap, M, p) {
  f <- if (L$normal_reading == "variance") identity else sqrt
  lag_bv <- setdiff(seq_len(1L + M * p), rowmap$intercept_bv)        # PHI lag rows, bv order
  vals <- f(var_ours[rowmap$bv_to_ours[lag_bv], , drop = FALSE])      # (lag rows bv, equations)
  if (L$slot_rule == "column-major") as.vector(vals) else as.vector(t(vals))
}

stage_probes <- function(env, results = NULL) {
  ids <- names(PROBES)
  if (is.null(results)) {
    evidence <- paste(
      "bayesianVARs is not installed (requireNamespace() is FALSE); gfortran on PATH: '",
      env$gfortran_on_path, "'; R CMD config FC: ", env$r_config_fc, "; FLIBS: ",
      env$r_config_flibs, " (", env$flibs_dirs_exist, "). ", env$cran_binary_note,
      ". The probe was not run; the documented convention is unverified.", sep = "")
    results <- lapply(ids, function(id) c(PROBES[[id]], list(
      status = "unresolved", answer = "(not run)", evidence = evidence)))
    names(results) <- ids
  }
  attrs <- c(p3_meta("bayesianVARs", "probe"), env)
  attrs$bayesianvars_version <- if (isTRUE(env$bayesianvars_available)) {
    as.character(packageVersion("bayesianVARs"))
  } else "not installed"
  attrs$consequence <- if (isTRUE(env$bayesianvars_available)) paste(
    "bayesianVARs ran: the probes are decided from its output (Phase 3 DESIGN sec. 9.8); the",
    "parity runs are bvars_bayesianvars_*.nc (R1, R2) and bvars_gl_*.nc (R3-R9); the exact",
    "system-wide Gibbs oracle fixtures bvars_swa_*.nc are kept unchanged") else paste(
    "bayesianVARs parity (DESIGN sec. 9.9) is blocked; the reduced-form sampler is compared",
    "instead with the exact system-wide Gibbs oracle fixtures bvars_swa_*.nc, which use",
    "rai-bvar's own conventions (P1-P5 do not arise there)")
  attrs$parity_calls <- paste(bvars_parity_calls(), collapse = "\n")
  stage_probe_fixture("probe_bayesianvars", attrs, results, ids)
}

# ------------------------------------------------------------------------------------------------
# Extraction of one bvar() run in rai-bvar's layout (by the answers of P1–P5)
# ------------------------------------------------------------------------------------------------

bv_predicted_rowmap <- function(M, p, L) {
  K <- 1L + M * p
  if (!identical(L$lag_order, "rai-bvar") || !(L$intercept_rule %in% c("first", "last"))) {
    stop("the PHI row order cannot be predicted (P3)")
  }
  b2o <- if (L$intercept_rule == "first") seq_len(K) else c(2:K, 1L)
  list(bv_to_ours = b2o, phi_rows = match(seq_len(K), b2o), intercept_bv = which(b2o == 1L))
}

# The lag-row values of a (K x M) rai-bvar array in bayesianVARs' slot order (P1, P3).
bv_slots <- function(values_ours, L, rowmap, M, p) {
  lag_bv <- setdiff(seq_len(1L + M * p), rowmap$intercept_bv)
  vals <- values_ours[rowmap$bv_to_ours[lag_bv], , drop = FALSE]
  if (L$slot_rule == "column-major") as.vector(vals) else as.vector(t(vals))
}
bv_slot_vector <- function(var_ours, L, rowmap, M, p) {
  bv_slots(if (L$normal_reading == "variance") var_ours else sqrt(var_ours), L, rowmap, M, p)
}

bv_homo <- function(L, shape = 3, scale = 2) {
  if (identical(L$homo_reading, "IG scale")) {
    sprintf("cbind(rep(%s, M), rep(%s, M))", num_text(shape), num_text(scale))
  } else if (identical(L$homo_reading, "scale of a Gamma on 1/lambda")) {
    sprintf("cbind(rep(%s, M), rep(%s, M))", num_text(shape), num_text(1 / scale))
  } else stop("the scale convention of cholesky_priorhomoscedastic is undecided (P5)")
}

bv_extract <- function(r, L, p, vol, t_sel, H, fc_seed, hyper = TRUE, vprior = FALSE) {
  fit <- r$fit
  f <- unclass(fit)
  Y <- r$data
  M <- ncol(Y)
  K <- 1L + M * p
  D <- r$D
  Tn <- nrow(Y) - p
  rmap <- bv_row_map(fit, Y, p, L$intercept_rule)
  P <- bv_phi(f, M, K, D)
  out <- list(row_map = rmap,
              floor_phi = mean(P[-rmap$intercept_bv, , ] == 0),
              nonfinite = sum(!is.finite(P)))
  out$Pi <- aperm(P[rmap$phi_rows, , , drop = FALSE], c(3, 1, 2))   # D x K x M
  rm(P)
  U <- draws_last(f$U, D)
  out$floor_u <- if (length(dim(U)) == 3L) {
    free <- apply(U, c(1, 2), function(z) any(z != 0)) & !diag(M)
    mean(apply(U, 3, function(u) u[free]) == 0)
  } else mean(U == 0)
  A <- array(NA_real_, c(D, M, M))
  for (d in seq_len(D)) A[d, , ] <- a_from_u(u_draw(U, d), L$u_cand, M)
  out$A <- A
  rm(U, A)
  lv <- bv_logvar(f, M, D)
  if (vol == "sv") {
    S <- bv_svpara(f, M, D, L$sv)
    out$mu <- S$mu
    out$phi <- S$phi
    out$sigma <- S$sigma
    out$h_sel <- array(NA_real_, c(D, length(t_sel), M))
    for (k in seq_along(t_sel)) {
      out$h_sel[, k, ] <- if (t_sel[k] == 0L) S$h0 else
        lv_at(lv, lv_row(t_sel[k], dim(lv)[1], Tn), seq_len(D))
    }
    keep <- seq(BV_PATH_THIN, D, by = BV_PATH_THIN)
    tt <- if (is.null(S$h0)) seq_len(Tn) else 0:Tn
    hp <- array(NA_real_, c(length(keep), length(tt), M))
    for (k in seq_along(tt)) {
      hp[, k, ] <- if (tt[k] == 0L) S$h0[keep, , drop = FALSE] else
        lv_at(lv, lv_row(tt[k], dim(lv)[1], Tn), keep)
    }
    out$h_path <- hp
    out$h_path_t <- tt
  } else {
    out$lam <- exp(lv_at(lv, 1L, seq_len(D)))
  }
  rm(lv)
  if (H > 0L) {
    set.seed(fc_seed)
    pr <- unclass(stats::predict(fit, ahead = seq_len(H), each = 1L, stable = FALSE,
                                 simulate_predictive = TRUE))
    pred <- draws_last(pr$predictions, D)
    if (dim(pred)[1] != H && dim(pred)[2] == H) pred <- aperm(pred, c(2, 1, 3))
    out$fc <- aperm(pred, c(3, 1, 2))                                  # D x H x M
  }
  if (hyper && !is.null(f$phi_hyperparameter)) {
    hm <- draws_last(f$phi_hyperparameter, D)
    nm <- if (length(dim(hm)) == 2L) dimnames(hm)[[1]] else NULL
    hm <- matrix(hm, ncol = D)
    out$hyper <- t(hm)
    out$hyper_names <- if (length(nm) == nrow(hm)) nm else paste0("hyper", seq_len(nrow(hm)))
    out$hyper_floor <- mean(hm == 0)
    out$hyper_nonfinite <- sum(!is.finite(hm))
  }
  if (vprior && !is.null(f$V_prior)) {
    vp <- draws_last(f$V_prior, D)
    out$vprior_dim <- dim(vp)[-length(dim(vp))]
    out$vprior <- t(matrix(vp, ncol = D))
  }
  out
}

stage_reduced_fixture <- function(name, chains, info, extra_vars = list(), extra_coords = list(),
                                  extra_blocks = list(), mixing_hard = TRUE) {
  M <- info$n
  p <- info$lags
  K <- 1L + M * p
  H <- info$horizon
  variables <- info$variables
  free <- which(lower.tri(diag(M)), arr.ind = TRUE)
  free <- free[order(free[, 1], free[, 2]), , drop = FALSE]
  chains_at <- function(field, idx) lapply(chains, function(ch) {
    x <- ch[[field]]
    switch(length(idx), x[, idx[1]], x[, idx[1], idx[2]])
  })
  blocks <- list(
    summary_vars("Pi", function(ix) chains_at("Pi", ix), c(K, M), c("regressor", "equation"),
                 paste("reduced-form coefficients Pi[regressor, equation] (y_t = Pi'x_t + u_t),",
                       "rai-bvar regressor order")),
    summary_vars("A", function(ix) chains_at("A", free[ix[1], ]), nrow(free), "a_free",
                 "free entries a_ij (i > j) of the unit lower-triangular A (A u_t = e_t)"))
  coords <- list(
    quantile = nc_array(P2_PROBS, "quantile"),
    regressor = nc_array(swa_regressor_names(variables, p), "regressor"),
    equation = nc_array(variables, "equation"),
    a_free = nc_array(sprintf("a%d%d", free[, 1], free[, 2]), "a_free"),
    a_row = nc_array(as.integer(free[, 1]), "a_free"),
    a_col = nc_array(as.integer(free[, 2]), "a_free"))
  if (H > 0L) {
    blocks <- c(blocks, list(summary_vars(
      "forecast", function(ix) lapply(chains, function(ch) ch$fc[, ix[1], ix[2]]), c(H, M),
      c("horizon", "equation"), "predictive y_{T+s}, one path per stored draw")))
    coords$horizon <- nc_array(seq_len(H), "horizon")
  }
  extra <- list()
  if (info$vol == "sv") {
    sv_names <- c("mu", "phi", "sigma")
    t_sel <- info$t_sel
    blocks <- c(blocks, list(
      summary_vars("sv", function(ix) chains_at(sv_names[ix[1]], ix[2]), c(3L, M),
                   c("sv_param", "equation"),
                   "SV parameters (mu, phi, sigma) per equation; sigma is the s.d. of the h innovation"),
      summary_vars("h", function(ix) chains_at("h_sel", ix), c(length(t_sel), M),
                   c("t_sel", "equation"), "log-variance h_it at t_sel")))
    coords$sv_param <- nc_array(sv_names, "sv_param")
    coords$t_sel <- nc_array(as.integer(t_sel), "t_sel")
    tt <- chains[[1]]$h_path_t
    coords$t <- nc_array(as.integer(tt), "t")
    HP <- do.call(abind_first, lapply(chains, `[[`, "h_path"))
    hq <- array(NA_real_, c(length(tt), M, length(P2_PROBS)))
    for (j in seq_len(M)) hq[, j, ] <- path_quantiles(HP[, , j])
    extra$h_path_q <- nc_array(hq, c("t", "equation", "quantile"), sprintf(
      "quantiles of h_it at t (every %dth draw, no MCSE)", BV_PATH_THIN))
  } else {
    blocks <- c(blocks, list(summary_vars(
      "lambda", function(ix) chains_at("lam", ix), M, "equation",
      "constant variances lambda_i of e_it")))
  }
  for (b in blocks) check_mixing_p3(name, b, hard = mixing_hard)
  for (b in extra_blocks) check_mixing_p3(paste(name, "(hyperparameters)"), b, hard = FALSE)
  mix <- mixing_summary(blocks)
  vars <- c(unlist(lapply(c(blocks, extra_blocks), `[[`, "vars"), recursive = FALSE), extra,
            extra_vars)
  attrs <- c(p3_meta("bayesianVARs", info$kind), info$attrs)
  attrs$mixing_max_rhat <- mix$max_rhat
  attrs$mixing_min_ess <- mix$min_ess
  attrs$mixing_ok <- mix$ok
  attrs$floor_fraction_phi <- max(vapply(chains, `[[`, numeric(1), "floor_phi"))
  attrs$floor_fraction_u <- max(vapply(chains, `[[`, numeric(1), "floor_u"))
  attrs$nonfinite_phi <- sum(vapply(chains, `[[`, numeric(1), "nonfinite"))
  attrs$runtime_seconds_per_run <- I(vapply(chains, function(ch) ch$seconds %||% NA_real_,
                                            numeric(1)))
  stage(name, attrs, c(coords, extra_coords), vars)
}

# Hyperparameter draws: every column thinned, binary columns as inclusion probabilities, grid
# columns as categorical probabilities, the continuous columns named by `select` summarised.
hyper_staging <- function(chains, select = NULL, grid = NULL) {
  nm <- chains[[1]]$hyper_names
  if (is.null(nm)) return(list(vars = list(), coords = list(), blocks = list(), class = "none"))
  th <- thinned_draws(lapply(chains, `[[`, "hyper"), BV_HYPER_KEEP)
  vars <- list(hyper_draws = nc_array(th$draws, c("chain", "hyper_draw", "hyper"), paste(
    "evenly thinned draws of every column of bayesianVARs' phi_hyperparameter (names in the",
    "coordinate hyper; the iteration index in hyper_draw)")))
  coords <- list(hyper = nc_array(nm, "hyper"), hyper_draw = nc_array(th$keep, "hyper_draw"))
  col <- function(k) lapply(chains, function(ch) ch$hyper[, k])
  cls <- vapply(seq_along(nm), function(k) {
    x <- unlist(col(k), use.names = FALSE)
    u <- unique(x)
    if (all(u %in% c(0, 1))) "binary" else if (length(u) == 1L) "constant" else
      if (!is.null(grid) && all(vapply(u, function(v) min(abs(v - grid)) < 1e-10, logical(1))))
        "grid" else "continuous"
  }, character(1))
  blocks <- list()
  bin <- which(cls == "binary")
  if (length(bin)) {
    st <- t(vapply(bin, function(k) {
      ch <- col(k)
      pr <- mean(unlist(ch))
      ess <- if (pr > 0 && pr < 1) ess_multi(ch) else NA_real_
      c(pr, if (is.finite(ess)) sqrt(pr * (1 - pr) / ess) else 0, ess %||% NA_real_)
    }, numeric(3)))
    coords$incl <- nc_array(nm[bin], "incl")
    vars$incl_prob <- nc_array(st[, 1], "incl", "posterior inclusion probability (mean of the 0/1 column)")
    vars$incl_prob_mcse <- nc_array(st[, 2], "incl", "binomial MCSE sqrt(p(1 - p)/ESS), multi-chain ESS")
    vars$incl_prob_ess <- nc_array(st[, 3], "incl", "multi-chain ESS of the indicator (NA if constant)")
  }
  gr <- which(cls == "grid")
  if (length(gr)) {
    pr <- array(NA_real_, c(length(gr), length(grid)))
    se <- pr
    for (g in seq_along(gr)) {
      cs <- categorical_summary(col(gr[g]), grid)
      pr[g, ] <- cs[, "prob"]
      se[g, ] <- cs[, "mcse"]
    }
    coords$grid_param <- nc_array(nm[gr], "grid_param")
    coords$grid_support <- nc_array(grid, "grid_support")
    vars$grid_prob <- nc_array(pr, c("grid_param", "grid_support"), "posterior probability of each grid point")
    vars$grid_prob_mcse <- nc_array(se, c("grid_param", "grid_support"), "binomial MCSE from the ESS of the indicator")
  }
  sel <- which(cls == "continuous" & (if (is.null(select)) FALSE else grepl(select, nm)))
  if (length(sel)) {
    coords$hyper_sel <- nc_array(nm[sel], "hyper_sel")
    blocks <- list(summary_vars("hypersel", function(ix) col(sel[ix[1]]), length(sel), "hyper_sel",
                                "selected continuous hyperparameters (coordinate hyper_sel)"))
  }
  list(vars = vars, coords = coords, blocks = blocks,
       class = paste(sprintf("%s: %s", nm, cls), collapse = "; "))
}

# ------------------------------------------------------------------------------------------------
# Probes P6–P10 (Phase 3 DESIGN §9.8)
# ------------------------------------------------------------------------------------------------

PROBES_P3 <- list(
  P6 = list(
    question = "Does bayesianVARs standardise the data internally?",
    documented = "not documented",
    dataset = "data/var_sv_n3.csv (n = 3, p = 2)",
    design = paste(
      "the R1 parameter set (informative Normal prior on PHI, cholesky_normal_sds = 1e6, SV",
      "defaults) on the data and on the data with series 1 x 100, the prior transformed",
      "equivariantly (s.d. of the coefficient of equation i on variable j x d_i/d_j, intercept",
      "s.d. x d_i); 4 runs x 10,000 draws each; the back-transformed PHI quantiles of the",
      "scaled fit against those of the base fit under the Phase 2 rule: pass => no internal",
      "standardisation (both sides use the data as committed); fail => both sides receive",
      "pre-standardised data")),
  P7 = list(
    question = "What are the grouping matrices of global_grouping = global, equation-wise, covariate-wise, olcl-lagwise?",
    documented = "manual: subgroups for semi-global shrinkage, or a lags*M x M indicator matrix",
    dataset = "data/var_sv_n3.csv (n = 3, p = 2)",
    design = paste(
      "recorded as data: the integer (lags*M x M) field of specify_prior_phi(prior = 'HS',",
      "global_grouping = g), mapped to rai-bvar's regressor order by P3, compared as a",
      "partition with DESIGN D-38; a differing matrix is passed to rai-bvar as a custom groups",
      "array (not a failure)")),
  P8 = list(
    question = "What do the global-local hyperparameters mean (NG_a, NG_b, NG_c; R2D2_a, R2D2_b; DL_a = '1/K')?",
    documented = "manual (one line per argument); GK (2025) App. B",
    dataset = "data/probe_gl.csv (n = 3, p = 2, T_eff = 8)",
    design = paste(
      "bayesianVARs side only (fixtures probe_bvars_p8_*.nc): one run set per family and",
      "setting (HS; NG (0.1, 1, 4) and (0.1, 1, 1); R2D2 (0.1, 4) and (0.1, 0.5); DL '1/K' and",
      "0.5), constant variances lambda ~ IG(3, 2), 4 runs x 50,000 draws; WP30 runs rai-bvar",
      "under each candidate mapping on the same data and adopts the one that alone passes the",
      "Phase 2 rule")),
  P9 = list(
    question = "Which tau0, tau1 does SSVS_semiautomatic = TRUE use?",
    documented = "manual: c0 and c1 scaled by the posterior variances from a flat conjugate Normal-Wishart prior",
    dataset = "data/var_sv_n3.csv (n = 3, p = 2)",
    design = paste(
      "the positive lags*M^2 fields of specify_prior_phi(prior = 'SSVS', SSVS_c0 = 0.01,",
      "SSVS_c1 = 100, SSVS_semiautomatic = TRUE) against c * s_hat for DESIGN D-44's s_hat",
      "(s.d., df T - k - n - 1) and variants (df T - k, T; variance; without the intercept",
      "column) to 1e-8 (relative): equal to D-44 => semi-automatic runs may be compared;",
      "otherwise the runs use SSVS_semiautomatic = FALSE with explicit tau (R7 does)")),
  P10 = list(
    question = "Is the intercept outside the shrinkage, is prior_intercept an s.d., and do all *_tol accept 0?",
    documented = "manual: prior_intercept 'prior standard deviation(s)'; the *_tol floors draws when > 0",
    dataset = "data/probe_tiny.csv (n = 2, p = 1, T_eff = 4) and data/var_sv_n3.csv (n = 3, p = 2)",
    design = paste(
      "(a) prior_intercept = 1e-3 on probe_tiny (Normal and HS priors on PHI, 2 runs x 20,000",
      "draws): the intercept quantiles against N(0, 1e-6) (s.d. reading) and N(0, 1e-3)",
      "(variance reading) under the Phase 2 rule; (b) the HS fit passes the s.d. reading =>",
      "the intercept is outside the shrinkage; (c) every family (HS, NG, R2D2, DL, SSVS) with",
      "PHI_tol, its own *_tol and cholesky_U_tol = 0 on var_sv_n3: the call succeeds with",
      "finite draws (else the smallest of 1e-300, 1e-100, 1e-50, 1e-18 accepted is used and",
      "the fraction of draws at the floor is recorded)"))
)

probe_p6 <- function(L) {
  ds <- bv_data("var_sv_n3.csv")
  Y <- ds$Y
  M <- 3L
  p <- 2L
  K <- 7L
  dsc <- c(100, 1, 1)
  prior <- swa_prior_arrays(M, p)
  var_s <- prior$var
  var_s[1, ] <- prior$var[1, ] * dsc^2
  for (r in 2:K) var_s[r, ] <- prior$var[r, ] * (dsc / dsc[((r - 2L) %% M) + 1L])^2
  prm <- bv_predicted_rowmap(M, p, L)
  draws <- if (QUICK) 1000L else 10000L
  call <- bv_call(p, draws, BV_PROBE_BURN,
                  'prior = "normal", normal_sds = bv_normal_sds, priormean = 0.5, PHI_tol = 0',
                  bv_sigma_args("sv", normal_sds = "1e6"), "prior_int")
  run <- function(Yd, vs, pint, seed0) lapply(1:4, function(ch) {
    r <- bv_eval(call, list(data = Yd, M = M, bv_normal_sds = bv_slot_vector(vs, L, prm, M, p),
                            prior_int = pint), seed0 + ch)
    rmap <- bv_row_map(r$fit, Yd, p, L$intercept_rule)
    if (!identical(rmap$bv_to_ours, prm$bv_to_ours)) stop("unexpected PHI row order")
    bv_phi(unclass(r$fit), M, K, draws)[prm$phi_rows, , , drop = FALSE]
  })
  base <- run(Y, prior$var, c(10, 10, 10), BV_PROBE_SEED + 600L)
  Ys <- Y %*% diag(dsc)
  colnames(Ys) <- colnames(Y)
  scl <- run(Ys, var_s, 10 * dsc, BV_PROBE_SEED + 700L)
  back <- lapply(scl, function(P) {
    for (r in seq_len(K)) {
      dj <- if (r == 1L) 1 else dsc[((r - 2L) %% M) + 1L]
      P[r, , ] <- P[r, , ] * dj / dsc
    }
    P
  })
  pairs <- list()
  for (r in seq_len(K)) for (i in seq_len(M)) {
    pairs[[length(pairs) + 1L]] <- list(a = mc_summary(lapply(base, function(P) P[r, i, ])),
                                        b = mc_summary(lapply(back, function(P) P[r, i, ])))
  }
  cmp <- parity_compare(pairs)
  list(
    status = "answered",
    answer = if (cmp$pass) paste(
      "no internal standardisation: PHI rescales equivariantly; both sides use the data as",
      "committed") else "PHI is not equivariant: both sides receive pre-standardised data",
    evidence = sprintf(
      "max |z| = %.2f over m = %d quantile comparisons (bound %.2f); call %s", cmp$max_abs_z,
      cmp$m, cmp$bound, call),
    layout = list(standardise = !cmp$pass), max_abs_z = cmp$max_abs_z)
}

flatten_fields <- function(x, path = "") {
  if (is.list(x) && !is.data.frame(x)) {
    nm <- names(x) %||% rep("", length(x))
    nm[!nzchar(nm)] <- paste0("[[", which(!nzchar(nm)), "]]")
    out <- list()
    for (i in seq_along(x)) out <- c(out, flatten_fields(x[[i]], paste0(path, "$", nm[i])))
    return(out)
  }
  stats::setNames(list(x), path)
}

d38_groups <- function(g, M, p) {
  G <- matrix(0L, p * M, M)
  for (l in seq_len(p)) for (m in seq_len(M)) for (i in seq_len(M)) {
    r <- (l - 1L) * M + m
    G[r, i] <- switch(g, global = 1L, `equation-wise` = i, `covariate-wise` = r,
                      `olcl-lagwise` = if (m == i) 2L * l - 1L else 2L * l)
  }
  G
}

same_partition <- function(a, b) {
  a <- as.vector(a)
  b <- as.vector(b)
  length(unique(a)) == length(unique(b)) && length(unique(paste(a, b))) == length(unique(a))
}

probe_p7 <- function(L) {
  if (is.null(L$rowmap_n3p2)) stop("needs P3")
  ds <- bv_data("var_sv_n3.csv")
  Y <- ds$Y
  M <- 3L
  p <- 2L
  rm2 <- L$rowmap_n3p2
  lag_bv <- setdiff(seq_len(1L + M * p), rm2$intercept_bv)
  ours_lag <- rm2$bv_to_ours[lag_bv] - 1L
  gs <- c("global", "equation-wise", "covariate-wise", "olcl-lagwise")
  fields <- list()
  desc <- character()
  for (g in gs) {
    pr <- bayesianVARs::specify_prior_phi(data = Y, lags = p, prior = "HS", global_grouping = g)
    fl <- flatten_fields(as_data(pr))
    desc <- c(desc, describe(pr, sprintf("prior_phi_HS_%s", g)))
    fields[[g]] <- Filter(function(v) is.numeric(v) && length(v) == p * M * M &&
                            all(is.finite(v)) && all(v == round(v)) && all(v >= 0), fl)
  }
  # the grouping field: present for every grouping, its number of distinct values varies with it
  common <- Reduce(intersect, lapply(fields, names))
  nuniq <- vapply(common, function(nm) vapply(gs, function(g) length(unique(as.vector(fields[[g]][[nm]]))),
                                              integer(1)), integer(length(gs)))
  nuniq <- matrix(nuniq, nrow = length(gs), dimnames = list(gs, common))
  varies <- common[apply(nuniq, 2, function(z) length(unique(z)) > 1L)]
  if (length(varies) != 1L) {
    return(list(status = "unresolved", answer = "(no unique grouping field)",
                evidence = sprintf("integer fields of length lags*M^2 present for every grouping: %s; distinct values per grouping: %s",
                                   paste(common, collapse = ", "),
                                   paste(sprintf("%s %s", colnames(nuniq), apply(nuniq, 2, paste, collapse = "/")), collapse = "; ")),
                describe = desc))
  }
  fld <- varies
  res <- list()
  for (g in gs) {
    v <- fields[[g]][[fld]]
    G <- if (!is.null(dim(v)) && all(dim(v) == c(p * M, M))) matrix(v, p * M, M) else
      if (L$slot_rule == "column-major") matrix(v, p * M, M) else matrix(v, p * M, M, byrow = TRUE)
    Go <- matrix(NA_integer_, p * M, M)
    Go[ours_lag, ] <- as.integer(G)
    res[[g]] <- list(found = TRUE, field = fld, G = Go, d38 = same_partition(Go, d38_groups(g, M, p)))
  }
  txt <- vapply(gs, function(g) sprintf("%s: %d groups, %s D-38", g, length(unique(as.vector(res[[g]]$G))),
                                       if (res[[g]]$d38) "equals" else "differs from"), character(1))
  list(
    status = "answered",
    answer = paste0("grouping field ", fld, " (group ids per coefficient); ", paste(txt, collapse = "; ")),
    evidence = paste(c(txt, vapply(gs, function(g) sprintf(
      "%s (rai-bvar lag rows x equations, column-major): %s", g, paste(res[[g]]$G, collapse = ",")),
      character(1))), collapse = "; "),
    layout = list(groups = lapply(res, `[[`, "G"), group_field = fld),
    groups = res, describe = desc)
}

probe_p9 <- function(L) {
  if (is.null(L$rowmap_n3p2)) stop("needs P3")
  ds <- bv_data("var_sv_n3.csv")
  Y <- ds$Y
  M <- 3L
  p <- 2L
  K <- 7L
  c0 <- 0.01
  c1 <- 100
  pr <- bayesianVARs::specify_prior_phi(data = Y, lags = p, prior = "SSVS", SSVS_c0 = c0,
                                        SSVS_c1 = c1, SSVS_semiautomatic = TRUE)
  fl <- flatten_fields(as_data(pr))
  L0 <- p * M * M
  cands <- list()
  for (nm in names(fl)) {
    v <- fl[[nm]]
    if (!is.numeric(v) || !length(v) || length(v) %% L0 != 0L || length(v) > 4L * L0) next
    if (!all(is.finite(v)) || !all(v > 0)) next
    for (k in seq_len(length(v) / L0)) cands[[sprintf("%s[%d]", nm, k)]] <- as.vector(v)[(k - 1L) * L0 + seq_len(L0)]
  }
  des <- swa_design(Y, p)
  s_hat <- function(X, df) {
    B <- solve(crossprod(X), crossprod(X, des$Y))
    S <- crossprod(des$Y - X %*% B)
    sqrt(outer(diag(solve(crossprod(X))), diag(S)) / df)
  }
  Tn <- nrow(des$X)
  noint <- des$X[, -1, drop = FALSE]
  variants <- list(
    "D-44 (s.d., df T - k - n - 1)" = s_hat(des$X, Tn - K - M - 1),
    "s.d., df T - k" = s_hat(des$X, Tn - K),
    "s.d., df T" = s_hat(des$X, Tn),
    "s.d., df T - k - n - 1, X without the intercept" =
      rbind(NA, s_hat(noint, Tn - (K - 1L) - M - 1)),
    "s.d., df T - k, X without the intercept" = rbind(NA, s_hat(noint, Tn - (K - 1L))))
  rmap <- L$rowmap_n3p2
  hits <- character()
  ratios <- character()
  for (vn in names(variants)) {
    sl <- bv_slots(variants[[vn]], L, rmap, M, p)
    for (cn in names(cands)) {
      for (form in c("c * s", "c * s^2", "c^2 * s^2")) for (cc in c(c0 = c0, c1 = c1)) {
        target <- switch(form, "c * s" = cc * sl, "c * s^2" = cc * sl^2, "c^2 * s^2" = cc^2 * sl^2)
        if (length(target) != length(cands[[cn]]) || anyNA(target)) next
        rel <- max(abs(cands[[cn]] / target - 1))
        if (is.finite(rel) && rel < 1e-8) hits <- c(hits, sprintf("%s = %s with c = %g, s from %s (max rel. diff %.1e)", cn, form, cc, vn, rel))
      }
      if (length(sl) == length(cands[[cn]]) && !anyNA(sl)) {
        ratios <- c(ratios, sprintf("%s / (c0 s, %s): %.6g .. %.6g", cn, vn,
                                    min(cands[[cn]] / (c0 * sl)), max(cands[[cn]] / (c0 * sl))))
      }
    }
  }
  d44 <- any(grepl("D-44", hits))
  list(
    status = if (length(cands)) "answered" else "unresolved",
    answer = if (!length(cands)) "(no tau field found in the prior object)" else if (d44)
      "tau0 = c0 * s_hat and tau1 = c1 * s_hat with D-44's s_hat (flat-prior normal-Wishart posterior s.d.)"
      else if (length(hits)) paste("differs from D-44; matches:", paste(unique(hits), collapse = "; "),
                                   "- the runs use SSVS_semiautomatic = FALSE with explicit tau")
      else "differs from D-44 and from every variant tried: the runs use SSVS_semiautomatic = FALSE with explicit tau",
    evidence = paste(c(sprintf("candidate fields: %s", paste(names(cands), collapse = ", ")),
                       hits, utils::head(ratios, 40)), collapse = "; "),
    layout = list(ssvs_semiautomatic_d44 = d44))
}

probe_p10 <- function(L) {
  sc <- 1e-3
  homo <- bv_homo(L)
  int_test <- function(phi_args, seed0) {
    rs <- lapply(1:2, function(ch) bv_probe_fit("probe_tiny.csv", 1L, phi_args,
                                                bv_sigma_args("constant", homo = homo),
                                                num_text(sc), seed = seed0 + ch))
    chains <- lapply(rs, function(r) {
      rmap <- bv_row_map(r$fit, r$data, 1L, L$intercept_rule)
      matrix(bv_phi(unclass(r$fit), 2L, 3L, r$D)[rmap$intercept_bv, , ], nrow = 2L)
    })
    out <- list()
    for (reading in c(sd = sc, variance = sqrt(sc))) {
      z <- unlist(lapply(1:2, function(j) {
        s <- mc_summary(lapply(chains, function(m) m[j, ]))
        (s$q - qnorm(P2_PROBS, 0, reading)) / s$mcse
      }))
      out[[length(out) + 1L]] <- max(abs(z))
    }
    names(out) <- c("sd", "variance")
    c(out, list(call = rs[[1]]$call))
  }
  a <- int_test('prior = "normal", normal_sds = 1, priormean = 0, PHI_tol = 0', BV_PROBE_SEED + 800L)
  b <- int_test('prior = "HS", priormean = 0, PHI_tol = 0', BV_PROBE_SEED + 810L)
  bound <- parity_bound(10)
  sd_reading <- a$sd <= bound && a$variance > bound
  outside <- b$sd <= bound
  # (c) tolerances 0
  ds <- bv_data("var_sv_n3.csv")
  fam_args <- list(
    HS = function(t) sprintf('prior = "HS", PHI_tol = %s', num_text(t)),
    NG = function(t) sprintf('prior = "NG", NG_a = 0.1, NG_b = 1, NG_c = 1, NG_tol = %s, PHI_tol = %s', num_text(t), num_text(t)),
    R2D2 = function(t) sprintf('prior = "R2D2", R2D2_a = 0.1, R2D2_b = 0.5, R2D2_tol = %s, PHI_tol = %s', num_text(t), num_text(t)),
    DL = function(t) sprintf('prior = "DL", DL_a = 0.5, DL_tol = %s, PHI_tol = %s', num_text(t), num_text(t)),
    SSVS = function(t) sprintf('prior = "SSVS", SSVS_c0 = 0.01, SSVS_c1 = 100, SSVS_semiautomatic = FALSE, SSVS_p = 0.5, PHI_tol = %s', num_text(t)))
  tol <- list()
  txt <- character()
  for (fam in names(fam_args)) {
    accepted <- NA_real_
    msgs <- character()
    for (t in c(0, 1e-300, 1e-100, 1e-50, 1e-18)) {
      r <- tryCatch(bv_probe_fit("var_sv_n3.csv", 2L, fam_args[[fam]](t),
                                 bv_sigma_args("sv", u_tol = t), "10",
                                 draws = if (QUICK) 500L else 2000L, burnin = 500L,
                                 seed = BV_PROBE_SEED + 900L),
                    error = function(e) e)
      if (inherits(r, "error")) {
        msgs <- c(msgs, sprintf("tol %g: %s", t, conditionMessage(r)))
        next
      }
      f <- unclass(r$fit)
      P <- draws_last(f$PHI, r$D)
      hp <- f$phi_hyperparameter
      fin <- all(is.finite(P)) && (is.null(hp) || all(is.finite(unclass(hp))))
      if (!fin) {
        msgs <- c(msgs, sprintf("tol %g: non-finite draws", t))
        next
      }
      accepted <- t
      txt <- c(txt, sprintf(
        "%s: tolerance %g accepted; fraction of PHI draws with |x| <= tol %.2e; hyperparameter draws <= tol %.2e",
        fam, t, mean(abs(P) <= t), if (is.null(hp)) 0 else mean(abs(unclass(hp)) <= t)))
      break
    }
    if (!is.finite(accepted)) txt <- c(txt, sprintf("%s: no tolerance accepted (%s)", fam, paste(msgs, collapse = " | ")))
    else if (length(msgs)) txt <- c(txt, sprintf("%s rejected: %s", fam, paste(msgs, collapse = " | ")))
    tol[[fam]] <- list(phi = accepted, fam = accepted, u = accepted)
  }
  all_tol <- all(vapply(tol, function(x) is.finite(x$phi), logical(1)))
  zero_tol <- all(vapply(tol, function(x) isTRUE(x$phi == 0), logical(1)))
  status <- if ((sd_reading || (a$variance <= bound && a$sd > bound)) && all_tol) "answered" else "unresolved"
  list(
    status = status,
    answer = if (status == "answered") paste0(
      "prior_intercept is ", if (sd_reading) "a standard deviation" else "a variance", "; the intercept is ",
      if (outside) "outside the shrinkage (HS fit: intercept posterior = its N(0, s_c^2) prior)" else
        "NOT outside the shrinkage", "; ", if (zero_tol) "every tolerance accepts 0" else
          "not every tolerance accepts 0 (smallest accepted values used, see evidence)")
      else "(undecided)",
    evidence = paste(c(sprintf(
      "Normal prior: max |z| vs N(0, s^2) %.2f (s.d. reading), %.2f (variance reading); HS: %.2f, %.2f; bound %.2f",
      a$sd, a$variance, b$sd, b$variance, bound), txt), collapse = "; "),
    layout = list(tol = tol, intercept_sd = sd_reading, intercept_outside = outside))
}

# ------------------------------------------------------------------------------------------------
# P8 data and the parity runs R1–R9
# ------------------------------------------------------------------------------------------------

P8_CONFIGS <- list(
  list(id = "hs", family = "HS", args = 'prior = "HS"'),
  list(id = "ng_a0.1_b1_c4", family = "NG", args = 'prior = "NG", NG_a = 0.1, NG_b = 1, NG_c = 4, NG_tol = 0'),
  list(id = "ng_a0.1_b1_c1", family = "NG", args = 'prior = "NG", NG_a = 0.1, NG_b = 1, NG_c = 1, NG_tol = 0'),
  list(id = "r2d2_a0.1_b4", family = "R2D2", args = 'prior = "R2D2", R2D2_a = 0.1, R2D2_b = 4, R2D2_tol = 0'),
  list(id = "r2d2_a0.1_b0.5", family = "R2D2", args = 'prior = "R2D2", R2D2_a = 0.1, R2D2_b = 0.5, R2D2_tol = 0'),
  list(id = "dl_a1overK", family = "DL", args = 'prior = "DL", DL_a = "1/K", DL_tol = 0'),
  list(id = "dl_a0.5", family = "DL", args = 'prior = "DL", DL_a = 0.5, DL_tol = 0'))

BV_GRID_SUPPORT <- seq(0.1, 1, by = 0.1)
BV_GRID_WEIGHTS <- exp(-BV_GRID_SUPPORT) / sum(exp(-BV_GRID_SUPPORT))
# Hyperparameter columns summarised (with MCSE) besides the thinned draws of all columns: the
# global scales and the hyperprior index (regular expressions on the column names).
BV_HYPER_SELECT <- "zeta|global|^a$|^a[0-9_]|xi|lambda"

BV_RUN_TABLE <- c(
  list(
    list(name = "bvars_bayesianvars_n3p2_sv", run = "R1", data = "var_sv_n3.csv", lags = 2L, vol = "sv", family = "normal"),
    list(name = "bvars_bayesianvars_n7p4_sv", run = "R1", data = "var_sv_n7.csv", lags = 4L, vol = "sv", family = "normal"),
    list(name = "bvars_bayesianvars_n3p2_const", run = "R2", data = "var_sv_n3.csv", lags = 2L, vol = "constant", family = "normal")),
  unlist(lapply(list(c("R3", "HS"), c("R4", "NG"), c("R5", "R2D2"), c("R6", "DL"), c("R7", "SSVS")),
                function(rf) lapply(c(3L, 7L), function(n) list(
                  name = sprintf("bvars_gl_%s_%s_global_n%dp2", tolower(rf[1]), tolower(rf[2]), n),
                  run = rf[1], data = sprintf("var_sv_n%d.csv", n), lags = 2L, vol = "sv",
                  family = rf[2], groups = "global")))
         , recursive = FALSE),
  list(
    list(name = "bvars_gl_r8_hs_olcl_n7p4", run = "R8", data = "var_sv_n7.csv", lags = 4L, vol = "sv", family = "HS", groups = "olcl-lagwise"),
    list(name = "bvars_gl_r9_nggrid_global_n7p2", run = "R9", data = "var_sv_n7.csv", lags = 2L, vol = "sv", family = "NG", groups = "global", grid = TRUE)))

bv_gl_args <- function(fx, L) {
  t <- L$tol[[fx$family]]$phi
  fam <- switch(fx$family,
    HS = 'prior = "HS"',
    NG = sprintf('prior = "NG", NG_a = %s, NG_b = 1, NG_c = 1, NG_tol = %s',
                 if (isTRUE(fx$grid)) "ng_grid" else "0.1", num_text(t)),
    R2D2 = sprintf('prior = "R2D2", R2D2_a = 0.1, R2D2_b = 0.5, R2D2_tol = %s', num_text(t)),
    DL = sprintf('prior = "DL", DL_a = 0.5, DL_tol = %s', num_text(t)),
    SSVS = 'prior = "SSVS", SSVS_c0 = 0.01, SSVS_c1 = 100, SSVS_semiautomatic = FALSE, SSVS_p = 0.5')
  sprintf('%s, priormean = 0, PHI_tol = %s, global_grouping = "%s"', fam, num_text(t), fx$groups)
}

bv_transform <- function(Y, L) {
  if (!isTRUE(L$standardise)) return(list(Y = Y, note = "none (P6: no internal standardisation)"))
  ctr <- colMeans(Y)
  scl <- apply(Y, 2, stats::sd)
  list(Y = sweep(sweep(Y, 2, ctr), 2, scl, "/"), center = ctr, scale = scl,
       note = "standardised by P6: (y - colMeans(y)) / apply(y, 2, sd)")
}

bv_run_fixture <- function(fx, L, seed, probe_answers, kind, label, chains_n = BV_RUNS_N,
                           draws = BV_DRAWS, burnin = BV_BURN, data_dir_file = NULL,
                           phi_args = NULL, vol_args = NULL, horizon = BV_HORIZON,
                           mixing_hard = TRUE, vprior = FALSE) {
  ds <- bv_data(fx$data)
  tr <- bv_transform(ds$Y, L)
  Y <- tr$Y
  M <- ncol(Y)
  p <- fx$lags
  K <- 1L + M * p
  Tn <- nrow(Y) - p
  h0 <- fx$vol == "sv" && !is.na(L$sv$rows[["h0"]])
  t_sel <- as.integer(c(if (h0) 0L, 1L, Tn %/% 4L, Tn %/% 2L, (3L * Tn) %/% 4L, Tn))
  prm <- bv_predicted_rowmap(M, p, L)
  vars <- list(data = Y, M = M)
  extra_vars <- list()
  extra_coords <- list()
  prior <- swa_prior_arrays(M, p)
  free <- which(lower.tri(diag(M)), arr.ind = TRUE)
  free <- free[order(free[, 1], free[, 2]), , drop = FALSE]
  sig <- vol_args %||% bv_sigma_args(fx$vol, homo = if (fx$vol == "constant") bv_homo(L) else "",
                                     u_tol = if (fx$family == "normal") 0 else L$tol[[fx$family]]$u)
  if (fx$family == "normal") {
    vars$bv_normal_sds <- bv_slot_vector(prior$var, L, prm, M, p)
    phi <- 'prior = "normal", normal_sds = bv_normal_sds, priormean = 0.5, PHI_tol = 0'
    extra_vars$bv_normal_sds <- nc_array(vars$bv_normal_sds, "bv_slot", paste(
      "the normal_sds vector passed to specify_prior_phi (P1 reading and order, P3 rows)"))
    extra_vars$prior_mean <- nc_array(prior$mean, c("regressor", "equation"), "prior mean of Pi")
    extra_vars$prior_var <- nc_array(prior$var, c("regressor", "equation"), "prior variance of Pi")
  } else {
    phi <- phi_args %||% bv_gl_args(fx, L)
    pm <- matrix(0, K, M)
    extra_vars$prior_mean <- nc_array(pm, c("regressor", "equation"), "prior mean of Pi (0)")
    if (isTRUE(fx$grid)) {
      vars$ng_grid <- cbind(BV_GRID_SUPPORT, BV_GRID_WEIGHTS)
      extra_vars$ng_grid_support <- nc_array(BV_GRID_SUPPORT, "grid_point", "NG_a grid: support (column 1 of ng_grid)")
      extra_vars$ng_grid_weight <- nc_array(BV_GRID_WEIGHTS, "grid_point", "NG_a grid: prior weights (column 2 of ng_grid)")
    }
    if (!is.null(L$group_field)) {   # this run's grouping matrix (P7's field), rai-bvar rows
      gm <- tryCatch({
        env <- new.env(parent = globalenv())
        assign("data", Y, envir = env)
        for (k in names(vars)) assign(k, vars[[k]], envir = env)
        pr <- eval(parse(text = sprintf("specify_prior_phi(data = data, lags = %dL, %s)", p, phi)), env)
        v <- flatten_fields(as_data(pr))[[L$group_field]]
        G <- if (!is.null(dim(v))) matrix(v, p * M, M) else if (L$slot_rule == "column-major")
          matrix(v, p * M, M) else matrix(v, p * M, M, byrow = TRUE)
        lag_bv <- setdiff(seq_len(K), prm$intercept_bv)
        Go <- matrix(NA_integer_, p * M, M)
        Go[prm$bv_to_ours[lag_bv] - 1L, ] <- as.integer(G)
        Go
      }, error = function(e) NULL)
      if (!is.null(gm)) extra_vars$groups_matrix <- nc_array(gm, c("lag_regressor", "equation"),
        "bayesianVARs' grouping matrix of this run (P7's field), rai-bvar lag-row order")
    }
  }
  extra_vars$a_prior_mean <- nc_array(rep(0, nrow(free)), "a_free", "prior mean of a_ij")
  extra_vars$a_prior_var <- nc_array(rep(1, nrow(free)), "a_free", "prior variance of a_ij")
  if (fx$vol == "constant") {
    extra_vars$ig_shape <- nc_array(rep(3, M), "equation", "lambda_i ~ IG(shape, scale)")
    extra_vars$ig_scale <- nc_array(rep(2, M), "equation", "lambda_i ~ IG(shape, scale)")
  }
  call <- bv_call(p, draws, burnin, phi, sig, "10")
  chains <- vector("list", chains_n)
  for (ch in seq_len(chains_n)) {
    r <- bv_eval(call, vars, seed + ch)
    r$data <- Y
    r$D <- draws
    ex <- bv_extract(r, L, p, fx$vol, t_sel, horizon, seed + 1000L + ch,
                     hyper = fx$family != "normal", vprior = vprior)
    if (!identical(ex$row_map$bv_to_ours, prm$bv_to_ours)) stop("PHI rows not in the predicted order")
    ex$seconds <- r$seconds
    rm(r)
    invisible(gc(verbose = FALSE))
    chains[[ch]] <- ex
    message(sprintf("    %s run %d: %.1fs", fx$name, ch, ex$seconds))
  }
  hy <- if (fx$family != "normal") hyper_staging(chains, BV_HYPER_SELECT,
                                                 if (isTRUE(fx$grid)) BV_GRID_SUPPORT else NULL)
    else list(vars = list(), coords = list(), blocks = list(), class = "none")
  if (vprior && !is.null(chains[[1]]$vprior)) {
    th <- thinned_draws(lapply(chains, `[[`, "vprior"), BV_HYPER_KEEP)
    extra_vars$vprior_draws <- nc_array(th$draws, c("chain", "hyper_draw", "vprior_slot"), sprintf(
      "thinned draws of V_prior (bayesianVARs' prior variances of PHI), flattened (column-major) from dim %s",
      paste(chains[[1]]$vprior_dim, collapse = " x ")))
    if (is.null(hy$coords$hyper_draw)) extra_coords$hyper_draw <- nc_array(th$keep, "hyper_draw")
  }
  info <- list(
    kind = kind, n = M, lags = p, horizon = horizon, vol = fx$vol, variables = ds$variables,
    t_sel = t_sel,
    attrs = list(
      run = fx$run, family = fx$family, groups = fx$groups %||% "none", vol = fx$vol,
      lags = p, n = M, T = Tn, k = K,
      data_file = file.path("data", ds$file), data_md5 = ds$md5, data_transform = tr$note,
      data_center = if (!is.null(tr$center)) I(tr$center) else NULL,
      data_scale = if (!is.null(tr$scale)) I(tr$scale) else NULL,
      seed = seed, chain_seeds = I(seed + seq_len(chains_n)),
      forecast_seeds = I(seed + 1000L + seq_len(chains_n)),
      draws = draws, burnin = burnin, chains = chains_n,
      runs = "independent bvar() runs, one chain each, seeds in chain_seeds",
      h_path_thin = BV_PATH_THIN, call = call, label = label,
      probe_answers = probe_answers,
      hyper_columns = hy$class,
      rai_bvar_spec = if (fx$family == "normal") paste(
        "BVAR(Linear(Minnesota(form='reduced')),", if (fx$vol == "sv") paste(
          "SV(mu_prior=Normal(0.0, 1e4), phi_prior=Beta(20, 1.5), sigma2_prior=ScaledChi2(1.0),",
          "h0='stationary', offset=0.0, exact=False))") else "Constant('ig'))",
        "through fit_mcmc with explicit arrays (prior_mean, prior_var, a_prior_*)") else paste(
          "BVAR(Linear(", fx$family, "(form='reduced', groups='", fx$groups, "', scale='none', ",
          "fixed hyperparameters as in `call`)), SV(mu_prior=Normal(0.0, 1e4), ",
          "phi_prior=Beta(20, 1.5), sigma2_prior=ScaledChi2(1.0), h0='stationary', offset=0.0, ",
          "exact=False)); intercept N(0, 10^2) (var_const = 100), a_ij ~ N(0, 1)", sep = ""),
      layout = paste(
        "regressor order: const, lag-1 block (variables in data order), ..., lag-p block;",
        "t_sel = {", if (h0) "0, ", "1, T %/% 4, T %/% 2, 3T %/% 4, T} (t = data row after the",
        "lags); forecast horizon s = 1..4 after the last data row; A u_t = e_t"),
      compare = paste(
        "Phase 2 parity rule (Phase 3 DESIGN Conventions, sec. 9.8): quantiles of Pi, A, h_t",
        "(t >= 1), mu, phi, sigma (or lambda), forecasts; GL: and the hyperparameters with the",
        "same meaning")))
  stage_reduced_fixture(fx$name, chains, info, c(extra_vars, hy$vars), c(extra_coords, hy$coords),
                        hy$blocks, mixing_hard = mixing_hard)
}

probe_answers_text <- function(probes) {
  paste(vapply(names(probes), function(id) sprintf("%s: %s - %s", id, probes[[id]]$status,
                                                   probes[[id]]$answer), character(1)),
        collapse = "\n")
}

run_bvars_p3 <- function(env) {
  probes <- run_bvars_probes()
  record_layout("bayesianvars_probes.txt", unlist(lapply(probes, `[[`, "describe")))
  if (want("probe_bayesianvars")) stage_probes(env, probes)
  L <- bv_layout(probes)
  p3 <- list()
  p3$P6 <- attempt_probe("P6", probe_p6(L))
  L$standardise <- isTRUE(p3$P6$layout$standardise)
  p3$P7 <- attempt_probe("P7", probe_p7(L))
  if (!is.null(p3$P7$describe)) record_layout("bayesianvars_prior_phi.txt", p3$P7$describe)
  if (identical(p3$P7$status, "answered")) L$groups <- p3$P7$layout$groups
  p3$P9 <- attempt_probe("P9", probe_p9(L))
  p3$P10 <- attempt_probe("P10", probe_p10(L))
  if (identical(p3$P10$status, "answered")) L$tol <- p3$P10$layout$tol
  p3$P8 <- list(status = "pending", answer = "data committed (probe_bvars_p8_*.nc), decision pending WP30",
                evidence = "")
  p3 <- p3[c("P6", "P7", "P8", "P9", "P10")]
  for (id in names(p3)) p3[[id]] <- c(PROBES_P3[[id]], p3[[id]])
  all_probes <- c(probes, p3)
  answers <- probe_answers_text(all_probes)
  message(answers)
  # P8 data
  p8_txt <- character()
  if (!is.null(L$tol) && !is.null(L$homo_reading)) {
    for (i in seq_along(P8_CONFIGS)) {
      cf <- P8_CONFIGS[[i]]
      name <- paste0("probe_bvars_p8_", cf$id)
      if (!want(name)) next
      r <- tryCatch({
        bv_run_fixture(list(name = name, run = "P8", data = "probe_gl.csv", lags = 2L,
                            vol = "constant", family = cf$family, groups = "global"),
                       L, 20261700L + 100L * i, answers, "probe_p8", paste("P8", cf$id),
                       phi_args = paste0(cf$args, ', priormean = 0, PHI_tol = 0, global_grouping = "global"'),
                       horizon = 0L, mixing_hard = FALSE, vprior = TRUE)
        "staged"
      }, error = function(e) paste("failed:", conditionMessage(e)))
      p8_txt <- c(p8_txt, sprintf("%s: %s", name, r))
    }
  } else p8_txt <- "not run: P5 or P10 undecided"
  all_probes$P8$evidence <- paste(p8_txt, collapse = "; ")
  # parity runs
  needs <- c("P1", "P2", "P3", "P4", "P5")
  run_txt <- character()
  for (i in seq_along(BV_RUN_TABLE)) {
    fx <- BV_RUN_TABLE[[i]]
    if (!want(fx$name)) next
    dep <- c(needs, if (fx$family != "normal") c("P6", "P10"), if (identical(fx$groups, "olcl-lagwise")) "P7")
    open <- dep[vapply(dep, function(id) !identical(all_probes[[id]]$status, "answered"), logical(1))]
    if (length(open)) {
      run_txt <- c(run_txt, sprintf("%s (%s): not run, probes %s undecided (D-59: not met)", fx$name, fx$run, paste(open, collapse = ", ")))
      next
    }
    kind <- if (fx$family == "normal") "bvars_bayesianvars" else "bvars_gl"
    r <- tryCatch({
      bv_run_fixture(fx, L, 20261500L + 100L * i, answers, kind, fx$run)
      "staged"
    }, error = function(e) paste("failed:", conditionMessage(e)))
    run_txt <- c(run_txt, sprintf("%s (%s): %s", fx$name, fx$run, r))
    message("  ", run_txt[length(run_txt)])
  }
  if (want("probe_bayesianvars_p3")) {
    attrs <- p3_meta("bayesianVARs", "probe_p3")
    attrs$bayesianvars_version <- as.character(packageVersion("bayesianVARs"))
    attrs$probe_answers <- answers
    attrs$runs <- paste(run_txt, collapse = "\n")
    attrs$p8_fixtures <- paste(p8_txt, collapse = "\n")
    vars <- list()
    coords <- list()
    if (!is.null(L$groups)) {
      gnames <- names(L$groups)
      G <- array(NA_integer_, c(length(gnames), 6L, 3L))
      for (g in seq_along(gnames)) if (!is.null(L$groups[[g]])) G[g, , ] <- L$groups[[g]]
      G[is.na(G)] <- 0L
      coords$grouping <- nc_array(gnames, "grouping")
      vars$p7_groups <- nc_array(G, c("grouping", "lag_regressor", "equation"), paste(
        "bayesianVARs' grouping matrices (var_sv_n3, p = 2) in rai-bvar lag-row order (0: not found)"))
      vars$p7_equals_d38 <- nc_array(vapply(gnames, function(g) as.integer(isTRUE(
        p3$P7$groups[[g]]$d38)), integer(1)), "grouping", "1 if the partition equals DESIGN D-38's")
    }
    stage_probe_fixture("probe_bayesianvars_p3", attrs, all_probes[c("P6", "P7", "P8", "P9", "P10")],
                        c("P6", "P7", "P8", "P9", "P10"), vars, coords)
  }
  invisible(list(probes = all_probes, L = L, runs = run_txt))
}

if ("bvars" %in% SECTIONS) {
  message("(f) bayesianVARs probes and parity runs; the system-wide Gibbs oracle")
  env <- bvars_environment()
  message("  bayesianVARs available: ", env$bayesianvars_available)
  if (isTRUE(env$bayesianvars_available)) {
    suppressPackageStartupMessages(library(bayesianVARs))
    message("  bayesianVARs ", packageVersion("bayesianVARs"))
    quiet_calls(run_bvars_p3(env))
  } else {
    stage_probes(env)
  }
  if (!NO_SWA) {
    sc <- swa_self_checks()
    z <- sv_single_step_check()
    message(sprintf("  stochvol single-step vs svsample: max |z| = %.2f", z))
    for (fx in SWA_FIXTURES) run_swa_fixture(fx, sc, z)
  } else {
    message("  --no-swa: the system-wide Gibbs oracle fixtures bvars_swa_*.nc are not regenerated")
  }
}

# =================================================================================================
# (h) bvars 1.1 (Woźniak): probes P11–P16 and the CommonSV parity runs (Phase 3 DESIGN §9.8)
# =================================================================================================
#
# Section `wbvars` (the name `bvars` above means bayesianVARs). bvars' model (manual): Y = A X + E,
# e_t ~ N(0, sigma_t^2 lambda_t Sigma), A | Sigma, V ~ MN(A0, Sigma, V), Sigma ~ IW(S0, nu0), V a
# matrix-GIG hyperparameter, sigma_t^2 a common volatility (cSV centred, ncSV non-centred SV).
# Only data fields are read: the documented getters get_prior(), get_starting_values(),
# get_data_matrices(), get_posterior(), get_last_draw(), whose values are lists of arrays.
# estimate() has no burn-in argument: every run is a burn-in estimate() and a second estimate()
# restarted from its last_draw. h, rho and omega exist only as the last_draw starting values,
# so their draws come from chained estimate() calls (P16 and the state runs).

WB_LAGS <- 4L
WB_DRAWS <- if (QUICK) 2000L else 50000L
WB_BURN <- if (QUICK) 500L else 10000L
WB_RUNS_N <- 4L
WB_HORIZON <- 4L
WB_CHAIN_CALLS <- as.integer(arg_value("wb-chain-calls", if (QUICK) "40" else "1000"))
WB_CHAIN_S <- 50L
WB_P16_BURN <- if (QUICK) 500L else 10000L
WB_P16_CALLS <- as.integer(arg_value("wb-p16-calls", if (QUICK) "300" else "10000"))
WB_SEED <- 20261800L
WB_VOLS <- c("cSV", "ncSV")
WB_STATE_FIELDS <- c("h", "rho", "omega", "sigma2v", "sigma2_omega", "s_", "S")

wb_read <- function(file, dir = FIXTURES) {
  path <- file.path(dir, file)
  frame <- read.csv(path, check.names = FALSE, stringsAsFactors = FALSE)
  Y <- as.matrix(frame[, -1, drop = FALSE])
  storage.mode(Y) <- "double"
  list(file = file, rel = if (identical(dir, FIXTURES)) file else file.path("data", file),
       md5 = unname(tools::md5sum(path)), variables = colnames(frame)[-1], Y = Y)
}

wb_spec <- function(Y, p, vol) {
  bvars::specify_bvar$new(data = Y, p = p, common_volatility = vol, distribution = "norm")
}
wb_est <- function(x, S) estimate(x, S = S, thin = 1L, show_progress = FALSE)
wb_prior <- function(spec) as_data(spec$get_prior()$get_prior())
wb_start <- function(spec) as_data(spec$get_starting_values()$get_starting_values())
wb_dm <- function(spec) as_data(spec$get_data_matrices())
wb_post <- function(post) as_data(post$get_posterior())
wb_state <- function(post) {
  s <- as_data(post$get_last_draw()$get_starting_values()$get_starting_values())
  s[intersect(WB_STATE_FIELDS, names(s))]
}

# The columns of A (rows of X, K x T) matched exactly to the rai-bvar design (const, lag blocks).
wb_row_map <- function(spec, Y, p) {
  X <- as.matrix(wb_dm(spec)$X)
  des <- swa_design(Y, p)$X
  if (ncol(X) != nrow(des) && nrow(X) == nrow(des)) X <- t(X)
  m <- apply(X, 1, function(row) {
    w <- which(apply(des, 2, function(o) max(abs(o - row))) < 1e-10)
    if (length(w) == 1L) w else NA_integer_
  })
  if (anyNA(m) || !setequal(m, seq_len(ncol(des)))) {
    stop("the rows of bvars' X do not match the design: ", paste(m, collapse = ","))
  }
  list(wb_to_ours = as.integer(m), ours_cols = match(seq_len(ncol(des)), m))
}

# Draws of A (N x K x S) as B[draw, regressor (rai-bvar order), equation].
wb_B <- function(A, rmap) {
  A <- A[, rmap$ours_cols, , drop = FALSE]
  aperm(A, c(3, 2, 1))
}

wb_fixv_code <- function(fixv) {
  if (is.null(fixv)) return("")
  paste0("spec$prior$Psi <- fixv$Psi; spec$prior$Gamma <- fixv$Gamma; ",
         "spec$prior$lambda <- fixv$lambda; spec$starting_values$V <- fixv$Omega; ")
}

# specify_bvar$new() fits AR models to every series and needs more rows than the T = 2 dataset of
# P16 has; there the specification is built on a carrier (the data plus 40 N(0, 1) rows) and its
# data matrices, prior and starting values are then replaced by those of the data themselves
# (bsvars::specify_data_matrices, specify_prior_bvar, specify_starting_values_bvar).
wb_spec_code <- function(vol, p, tiny = FALSE) {
  if (!tiny) {
    return(sprintf(paste0("spec <- specify_bvar$new(data = data, p = %dL, common_volatility = \"%s\", ",
                          "distribution = \"norm\"); "), p, vol))
  }
  paste0(
    "carrier <- rbind(data, matrix(stats::rnorm(40L * ncol(data)), 40L, ncol(data))); ",
    sprintf(paste0("spec <- specify_bvar$new(data = carrier, p = %dL, common_volatility = \"%s\", ",
                   "distribution = \"norm\"); "), p, vol),
    sprintf("spec$data_matrices <- bsvars::specify_data_matrices$new(data, p = %dL); ", p),
    sprintf("spec$prior <- specify_prior_bvar$new(N = ncol(data), p = %dL, is_homoskedastic = FALSE); ", p),
    sprintf(paste0("spec$starting_values <- specify_starting_values_bvar$new(N = ncol(data), p = %dL, ",
                   "T = nrow(data) - %dL, is_homoskedastic = FALSE); "), p, p))
}

wb_run_code <- function(vol, p, burn, draws, fixv = NULL, tiny = FALSE) {
  paste0(wb_spec_code(vol, p, tiny),
         wb_fixv_code(fixv),
         sprintf("burn <- estimate(spec, S = %dL, thin = 1L, show_progress = FALSE); ", burn),
         if (draws > 0L) sprintf("post <- estimate(burn, S = %dL, thin = 1L, show_progress = FALSE)",
                                 draws) else "post <- burn")
}

wb_eval <- function(code, vars, seed) {
  env <- new.env(parent = globalenv())
  for (k in names(vars)) assign(k, vars[[k]], envir = env)
  set.seed(seed)
  t0 <- proc.time()[["elapsed"]]
  eval(parse(text = code), env)
  list(env = env, seconds = proc.time()[["elapsed"]] - t0)
}

# Chained estimate(·, S = s) calls after a burn-in: the last_draw state after every call.
wb_chain <- function(Y, p, vol, burn, calls, s, seed, fixv = NULL, keep_post = TRUE, tiny = FALSE) {
  r <- wb_eval(wb_run_code(vol, p, burn, 0L, fixv, tiny), list(data = Y, fixv = fixv), seed)
  post <- r$env$post
  states <- vector("list", calls)
  posts <- vector("list", calls)
  t0 <- proc.time()[["elapsed"]]
  for (k in seq_len(calls)) {
    post <- wb_est(post, s)
    states[[k]] <- wb_state(post)
    if (keep_post) {
      po <- wb_post(post)
      posts[[k]] <- list(A = po$A, Sigma = po$Sigma)
    }
  }
  list(states = states, posts = posts, seconds_per_call = (proc.time()[["elapsed"]] - t0) / calls,
       burn_seconds = r$seconds, spec = r$env$spec)
}

state_matrix <- function(states, field) {   # (calls x length) numeric
  x <- lapply(states, function(s) as.numeric(s[[field]]))
  if (any(vapply(x, length, 1L) == 0L)) return(NULL)
  do.call(rbind, x)
}

# Posterior draws (A, Sigma) of chained calls stacked along the draw axis.
stack_posts <- function(posts, field) {
  arrs <- lapply(posts, `[[`, field)
  d <- dim(arrs[[1]])
  out <- array(NA_real_, c(d[1], d[2], sum(vapply(arrs, function(a) dim(a)[3], 1L))))
  at <- 0L
  for (a in arrs) {
    out[, , at + seq_len(dim(a)[3])] <- a
    at <- at + dim(a)[3]
  }
  out
}

# ------------------------------------------------------------------------------------------------
# Probability-integral-transform tests of single Gibbs updates (P13, P14)
# ------------------------------------------------------------------------------------------------
#
# If a state component x is drawn from its full conditional given the other components,
# u = F(x | conditioning values) is U(0, 1) and independent across iterations whatever the state,
# so a KS test of the u of all iterations against U(0, 1) (alpha = 1e-3) checks a candidate
# conditional: the prior, the initial condition and the order of the updates (whether a
# component conditions on the value of this sweep or of the previous one). The conditional CDF is
# computed on a fine grid (trapezoid rule) from its log density.

pit_grid <- function(x, logdens, grid) {
  vapply(seq_along(x), function(i) {
    ld <- logdens(i)
    if (!is.finite(x[i]) || anyNA(ld) || !any(is.finite(ld))) return(NA_real_)
    w <- exp(ld - max(ld[is.finite(ld)]))
    w[!is.finite(w)] <- 0
    cw <- c(0, cumsum((w[-1] + w[-length(w)]) / 2 * diff(grid)))
    if (!(cw[length(cw)] > 0)) return(NA_real_)
    stats::approx(grid, cw / cw[length(cw)], xout = x[i], rule = 2)$y
  }, numeric(1))
}

ks_p <- function(u) {
  u <- u[is.finite(u)]
  if (length(u) < 50L) return(NA_real_)
  suppressWarnings(stats::ks.test(u, "punif")$p.value)
}

RHO_GRID <- seq(-1, 1, length.out = 4001L)

# ρ | h, σ² for h_t = ρ h_{t-1} + σ v_t, ρ ~ U(-1, 1): H0 h_0 = 0 (h_1 ~ N(0, σ²)); H1 stationary
# h_1 ~ N(0, σ²/(1 - ρ²)).
rho_logdens <- function(h, s2, init) {
  Tn <- length(h)
  sxx <- if (Tn > 1L) sum(h[-Tn]^2) else 0
  sxy <- if (Tn > 1L) sum(h[-1] * h[-Tn]) else 0
  ld <- -(RHO_GRID^2 * sxx - 2 * RHO_GRID * sxy) / (2 * s2)
  if (init == "stationary") ld <- ld + 0.5 * log1p(-RHO_GRID^2) + RHO_GRID^2 * h[1]^2 / (2 * s2)
  ld
}

wb_pit_rho <- function(runs, vol) {
  out <- list()
  for (init in c("h0_zero", "stationary")) {
    combos <- if (vol == "cSV") {
      expand.grid(h = c("same", "prev"), s2 = c("sigma2v_same", "sigma2v_prev", "omega2_same",
                                                 "omega2_prev"), stringsAsFactors = FALSE)
    } else {
      expand.grid(h = c("same", "prev"), s2 = c("unit_h", "unit_h_over_omega"),
                  stringsAsFactors = FALSE)
    }
    for (k in seq_len(nrow(combos))) {
      cb <- combos[k, ]
      u <- unlist(lapply(runs, function(st) {
        H <- state_matrix(st, "h")
        rho <- as.numeric(state_matrix(st, "rho"))
        om <- state_matrix(st, "omega")
        s2v <- state_matrix(st, "sigma2v")
        if (is.null(H) || !length(rho)) return(NULL)
        n <- nrow(H)
        idx <- 2:n
        hi <- if (cb$h == "same") idx else idx - 1L
        pit_grid(rho[idx], function(i) {
          s <- idx[i]
          hh <- H[hi[i], ]
          if (cb$s2 == "unit_h_over_omega") hh <- if (is.null(om)) NA_real_ else hh / om[hi[i]]
          s2 <- switch(cb$s2,
                       sigma2v_same = if (is.null(s2v)) NA_real_ else s2v[s],
                       sigma2v_prev = if (is.null(s2v)) NA_real_ else s2v[s - 1L],
                       omega2_same = if (is.null(om)) NA_real_ else om[s]^2,
                       omega2_prev = if (is.null(om)) NA_real_ else om[s - 1L]^2,
                       1)
          if (!is.finite(s2) || any(!is.finite(hh))) return(rep(NA_real_, length(RHO_GRID)))
          rho_logdens(hh, s2, if (init == "h0_zero") "h0_zero" else "stationary")
        }, RHO_GRID)
      }))
      out[[sprintf("%s | h %s, %s", init, cb$h, cb$s2)]] <- ks_p(u)
    }
  }
  out
}

# σ_ω² | ω, s_ for ω ~ N(0, σ_ω²), σ_ω² ~ G(sv_a, scale s_) or G(sv_a, rate s_).
wb_pit_sigma2_omega <- function(runs, sv_a) {
  ygrid <- seq(-40, 15, length.out = 8001L)
  out <- list()
  for (form in c("scale", "rate")) for (om_at in c("same", "prev")) for (s_at in c("same", "prev")) {
    u <- unlist(lapply(runs, function(st) {
      s2o <- as.numeric(state_matrix(st, "sigma2_omega"))
      om <- as.numeric(state_matrix(st, "omega"))
      ss <- as.numeric(state_matrix(st, "s_"))
      if (!length(s2o) || !length(om) || !length(ss)) return(NULL)
      idx <- 2:length(s2o)
      pit_grid(log(s2o[idx]), function(i) {
        s <- idx[i]
        w <- om[if (om_at == "same") s else s - 1L]
        sc <- ss[if (s_at == "same") s else s - 1L]
        inv <- if (form == "scale") 1 / sc else sc
        (sv_a - 0.5) * ygrid - exp(ygrid) * inv - w^2 * exp(-ygrid) / 2
      }, ygrid)
    }))
    out[[sprintf("G(sv_a, %s s_) | omega %s, s_ %s", form, om_at, s_at)]] <- ks_p(u)
  }
  out
}

repeat_fraction <- function(runs, field) {
  x <- unlist(lapply(runs, function(st) {
    v <- as.numeric(state_matrix(st, field))
    if (length(v) < 2L) return(NULL)
    as.numeric(diff(v) == 0)
  }))
  if (length(x)) mean(x) else NA_real_
}

fmt_p <- function(lst) paste(sprintf("%s: p = %.3g", names(lst), unlist(lst)), collapse = "; ")

# ------------------------------------------------------------------------------------------------
# Probes P11–P16
# ------------------------------------------------------------------------------------------------

PROBES_WB <- list(
  P11 = list(
    question = "What are the defaults of specify_prior_bvar and specify_starting_values_bvar?",
    documented = "manual: fields A, S, nu, Psi, Gamma, lambda, sv_a, sv_s; starting values A, Sigma, V, h, rho, omega, sigma2v, S, sigma2_omega, s_, lambda, df",
    dataset = "qd_small_levels.csv (FRED-QD: GDPC1, GDPCTPI, FEDFUNDS, 1960Q1-2019Q4, 100 x log levels and percent), p = 4",
    design = "recorded as data (the get_prior() and get_starting_values() lists) for homoskedastic, cSV and ncSV"),
  P12 = list(
    question = "Can V be held fixed?",
    documented = "manual: V has a matrix-GIG prior with Psi, Gamma (K x K scale matrices) and lambda (shape); its draws are the documented output",
    dataset = "qd_small_levels.csv, p = 4, cSV",
    design = paste(
      "Psi = c Omega, Gamma = c Omega^-1 (and the swapped orientation), lambda = (K + 1)/2,",
      "starting V = Omega = the default starting V, c in {1e4, 1e6, 1e8}: V is held fixed if",
      "the posterior draws have sd(V_ij) <= 1e-3 sqrt(Omega_ii Omega_jj) for every (i, j); the",
      "smallest such c whose posterior mean is Omega (to the same tolerance) is used")),
  P13 = list(
    question = "What is the common-volatility normalisation (E h = 0, initial condition) under cSV and ncSV?",
    documented = "manual: sigma_t^2 'estimated and following (non-)centred stochastic volatility'; h is a T-vector of starting values",
    dataset = "qd_small_levels.csv (scale experiment) and data/probe_t2n5.csv (n = 5, p = 1, T = 2; chained draws)",
    design = paste(
      "(a) the length of h (T: no separate h_0 state); (b) with V fixed (P12), data x 10 and",
      "S0 x 100, V' = D V D (D = 1 on the constant, 1/10 on the lags): A and Sigma/100",
      "equivariant under the Phase 2 rule => the level of the volatility is carried by Sigma",
      "and h has no free level (E h = 0); (c) the initial condition by the PIT test of every",
      "rho draw against its full conditional given h (rho ~ U(-1, 1)) under h_0 = 0 or a",
      "stationary h_1, for each update order (KS alpha = 1e-3): exactly one initial condition",
      "passes => decided")),
  P14 = list(
    question = "What are the rho and omega / sigma_omega^2 priors, and does a setting fix sigma_omega^2?",
    documented = "manual: sv_a 'shape', sv_s 'scale' of the gamma prior in the hierarchical prior for the common SV; s_ and sigma2_omega are starting values",
    dataset = "data/probe_t2n5.csv (chained draws of P16)",
    design = paste(
      "rho: the P13 PIT test (passes only if rho ~ U(-1, 1)); sigma_omega^2: PIT test against",
      "the conditional of omega ~ N(0, sigma_omega^2), sigma_omega^2 ~ G(sv_a, scale or rate",
      "s_); fixing: sv_a x 1e6 with sv_s x {1e-6, 1, 1e6}: a posterior CV of sigma_omega^2",
      "(and sigma2v) <= 1e-3 fixes it; repeat fractions show Metropolis steps")),
  P15 = list(
    question = "Is Sigma a covariance or a scale, what is the prior mean (stationary), where is the intercept?",
    documented = "manual: e_t ~ N(0, sigma_t^2 lambda_t Sigma); stationary = FALSE sets a random-walk prior mean, TRUE white noise",
    dataset = "data/probe_corr.csv (n = 2, p = 1, T = 600, Sigma = [[1, 1.6], [1.6, 4]], intercepts 5, -3) and qd_small_levels.csv",
    design = paste(
      "homoskedastic normal fit: posterior mean of Sigma within 5% of the residual covariance",
      "(OLS, df T - k) => covariance; prior A0 of stationary = FALSE / TRUE; the row of X",
      "equal to 1 is the intercept, its A column has posterior mean (5, -3) +- 0.5")),
  P16 = list(
    question = "Is bvars' h sampler exact?",
    documented = "manual: the starting values include S, 'the auxiliary mixture component indicator' (an Omori-type mixture sampler)",
    dataset = "data/probe_t2n5.csv (n = 5, p = 1, T = 2)",
    design = paste(
      "chained estimate(., S = 1) calls after a 10,000-draw burn-in, recording h, rho, omega",
      "(and sigma2v, sigma2_omega, s_, S) after each, k = 4 seeded runs per volatility;",
      "the extraction is checked against single estimate() runs (A and Sigma quantiles, Phase",
      "2 rule); the oracle (4-D quadrature with the closed-form p(Y | h), N08 sec. A4; KS",
      "alpha = 1e-3 on ESS-thinned draws) is WP38's"))
)

wb_probe_p11 <- function() {
  q <- wb_read("qd_small_levels.csv")
  lines <- character()
  vars <- list()
  vals <- list()
  for (vol in c("homoskedastic", "cSV", "ncSV")) {
    spec <- wb_spec(q$Y, WB_LAGS, vol)
    pf <- wb_prior(spec)
    sf <- wb_start(spec)
    dm <- wb_dm(spec)
    lines <- c(lines, describe(pf, paste0(vol, "$prior")), describe(sf, paste0(vol, "$starting_values")),
               describe(dm, paste0(vol, "$data_matrices")),
               sprintf("%s: get_normal() %s, get_homoskedastic() %s, get_centred_sv() %s", vol,
                       spec$get_normal(), spec$get_homoskedastic(), spec$get_centred_sv()))
    for (nm in names(pf)) if (is.numeric(pf[[nm]]) || is.logical(pf[[nm]])) {
      vars[[sprintf("p11_%s_prior_%s", vol, nm)]] <- nc_auto(pf[[nm]], sprintf(
        "bvars default prior field %s (common_volatility = %s, qd data, p = 4)", nm, vol))
    }
    for (nm in names(sf)) if (is.numeric(sf[[nm]]) || is.logical(sf[[nm]])) {
      vars[[sprintf("p11_%s_start_%s", vol, nm)]] <- nc_auto(sf[[nm]], sprintf(
        "bvars default starting value %s (common_volatility = %s, qd data, p = 4)", nm, vol))
    }
    vals[[vol]] <- list(prior = pf, start = sf)
  }
  scal <- function(l) paste(vapply(names(l), function(nm) {
    v <- l[[nm]]
    if (is.numeric(v) && length(v) == 1L) sprintf("%s = %.6g", nm, v) else
      sprintf("%s: %s", nm, if (is.null(dim(v))) paste("length", length(v)) else paste(dim(v), collapse = " x "))
  }, character(1)), collapse = ", ")
  list(status = "answered",
       answer = paste(vapply(names(vals), function(v) sprintf("%s: prior {%s}; starting values {%s}", v,
                                                              scal(vals[[v]]$prior), scal(vals[[v]]$start)),
                             character(1)), collapse = " || "),
       evidence = "the full fields are the p11_* variables of this fixture and layout/bvars_fields.txt",
       vars = vars, describe = lines, vals = vals)
}

wb_probe_p15 <- function() {
  q <- wb_read("qd_small_levels.csv")
  rmq <- wb_row_map(wb_spec(q$Y, WB_LAGS, "homoskedastic"), q$Y, WB_LAGS)
  cr <- wb_read("probe_corr.csv", P2_DATA)
  r <- wb_eval(wb_run_code("homoskedastic", 1L, if (QUICK) 500L else 2000L, if (QUICK) 1000L else 5000L),
               list(data = cr$Y), WB_SEED + 15L)
  rmc <- wb_row_map(r$env$spec, cr$Y, 1L)
  po <- wb_post(r$env$post)
  Sig <- apply(po$Sigma, c(1, 2), mean)
  A <- apply(po$A, c(1, 2), mean)
  des <- swa_design(cr$Y, 1L)
  B <- solve(crossprod(des$X), crossprod(des$X, des$Y))
  Sh <- crossprod(des$Y - des$X %*% B) / (nrow(des$X) - ncol(des$X))
  err_cov <- max(abs(Sig - Sh)) / max(abs(Sh))
  int <- A[, rmc$ours_cols[1]]
  int_ok <- max(abs(int - c(5, -3))) < 0.5
  K <- 1L + 3L * WB_LAGS
  a0 <- function(stationary) {
    pr <- bvars::specify_prior_bvar$new(N = 3L, p = WB_LAGS, stationary = rep(stationary, 3L))
    as_data(pr$get_prior())$A
  }
  rw <- a0(FALSE)[, rmq$ours_cols]
  wn <- a0(TRUE)[, rmq$ours_cols]
  own1 <- cbind(1:3, 1L + 1:3)        # (equation i, regressor y_i.L1) in rai-bvar order
  own_mask <- matrix(FALSE, 3L, K)
  own_mask[own1] <- TRUE
  rw_ok <- all(rw[own_mask] == 1) && all(rw[!own_mask] == 0)
  wn_ok <- all(wn == 0)
  status <- if (err_cov < 0.05 && int_ok && rw_ok && wn_ok) "answered" else "unresolved"
  list(
    status = status,
    answer = if (status == "answered") paste(
      "Sigma is the error covariance (homoskedastic normal; under SV the covariance at h = 0);",
      "stationary = FALSE gives A0 = 1 on the own first lag and 0 elsewhere, TRUE gives 0;",
      sprintf("the intercept is row %d of X (column %d of A); X rows match the rai-bvar regressors as (%s)",
              rmq$ours_cols[1], rmq$ours_cols[1], paste(rmq$wb_to_ours, collapse = ", "))) else "(undecided)",
    evidence = sprintf(paste(
      "probe_corr: posterior mean Sigma %s vs residual covariance %s (max rel. diff %.3f);",
      "intercept column mean %s; A0 (stationary = FALSE) in rai-bvar order %s; A0 (TRUE) all",
      "zero %s"), paste(signif(Sig, 4), collapse = ","), paste(signif(Sh, 4), collapse = ","),
      err_cov, paste(signif(int, 4), collapse = ","), paste(rw, collapse = ","), wn_ok),
    rmq = rmq)
}

wb_probe_p12 <- function() {
  q <- wb_read("qd_small_levels.csv")
  K <- 1L + 3L * WB_LAGS
  Om <- wb_start(wb_spec(q$Y, WB_LAGS, "cSV"))$V
  Oi <- solve(Om)
  sc <- sqrt(outer(diag(Om), diag(Om)))
  sci <- sqrt(outer(diag(Oi), diag(Oi)))
  settings <- list()
  for (cc in c(1e4, 1e6, 1e8)) for (orient in c("Psi = c Omega", "Psi = c Omega^-1")) {
    settings[[sprintf("c = %.0e, %s", cc, orient)]] <- list(
      c = cc, orient = orient, Omega = Om, lambda = (K + 1) / 2,
      Psi = if (orient == "Psi = c Omega") cc * Om else cc * Oi,
      Gamma = if (orient == "Psi = c Omega") cc * Oi else cc * Om)
  }
  settings[["control: default prior and starting V"]] <- list(control = "")
  settings[["control: default prior, starting V = 2 Omega"]] <- list(
    control = "spec$starting_values$V <- 2 * Omega; ")
  res <- lapply(names(settings), function(nm) tryCatch({
    st <- settings[[nm]]
    code <- if (is.null(st$control)) wb_run_code("cSV", WB_LAGS, if (QUICK) 200L else 1000L,
                                                 if (QUICK) 300L else 2000L, st) else
      paste0(wb_spec_code("cSV", WB_LAGS), st$control,
             sprintf("burn <- estimate(spec, S = %dL, thin = 1L, show_progress = FALSE); ", if (QUICK) 200L else 1000L),
             sprintf("post <- estimate(burn, S = %dL, thin = 1L, show_progress = FALSE)", if (QUICK) 300L else 2000L))
    r <- wb_eval(code, list(data = q$Y, fixv = st, Omega = Om), WB_SEED + 12L)
    V <- wb_post(r$env$post)$V
    sdV <- apply(V, c(1, 2), stats::sd)
    mV <- apply(V, c(1, 2), mean)
    list(ok = TRUE, ratio = max(sdV / sc), err = max(abs(mV - Om) / sc),
         err_inv = max(abs(mV - Oi) / sci), mean = mV)
  }, error = function(e) list(ok = FALSE, msg = conditionMessage(e))))
  names(res) <- names(settings)
  txt <- vapply(names(res), function(nm) {
    x <- res[[nm]]
    if (!x$ok) sprintf("%s: failed (%s)", nm, x$msg) else
      sprintf("%s: max sd(V_ij)/sqrt(Om_ii Om_jj) %.2e, max |mean - Omega| (scaled) %.2e, vs Omega^-1 %.2e",
              nm, x$ratio, x$err, x$err_inv)
  }, character(1))
  ctrl <- grepl("^control", names(res))
  pass <- names(res)[!ctrl & vapply(res, function(x) isTRUE(x$ok) && x$ratio <= 1e-3, logical(1))]
  default_moves <- isTRUE(res[[which(ctrl)[1]]]$ratio > 1e-3)
  at_om <- pass[vapply(pass, function(nm) res[[nm]]$err <= 1e-3, logical(1))]
  best <- if (length(at_om)) at_om[1] else NULL
  ran <- all(vapply(res, `[[`, logical(1), "ok"))
  list(
    status = if (length(best) || ran) "answered" else "unresolved",
    answer = if (length(best)) sprintf(paste(
      "V can be held fixed at the default starting V (Omega): %s, lambda = (K + 1)/2, starting V =",
      "Omega%s"), best, if (default_moves) "" else
        "; note: V does not move under the default prior either (see the control runs)")
      else if (ran) "V cannot be held fixed by the matrix-GIG settings tried (P12 does not hold)" else "(undecided)",
    evidence = paste(txt, collapse = "; "),
    fixv = if (length(best)) settings[[best]] else NULL, Omega = Om, holds = length(best) > 0L)
}

wb_probe_p16 <- function() {
  d <- wb_read("probe_t2n5.csv", P2_DATA)
  out <- list()
  for (v in seq_along(WB_VOLS)) {
    vol <- WB_VOLS[v]
    runs <- lapply(seq_len(WB_RUNS_N), function(k) {
      wb_chain(d$Y, 1L, vol, WB_P16_BURN, WB_P16_CALLS, 1L, WB_SEED + 1600L + 100L * v + k, tiny = TRUE)
    })
    message(sprintf("    P16 %s: %.4fs per chained call", vol, mean(vapply(runs, `[[`, 1, "seconds_per_call"))))
    ref <- lapply(seq_len(WB_RUNS_N), function(k) {
      r <- wb_eval(wb_run_code(vol, 1L, WB_P16_BURN, WB_P16_CALLS, tiny = TRUE), list(data = d$Y),
                   WB_SEED + 1650L + 100L * v + k)
      wb_post(r$env$post)
    })
    # extraction check: A and Sigma quantiles, chained vs single runs
    chA <- lapply(runs, function(r) stack_posts(r$posts, "A"))
    chS <- lapply(runs, function(r) stack_posts(r$posts, "Sigma"))
    pairs <- list()
    N <- dim(chA[[1]])[1]
    Kk <- dim(chA[[1]])[2]
    for (i in seq_len(N)) for (j in seq_len(Kk)) pairs[[length(pairs) + 1L]] <- list(
      a = mc_summary(lapply(chA, function(a) a[i, j, ])), b = mc_summary(lapply(ref, function(p) p$A[i, j, ])))
    for (i in seq_len(N)) for (j in seq_len(i)) pairs[[length(pairs) + 1L]] <- list(
      a = mc_summary(lapply(chS, function(a) a[i, j, ])), b = mc_summary(lapply(ref, function(p) p$Sigma[i, j, ])))
    cmp <- parity_compare(pairs)
    states <- lapply(runs, `[[`, "states")
    out[[vol]] <- list(states = states, cmp = cmp,
                       seconds_per_call = mean(vapply(runs, `[[`, 1, "seconds_per_call")),
                       seeds = WB_SEED + 1600L + 100L * v + seq_len(WB_RUNS_N),
                       ref_seeds = WB_SEED + 1650L + 100L * v + seq_len(WB_RUNS_N),
                       prior = wb_prior(runs[[1]]$spec))
  }
  out
}

# Posterior CV of sigma2_omega / sigma2v under scaled sv_a, sv_s (P14: does a setting fix it?).
wb_fix_scan <- function() {
  d <- wb_read("probe_t2n5.csv", P2_DATA)
  out <- character()
  cvs <- numeric()
  for (vol in WB_VOLS) {
    pf <- wb_prior(wb_eval(wb_spec_code(vol, 1L, tiny = TRUE), list(data = d$Y), WB_SEED + 1399L)$env$spec)
    for (ms in c(1e-6, 1, 1e6)) {
      st <- tryCatch({
        code <- paste0(wb_spec_code(vol, 1L, tiny = TRUE),
                       "spec$prior$sv_a <- sv_a; spec$prior$sv_s <- sv_s; ",
                       "burn <- estimate(spec, S = 2000L, thin = 1L, show_progress = FALSE); post <- burn")
        r <- wb_eval(code, list(data = d$Y, sv_a = pf$sv_a * 1e6, sv_s = pf$sv_s * ms), WB_SEED + 1400L)
        post <- r$env$post
        s <- vector("list", if (QUICK) 30L else 200L)
        for (k in seq_along(s)) {
          post <- wb_est(post, 10L)
          s[[k]] <- wb_state(post)
        }
        cv <- function(f) {
          x <- as.numeric(state_matrix(s, f))
          if (!length(x)) NA_real_ else stats::sd(x) / abs(mean(x))
        }
        cvs <- c(cvs, min(cv("sigma2_omega"), cv("sigma2v"), na.rm = TRUE))
        sprintf("%s, sv_a x 1e6, sv_s x %g: CV(sigma2_omega) %.3g, CV(sigma2v) %.3g, CV(omega) %.3g, mean sigma2_omega %.4g",
                vol, ms, cv("sigma2_omega"), cv("sigma2v"), cv("omega"),
                mean(as.numeric(state_matrix(s, "sigma2_omega")) %||% NA))
      }, error = function(e) sprintf("%s, sv_s x %g: failed (%s)", vol, ms, conditionMessage(e)))
      out <- c(out, st)
    }
  }
  list(text = out, fixed = any(is.finite(cvs) & cvs <= 1e-3))
}

# The scale experiment of P13 (b): A and Sigma/100 equivariant when the data are multiplied by 10.
wb_scale_experiment <- function(fixv, rmq) {
  if (is.null(fixv)) return(list(text = "not run: V cannot be held fixed (P12)", pass = NA))
  q <- wb_read("qd_small_levels.csv")
  K <- 1L + 3L * WB_LAGS
  cfac <- 10
  dvec <- rep(1 / cfac, K)
  dvec[rmq$ours_cols[1]] <- 1                          # the constant (wb order)
  fx2 <- fixv
  fx2$Omega <- fixv$Omega * outer(dvec, dvec)
  Om2i <- solve(fx2$Omega)
  fx2$Psi <- if (fixv$orient == "Psi = c Omega") fixv$c * fx2$Omega else fixv$c * Om2i
  fx2$Gamma <- if (fixv$orient == "Psi = c Omega") fixv$c * Om2i else fixv$c * fx2$Omega
  txt <- character()
  pass <- logical()
  for (vol in WB_VOLS) {
    S0 <- wb_prior(wb_spec(q$Y, WB_LAGS, vol))$S
    draws <- if (QUICK) 1000L else 10000L
    burn <- if (QUICK) 500L else 2000L
    fit <- function(Y, fv, S0x, seed0) lapply(seq_len(WB_RUNS_N), function(k) {
      code <- paste0(sprintf("spec <- specify_bvar$new(data = data, p = %dL, common_volatility = \"%s\", distribution = \"norm\"); ",
                             WB_LAGS, vol), wb_fixv_code(fv), "spec$prior$S <- S0x; ",
                     sprintf("burn <- estimate(spec, S = %dL, thin = 1L, show_progress = FALSE); ", burn),
                     sprintf("post <- estimate(burn, S = %dL, thin = 1L, show_progress = FALSE)", draws))
      wb_post(wb_eval(code, list(data = Y, fixv = fv, S0x = S0x), seed0 + k)$env$post)
    })
    base <- fit(q$Y, fixv, S0, WB_SEED + 1300L)
    scl <- fit(q$Y * cfac, fx2, S0 * cfac^2, WB_SEED + 1350L)
    pairs <- list()
    cst <- rmq$ours_cols[1]
    for (i in 1:3) for (j in seq_len(K)) pairs[[length(pairs) + 1L]] <- list(
      a = mc_summary(lapply(base, function(p) p$A[i, j, ])),
      b = mc_summary(lapply(scl, function(p) p$A[i, j, ] / (if (j == cst) cfac else 1))))
    for (i in 1:3) for (j in seq_len(i)) pairs[[length(pairs) + 1L]] <- list(
      a = mc_summary(lapply(base, function(p) p$Sigma[i, j, ])),
      b = mc_summary(lapply(scl, function(p) p$Sigma[i, j, ] / cfac^2)))
    cmp <- parity_compare(pairs)
    pass <- c(pass, cmp$pass)
    txt <- c(txt, sprintf("%s: max |z| %.2f over m = %d (bound %.2f)", vol, cmp$max_abs_z, cmp$m, cmp$bound))
  }
  list(text = paste(txt, collapse = "; "), pass = all(pass))
}

run_wbvars_probes <- function() {
  pr <- list()
  pr$P11 <- attempt_probe("P11", wb_probe_p11())
  pr$P15 <- attempt_probe("P15", wb_probe_p15())
  pr$P12 <- attempt_probe("P12", wb_probe_p12())
  p16 <- tryCatch(wb_probe_p16(), error = function(e) e)
  scan <- tryCatch(wb_fix_scan(), error = function(e) list(text = paste("scan failed:", conditionMessage(e)),
                                                           fixed = FALSE))
  rmq <- pr$P15$rmq
  scale <- if (!is.null(rmq)) tryCatch(wb_scale_experiment(pr$P12$fixv, rmq),
                                       error = function(e) list(text = paste("failed:", conditionMessage(e)), pass = NA))
    else list(text = "not run: P15 undecided", pass = NA)
  # P13 / P14 from the chained draws
  pit <- list()
  hlen <- NA
  if (!inherits(p16, "error")) for (vol in WB_VOLS) {
    st <- p16[[vol]]$states
    hlen <- length(st[[1]][[1]]$h)
    pit[[vol]] <- list(
      rho = wb_pit_rho(st, vol),
      s2o = if (!is.null(p16[[vol]]$prior$sv_a)) wb_pit_sigma2_omega(st, p16[[vol]]$prior$sv_a) else list(),
      rep_rho = repeat_fraction(st, "rho"), rep_omega = repeat_fraction(st, "omega"),
      rep_h = repeat_fraction(st, "h"), rep_s2v = repeat_fraction(st, "sigma2v"),
      rep_s2o = repeat_fraction(st, "sigma2_omega"))
  }
  pass_any <- function(lst, pat) {
    v <- unlist(lst[grepl(pat, names(lst), fixed = TRUE)])
    any(is.finite(v) & v > 1e-3)
  }
  init <- vapply(WB_VOLS, function(vol) {
    if (is.null(pit[[vol]])) return("undecided")
    a <- pass_any(pit[[vol]]$rho, "h0_zero")
    b <- pass_any(pit[[vol]]$rho, "stationary")
    if (isTRUE(pit[[vol]]$rep_rho > 0.01)) "undecided (rho has Metropolis repeats)" else
      if (a && !b) "h_0 = 0" else if (b && !a) "stationary" else "undecided"
  }, character(1))
  pit_txt <- vapply(names(pit), function(vol) sprintf(
    "%s: rho PIT (U(-1, 1) prior) %s; sigma2_omega PIT %s; repeat fractions rho %.3g, omega %.3g, h %.3g, sigma2v %.3g, sigma2_omega %.3g",
    vol, fmt_p(pit[[vol]]$rho), fmt_p(pit[[vol]]$s2o), pit[[vol]]$rep_rho, pit[[vol]]$rep_omega,
    pit[[vol]]$rep_h, pit[[vol]]$rep_s2v, pit[[vol]]$rep_s2o), character(1))
  p13_decided <- all(init %in% c("h_0 = 0", "stationary")) && isTRUE(scale$pass)
  pr$P13 <- list(
    status = if (p13_decided) "answered" else "unresolved",
    answer = if (p13_decided) sprintf(
      "h has %d entries (t = 1..T, no h_0 state); the volatility level is carried by Sigma (E h = 0); initial condition: cSV %s, ncSV %s",
      hlen, init[["cSV"]], init[["ncSV"]]) else sprintf(
        "(undecided) h has %s entries; scale experiment pass = %s; initial condition: cSV %s, ncSV %s",
        hlen, scale$pass, init[["cSV"]], init[["ncSV"]]),
    evidence = paste(c(paste("scale experiment:", scale$text), pit_txt), collapse = " || "))
  rho_unif <- all(init %in% c("h_0 = 0", "stationary"))
  s2o_pass <- vapply(names(pit), function(vol) {
    v <- unlist(pit[[vol]]$s2o)
    if (!length(v)) NA_character_ else if (any(is.finite(v) & v > 1e-3)) names(v)[which.max(v)] else "none"
  }, character(1))
  pr$P14 <- list(
    status = if (rho_unif && all(!is.na(s2o_pass))) "answered" else "unresolved",
    answer = if (rho_unif) sprintf(
      "rho ~ U(-1, 1) (PIT passes); sigma_omega^2 conditional: %s; a setting that fixes sigma_omega^2: %s",
      paste(sprintf("%s %s", names(s2o_pass), s2o_pass), collapse = ", "),
      if (isTRUE(scan$fixed)) "found (see evidence)" else "none among those tried")
      else "(undecided: the rho prior is not identified by the PIT tests)",
    evidence = paste(c(pit_txt, paste("fixing scan:", paste(scan$text, collapse = "; "))), collapse = " || "))
  pr$P16 <- if (inherits(p16, "error")) list(status = "unresolved", answer = "(extraction failed)",
                                            evidence = conditionMessage(p16)) else list(
    status = "pending",
    answer = "data committed (probe_wbvars_p16.nc), decision pending WP38 (quadrature oracle)",
    evidence = paste(vapply(WB_VOLS, function(vol) sprintf(
      "%s: %d runs x %d chained draws; extraction check max |z| %.2f over m = %d (bound %.2f, pass %s); %.4fs per call",
      vol, WB_RUNS_N, WB_P16_CALLS, p16[[vol]]$cmp$max_abs_z, p16[[vol]]$cmp$m, p16[[vol]]$cmp$bound,
      p16[[vol]]$cmp$pass, p16[[vol]]$seconds_per_call), character(1)), collapse = "; "))
  pr <- pr[names(PROBES_WB)]
  for (id in names(pr)) pr[[id]] <- c(PROBES_WB[[id]], pr[[id]])
  list(probes = pr, p16 = p16, init = init, scale = scale, pit = pit, scan = scan)
}

wb_gating <- function(res) {
  pr <- res$probes
  p12 <- isTRUE(pr$P12$holds)
  maps <- identical(pr$P13$status, "answered") && identical(pr$P14$status, "answered") &&
    all(res$init == "stationary") && grepl("fixes sigma_omega\\^2: found", pr$P14$answer)
  if (p12 && maps) paste(
    "option 1 (P12 holds and P13-P14 map the SV prior): posterior quantiles of B and Sigma under",
    "the Phase 2 rule; h_t, rho, sigma_h gate only if P16 shows the sampler exact (pending WP38);",
    "until then they are reported, not gating, and the 1- and 4-step predictive quantiles join",
    "the gate") else paste(
      "option 2: the 1- and 4-step predictive quantiles under the Phase 2 rule, same data and",
      "draw counts, because", if (!p12) "V cannot be held fixed (P12)" else
        "P13-P14 do not map bvars' SV prior onto a CommonSV option (initial condition and/or the",
      if (p12) "hierarchical sigma_omega^2 prior; see P13, P14)" else "")
}

stage_wbvars_p16 <- function(res) {
  p16 <- res$p16
  vars <- list()
  coords <- list(run = nc_array(seq_len(WB_RUNS_N), "run"), iter = nc_array(seq_len(WB_P16_CALLS), "iter"))
  attrs <- p3_meta("bvars", "probe_wbvars_p16")
  d <- wb_read("probe_t2n5.csv", P2_DATA)
  attrs$data_file <- d$rel
  attrs$data_md5 <- d$md5
  attrs$seed <- WB_SEED + 1600L
  attrs$lags <- 1L
  attrs$burnin <- WB_P16_BURN
  attrs$chained_calls <- WB_P16_CALLS
  attrs$chain_note <- paste(
    "each run: estimate(spec, S = burnin) then chained estimate(post, S = 1) calls; after each",
    "call the last_draw starting values h, rho, omega, sigma2v, sigma2_omega, s_ and S are",
    "recorded (iteration = call). Single-run references (ref_seeds) check the extraction.")
  for (vol in WB_VOLS) {
    st <- p16[[vol]]$states
    for (f in WB_STATE_FIELDS) {
      m <- lapply(st, function(s) state_matrix(s, f))
      if (any(vapply(m, is.null, logical(1)))) next
      w <- ncol(m[[1]])
      arr <- array(NA_real_, c(WB_RUNS_N, WB_P16_CALLS, w))
      for (k in seq_len(WB_RUNS_N)) arr[k, , ] <- m[[k]]
      if (f == "S") storage.mode(arr) <- "integer"
      dims <- c("run", "iter", paste0("n", w))
      vars[[sprintf("p16_%s_%s", vol, f)]] <- nc_array(if (w == 1L) arr[, , 1] else arr,
                                                       if (w == 1L) dims[1:2] else dims,
                                                       sprintf("%s draws of last_draw$%s (%s)", vol, f, vol))
    }
    attrs[[paste0(vol, "_seeds")]] <- I(p16[[vol]]$seeds)
    attrs[[paste0(vol, "_ref_seeds")]] <- I(p16[[vol]]$ref_seeds)
    attrs[[paste0(vol, "_extraction_max_abs_z")]] <- p16[[vol]]$cmp$max_abs_z
    attrs[[paste0(vol, "_extraction_m")]] <- p16[[vol]]$cmp$m
    attrs[[paste0(vol, "_extraction_pass")]] <- as.integer(p16[[vol]]$cmp$pass)
    attrs[[paste0(vol, "_seconds_per_call")]] <- p16[[vol]]$seconds_per_call
    pf <- p16[[vol]]$prior
    for (nm in names(pf)) if (is.numeric(pf[[nm]])) {
      vars[[sprintf("p16_%s_prior_%s", vol, nm)]] <- nc_auto(pf[[nm]], sprintf("bvars default prior %s (%s, probe_t2n5)", nm, vol))
    }
  }
  attrs$pit_tests <- paste(vapply(names(res$pit), function(vol) sprintf("%s rho: %s; sigma2_omega: %s", vol,
                                                                        fmt_p(res$pit[[vol]]$rho), fmt_p(res$pit[[vol]]$s2o)),
                                  character(1)), collapse = "\n")
  stage("probe_wbvars_p16", attrs, coords, vars)
}

wb_parity_fixture <- function(vol, fixv, rmq, probe_answers, gating, idx) {
  q <- wb_read("qd_small_levels.csv")
  Y <- q$Y
  N <- ncol(Y)
  p <- WB_LAGS
  K <- 1L + N * p
  Tn <- nrow(Y) - p
  seed <- WB_SEED + 2000L + 100L * idx
  code <- wb_run_code(vol, p, WB_BURN, WB_DRAWS, fixv)
  runs <- lapply(seq_len(WB_RUNS_N), function(k) {
    r <- wb_eval(code, list(data = Y, fixv = fixv), seed + k)
    rm <- wb_row_map(r$env$spec, Y, p)
    if (!identical(rm$ours_cols, rmq$ours_cols)) stop("unexpected X row order")
    po <- wb_post(r$env$post)
    set.seed(seed + 500L + k)
    fc <- unclass(forecast(r$env$post, horizon = WB_HORIZON))
    F <- draws_last(fc$forecasts, WB_DRAWS)
    if (dim(F)[1] != N && dim(F)[2] == N) F <- aperm(F, c(2, 1, 3))   # (N, H, S)
    message(sprintf("    wbvars_%s run %d: %.1fs", tolower(vol), k, r$seconds))
    list(B = wb_B(po$A, rmq), Sigma = aperm(po$Sigma, c(3, 1, 2)), fc = aperm(F, c(3, 2, 1)),
         Vmean = apply(po$V, c(1, 2), mean), Vsd = apply(po$V, c(1, 2), stats::sd),
         seconds = r$seconds, prior = wb_prior(r$env$spec))
  })
  # chained state runs: h, rho, omega, ... every WB_CHAIN_S draws
  sruns <- lapply(seq_len(WB_RUNS_N), function(k) {
    wb_chain(Y, p, vol, WB_BURN, WB_CHAIN_CALLS, WB_CHAIN_S, seed + 50L + k, fixv)
  })
  sigma_pairs <- which(lower.tri(diag(N), diag = TRUE), arr.ind = TRUE)
  sigma_pairs <- sigma_pairs[order(sigma_pairs[, 1], sigma_pairs[, 2]), , drop = FALSE]
  chains_B <- function(ix) lapply(runs, function(r) r$B[, ix[1], ix[2]])
  blocks <- list(
    summary_vars("B", chains_B, c(K, N), c("regressor", "equation"),
                 "B = t(A)[regressor, equation] (y_t = B'x_t + u_t), rai-bvar regressor order"),
    summary_vars("Sigma", function(ix) lapply(runs, function(r) r$Sigma[, sigma_pairs[ix[1], 1], sigma_pairs[ix[1], 2]]),
                 nrow(sigma_pairs), "sigma_pair", "Sigma[i, j], i >= j (row-major), the covariance at h = 0"),
    summary_vars("forecast", function(ix) lapply(runs, function(r) r$fc[, ix[1], ix[2]]), c(WB_HORIZON, N),
                 c("horizon", "equation"), "predictive y_{T+s} (forecast(post, horizon = 4))"))
  for (b in blocks) check_mixing_p3(paste0("wbvars_", vol), b)
  # state blocks from the chained runs
  t_sel <- as.integer(c(1L, Tn %/% 4L, Tn %/% 2L, (3L * Tn) %/% 4L, Tn))
  sblocks <- list()
  scoords <- list()
  H <- lapply(sruns, function(r) state_matrix(r$states, "h"))
  if (!any(vapply(H, is.null, logical(1)))) {
    sblocks$h <- summary_vars("h", function(ix) lapply(H, function(m) m[, t_sel[ix[1]]]), length(t_sel), "t_sel",
                              sprintf("common log-volatility h_t at t_sel (last_draw after every %d draws)", WB_CHAIN_S))
    scoords$t_sel <- nc_array(t_sel, "t_sel")
  }
  sc <- intersect(c("rho", "omega", "sigma2v", "sigma2_omega", "s_"), names(sruns[[1]]$states[[1]]))
  sc <- sc[vapply(sc, function(f) length(sruns[[1]]$states[[1]][[f]]) == 1L, logical(1))]
  if (length(sc)) {
    sblocks$state <- summary_vars("state", function(ix) lapply(sruns, function(r) as.numeric(state_matrix(r$states, sc[ix[1]]))),
                                  length(sc), "state_param", sprintf("SV parameters (last_draw after every %d draws)", WB_CHAIN_S))
    scoords$state_param <- nc_array(sc, "state_param")
  }
  for (b in sblocks) check_mixing_p3(paste0("wbvars_", vol, " (states)"), b, hard = FALSE)
  # extraction check on the parity data: B, Sigma from the chained runs vs the main runs
  cB <- lapply(sruns, function(r) wb_B(stack_posts(r$posts, "A"), rmq))
  cS <- lapply(sruns, function(r) aperm(stack_posts(r$posts, "Sigma"), c(3, 1, 2)))
  cmp_pairs <- list()
  for (k in seq_len(K)) for (i in seq_len(N)) cmp_pairs[[length(cmp_pairs) + 1L]] <- list(
    a = mc_summary(lapply(runs, function(r) r$B[, k, i])), b = mc_summary(lapply(cB, function(b) b[, k, i])))
  for (s in seq_len(nrow(sigma_pairs))) cmp_pairs[[length(cmp_pairs) + 1L]] <- list(
    a = mc_summary(lapply(runs, function(r) r$Sigma[, sigma_pairs[s, 1], sigma_pairs[s, 2]])),
    b = mc_summary(lapply(cS, function(x) x[, sigma_pairs[s, 1], sigma_pairs[s, 2]])))
  cmp <- parity_compare(cmp_pairs)
  pf <- runs[[1]]$prior
  coords <- c(list(
    quantile = nc_array(P2_PROBS, "quantile"),
    regressor = nc_array(swa_regressor_names(q$variables, p), "regressor"),
    regressor2 = nc_array(swa_regressor_names(q$variables, p), "regressor2"),
    equation = nc_array(q$variables, "equation"),
    equation2 = nc_array(q$variables, "equation2"),
    sigma_pair = nc_array(sprintf("s%d%d", sigma_pairs[, 1], sigma_pairs[, 2]), "sigma_pair"),
    horizon = nc_array(seq_len(WB_HORIZON), "horizon")), scoords)
  oc <- rmq$ours_cols
  vars <- c(unlist(lapply(c(blocks, sblocks), `[[`, "vars"), recursive = FALSE), list(
    prior_B0 = nc_array(t(pf$A)[oc, , drop = FALSE], c("regressor", "equation"), "prior mean B0 = t(A0), rai-bvar order"),
    prior_S0 = nc_array(pf$S, c("equation", "equation2"), "inverse-Wishart scale S0 of Sigma"),
    V_mean = nc_array(Reduce(`+`, lapply(runs, `[[`, "Vmean"))[oc, oc] / WB_RUNS_N, c("regressor", "regressor2"),
                      "posterior mean of V (rai-bvar order)"),
    V_sd = nc_array(sqrt(Reduce(`+`, lapply(runs, function(r) r$Vsd^2))[oc, oc] / WB_RUNS_N), c("regressor", "regressor2"),
                    "posterior s.d. of V (rai-bvar order; pooled within-run)")))
  if (!is.null(fixv)) vars$V_target <- nc_array(fixv$Omega[oc, oc], c("regressor", "regressor2"),
                                                "Omega: the value at which V is held (P12)")
  attrs <- c(p3_meta("bvars", "wbvars"), list(
    vol = vol, lags = p, n = N, T = Tn, k = K, data_file = q$rel, data_md5 = q$md5,
    data_transform = "none (FRED-QD small set as committed: 100 x log GDPC1, 100 x log GDPCTPI, FEDFUNDS)",
    seed = seed, chain_seeds = I(seed + seq_len(WB_RUNS_N)), forecast_seeds = I(seed + 500L + seq_len(WB_RUNS_N)),
    state_seeds = I(seed + 50L + seq_len(WB_RUNS_N)), draws = WB_DRAWS, burnin = WB_BURN, chains = WB_RUNS_N,
    runs = "independent runs: a burn-in estimate() then an estimate() restarted from its last_draw",
    call = code, forecast_call = sprintf("forecast(post, horizon = %dL)", WB_HORIZON),
    state_runs = sprintf("%d runs x %d chained estimate(post, S = %d) calls after the burn-in", WB_RUNS_N,
                         WB_CHAIN_CALLS, WB_CHAIN_S),
    v_fixed = as.integer(!is.null(fixv)),
    v_fix_setting = if (!is.null(fixv)) sprintf("%s, c = %g, lambda = %g", fixv$orient, fixv$c, fixv$lambda) else "default (V sampled)",
    prior_nu0 = pf$nu %||% NA_real_, prior_sv_a = pf$sv_a %||% NA_real_, prior_sv_s = pf$sv_s %||% NA_real_,
    prior_lambda = pf$lambda %||% NA_real_,
    extraction_max_abs_z = cmp$max_abs_z, extraction_m = cmp$m, extraction_pass = as.integer(cmp$pass),
    state_seconds_per_call = mean(vapply(sruns, `[[`, 1, "seconds_per_call")),
    probe_answers = probe_answers, gating = gating,
    rai_bvar_spec = paste(
      "BVAR(Linear(Minnesota(form='conjugate') with explicit arrays B0 = prior_B0, Omega =",
      "V_target (V held fixed), S0 = prior_S0, nu0 = prior_nu0), CommonSV(<mapped by P13-P14>))"),
    layout = "regressor order const, lag-1 block, ..., lag-p block; Sigma pairs (i, j), i >= j, row-major; t_sel = {1, T/4, T/2, 3T/4, T}"))
  mix <- mixing_summary(blocks)
  attrs$mixing_max_rhat <- mix$max_rhat
  attrs$mixing_min_ess <- mix$min_ess
  attrs$mixing_ok <- mix$ok
  stage(sprintf("wbvars_%s", tolower(vol)), attrs, coords, vars)
}

stage_wbvars_probes <- function(res, gating, runs_txt) {
  pr <- res$probes
  attrs <- p3_meta("bvars", "probe_wbvars")
  attrs$bvars_version <- as.character(packageVersion("bvars"))
  attrs$bsvars_version <- as.character(packageVersion("bsvars"))
  attrs$gating_comparison <- gating
  attrs$runs <- paste(runs_txt, collapse = "\n")
  attrs$p13_scale_experiment <- res$scale$text
  attrs$p14_fix_scan <- paste(res$scan$text, collapse = "\n")
  attrs$data_choice <- paste(
    "FRED-QD small set: tests/parity/fixtures/qd_small_levels.csv (the Phase 1 parity data, n = 3,",
    "GDPC1, GDPCTPI, FEDFUNDS, 1960Q1-2019Q4), p = 4, as committed")
  vars <- pr$P11$vars %||% list()
  stage_probe_fixture("probe_wbvars", attrs, pr, names(PROBES_WB), vars)
}

run_wbvars <- function() {
  res <- run_wbvars_probes()
  if (!is.null(res$probes$P11$describe)) record_layout("bvars_fields.txt", res$probes$P11$describe)
  gating <- wb_gating(res)
  message("  gating comparison: ", gating)
  answers <- probe_answers_text(res$probes)
  message(answers)
  runs_txt <- character()
  rmq <- res$probes$P15$rmq
  for (i in seq_along(WB_VOLS)) {
    name <- sprintf("wbvars_%s", tolower(WB_VOLS[i]))
    if (!want(name)) next
    if (is.null(rmq)) {
      runs_txt <- c(runs_txt, sprintf("%s: not run, P15 undecided", name))
      next
    }
    r <- tryCatch({
      wb_parity_fixture(WB_VOLS[i], res$probes$P12$fixv, rmq, answers, gating, i)
      "staged"
    }, error = function(e) paste("failed:", conditionMessage(e)))
    runs_txt <- c(runs_txt, sprintf("%s: %s", name, r))
    message("  ", runs_txt[length(runs_txt)])
  }
  if (want("probe_wbvars_p16") && !inherits(res$p16, "error")) stage_wbvars_p16(res)
  if (want("probe_wbvars")) stage_wbvars_probes(res, gating, runs_txt)
  invisible(res)
}

if ("wbvars" %in% SECTIONS) {
  message("(h) bvars (Wozniak) probes P11-P16 and the CommonSV parity runs")
  if (!requireNamespace("bvars", quietly = TRUE)) stop("section wbvars needs the R package bvars")
  suppressPackageStartupMessages(library(bvars))
  message("  bvars ", packageVersion("bvars"), ", bsvars ", packageVersion("bsvars"))
  quiet_calls(run_wbvars())
}

# ------------------------------------------------------------------------------------------------

failed <- Filter(function(x) !x$ok && x$hard, checks)
soft <- Filter(function(x) !x$ok && !x$hard, checks)
message(sprintf("%d consistency checks, %d failed, %d soft warnings", length(checks),
                length(failed), length(soft)))
for (x in failed) message("  FAILED: ", x$check, " = ", format(x$value, digits = 3))
for (x in soft) message("  warning (not fatal): ", x$check, " = ", format(x$value, digits = 3))
if ("bvar" %in% SECTIONS) {
  worst <- function(pattern) {
    v <- vapply(Filter(function(x) grepl(pattern, x$check), checks), `[[`, numeric(1), "value")
    if (length(v)) max(v) else NA
  }
  message(sprintf("max rel diff log ML BVAR vs exact (where exact): %.2e",
                  worst("rel diff log ML bvar vs exact")))
  message(sprintf("max scaled diff posterior mean BVAR vs exact: %.2e",
                  worst("posterior mean bvar vs exact")))
}
if (length(failed)) quit(status = 1)
