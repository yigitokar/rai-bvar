#!/usr/bin/env Rscript
# =================================================================================================
# make_r_fixtures.R: numerical-oracle fixtures from R packages (BVAR; stochvol; bayesianVARs)
# =================================================================================================
#
# Three sections, selected with --sections (default: all three). Not run in CI. Reproducible:
# every random number comes from a fixed seed.
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
#   bvars     (f) bayesianVARs parity runs and the convention probes P1–P5 when bayesianVARs is
#                 installed; otherwise the probes are recorded as unresolved, with the evidence,
#                 and the bayesianVARs runs are replaced by an exact system-wide Gibbs oracle
#                 (N04 §9.5) written here, with stochvol's single-step SV update
#                 -> tests/parity/fixtures/probe_bayesianvars.nc, bvars_swa_*.nc
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
#   python -I tools/make_sv_parity_data.py all            # data, the Phase 2 sections, pack
#
# Requirements: R >= 4.1 with BVAR 1.0.5, mvtnorm, jsonlite and coda (bvar section) and stochvol
# (stochvol and bvars sections), e.g. in the user library:
#   lib <- Sys.getenv("R_LIBS_USER"); dir.create(lib, recursive = TRUE, showWarnings = FALSE)
#   install.packages(c("BVAR", "jsonlite", "coda", "stochvol"), lib = lib,
#                    repos = "https://cloud.r-project.org")
# (On macOS without gfortran, install the CRAN *binary* of mvtnorm first. bayesianVARs has no
# CRAN binary for R 4.1 and does not build without gfortran; see section (f).)
#
# Licensing (SPEC §2.1): BVAR and stochvol are GPL (BVAR GPL-3, stochvol GPL >= 2), bayesianVARs is
# GPL-3. This script *runs* them, including a few internal BVAR functions through `:::`, as
# numerical oracles; their outputs are data. Nothing in this file is part of the rai-bvar library,
# and no library code may be derived from their sources.
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

args <- commandArgs(trailingOnly = TRUE)
RUN_MH <- !("--no-mh" %in% args)
QUICK <- "--quick" %in% args
arg_value <- function(name, default) {
  hit <- grep(paste0("^--", name, "="), args, value = TRUE)
  if (length(hit)) sub(paste0("^--", name, "="), "", hit[length(hit)]) else default
}
ALL_SECTIONS <- c("bvar", "stochvol", "bvars")
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
  stage(name, attrs, coords, vars)
}

if ("stochvol" %in% SECTIONS) {
  message("(e) stochvol (", SV_CHAINS, " chains x ", SV_DRAWS, " draws)")
  if (as.character(packageVersion("stochvol")) != "3.2.9") {
    message("  note: DESIGN sec. 9.9 names stochvol 3.2.9; using ", packageVersion("stochvol"))
  }
  direct_check <- sv_direct_call_check()
  SV_ONLY <- arg_value("sv-only", "")   # e.g. --sv-only=stochvol_i_exact,stochvol_iv_exact
  SV_ONLY <- if (nzchar(SV_ONLY)) strsplit(SV_ONLY, ",", fixed = TRUE)[[1]] else character()
  i <- 0L
  for (fx in SV_FIXTURES) for (run_name in fx$runs) {
    if (!length(SV_ONLY) || sprintf("stochvol_%s_%s", fx$data, run_name) %in% SV_ONLY) {
      run_stochvol_fixture(fx$data, run_name, SV_SEED0 + i, direct_check)
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

PROBES <- list(
  P1 = list(
    question = paste(
      "Are specify_prior_phi(normal_sds) and specify_prior_sigma(cholesky_normal_sds) prior",
      "variances (as documented) or standard deviations (as named)?"),
    documented = paste(
      "bayesianVARs 0.1.8 manual: normal_sds are 'prior variances' (length lags*M^2, intercept",
      "excluded); cholesky_normal_sds 'prior variances for the free off-diagonal elements in U'",
      "(N04 sec. 3.1-3.2, sec. 10 item 4)"),
    dataset = "data/probe_tiny.csv (n = 2, p = 1, T_eff = 4)",
    design = paste(
      "prior-dominated fit: normal_sds = cholesky_normal_sds = 1e-4, prior_intercept = 1e3;",
      "posterior variance of each lag coefficient and of u_12 is ~1e-4 if the argument is a",
      "variance and ~1e-8 if it is a standard deviation")),
  P2 = list(
    question = "Is U stored as U or U^-1, and with which sign (A = t(U) or A = 2I - t(U))?",
    documented = paste(
      "Sigma_t = U'^-1 D_t U^-1 with U upper unitriangular (manual; GK eq. 2), hence A = U';",
      "GK App. B prints the U step without the minus sign, which would be the conditional of -u",
      "(N04 sec. 3.4)"),
    dataset = "data/probe_corr.csv (n = 2, p = 1, T = 600, corr(u1, u2) = +0.8)",
    design = paste(
      "map every draw with A = t(U), Sigma = A^-1 D A^-T; posterior mean correlation of u1, u2",
      "is ~ +0.8 if A = t(U), ~ -0.8 if the sign is flipped; otherwise U^-1 is stored")),
  P3 = list(
    question = "Which row of the returned PHI draws holds the intercept?",
    documented = "undocumented (N04 sec. 3.4, sec. 10 item 4)",
    dataset = "data/probe_corr.csv (intercepts 5 and -3, own lag 0.3)",
    design = paste(
      "the PHI row whose posterior mean is ~ (5, -3) is the intercept row; the lag rows are",
      "~ diag(0.3)")),
  P4 = list(
    question = "What is the layout of the returned sv_para draws?",
    documented = "undocumented (N04 sec. 9.8, sec. 10 item 4)",
    dataset = paste(
      "data/probe_sv.csv (n = 3, p = 1, T = 400, mu = (-3, 0, 2), phi = (0.8, 0.9, 0.97),",
      "sigma = (0.4, 0.3, 0.15))"),
    design = paste("match the posterior means of each sv_para slice to the distinct true",
                   "(mu, phi, sigma)")),
  P5 = list(
    question = paste(
      "Is cholesky_priorhomoscedastic = cbind(shape, scale) the inverse-gamma scale beta0 of",
      "p(lambda) ~ lambda^(-alpha0-1) exp(-beta0/lambda)? What is used when it is NA?"),
    documented = paste(
      "manual: an M x 2 matrix of (shape, scale), default NA with an undocumented substitute;",
      "the factor-model twin argument is documented as (shape, rate) (N04 sec. 3.3)"),
    dataset = "data/probe_tiny.csv (n = 2, p = 1, T_eff = 4)",
    design = paste(
      "cholesky_heteroscedastic = FALSE, cholesky_priorhomoscedastic = cbind(50, 100):",
      "posterior mean of lambda ~ 2 (IG scale) or ~ 2e-4 (scale of a Gamma on 1/lambda)"))
)

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

# Executed only when bayesianVARs is installed (never run so far: review on first use). Every
# probe is wrapped in tryCatch; the str() of each fit is kept as layout evidence either way.
run_bvars_probes <- function() {
  bv <- asNamespace("bayesianVARs")
  draws <- if (QUICK) 2000L else 20000L
  fit_probe <- function(file, lags, prior_phi_args, prior_sigma_args, intercept) {
    Y <- as.matrix(read_p2_data(file)$frame[, -1])
    set.seed(20261400L)
    phi <- do.call(bv$specify_prior_phi, c(list(data = Y, lags = lags), prior_phi_args))
    sig <- do.call(bv$specify_prior_sigma, c(list(data = Y, type = "cholesky"), prior_sigma_args))
    bv$bvar(Y, lags = lags, draws = draws, burnin = 2000L, thin = 1L, sv_keep = "all",
            prior_intercept = intercept, prior_phi = phi, prior_sigma = sig, quiet = TRUE)
  }
  layout <- function(fit) paste(capture.output(str(unclass(fit), max.level = 1, give.attr = FALSE)),
                                collapse = "\n")
  draws_axis <- function(x) which(dim(x) == draws)[1]
  post_moment <- function(x, f) apply(x, setdiff(seq_along(dim(x)), draws_axis(x)), f)
  res <- list()
  attempt <- function(id, expr) {
    res[[id]] <<- tryCatch(expr, error = function(e) list(
      status = "unresolved", answer = "(probe failed)",
      evidence = paste("probe failed:", conditionMessage(e))))
  }
  homo_prior <- list(cholesky_U_prior = "normal", cholesky_normal_sds = 1,
                     cholesky_heteroscedastic = FALSE,
                     cholesky_priorhomoscedastic = cbind(rep(3, 2), rep(2, 2)))
  attempt("P1", {
    fit <- fit_probe("probe_tiny.csv", 1L,
                     list(prior = "normal", normal_sds = rep(1e-4, 4), priormean = 0),
                     modifyList(homo_prior, list(cholesky_normal_sds = 1e-4)), 1e3)
    v <- post_moment(fit$PHI, var)
    vu <- post_moment(fit$U, var)
    lag_v <- median(sort(as.vector(v))[1:4])
    answer <- if (abs(log10(lag_v) + 4) < 1) "variance" else if (abs(log10(lag_v) + 8) < 1)
      "standard deviation" else "undecided"
    list(status = if (answer == "undecided") "unresolved" else "answered", answer = answer,
         evidence = sprintf("posterior var of PHI entries: %s; of U entries: %s\n%s",
                            paste(signif(v, 3), collapse = ", "),
                            paste(signif(vu, 3), collapse = ", "), layout(fit)))
  })
  attempt("P2", {
    fit <- fit_probe("probe_corr.csv", 1L, list(prior = "normal", normal_sds = rep(1, 4)),
                     homo_prior, 100)
    ax <- draws_axis(fit$U)
    corr <- vapply(seq_len(draws), function(d) {
      U <- matrix(asplit(fit$U, ax)[[d]], 2)
      A <- t(U)
      Dd <- diag(as.vector(exp(asplit(fit$logvar, draws_axis(fit$logvar))[[d]][1, ])), 2)
      S <- solve(A, t(solve(A, Dd)))
      S[1, 2] / sqrt(S[1, 1] * S[2, 2])
    }, numeric(1))
    m <- mean(corr)
    answer <- if (abs(m - 0.8) < 0.1) "A = t(U)" else if (abs(m + 0.8) < 0.1)
      "A = 2I - t(U) (sign flipped)" else "U^-1 stored (or other)"
    list(status = "answered", answer = answer,
         evidence = sprintf("posterior mean corr(u1, u2) under A = t(U): %.3f\n%s", m,
                            layout(fit)))
  })
  attempt("P3", {
    fit <- fit_probe("probe_corr.csv", 1L, list(prior = "normal", normal_sds = rep(1, 4)),
                     homo_prior, 100)
    pm <- post_moment(fit$PHI, mean)
    row <- which.min(rowSums((pm - matrix(c(5, -3), nrow(pm), 2, byrow = TRUE))^2))
    list(status = "answered", answer = sprintf("intercept in row %d of %d", row, nrow(pm)),
         evidence = paste(capture.output(print(pm)), collapse = "\n"))
  })
  attempt("P4", {
    fit <- fit_probe("probe_sv.csv", 1L, list(prior = "normal", normal_sds = rep(1, 9)),
                     list(cholesky_U_prior = "normal", cholesky_normal_sds = 1,
                          cholesky_heteroscedastic = TRUE, cholesky_priormu = c(0, 100),
                          cholesky_priorphi = c(20, 1.5), cholesky_priorsigma2 = c(0.5, 0.5),
                          cholesky_priorh0 = "stationary"), 100)
    pm <- post_moment(fit$sv_para, mean)
    list(status = "unresolved", answer = "(decide by hand from the evidence)",
         evidence = paste(c("posterior means of sv_para (match to mu = (-3, 0, 2), phi = (0.8,",
                            "0.9, 0.97), sigma = (0.4, 0.3, 0.15) by hand):",
                            capture.output(print(pm)), layout(fit)), collapse = "\n"))
  })
  attempt("P5", {
    fit <- fit_probe("probe_tiny.csv", 1L, list(prior = "normal", normal_sds = rep(1, 4)),
                     modifyList(homo_prior, list(cholesky_priorhomoscedastic =
                                                   cbind(rep(50, 2), rep(100, 2)))), 100)
    lam <- post_moment(exp(fit$logvar), mean)
    m <- median(lam)
    answer <- if (abs(log10(m) - log10(2)) < 0.5) "IG scale" else
      if (abs(log10(m) - log10(2e-4)) < 0.5) "scale of a Gamma on 1/lambda" else "undecided"
    list(status = if (answer == "undecided") "unresolved" else "answered", answer = answer,
         evidence = sprintf("posterior mean of lambda: %s", paste(signif(lam, 3),
                                                                 collapse = ", ")))
  })
  res
}

stage_probes <- function(env) {
  ids <- names(PROBES)
  if (isTRUE(env$bayesianvars_available)) {
    results <- run_bvars_probes()
  } else {
    evidence <- paste(
      "bayesianVARs is not installed (requireNamespace() is FALSE); gfortran on PATH: '",
      env$gfortran_on_path, "'; R CMD config FC: ", env$r_config_fc, "; FLIBS: ",
      env$r_config_flibs, " (", env$flibs_dirs_exist, "). ", env$cran_binary_note,
      ". The probe was not run; the documented convention is unverified.", sep = "")
    results <- lapply(ids, function(id) list(status = "unresolved", answer = "(not run)",
                                             evidence = evidence))
    names(results) <- ids
  }
  attrs <- c(p2_meta("stochvol"), list(kind = "probe"), env)
  attrs$bayesianvars_version <- if (isTRUE(env$bayesianvars_available)) {
    as.character(packageVersion("bayesianVARs"))
  } else "not installed"
  attrs$consequence <- paste(
    "bayesianVARs parity (DESIGN sec. 9.9) is blocked; the reduced-form sampler is compared",
    "instead with the exact system-wide Gibbs oracle fixtures bvars_swa_*.nc, which use",
    "rai-bvar's own conventions (P1-P5 do not arise there)")
  attrs$parity_calls <- paste(bvars_parity_calls(), collapse = "\n")
  for (id in ids) {
    pr <- PROBES[[id]]
    rs <- results[[id]]
    attrs[[paste0(id, "_question")]] <- pr$question
    attrs[[paste0(id, "_documented")]] <- pr$documented
    attrs[[paste0(id, "_dataset")]] <- pr$dataset
    attrs[[paste0(id, "_design")]] <- pr$design
    attrs[[paste0(id, "_status")]] <- rs$status
    attrs[[paste0(id, "_answer")]] <- rs$answer
    attrs[[paste0(id, "_evidence")]] <- rs$evidence
  }
  status <- vapply(ids, function(id) as.integer(results[[id]]$status == "answered"), integer(1))
  stage("probe_bayesianvars", attrs,
        list(probe = nc_array(ids, "probe")),
        list(answered = nc_array(unname(status), "probe", "1 if answered, 0 if unresolved")))
}

if ("bvars" %in% SECTIONS) {
  message("(f) bayesianVARs probes / system-wide Gibbs oracle")
  env <- bvars_environment()
  message("  bayesianVARs available: ", env$bayesianvars_available)
  stage_probes(env)
  sc <- swa_self_checks()
  z <- sv_single_step_check()
  message(sprintf("  stochvol single-step vs svsample: max |z| = %.2f", z))
  for (fx in SWA_FIXTURES) run_swa_fixture(fx, sc, z)
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
