#!/usr/bin/env Rscript

# Gaussian mean-change comparison:
#   1. Section 3.1 ARL e-detector from the supplied PDF, with A = 1000.
#   2. Shekhar-Ramdas repeated-CS detector (arXiv:2309.09111), with alpha = 1/A.
#
# Model: X_1,...,X_T ~ N(0, sigma^2), X_{T+1},... ~ N(delta, sigma^2).
# The reported delay is conditional on tau > T. The script also reports false
# alarms before/at T and post-change censoring at max_time.

parse_args <- function(defaults) {
  args <- commandArgs(trailingOnly = TRUE)
  opts <- defaults
  for (arg in args) {
    if (!startsWith(arg, "--")) next
    kv <- strsplit(sub("^--", "", arg), "=", fixed = TRUE)[[1]]
    if (length(kv) != 2 || !(kv[1] %in% names(opts))) next
    key <- kv[1]
    value <- kv[2]
    if (key == "deltas") {
      opts[[key]] <- as.numeric(strsplit(value, ",", fixed = TRUE)[[1]])
    } else if (is.integer(defaults[[key]])) {
      opts[[key]] <- as.integer(value)
    } else {
      opts[[key]] <- as.numeric(value)
    }
  }
  opts
}

log_sum_exp <- function(z) {
  zmax <- max(z)
  zmax + log(sum(exp(z - zmax)))
}

suffix_means <- function(prefix, t) {
  before <- if (t == 1L) 0 else c(0, prefix[seq_len(t - 1L)])
  n <- seq.int(t, 1L)
  means <- (prefix[t] - before) / n
  list(n = n, means = means)
}

section31_log_stat <- function(prefix, t, sigma = 1, rho = 1, bisect_steps = 35L) {
  sm <- suffix_means(prefix, t)
  n <- sm$n
  means <- sm$means
  
  rho2 <- rho^2
  sig2 <- sigma^2
  denom <- 1 + rho2 * sig2 * n
  logw <- -0.5 * log(denom)
  a <- rho2 * n^2 / (2 * denom)
  
  lo <- min(means)
  hi <- max(means)
  if (lo == hi) {
    theta <- lo
  } else {
    for (unused in seq_len(bisect_steps)) {
      mid <- 0.5 * (lo + hi)
      d <- mid - means
      z <- logw + a * d^2
      zmax <- max(z)
      ew <- exp(z - zmax)
      deriv <- sum(2 * a * d * ew) / sum(ew)
      if (deriv < 0) {
        lo <- mid
      } else {
        hi <- mid
      }
    }
    theta <- 0.5 * (lo + hi)
  }
  
  d <- theta - means
  log_sum_exp(logw + a * d^2)
}

repeated_cs_crossed <- function(prefix, t, alpha, sigma = 1, rho = 1) {
  sm <- suffix_means(prefix, t)
  n <- sm$n
  means <- sm$means
  
  rho2 <- rho^2
  sig2 <- sigma^2
  denom <- 1 + rho2 * sig2 * n
  radius <- sqrt(
    (2 * denom / (rho2 * n^2)) *
      (log(1 / alpha) + 0.5 * log(denom))
  )
  
  max(means - radius) > min(means + radius)
}

run_detectors <- function(x, A = 1000, sigma = 1, rho = 1) {
  max_time <- length(x)
  prefix <- cumsum(x)
  log_A <- log(A)
  alpha <- 1 / A
  
  tau_section31 <- max_time + 1L
  tau_repeated_cs <- max_time + 1L
  
  for (t in seq_len(max_time)) {
    if (tau_section31 > max_time &&
        section31_log_stat(prefix, t, sigma, rho) >= log_A) {
      tau_section31 <- t
    }
    
    if (tau_repeated_cs > max_time &&
        repeated_cs_crossed(prefix, t, alpha, sigma, rho)) {
      tau_repeated_cs <- t
    }
    
    if (tau_section31 <= max_time && tau_repeated_cs <= max_time) break
  }
  
  c(section31 = tau_section31, repeated_cs = tau_repeated_cs)
}

simulate_one_delta <- function(delta, opts) {
  delays_section31 <- numeric(opts$reps)
  delays_repeated_cs <- numeric(opts$reps)
  false_section31 <- 0L
  false_repeated_cs <- 0L
  cens_section31 <- 0L
  cens_repeated_cs <- 0L
  cond_section31 <- 0L
  cond_repeated_cs <- 0L
  
  for (b in seq_len(opts$reps)) {
    x <- c(
      rnorm(opts$change_time, mean = 0, sd = opts$sigma),
      rnorm(opts$max_time - opts$change_time, mean = delta, sd = opts$sigma)
    )
    
    tau <- run_detectors(x, A = opts$A, sigma = opts$sigma, rho = opts$rho)
    
    if (tau["section31"] <= opts$change_time) {
      false_section31 <- false_section31 + 1L
    } else {
      cond_section31 <- cond_section31 + 1L
      if (tau["section31"] > opts$max_time) cens_section31 <- cens_section31 + 1L
      delays_section31[cond_section31] <- min(tau["section31"], opts$max_time + 1L) -
        opts$change_time
    }
    
    if (tau["repeated_cs"] <= opts$change_time) {
      false_repeated_cs <- false_repeated_cs + 1L
    } else {
      cond_repeated_cs <- cond_repeated_cs + 1L
      if (tau["repeated_cs"] > opts$max_time) cens_repeated_cs <- cens_repeated_cs + 1L
      delays_repeated_cs[cond_repeated_cs] <- min(tau["repeated_cs"], opts$max_time + 1L) -
        opts$change_time
    }
  }
  
  data.frame(
    delta = delta,
    section31_cadd = mean(delays_section31[seq_len(cond_section31)]),
    repeated_cs_cadd = mean(delays_repeated_cs[seq_len(cond_repeated_cs)]),
    section31_false_alarm_rate = false_section31 / opts$reps,
    repeated_cs_false_alarm_rate = false_repeated_cs / opts$reps,
    section31_conditioned_n = cond_section31,
    repeated_cs_conditioned_n = cond_repeated_cs,
    section31_censored_after_change = cens_section31,
    repeated_cs_censored_after_change = cens_repeated_cs
  )
}

defaults <- list(
  A = 1000,
  reps = 50L,
  change_time = 500L,
  max_time = 2500L,
  sigma = 1,
  rho = 1,
  seed = 20260711L,
  deltas = c(1, 1.25, 1.5, 2)
)

opts <- parse_args(defaults)
set.seed(opts$seed)

message("Settings:")
message("  A = ", opts$A, "; repeated-CS alpha = 1/A = ", signif(1 / opts$A, 4))
message("  reps = ", opts$reps, "; change_time = ", opts$change_time,
        "; max_time = ", opts$max_time)
message("  sigma = ", opts$sigma, "; rho = ", opts$rho)
message("  deltas = ", paste(opts$deltas, collapse = ", "))

rows <- vector("list", length(opts$deltas))
for (i in seq_along(opts$deltas)) {
  message("Running delta = ", opts$deltas[i])
  rows[[i]] <- simulate_one_delta(opts$deltas[i], opts)
}
results <- do.call(rbind, rows)

print(results, row.names = FALSE, digits = 4)

out_file <- file.path("outputs", "gaussian_mean_change_cadd_table.csv")
if (!dir.exists(dirname(out_file))) dir.create(dirname(out_file), recursive = TRUE)
write.csv(results, out_file, row.names = FALSE)
message("Wrote: ", out_file)
