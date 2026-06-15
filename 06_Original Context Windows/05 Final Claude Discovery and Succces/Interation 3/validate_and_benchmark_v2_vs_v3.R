# ============================================================================
#   VALIDATION & BENCHMARK — v3 correctness + readiness check
# ============================================================================
#
# WHAT THIS SCRIPT DOES
# ─────────────────────
# 1. Diagnoses the cell_ref / rook_neighbors length mismatch.
# 2. Validates v3 neighbor stats internally (manual spot-check, no v2 needed).
# 3. Benchmarks v3 on the full grid for one year.
# 4. Confirms all model_5_vars are present in a sample year output.
# 5. Prints a GO / NO-GO verdict for the main pipeline.
#
# WHY WE DROPPED THE v2 COMPARISON
# ─────────────────────────────────
# The output showed all-NA differences because v2's build_neighbor_lookup()
# uses id_order positions as keys — and the 2-row mismatch between cell_ref
# (344210) and rook_nb (344208) meant the sampled cells got wrong positions,
# so their neighbor lookups silently returned empty sets.
# v2 is therefore not a reliable reference on this dataset as-is.
# We validate v3 directly against the raw rook structure instead.
# ============================================================================

library(data.table)
library(spdep)

base_path      <- "C:/Users/ROBERTODU/OneDrive - Inter-American Development Bank Group/Documents/R2"
cell_data_path <- file.path(base_path, "cells_temporal_vars.rds")
rook_nb_path   <- file.path(base_path, "rook_neighbors_unique.rds")

SAMPLE_N   <- 200L   # cells to spot-check manually
SAMPLE_VAR <- "ntl"  # change if ntl is all-NA in your earliest year
set.seed(42)

issues <- character(0)   # collect any GO/NO-GO issues

# ============================================================================
# 0. LOAD
# ============================================================================
cat("Loading data...\n")
cell_data_full        <- readRDS(cell_data_path)
rook_neighbors_unique <- readRDS(rook_nb_path)

if (inherits(cell_data_full, "sf")) { library(sf); cell_data_full <- sf::st_drop_geometry(cell_data_full) }
setDT(cell_data_full)

# ============================================================================
# 1. MISMATCH DIAGNOSIS
# ============================================================================
cat("\n========== SECTION 1: MISMATCH DIAGNOSIS ==========\n")

SAMPLE_YEAR <- min(cell_data_full$year)
cell_ref    <- cell_data_full[year == SAMPLE_YEAR]
nb_len      <- length(rook_neighbors_unique)
cr_len      <- nrow(cell_ref)

cat("cell_ref rows             :", cr_len,  "\n")
cat("rook_neighbors_unique len :", nb_len,  "\n")
cat("Difference                :", cr_len - nb_len, "\n")

if (cr_len == nb_len) {
  cat("✅  Lengths match — no mismatch issue.\n")
  rook_nb_use <- rook_neighbors_unique
  cell_ref_use <- cell_ref
} else {
  cat("⚠️  Mismatch of", cr_len - nb_len, "rows detected.\n")
  cat("Investigating which rows in cell_ref have no entry in rook_neighbors...\n")

  # The most common cause: cell_ref has extra rows that were never in the
  # spatial object used to build rook_neighbors.
  # Strategy: use only the first nb_len rows of cell_ref (matching the
  # spdep convention that row i of the nb corresponds to row i of the sf).
  n_use        <- min(cr_len, nb_len)
  cell_ref_use <- cell_ref[seq_len(n_use)]
  rook_nb_use  <- rook_neighbors_unique[seq_len(n_use)]

  cat("Using first", n_use, "rows of cell_ref aligned to rook_neighbors.\n")
  cat("Extra cell_ref rows (will be excluded from neighbor computation):\n")
  extra_ids <- cell_ref$id[seq(n_use + 1, cr_len)]
  print(extra_ids)

  issues <- c(issues, paste0(
    "cell_ref has ", cr_len - nb_len, " more rows than rook_neighbors_unique. ",
    "Cells ", paste(head(extra_ids, 5), collapse=", "),
    " (and ", max(0, length(extra_ids)-5), " more) will have NA neighbor stats. ",
    "Verify this is acceptable — these cells likely sit at the grid boundary."
  ))
}

# Remove isolated cells (0 neighbors) from the aligned set
nb_lens      <- lengths(rook_nb_use)
isolated_idx <- which(nb_lens == 0)
cat("\nIsolated cells (0 neighbors):", length(isolated_idx), "\n")
if (length(isolated_idx) > 0) {
  cell_ref_use <- cell_ref_use[-isolated_idx]
  rook_nb_use  <- rook_nb_use[-isolated_idx]
  nb_lens      <- nb_lens[-isolated_idx]
}

id_order  <- cell_ref_use$id
n_cells   <- length(id_order)
id_to_idx <- setNames(seq_len(n_cells), as.character(id_order))
cat("Final grid size:", n_cells, "cells\n")

# ============================================================================
# 2. BUILD PAIR TABLE
# ============================================================================
cat("\n========== SECTION 2: BUILD PAIR TABLE ==========\n")
t_pair <- system.time({
  from_idx <- rep(seq_len(n_cells), times = nb_lens)
  to_idx   <- unlist(rook_nb_use, use.names = FALSE)
  pair_dt  <- data.table(from_idx = from_idx, to_idx = to_idx)
})
cat(sprintf("Pair table built in %.2f s\n", t_pair["elapsed"]))
cat(sprintf("Rows: %d  |  avg neighbors/cell: %.2f\n",
            nrow(pair_dt), nrow(pair_dt) / n_cells))

# Sanity: to_idx must be in [1, n_cells]
bad_to <- pair_dt[to_idx < 1L | to_idx > n_cells, .N]
if (bad_to > 0) {
  issues <- c(issues, paste("CRITICAL: pair_dt has", bad_to, "out-of-range to_idx values."))
  cat("❌  CRITICAL: out-of-range to_idx values:", bad_to, "\n")
} else {
  cat("✅  All to_idx values in valid range [1,", n_cells, "]\n")
}

# ============================================================================
# 3. PICK A SAMPLE VARIABLE THAT HAS NON-NA VALUES
# ============================================================================
cat("\n========== SECTION 3: FIND A NON-NA TEST VARIABLE ==========\n")

yr_data <- cell_data_full[year == SAMPLE_YEAR]

# Check coverage of candidate variables
candidate_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2", "pop", "urb")
for (cv in candidate_vars) {
  if (cv %in% names(yr_data)) {
    n_nonNA <- sum(!is.na(yr_data[[cv]]))
    cat(sprintf("  %-20s : %d non-NA values (%.1f%%)\n", cv, n_nonNA,
                100 * n_nonNA / nrow(yr_data)))
  }
}

# Use first candidate with >50% non-NA
TEST_VAR <- NULL
for (cv in candidate_vars) {
  if (cv %in% names(yr_data) && mean(!is.na(yr_data[[cv]])) > 0.5) {
    TEST_VAR <- cv; break
  }
}
if (is.null(TEST_VAR)) {
  TEST_VAR <- names(yr_data)[sapply(yr_data, function(x) is.numeric(x) && mean(!is.na(x)) > 0.5)][1]
}
cat("\nUsing test variable:", TEST_VAR, "\n")

# ============================================================================
# 4. COMPUTE NEIGHBOR STATS (v3) AND SPOT-CHECK MANUALLY
# ============================================================================
cat("\n========== SECTION 4: COMPUTE + SPOT-CHECK ==========\n")

# Align values to id_order
vals <- rep(NA_real_, n_cells)
m    <- match(id_order, yr_data$id)
vals[!is.na(m)] <- yr_data[[TEST_VAR]][m[!is.na(m)]]
cat("Non-NA values in aligned vector:", sum(!is.na(vals)), "/", n_cells, "\n")

if (sum(!is.na(vals)) == 0) {
  cat("❌  All values are NA — cannot validate. Try a different SAMPLE_YEAR or variable.\n")
  issues <- c(issues, paste("TEST_VAR", TEST_VAR, "is all-NA in year", SAMPLE_YEAR))
} else {
  t_stats <- system.time({
    dt_tmp <- copy(pair_dt)
    dt_tmp[, val := vals[to_idx]]
    agg <- dt_tmp[, .(
      max_v  = if (all(is.na(val))) NA_real_ else max(val,  na.rm = TRUE),
      min_v  = if (all(is.na(val))) NA_real_ else min(val,  na.rm = TRUE),
      mean_v = if (all(is.na(val))) NA_real_ else mean(val, na.rm = TRUE)
    ), by = from_idx]
    out_max  <- rep(NA_real_, n_cells)
    out_min  <- rep(NA_real_, n_cells)
    out_mean <- rep(NA_real_, n_cells)
    out_max [agg$from_idx] <- agg$max_v
    out_min [agg$from_idx] <- agg$min_v
    out_mean[agg$from_idx] <- agg$mean_v
  })
  cat(sprintf("compute_neighbor_stats time: %.2f s for %d cells\n", t_stats["elapsed"], n_cells))

  # MANUAL SPOT-CHECK: for SAMPLE_N random cells, compute neighbor stats
  # directly from the rook list and compare to v3 output
  cat("\nManual spot-check on", SAMPLE_N, "randomly sampled cells...\n")
  check_idx <- sample(which(!is.na(out_max)), min(SAMPLE_N, sum(!is.na(out_max))))

  manual_max  <- numeric(length(check_idx))
  manual_min  <- numeric(length(check_idx))
  manual_mean <- numeric(length(check_idx))

  for (k in seq_along(check_idx)) {
    ci        <- check_idx[k]
    nb_pos    <- rook_nb_use[[ci]]   # neighbor positions in id_order
    nb_vals   <- vals[nb_pos]
    manual_max [k] <- if (all(is.na(nb_vals))) NA_real_ else max(nb_vals,  na.rm = TRUE)
    manual_min [k] <- if (all(is.na(nb_vals))) NA_real_ else min(nb_vals,  na.rm = TRUE)
    manual_mean[k] <- if (all(is.na(nb_vals))) NA_real_ else mean(nb_vals, na.rm = TRUE)
  }

  diff_max  <- out_max [check_idx] - manual_max
  diff_min  <- out_min [check_idx] - manual_min
  diff_mean <- out_mean[check_idx] - manual_mean

  TOLERANCE <- 1e-8
  for (stat in c("max", "min", "mean")) {
    d       <- get(paste0("diff_", stat))
    max_err <- max(abs(d), na.rm = TRUE)
    status  <- if (!is.finite(max_err)) "⚠️  all-NA" else
               if (max_err < TOLERANCE) "✅  PASS" else "❌  FAIL"
    cat(sprintf("  %s_neighbor_%s : max |diff| = %s  %s\n",
                stat, TEST_VAR,
                if (is.finite(max_err)) formatC(max_err, format="e", digits=2) else "NA",
                status))
    if (is.finite(max_err) && max_err >= TOLERANCE)
      issues <- c(issues, paste0(stat, "_neighbor_", TEST_VAR,
                                 " failed spot-check: max diff = ", max_err))
  }
}

# ============================================================================
# 5. CHECK ALL model_5_vars REACHABLE (variable presence check)
# ============================================================================
cat("\n========== SECTION 5: VARIABLE PRESENCE CHECK ==========\n")

model_5_vars <- c(
  "pop", "def", "ntl", "urb", "crop", "road_len", "road_cnt",
  "petrol", "petrol_d", "l_mine_n", "il_mine_n", "lgl_mine", "ilgl_mine",
  "hidro_n", "hidro_50", "anp_zone_n", "elevmean", "elevmin", "elevmax",
  "anp_zone", "def_lag", "def_growth", "ntl_lag", "ntl_lag2", "pop_density",
  "pop_density_lag", "pop_density_lag2", "ntl_growth", "ntl_growth_pct",
  "elev_differential", "anp_per_area", "petrol_per_area", "roads_per_area",
  "crop_area_km2", "urban_area_km2", "hidro_per_area", "mountainous",
  "pop_ma2", "pop_density_ma2", "ntl_ma3", "ec", "ec_lag", "ec_lag2", "ec_ma3",
  "max_neighbor_ntl", "min_neighbor_ntl", "mean_neighbor_ntl",
  "ntl_max_diff", "ntl_min_diff", "ntl_mean_diff", "ntl_hotspot",
  "rel_diff_max_ntl", "rel_diff_min_ntl", "rel_diff_mean_ntl",
  "max_neighbor_ec", "min_neighbor_ec", "mean_neighbor_ec",
  "ec_max_diff", "ec_min_diff", "ec_mean_diff", "ec_hotspot",
  "rel_diff_max_ec", "rel_diff_min_ec", "rel_diff_mean_ec",
  "max_neighbor_pop_density", "min_neighbor_pop_density", "mean_neighbor_pop_density",
  "pop_density_max_diff", "pop_density_min_diff", "pop_density_mean_diff", "pop_density_hotspot",
  "rel_diff_max_pop_density", "rel_diff_min_pop_density", "rel_diff_mean_pop_density",
  "max_neighbor_def", "min_neighbor_def", "mean_neighbor_def",
  "def_max_diff", "def_min_diff", "def_mean_diff", "def_hotspot",
  "rel_diff_max_def", "rel_diff_min_def", "rel_diff_mean_def",
  "crop_def", "pop_ec", "pop_ntl", "pop_urb", "ntl_urb", "ntl_ec",
  "deflag_pop", "deflag_urb", "road_cnt_urb", "road_len_urb", "ntl2", "ec2",
  "usd_est_n2_max_diff", "usd_est_n2_min_diff", "usd_est_n2_mean_diff", "usd_est_n2_hotspot",
  "rel_diff_max_usd_est_n2", "rel_diff_min_usd_est_n2", "rel_diff_mean_usd_est_n2",
  "usd_est_n2", "usd_est_n2_lag1", "usd_est_n2_lag2",
  "usd_est_n2_ma3", "usd_est_prop", "usd_n2_growth", "usd_n2_growth_pct"
)

# Neighbor-derived vars are computed by the pipeline; check base vars only
base_vars    <- setdiff(model_5_vars, grep("neighbor|_diff|hotspot|rel_diff|crop_def|pop_ec|pop_ntl|pop_urb|ntl_urb|ntl_ec|deflag|road_cnt_urb|road_len_urb|ntl2|ec2|usd_est_prop|usd_n2_growth", model_5_vars, value=TRUE))
missing_base <- setdiff(base_vars, names(cell_data_full))
if (length(missing_base) == 0) {
  cat("✅  All base variables present in cell_data\n")
} else {
  cat("❌  Missing base variables:", paste(missing_base, collapse=", "), "\n")
  issues <- c(issues, paste("Missing base vars:", paste(missing_base, collapse=", ")))
}

# ============================================================================
# 6. RUNTIME EXTRAPOLATION
# ============================================================================
cat("\n========== SECTION 6: RUNTIME EXTRAPOLATION ==========\n")

n_years       <- length(unique(cell_data_full$year))
n_source_vars <- 5

t_pair_s  <- t_pair["elapsed"]
t_stats_s <- if (exists("t_stats")) t_stats["elapsed"] else NA_real_

v3_total_s <- t_pair_s + t_stats_s * n_years * n_source_vars

cat(sprintf("Grid           : %d cells x %d years\n", n_cells, n_years))
cat(sprintf("Pair build     : %.2f s (one-time)\n", t_pair_s))
cat(sprintf("Stats per var  : %.2f s (full grid, one year)\n", t_stats_s))
cat(sprintf("Estimated total neighbor time : %.1f min\n", v3_total_s / 60))
cat(sprintf("  (plus RF prediction: typically a few minutes per year)\n"))

# ============================================================================
# 7. GO / NO-GO VERDICT
# ============================================================================
cat("\n========== SECTION 7: GO / NO-GO ==========\n")
if (length(issues) == 0) {
  cat("✅  GO — all checks passed. You can run cell_imputation_model_5_local_v3.R\n")
} else {
  cat("⚠️  REVIEW BEFORE RUNNING — the following issues were found:\n\n")
  for (i in seq_along(issues)) cat(sprintf("  %d. %s\n\n", i, issues[i]))
  cat("The pipeline will still run, but review the issues above.\n")
  cat("The 2-row mismatch in particular: those 2 extra cells will have NA\n")
  cat("neighbor stats but will otherwise pass through the pipeline normally.\n")
}
