# ============================================================================
#   FULL CELL-LEVEL RF PIPELINE — v3 (MAJOR OPTIMIZATION)
# ============================================================================
#
# WHAT CHANGED FROM v2
# ─────────────────────────────────────────────────────────────────────────────
#  v2 bottleneck (still present):
#    build_neighbor_lookup() is an lapply() over ALL rows of cell_data
#    (~6.46 M rows).  For each row it:
#      1. looks up the cell's position in id_order (string hash),
#      2. looks up that cell's neighbors in rook_neighbors_unique,
#      3. pastes neighbor_id + year to build lookup keys,
#      4. scans idx_lookup for every key (string hash again).
#    This creates a list of 6.46 M integer vectors, consuming gigabytes of
#    RAM, and the lapply itself runs in pure R interpreter time.
#
#  v3 replaces that with:
#    - A SINGLE long neighbor-pair table (cell_id | neighbor_cell_id) built
#      ONCE from the rook adjacency object — no year dimension at this step.
#    - Year-by-year processing: for each year, join the pair table with that
#      year's values and aggregate with data.table (C-level).  RAM peak is
#      one year (~340 k rows) instead of all years at once.
#    - data.table is used throughout feature construction, eliminating
#      dplyr copies.
#    - Checkpointing per year so the run can resume.
#
#  Other improvements:
#    - usd_est_prop computed with data.table for speed.
#    - Interaction terms computed in-place with data.table :=.
#    - fwrite used for output (already in v2).
#    - gc() placed only at genuine memory boundaries.
#
# ============================================================================

library(data.table)
library(tidyverse)   # kept for any helpers not yet replaced
library(sf)
library(spdep)
library(zoo)
library(randomForest)

# ── PATHS ─────────────────────────────────────────────────────────────────────
base_path <- "C:/Users/ROBERTODU/OneDrive - Inter-American Development Bank Group/Documents/R2"
setwd(base_path)

cell_data_path     <- file.path(base_path, "cells_temporal_vars.rds")
rook_nb_path       <- file.path(base_path, "rook_neighbors_unique.rds")
model_path         <- file.path(base_path, "model_5_all_countries.RData")
output_path        <- file.path(base_path, "RF_imputated_cell_level.csv")
checkpoint_folder  <- file.path(base_path, "checkpoints_v3")
log_path           <- file.path(base_path, "cell_imputation_model_5_local_v3.log")
diagnostics_dir    <- file.path(base_path, "run_diagnostics_v3")
stage_metrics_path <- file.path(diagnostics_dir, "cell_imputation_stage_metrics_v3.csv")
run_summary_path   <- file.path(diagnostics_dir, "cell_imputation_run_summary_v3.txt")
failure_dump_path  <- file.path(diagnostics_dir, "cell_imputation_failure_dump_v3.rds")

required_files <- c(cell_data_path, rook_nb_path, model_path)
missing_files  <- required_files[!file.exists(required_files)]
if (length(missing_files) > 0) {
  stop(paste(
    "Missing required input file(s):",
    paste(basename(missing_files), collapse = ", "),
    "\nCurrent working directory:", base_path
  ))
}

dir.create(checkpoint_folder, showWarnings = FALSE, recursive = TRUE)
dir.create(diagnostics_dir,   showWarnings = FALSE, recursive = TRUE)

log_con <- file(log_path, open = "wt")
sink(log_con, split = TRUE)
sink(log_con, type = "message")
on.exit({ sink(type = "message"); sink(); close(log_con) }, add = TRUE)

log_message <- function(...) {
  cat(format(Sys.time(), "[%Y-%m-%d %H:%M:%S] "), ..., "\n", sep = "")
}

object_size_mb <- function(x) round(as.numeric(object.size(x)) / 1024^2, 2)

current_stage    <- "initialization"
stage_started_at <- Sys.time()

append_text_line <- function(path, text) {
  cat(text, "\n", file = path, append = TRUE, sep = "")
}

record_stage_metric <- function(stage, status, started_at, data_obj = NULL, notes = "") {
  metric <- data.frame(
    timestamp       = format(Sys.time(), "%Y-%m-%d %H:%M:%S"),
    stage           = stage,
    status          = status,
    elapsed_seconds = round(as.numeric(difftime(Sys.time(), started_at, units = "secs")), 2),
    rows            = if (is.null(data_obj)) NA_integer_ else nrow(data_obj),
    cols            = if (is.null(data_obj)) NA_integer_ else ncol(data_obj),
    object_size_mb  = if (is.null(data_obj)) NA_real_    else object_size_mb(data_obj),
    notes           = notes,
    stringsAsFactors = FALSE
  )
  write.table(metric, file = stage_metrics_path, sep = ",", row.names = FALSE,
              col.names = !file.exists(stage_metrics_path),
              append    = file.exists(stage_metrics_path))
}

start_stage <- function(stage_name) {
  current_stage    <<- stage_name
  stage_started_at <<- Sys.time()
  log_message("Starting stage: ", stage_name)
}

finish_stage <- function(stage_name, data_obj = NULL, notes = "") {
  record_stage_metric(stage_name, "completed", stage_started_at, data_obj, notes)
  log_message("Completed stage: ", stage_name,
              if (nzchar(notes)) paste0(" (", notes, ")") else "")
}

write_run_summary <- function(status, error_message = NULL) {
  summary_lines <- c(
    paste("Run status:", status),
    paste("Timestamp:", format(Sys.time(), "%Y-%m-%d %H:%M:%S")),
    paste("Script version: v3"),
    paste("Current stage:", current_stage),
    paste("Working directory:", base_path)
  )
  if (!is.null(error_message)) {
    summary_lines <- c(summary_lines, paste("Error message:", error_message))
  }
  writeLines(summary_lines, con = run_summary_path)
}

capture_failure_diagnostics <- function(error_obj) {
  error_message <- conditionMessage(error_obj)
  call_stack    <- vapply(sys.calls(), function(call) paste(deparse(call), collapse = " "), character(1))
  record_stage_metric(current_stage, "failed", stage_started_at, notes = error_message)
  write_run_summary("FAILED", error_message)
  saveRDS(list(timestamp = Sys.time(), stage = current_stage,
               error_message = error_message, call_stack = call_stack,
               session_info  = utils::capture.output(sessionInfo())),
          failure_dump_path)
  log_message("Run failed in stage ", current_stage, ": ", error_message)
}

safe_remove <- function(...) {
  nms <- vapply(as.list(substitute(list(...)))[-1], deparse, character(1))
  ex  <- nms[vapply(nms, exists, logical(1), inherits = FALSE)]
  if (length(ex) > 0) { rm(list = ex, envir = .GlobalEnv); invisible(gc()) }
}

prediction_vars_for_year <- function(rf_model, fallback_vars) {
  if (!is.null(rf_model$importance)) return(intersect(rownames(rf_model$importance), fallback_vars))
  fallback_vars
}

# ── KEY HELPER: build a STATIC cell-pair table from the rook nb object ─────────
# This is done once, in O(total_neighbor_pairs) time.
# Result: data.table with columns  cell_idx | neighbor_idx
#         where idx refers to position in id_order (1-based).
# Uses lengths() rather than card() — safe for spdep nb objects AND plain lists.
build_pair_table <- function(id_order, rook_neighbors) {
  lens <- lengths(rook_neighbors)   # works for nb class and plain list
  if (length(lens) != length(id_order)) {
    stop(paste0(
      "build_pair_table: length mismatch — id_order has ", length(id_order),
      " cells but rook_neighbors has ", length(lens), " entries. ",
      "Ensure isolated cells are removed from both before calling this function."
    ))
  }
  from_idx <- rep(seq_along(id_order), times = lens)
  to_idx   <- unlist(rook_neighbors, use.names = FALSE)
  data.table(from_idx = from_idx, to_idx = to_idx)
}

# ── NEIGHBOR STATS PER YEAR (vectorized, no row-wise loop) ───────────────────
# values_vec: numeric vector of length length(id_order), in id_order order
# pair_dt:    data.table(from_idx, to_idx) from build_pair_table()
# Returns data.table: row_idx | max_v | min_v | mean_v
compute_neighbor_stats_fast <- function(values_vec, pair_dt) {
  # Attach neighbor value to each pair
  dt <- copy(pair_dt)
  dt[, val := values_vec[to_idx]]

  # Aggregate by center cell
  agg <- dt[, .(
    max_v  = if (all(is.na(val))) NA_real_ else max(val,  na.rm = TRUE),
    min_v  = if (all(is.na(val))) NA_real_ else min(val,  na.rm = TRUE),
    mean_v = if (all(is.na(val))) NA_real_ else mean(val, na.rm = TRUE)
  ), by = from_idx]

  # Scatter back to full-length vectors (cells with no neighbors stay NA)
  n <- length(values_vec)
  out_max  <- rep(NA_real_, n)
  out_min  <- rep(NA_real_, n)
  out_mean <- rep(NA_real_, n)
  out_max [agg$from_idx] <- agg$max_v
  out_min [agg$from_idx] <- agg$min_v
  out_mean[agg$from_idx] <- agg$mean_v

  list(max = out_max, min = out_min, mean = out_mean)
}

# ── ADD NEIGHBOR FEATURE COLUMNS IN PLACE (data.table) ───────────────────────
add_neighbor_features_dt <- function(dt_year, var_name, stats, id_to_idx) {
  # Map cell id -> index in id_order so we can pull the right stat value
  idx <- id_to_idx[as.character(dt_year$id)]   # integer, NA for unknowns

  v  <- dt_year[[var_name]]
  mx <- stats$max [idx]
  mn <- stats$min [idx]
  me <- stats$mean[idx]

  dt_year[, (paste0("max_neighbor_",  var_name)) := mx]
  dt_year[, (paste0("min_neighbor_",  var_name)) := mn]
  dt_year[, (paste0("mean_neighbor_", var_name)) := me]

  max_diff  <- v - mx
  min_diff  <- v - mn
  mean_diff <- v - me

  dt_year[, (paste0(var_name, "_max_diff"))  := max_diff]
  dt_year[, (paste0(var_name, "_min_diff"))  := min_diff]
  dt_year[, (paste0(var_name, "_mean_diff")) := mean_diff]
  dt_year[, (paste0(var_name, "_hotspot"))   := fifelse(max_diff > 0, 1L, 0L)]
  dt_year[, (paste0("rel_diff_max_",  var_name)) := max_diff  / (mx + 1e-6) * 100]
  dt_year[, (paste0("rel_diff_min_",  var_name)) := min_diff  / (mn + 1e-6) * 100]
  dt_year[, (paste0("rel_diff_mean_", var_name)) := mean_diff / (me + 1e-6) * 100]

  invisible(dt_year)
}

# ── EXPECTED MODEL VARIABLES ─────────────────────────────────────────────────
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

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# ============================================================================
# MAIN PIPELINE
# ============================================================================

write_run_summary("STARTED")
total_start <- Sys.time()

tryCatch({

# --- 0. LOAD DATA -------------------------------------------------------------

start_stage("load_data")
cell_data             <- readRDS(cell_data_path)
rook_neighbors_unique <- readRDS(rook_nb_path)
load(model_path)   # loads rf_models_per_year (or equivalent)

if (inherits(cell_data, "sf")) {
  log_message("Dropping geometry to save memory...")
  cell_data <- st_drop_geometry(cell_data)
}

# Convert to data.table immediately — avoids dplyr copies throughout
setDT(cell_data)
log_message("cell_data rows: ", nrow(cell_data), "  cols: ", ncol(cell_data))
finish_stage("load_data", cell_data,
             paste("rook_nb_mb=", object_size_mb(rook_neighbors_unique)))

# --- 1. BUILD id_order + REMOVE ISOLATED CELLS --------------------------------

start_stage("rebuild_id_order")
min_year <- min(cell_data$year)
cell_ref <- cell_data[year == min_year]   # one row per cell

# ── Handle length mismatch between cell_ref and rook_neighbors ────────────────
# spdep builds the nb object from a spatial object with N rows. If cell_data
# has more rows than that object (e.g. 2 extra boundary cells), the nb will
# be shorter. Align by truncating cell_ref to nb length; extra cells will pass
# through the pipeline with NA neighbor stats.
nb_len_full <- length(rook_neighbors_unique)
cr_len      <- nrow(cell_ref)
if (cr_len != nb_len_full) {
  log_message(sprintf(
    "WARNING: cell_ref has %d rows but rook_neighbors has %d entries (diff=%d). ",
    cr_len, nb_len_full, cr_len - nb_len_full
  ), "Truncating cell_ref to nb length. Extra cells get NA neighbor stats.")
  cell_ref <- cell_ref[seq_len(nb_len_full)]
}

nb_lens      <- lengths(rook_neighbors_unique)   # safe for nb + plain list
isolated_idx <- which(nb_lens == 0)
if (length(isolated_idx) > 0) {
  log_message("Removing ", length(isolated_idx), " isolated cells.")
  isolated_ids          <- cell_ref$id[isolated_idx]
  cell_data             <- cell_data[!id %in% isolated_ids]
  cell_ref              <- cell_ref [-isolated_idx]
  keep_idx              <- setdiff(seq_along(rook_neighbors_unique), isolated_idx)
  rook_neighbors_unique <- rook_neighbors_unique[keep_idx]
  nb_lens               <- nb_lens[-isolated_idx]
}

id_order <- cell_ref$id   # integer vector, length = n_cells
n_cells  <- length(id_order)
log_message("Unique cells: ", n_cells, "  Total rows: ", nrow(cell_data))
log_message("Years: ", paste(sort(unique(cell_data$year)), collapse = ", "))

# Named integer map: cell id (as character) -> position in id_order
id_to_idx <- setNames(seq_len(n_cells), as.character(id_order))

finish_stage("rebuild_id_order", cell_data, paste("n_cells=", n_cells))

# --- 2. COMPUTE usd_est_prop (data.table) ------------------------------------

start_stage("compute_usd_est_prop")
cell_data[, grp_sum := sum(usd_est_n2, na.rm = TRUE),
          by = .(NUTS0, NUTS1, NUTS2, NUTS_pe, year)]
cell_data[, usd_est_prop := usd_est_n2 / grp_sum]
cell_data[is.na(usd_est_prop) | is.nan(usd_est_prop), usd_est_prop := 0]
cell_data[, grp_sum := NULL]
finish_stage("compute_usd_est_prop", cell_data)

# --- 3. BUILD STATIC PAIR TABLE (once, no year dimension) --------------------

start_stage("build_pair_table")
log_message("Building cell-pair table from rook adjacency (one-time, O(pairs))...")
pair_dt <- build_pair_table(id_order, rook_neighbors_unique)
log_message("Pair table rows: ", nrow(pair_dt))
finish_stage("build_pair_table", pair_dt, paste("pairs=", nrow(pair_dt)))

# Free rook object — no longer needed
safe_remove(rook_neighbors_unique)

# --- 4. YEAR-BY-YEAR: NEIGHBOR FEATURES + INTERACTION TERMS + RF PREDICTION ---

years     <- sort(unique(cell_data$year))
n_years   <- length(years)

# Check which years already have checkpoints
done_years <- c()
ckpt_master <- file.path(checkpoint_folder, "predictions_master.rds")
if (file.exists(ckpt_master)) {
  log_message("Found existing predictions master checkpoint — loading...")
  pred_master <- readRDS(ckpt_master)
  done_years  <- unique(pred_master$year)
  log_message("Already completed years: ", paste(sort(done_years), collapse = ", "))
} else {
  pred_master <- NULL
}

for (yr in years) {
  yr_chr   <- as.character(yr)
  yr_ckpt  <- file.path(checkpoint_folder, paste0("year_", yr, "_predictions.rds"))

  if (yr %in% done_years) {
    log_message("[Year ", yr, "] Skipping — checkpoint already exists.")
    next
  }

  current_stage    <- paste0("year_", yr)
  stage_started_at <- Sys.time()
  log_message("=== Processing year ", yr, " (", which(years == yr), "/", n_years, ") ===")

  # Subset this year
  dt_yr <- cell_data[year == yr]
  setkey(dt_yr, id)

  # ── 4a. Neighbor stats for each source variable ────────────────────────────
  # For each variable:
  #   - extract values in id_order order (aligning with pair_dt indices)
  #   - compute max/min/mean across neighbors via data.table aggregation
  #   - scatter back and attach derived columns

  for (var_name in neighbor_source_vars) {
    vs_start <- Sys.time()

    # Values aligned to id_order (some cells may not be in dt_yr — leave as NA)
    vals <- rep(NA_real_, n_cells)
    # Match by id
    m    <- match(id_order, dt_yr$id)
    vals[!is.na(m)] <- dt_yr[[var_name]][m[!is.na(m)]]

    stats <- compute_neighbor_stats_fast(vals, pair_dt)
    add_neighbor_features_dt(dt_yr, var_name, stats, id_to_idx)

    log_message("  ", var_name, " neighbor features: ",
                round(difftime(Sys.time(), vs_start, units = "secs"), 1), "s")
  }

  # ── 4b. Interaction terms (in-place, no copy) ──────────────────────────────
  dt_yr[, crop_def     := crop    * def]
  dt_yr[, pop_ec       := pop     * ec]
  dt_yr[, pop_ntl      := pop     * ntl]
  dt_yr[, pop_urb      := pop     * urb]
  dt_yr[, ntl_urb      := ntl     * urb]
  dt_yr[, ntl_ec       := ntl     * ec]
  dt_yr[, deflag_pop   := def_lag * pop]
  dt_yr[, deflag_urb   := def_lag * urb]
  dt_yr[, road_cnt_urb := road_cnt * urb]
  dt_yr[, road_len_urb := road_len * urb]
  dt_yr[, ntl2         := ntl^2]
  dt_yr[, ec2          := ec^2]

  # ── 4c. Variable check ─────────────────────────────────────────────────────
  missing_vars <- setdiff(model_5_vars, names(dt_yr))
  if (length(missing_vars) > 0) {
    log_message("WARNING year ", yr, " — missing vars: ", paste(missing_vars, collapse = ", "))
  }

  # ── 4d. RF prediction ─────────────────────────────────────────────────────
  dt_yr[, consolidated := NA_real_]

  if (!yr_chr %in% names(rf_models_per_year)) {
    log_message("  No RF model for year ", yr, " — skipping prediction.")
  } else {
    rf_model   <- rf_models_per_year[[yr_chr]]
    model_vars <- prediction_vars_for_year(rf_model, model_5_vars)
    avail_vars <- intersect(model_vars, names(dt_yr))

    tryCatch({
      test_set <- dt_yr[, ..avail_vars]
      preds    <- predict(rf_model, newdata = as.data.frame(test_set))   # vectorized
      dt_yr[, consolidated := preds]
      log_message("  Predicted ", length(preds), " rows.  Range: [",
                  round(min(preds), 3), ", ", round(max(preds), 3), "]")
    }, error = function(e) {
      log_message("  RF PREDICTION ERROR year ", yr, ": ", conditionMessage(e))
      missing_in_data <- setdiff(rownames(importance(rf_model)), names(dt_yr))
      if (length(missing_in_data) > 0) log_message("  Missing in data: ", paste(missing_in_data, collapse = ", "))
    })
  }

  # ── 4e. Save year checkpoint ───────────────────────────────────────────────
  out_cols <- union(c("id", "year", "consolidated"), intersect(model_5_vars, names(dt_yr)))
  saveRDS(dt_yr[, ..out_cols], yr_ckpt)

  # Accumulate master results
  if (is.null(pred_master)) {
    pred_master <- dt_yr[, ..out_cols]
  } else {
    pred_master <- rbindlist(list(pred_master, dt_yr[, ..out_cols]), use.names = TRUE, fill = TRUE)
  }
  saveRDS(pred_master, ckpt_master)

  elapsed_yr <- round(difftime(Sys.time(), stage_started_at, units = "mins"), 2)
  record_stage_metric(current_stage, "completed", stage_started_at, dt_yr,
                      paste0("elapsed_min=", elapsed_yr))
  log_message("  Year ", yr, " done in ", elapsed_yr, " min")

  rm(dt_yr, vals, stats); gc()
}

# --- 5. VARIABLE CHECK (on assembled results) ---------------------------------

start_stage("variable_check")
missing_vars <- setdiff(model_5_vars, names(pred_master))
if (length(missing_vars) > 0) {
  log_message("FINAL CHECK — missing vars: ", paste(missing_vars, collapse = ", "))
} else {
  log_message("All model 5 variables present in output.")
}
finish_stage("variable_check", pred_master,
             paste("model_var_count=", length(model_5_vars)))

# --- 6. SAVE FINAL OUTPUT -----------------------------------------------------

start_stage("save_results")
log_message("Writing output with fwrite...")
fwrite(pred_master, output_path)
finish_stage("save_results", pred_master)

log_message("========== DONE ==========")
log_message("Predicted rows: ", sum(!is.na(pred_master$consolidated)))
log_message("NA rows: ",        sum( is.na(pred_master$consolidated)))
log_message("Output: ", output_path)
log_message("Total time: ",
            round(difftime(Sys.time(), total_start, units = "hours"), 2), " hours")
log_message("==========================")
write_run_summary("COMPLETED")

}, error = function(e) {
  capture_failure_diagnostics(e)
  stop(e)
})
