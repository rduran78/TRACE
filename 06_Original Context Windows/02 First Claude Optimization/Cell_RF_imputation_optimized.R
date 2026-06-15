# =============================================================================
# Cell_RF_imputation_optimized.R
# Strategies applied:
#   1. Removed unused st_read() (shapefile was never used in predictions)
#   2. Year-by-year processing: load one model slice at a time, rm() + gc() between
#   4. Chunked prediction within each year to avoid large matrix allocations
# =============================================================================

library(randomForest)
library(data.table)
library(tidyverse)

# -----------------------------------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------------------------------

RDATA_PATH  <- '/Volumes/Toshi 1Tb/R_save_files/model_5_all_countries.RData'
OUTPUT_PATH <- 'RF_imputated_db.csv'
CHUNK_SIZE  <- 50000   # rows per prediction chunk — reduce if RAM is still tight
                       # (e.g. 25000 or 10000 for machines with <16 GB RAM)

# -----------------------------------------------------------------------------
# STEP 1: Load the .RData once, but immediately split and save per-year files
#         so that in subsequent runs (or after a crash) you don't need to reload
#         the full object. Skip this block if per-year files already exist.
# -----------------------------------------------------------------------------

per_year_dir <- "per_year_data"
if (!dir.exists(per_year_dir)) dir.create(per_year_dir)

per_year_files_exist <- length(list.files(per_year_dir, pattern = "\\.rds$")) > 0

if (!per_year_files_exist) {
  
  cat("== Loading full .RData for one-time split (this may take a while) ==\n")
  load(RDATA_PATH)
  # Expected objects after load: rf_models_per_year, joined_data, pred_db
  
  years <- unique(pred_db$year)
  cat(paste0("Years found: ", paste(sort(years), collapse = ", "), "\n"))
  
  # Save pred_db skeleton (without geometry, just the index/year columns needed)
  # We only need id + year columns to reconstruct the final result
  pred_skeleton <- pred_db %>% select(any_of(c("id", "year", "objectid", "NUTS0",
                                                "NUTS1", "NUTS2")))
  saveRDS(pred_skeleton, file.path(per_year_dir, "pred_skeleton.rds"))
  
  # Save each year's data slice and model separately
  for (yr in years) {
    cat(paste0("  Saving year ", yr, "...\n"))
    
    yr_key <- as.character(yr)
    
    # Data slice for this year
    data_slice <- joined_data %>% filter(year == yr)
    saveRDS(data_slice, file.path(per_year_dir, paste0("data_", yr_key, ".rds")))
    
    # Model for this year (if it exists)
    if (yr_key %in% names(rf_models_per_year)) {
      saveRDS(rf_models_per_year[[yr_key]],
              file.path(per_year_dir, paste0("model_", yr_key, ".rds")))
    }
  }
  
  # Free everything from the full load
  rm(rf_models_per_year, joined_data, pred_db, pred_skeleton, data_slice)
  gc()
  cat("== Per-year files saved. Full .RData released from memory. ==\n\n")
  
} else {
  cat("== Per-year files already exist — skipping full .RData load. ==\n\n")
}

# -----------------------------------------------------------------------------
# STEP 2: Predict year by year, one model + one data slice in memory at a time
# -----------------------------------------------------------------------------

pred_skeleton <- readRDS(file.path(per_year_dir, "pred_skeleton.rds"))
years         <- sort(unique(pred_skeleton$year))
all_results   <- vector("list", length(years))
names(all_results) <- as.character(years)

for (yr in years) {
  
  yr_key     <- as.character(yr)
  model_file <- file.path(per_year_dir, paste0("model_", yr_key, ".rds"))
  data_file  <- file.path(per_year_dir, paste0("data_",  yr_key, ".rds"))
  
  cat(paste0("[", yr, "] "))
  
  # --- Check files exist ---
  if (!file.exists(model_file)) {
    cat("WARNING: No model file found — skipping.\n")
    next
  }
  if (!file.exists(data_file)) {
    cat("WARNING: No data file found — skipping.\n")
    next
  }
  
  # --- Load only this year ---
  cat("Loading model... ")
  rf_model  <- readRDS(model_file)
  
  cat("Loading data... ")
  test_set  <- readRDS(data_file)
  n_rows    <- nrow(test_set)
  cat(paste0(n_rows, " rows. "))
  
  # --- STEP 4: Chunked prediction ---
  n_chunks  <- ceiling(n_rows / CHUNK_SIZE)
  preds     <- numeric(n_rows)
  
  cat(paste0("Predicting in ", n_chunks, " chunk(s)...\n"))
  
  for (chunk_i in seq_len(n_chunks)) {
    idx_start <- (chunk_i - 1) * CHUNK_SIZE + 1
    idx_end   <- min(chunk_i * CHUNK_SIZE, n_rows)
    
    chunk_data       <- test_set[idx_start:idx_end, , drop = FALSE]
    preds[idx_start:idx_end] <- predict(rf_model, newdata = chunk_data)
    
    cat(paste0("  Chunk ", chunk_i, "/", n_chunks,
               " (rows ", idx_start, "-", idx_end, ") done.\n"))
    
    rm(chunk_data); gc()
  }
  
  all_results[[yr_key]] <- preds
  
  # --- Free this year's objects before next iteration ---
  rm(rf_model, test_set, preds)
  gc()
  
  cat(paste0("[", yr, "] Complete.\n\n"))
}

# -----------------------------------------------------------------------------
# STEP 3: Assemble final output and write CSV
# -----------------------------------------------------------------------------

cat("== Assembling final results... ==\n")

pred_skeleton$consolidated <- NA_real_

for (yr_key in names(all_results)) {
  if (!is.null(all_results[[yr_key]])) {
    yr <- as.numeric(yr_key)
    pred_skeleton$consolidated[pred_skeleton$year == yr] <- all_results[[yr_key]]
  }
}

cat(paste0("Writing output to: ", OUTPUT_PATH, "\n"))
fwrite(pred_skeleton, OUTPUT_PATH)   # fwrite is much faster than write.csv for large files

cat("== Done! ==\n")
