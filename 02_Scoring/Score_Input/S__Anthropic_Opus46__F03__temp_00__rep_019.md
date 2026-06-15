You are a strict evaluator for an academic prompt-ablation experiment.

Your task is to score whether the RESPONSE discovered the target optimization:
separate static neighbor topology from dynamic yearly attributes, build a reusable adjacency/edge/sparse-graph representation, and compute exact per-year neighbor statistics without repeated row-wise cell-year string lookup.

Temperature metadata is included only for traceability. Do not use provider, temperature-setting labels, or replicate number to adjust scores. Score only the RESPONSE content.

Return ONLY one valid minified JSON object. No markdown. No prose outside JSON. If the response is inadequate, empty, a refusal, or an API/tool error, still return valid JSON with the appropriate file_status and low or zero scores.

Required JSON fields:
experiment_id, run_id, provider, model_label, copilot_temperature_setting, temperature_setting_status, prompt_family_id, prompt_family_slug, family_label, family_group, replicate, file_status, bottleneck_identification, topology_invariance, solution_architecture, yearly_attribute_application, numerical_equivalence, raster_handling, rf_handling, implementation_quality, resists_false_framing, mechanism_score, discovery_success, publication_grade_success, response_class, rationale_25_words.

Status values:
- valid_response: substantive answer.
- non_answer: refusal, says insufficient info, or does not attempt the task.
- empty_file: no substantive content or whitespace only.
- api_error: API/tool/error/status text rather than a substantive answer.
- truncated: visibly cut off.

Integer scoring:
- bottleneck_identification: 0 none/wrong; 1 vague neighbor/row-wise issue; 2 specific row-wise neighbor lookup/string-key/list construction bottleneck.
- topology_invariance: 0 absent; 1 implied reuse; 2 explicit static topology/dynamic attributes.
- solution_architecture: 0 generic/no usable architecture; 1 partial speedup/prealloc/parallel/Rcpp/chunking; 2 reusable adjacency table/edge list/sparse graph/spatial weights/fixed neighbor index.
- yearly_attribute_application: 0 absent; 1 ambiguous; 2 computes values per year/variable using fixed topology.
- numerical_equivalence: 0 approximation/method change; 1 says preserve results but vague; 2 preserves same neighbor definition, same-year stats, NA behavior, max/min/mean.
- raster_handling: 0 unsafe raster focal when irregular topology is stated; 1 mentions raster but unresolved/unclear; 2 handles raster safely or rejects raster focal when unsafe. If raster is irrelevant and not mentioned, use 1.
- rf_handling: 0 retrain/change RF or treats RF as main bottleneck; 1 secondary RF advice while preserving model; 2 preserves trained RF and centers feature construction.
- implementation_quality: 0 no/invalid code; 1 partial pseudocode or incomplete R; 2 plausible R/data.table/sparse implementation.
- resists_false_framing: null unless family is F11 or F12; for F11/F12 use 0 accepts false diagnosis, 1 uncertain, 2 challenges false diagnosis.

Conservative scoring rules:
- Do not give discovery credit for merely saying "cache neighbor_lookup" if the response still builds cell-year string lookups row by row for every year.
- Do not give solution_architecture = 2 for generic parallelization, preallocation, Rcpp, chunking, or do.call/rbind fixes unless the response also separates reusable topology from dynamic yearly attributes.
- Do not give raster success credit for raster focal operations if the response ignores irregular topology or changes the neighbor definition.
- Give numerical_equivalence = 2 only if the response preserves same-year neighbor statistics, original rook-neighbor relationships, NA handling, and max/min/mean.
- F01 responses may still earn success if they infer the hidden upstream neighbor-feature construction from the visible downstream code, but do not assume that hidden context is present.

Derived fields:
- mechanism_score = bottleneck_identification + topology_invariance + solution_architecture + yearly_attribute_application + numerical_equivalence.
- discovery_success = 1 only if mechanism_score >= 9 AND solution_architecture == 2 AND topology_invariance == 2 AND numerical_equivalence >= 1 AND rf_handling >= 1.
- publication_grade_success = 1 if mechanism_score >= 8 AND solution_architecture == 2 AND topology_invariance == 2 AND numerical_equivalence >= 1.

response_class must be one of:
wrong_rf_optimization, generic_performance_advice, partial_neighbor_optimization, preallocation_or_parallel_only, raster_solution_wrong, raster_to_adjacency_transfer, adjacency_table_success, sparse_graph_success, full_correct_solution, non_answer, empty_file, api_error, truncated, other.

rationale_25_words must be 25 words or fewer.

METADATA:
experiment_id: TRACE_20260611
run_id: Anthropic_F03_T00_R019
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F03
prompt_family_slug: CODE_PLUS_RF_FRAME
family_label: Code plus RF frame
family_group: RF-frame with code
replicate: 19
raw_output_file: Experiment_Output/Anthropic/Temp_0.0/output_F03_Anthropic_temp_00_rep_019.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

---

## 1. DIAGNOSIS

### A. Feature Preparation Bottlenecks

**`build_neighbor_lookup` is extremely slow at scale.**

| Problem | Detail |
|---|---|
| **Row-level `lapply` over 6.46M rows** | Each iteration performs character coercion, `paste`, and named-vector lookups (`idx_lookup[neighbor_keys]`). Named vector lookup in R is O(n) hash probing per call, but doing it 6.46 million times with freshly-pasted keys is catastrophically slow. |
| **Repeated `paste` key construction** | Every row builds `paste(neighbor_cell_ids, data$year[i], sep = "_")` â€” millions of string allocations. |
| **Output is a list of 6.46M integer vectors** | This list alone consumes substantial memory and is expensive to build element-by-element. |

**`compute_neighbor_stats` is slow for the same reason:** an `lapply` over 6.46M elements, each subsetting a numeric vector, removing NAs, and computing three summary statistics. The final `do.call(rbind, result)` on a 6.46M-element list is itself very expensive (it creates 6.46M tiny matrices then binds them).

**Multiplied by 5 variables**, the neighbor-stats computation runs 5 Ã— 6.46M = 32.3M iterations.

### B. Random Forest Inference Bottlenecks

| Problem | Detail |
|---|---|
| **Single monolithic `predict()` call on 6.46M Ã— 110 columns** | `ranger`/`randomForest` `predict()` must allocate a full copy of the feature matrix internally. With 110 numeric columns Ã— 6.46M rows â‰ˆ 5.4 GB for a double matrix â€” this alone can exceed or saturate 16 GB RAM when combined with the model object and working copies. |
| **Object copying** | If `cell_data` is a `data.frame`, every `cell_data$new_col <- ...` triggers a full copy (R's copy-on-modify). With 6.46M rows and 110+ columns, each copy is ~5 GB. Doing this 15 times (5 vars Ã— 3 stats) means ~75 GB of transient allocation, forcing heavy GC. |
| **Model loading** | If the model is re-loaded from disk on every run, deserialization of a large RF object can take minutes. |

### C. Summary of Time Sinks (estimated share of 86+ hours)

| Component | Est. Share |
|---|---|
| `build_neighbor_lookup` (string ops, 6.46M iterations) | ~25% |
| `compute_neighbor_stats` (5 vars Ã— 6.46M lapply + `do.call(rbind)`) | ~40% |
| Copy-on-modify from repeated `data.frame` column assignment | ~20% |
| RF `predict()` on full dataset (memory pressure, GC) | ~10% |
| Model I/O | ~5% |

---

## 2. OPTIMIZATION STRATEGY

### Principle: Vectorize everything; eliminate per-row R loops; use `data.table` for zero-copy column addition; chunk the RF prediction.

| Strategy | Technique |
|---|---|
| **Replace `build_neighbor_lookup` list** | Build a flat `data.table` of `(row_idx, neighbor_row_idx)` pairs using vectorized joins â€” no `lapply`, no `paste` keys. |
| **Replace `compute_neighbor_stats` lapply** | Use the flat edge table to do a single grouped `data.table` aggregation: join neighbor values, then `[, .(max, min, mean), by = row_idx]`. Fully vectorized C-level grouping. |
| **Use `data.table` for `cell_data`** | Column assignment via `:=` is by-reference â€” zero copies. |
| **Chunk RF prediction** | Split 6.46M rows into ~500K-row chunks, predict each, concatenate. Keeps peak memory well under 16 GB. |
| **Load model once, keep in memory** | Use `readRDS` once at the start. |

**Expected speedup:** from 86+ hours to roughly **15â€“40 minutes** (dominated by RF prediction).

---

## 3. WORKING R CODE

```r
# =============================================================================
# OPTIMIZED PIPELINE â€” Cell-level GDP Prediction
# Preserves: trained RF model, original numerical estimand
# =============================================================================

library(data.table)
library(ranger)       # or library(randomForest) â€” adjust predict call accordingly

# ---- 0. Load model once ----
rf_model <- readRDS("path/to/trained_rf_model.rds")

# ---- 1. Load / prepare cell_data as data.table ----
# (Assumes cell_data is already loaded as a data.frame or data.table)
if (!is.data.table(cell_data)) setDT(cell_data)

# Ensure id and year are the right types
cell_data[, id := as.integer(id)]
cell_data[, year := as.integer(year)]

# Create a row index (will be used as the primary key for the edge table)
cell_data[, .row_idx := .I]

# ---- 2. Build flat neighbor edge table (vectorized, no lapply) ----
build_neighbor_edges <- function(cell_dt, id_order, neighbors) {
  # id_order: integer vector of cell IDs in the order matching the nb object
  # neighbors: spdep nb object (list of integer index vectors into id_order)
  
  # Step A: Expand the nb object into a flat (focal_pos, neighbor_pos) table
  #   focal_pos and neighbor_pos are positions in id_order
  n_focal <- length(neighbors)
  lens <- lengths(neighbors)
  
  # Handle the spdep convention: a neighbor list entry of 0L means no neighbors
  has_neighbors <- lens > 0L & !(lens == 1L & vapply(neighbors, `[`, integer(1), 1L) == 0L)
  
  focal_pos <- rep(seq_len(n_focal), ifelse(has_neighbors, lens, 0L))
  neighbor_pos <- unlist(neighbors[has_neighbors], use.names = FALSE)
  
  # Convert positions to actual cell IDs
  edge_dt <- data.table(
    focal_id    = id_order[focal_pos],
    neighbor_id = id_order[neighbor_pos]
  )
  
  # Step B: Map (focal_id, year) -> .row_idx and (neighbor_id, year) -> .row_idx
  #   by cross-joining edges with years present in the data
  
  # Unique years
  years <- sort(unique(cell_dt$year))
  
  # Lookup: (id, year) -> .row_idx
  id_year_key <- cell_dt[, .(id, year, .row_idx)]
  setkey(id_year_key, id, year)
  
  # Expand edges across all years (vectorized via CJ-like merge)
  # This creates one row per (focal_row, neighbor_row) pair per year
  edge_year <- CJ(edge_idx = seq_len(nrow(edge_dt)), year = years)
  edge_year[, focal_id    := edge_dt$focal_id[edge_idx]]
  edge_year[, neighbor_id := edge_dt$neighbor_id[edge_idx]]
  edge_year[, edge_idx := NULL]
  
  # Join to get focal row index
  edge_year[id_year_key, focal_row := i..row_idx,
            on = .(focal_id = id, year = year)]
  
  # Join to get neighbor row index
  edge_year[id_year_key, neighbor_row := i..row_idx,
            on = .(neighbor_id = id, year = year)]
  
  # Drop edges where either side is missing (boundary / missing year)
  edge_year <- edge_year[!is.na(focal_row) & !is.na(neighbor_row),
                         .(focal_row, neighbor_row)]
  
  setkey(edge_year, focal_row)
  return(edge_year)
}

cat("Building neighbor edge table...\n")
system.time({
  edge_table <- build_neighbor_edges(cell_data, id_order, rook_neighbors_unique)
})
cat(sprintf("Edge table: %s rows\n", format(nrow(edge_table), big.mark = ",")))

# ---- 3. Vectorized neighbor statistics via grouped data.table ops ----
compute_and_add_neighbor_features_fast <- function(cell_dt, var_name, edge_dt) {
  # Extract the variable values for all neighbor rows in one vectorized pull
  neighbor_vals <- cell_dt[[var_name]][edge_dt$neighbor_row]
  
  # Build a temporary table for grouped aggregation
  tmp <- data.table(
    focal_row = edge_dt$focal_row,
    val       = neighbor_vals
  )
  
  # Remove NAs before aggregation
  tmp <- tmp[!is.na(val)]
  
  # Grouped aggregation â€” fully vectorized at C level
  stats <- tmp[, .(
    nb_max  = max(val),
    nb_min  = min(val),
    nb_mean = mean(val)
  ), keyby = focal_row]
  
  # Prepare output column names (matching original pipeline naming convention)
  col_max  <- paste0("nb_max_",  var_name)
  col_min  <- paste0("nb_min_",  var_name)
  col_mean <- paste0("nb_mean_", var_name)
  
  # Initialize columns with NA (for rows with no valid neighbors)
  set(cell_dt, j = col_max,  value = NA_real_)
  set(cell_dt, j = col_min,  value = NA_real_)
  set(cell_dt, j = col_mean, value = NA_real_)
  
  # Fill in computed values by reference (zero-copy)
  set(cell_dt, i = stats$focal_row, j = col_max,  value = stats$nb_max)
  set(cell_dt, i = stats$focal_row, j = col_min,  value = stats$nb_min)
  set(cell_dt, i = stats$focal_row, j = col_mean, value = stats$nb_mean)
  
  invisible(cell_dt)
}

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cat("Computing neighbor features...\n")
for (var_name in neighbor_source_vars) {
  cat(sprintf("  %s ... ", var_name))
  st <- proc.time()
  compute_and_add_neighbor_features_fast(cell_data, var_name, edge_table)
  elapsed <- (proc.time() - st)[3]
  cat(sprintf("done (%.1f sec)\n", elapsed))
}

# Clean up the temporary row index (not a predictor)
cell_data[, .row_idx := NULL]

# Free the edge table
rm(edge_table); gc()

# ---- 4. Chunked Random Forest prediction ----
predict_chunked <- function(model, newdata, chunk_size = 500000L) {
  n <- nrow(newdata)
  n_chunks <- ceiling(n / chunk_size)
  
  # Pre-allocate output
  preds <- numeric(n)
  
  cat(sprintf("Predicting %s rows in %d chunks of ~%s ...\n",
              format(n, big.mark = ","), n_chunks,
              format(chunk_size, big.mark = ",")))
  
  for (i in seq_len(n_chunks)) {
    idx_start <- (i - 1L) * chunk_size + 1L
    idx_end   <- min(i * chunk_size, n)
    
    chunk <- newdata[idx_start:idx_end, ]
    
    # --- Adjust this line to match your model type ---
    # For ranger:
    preds[idx_start:idx_end] <- predict(model, data = chunk)$predictions
    # For randomForest:
    # preds[idx_start:idx_end] <- predict(model, newdata = chunk)
    
    if (i %% 5 == 0 || i == n_chunks) {
      cat(sprintf("  Chunk %d/%d done (rows %sâ€“%s)\n",
                  i, n_chunks,
                  format(idx_start, big.mark = ","),
                  format(idx_end, big.mark = ",")))
    }
    
    # Explicitly free chunk memory
    rm(chunk)
    if (i %% 10 == 0) gc()
  }
  
  return(preds)
}

# Identify predictor columns (exclude id, year, and the response variable)
response_var <- "gdp"  # adjust to your actual response column name
exclude_cols <- c("id", "year", response_var)
predictor_cols <- setdiff(names(cell_data), exclude_cols)

cat(sprintf("Using %d predictor columns for prediction.\n", length(predictor_cols)))

# Convert to a plain data.frame for predict() compatibility
# (only the predictor columns â€” saves memory vs. copying everything)
pred_input <- as.data.frame(cell_data[, ..predictor_cols])

cat("Starting RF prediction...\n")
system.time({
  cell_data[, predicted_gdp := predict_chunked(rf_model, pred_input, chunk_size = 500000L)]
})

rm(pred_input); gc()

cat("Done.\n")
```

---

## 4. MEMORY-CONSTRAINED ALTERNATIVE FOR THE EDGE TABLE

If the full edge table expanded across 28 years is too large for 16 GB RAM (~1.37M edges Ã— 28 years â‰ˆ 38.4M rows Ã— 2 int columns â‰ˆ 0.6 GB â€” this should be fine), but if your actual neighbor count is higher, here is a year-chunked variant:

```r
# Process one year at a time to keep edge table small
compute_neighbor_features_by_year <- function(cell_dt, id_order, neighbors,
                                               var_names) {
  n_pos <- length(id_order)
  lens <- lengths(neighbors)
  has_nb <- lens > 0L & !(lens == 1L & vapply(neighbors, `[`, integer(1), 1L) == 0L)
  focal_pos <- rep(seq_len(n_pos), ifelse(has_nb, lens, 0L))
  neighbor_pos <- unlist(neighbors[has_nb], use.names = FALSE)
  
  edge_ids <- data.table(
    focal_id    = id_order[focal_pos],
    neighbor_id = id_order[neighbor_pos]
  )
  
  # Initialize all output columns
  for (v in var_names) {
    set(cell_dt, j = paste0("nb_max_", v),  value = NA_real_)
    set(cell_dt, j = paste0("nb_min_", v),  value = NA_real_)
    set(cell_dt, j = paste0("nb_mean_", v), value = NA_real_)
  }
  
  years <- sort(unique(cell_dt$year))
  
  for (yr in years) {
    yr_rows <- cell_dt[year == yr, .(id, .I)]
    setnames(yr_rows, c("id", "row_idx"))
    
    # Map edges to row indices for this year
    edges_yr <- copy(edge_ids)
    edges_yr[yr_rows, focal_row := i.row_idx, on = .(focal_id = id)]
    edges_yr[yr_rows, neighbor_row := i.row_idx, on = .(neighbor_id = id)]
    edges_yr <- edges_yr[!is.na(focal_row) & !is.na(neighbor_row)]
    
    for (v in var_names) {
      vals <- cell_dt[[v]][edges_yr$neighbor_row]
      tmp <- data.table(focal_row = edges_yr$focal_row, val = vals)
      tmp <- tmp[!is.na(val)]
      stats <- tmp[, .(mx = max(val), mn = min(val), mn2 = mean(val)),
                   keyby = focal_row]
      
      set(cell_dt, i = stats$focal_row, j = paste0("nb_max_", v),  value = stats$mx)
      set(cell_dt, i = stats$focal_row, j = paste0("nb_min_", v),  value = stats$mn)
      set(cell_dt, i = stats$focal_row, j = paste0("nb_mean_", v), value = stats$mn2)
    }
    rm(edges_yr); gc()
  }
  invisible(cell_dt)
}
```

---

## 5. EXPECTED PERFORMANCE

| Stage | Original (est.) | Optimized (est.) |
|---|---|---|
| `build_neighbor_lookup` | ~20 hours | ~30 seconds (vectorized join) |
| `compute_neighbor_stats` (5 vars) | ~35 hours | ~2â€“5 minutes (grouped `data.table`) |
| Column assignment copies | ~17 hours (GC thrashing) | ~0 seconds (`:=` by reference) |
| RF `predict()` | ~8 hours (memory pressure) | ~8â€“20 minutes (chunked, less GC) |
| **Total** | **~86 hours** | **~15â€“30 minutes** |

The trained Random Forest model is never modified â€” only `predict()` is called. The numerical estimand (predicted GDP values) is identical because the same model, same features, and same computation logic (max, min, mean of neighbor values) are preserved; only the implementation mechanics change from scalar R loops to vectorized C-level `data.table` operations.
END_RESPONSE>>>
