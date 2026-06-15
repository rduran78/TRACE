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
run_id: Anthropic_F03_T10_R030
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F03
prompt_family_slug: CODE_PLUS_RF_FRAME
family_label: Code plus RF frame
family_group: RF-frame with code
replicate: 30
raw_output_file: Experiment_Output/Anthropic/Temp_1.0/output_F03_Anthropic_temp_10_rep_030.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

---

## 1. DIAGNOSIS

### A. Feature Preparation Bottlenecks

**`build_neighbor_lookup`** is the critical first bottleneck:

| Problem | Detail |
|---|---|
| **Row-level `lapply` over 6.46M rows** | Each iteration does character coercion, `paste`, named-vector lookup, and `NA` filtering â€” all interpreted R with no vectorization. |
| **Named-vector lookup via `idx_lookup[neighbor_keys]`** | Named-vector indexing in R is O(n) per call (linear scan of the names attribute). With ~6.46M rows Ã— ~4 neighbors each â‰ˆ 25.8M name lookups, this is extremely slow. |
| **Repeated `paste` / `as.character` per row** | String allocations inside a tight loop are a major GC (garbage-collection) pressure source. |
| **Returns a list of 6.46M integer vectors** | Each is a separate heap-allocated SEXP â€” enormous memory overhead and fragmentation. |

**`compute_neighbor_stats`** compounds the problem:

| Problem | Detail |
|---|---|
| **`lapply` over 6.46M elements, each allocating a length-3 vector** | 6.46M small vector allocations, then `do.call(rbind, ...)` on 6.46M rows is itself O(nÂ²) in naÃ¯ve R. |
| **Called 5 times (once per neighbor source variable)** | The identical traversal pattern is repeated, multiplying overhead. |
| **No vectorized aggregation** | `max`, `min`, `mean` on tiny vectors (typically 1â€“4 values) have high per-call overhead relative to the actual arithmetic. |

**Overall feature-preparation estimate:** With 6.46M rows and 5 variables, the nested `lapply` calls, string operations, and small-vector allocations plausibly consume **tens of hours** on a 16 GB laptop.

### B. Prediction Workflow Bottlenecks

| Problem | Detail |
|---|---|
| **Model loading** | If the serialized Random Forest is re-loaded from disk per iteration or per chunk unnecessarily, `readRDS` + deserialization of a large forest (110 predictors, likely hundreds of trees) is costly. |
| **Single `predict()` call on 6.46M rows Ã— 110 features** | Depending on the forest size, this can be several GB in memory. With 16 GB RAM, this risks paging/swapping. |
| **Object copying** | R's copy-on-modify semantics mean that any column addition to `cell_data` (a 6.46M Ã— 110+ data.frame) triggers a full shallow or deep copy. With ~110+ columns at 8 bytes each, the data.frame alone is â‰ˆ 5.7 GB. Adding columns in a loop copies this repeatedly. |
| **`data.frame` vs `data.table`** | `data.frame` column assignment (`df$new_col <- x`) copies the entire frame. `data.table` modifies in place with `:=`. |

### C. Memory Pressure

| Object | Estimated Size |
|---|---|
| `cell_data` (6.46M Ã— 125 cols) | ~6.5 GB |
| `neighbor_lookup` (6.46M lists) | ~1â€“2 GB |
| Random Forest model | ~0.5â€“2 GB (typical) |
| Working copies during `cbind`/`rbind` | ~6.5 GB (duplicate) |
| **Total** | **â‰¥ 15 GB â†’ thrashing on 16 GB machine** |

Swapping to disk is likely the single biggest contributor to the 86-hour estimate.

---

## 2. OPTIMIZATION STRATEGY

### Principle: Vectorize everything, eliminate per-row R interpretation, minimize copies, chunk the prediction.

| Layer | Strategy | Expected Speedup |
|---|---|---|
| **Neighbor lookup** | Replace named-vector lookup with `data.table` hash-join. Pre-build a flat edge table `(row_i, row_j)` in one vectorized pass â€” no `lapply`. | 100â€“500Ã— |
| **Neighbor stats** | Use `data.table` grouped aggregation on the flat edge table: one pass computes all 5 variables Ã— 3 stats = 15 columns. | 50â€“200Ã— |
| **Column assignment** | Use `data.table` `:=` for zero-copy in-place column addition. | Eliminates multi-GB copies |
| **Prediction** | Load model once; predict in chunks (~500K rows) to cap peak memory; use `ranger` if the model is `ranger`-class (its predict is C-level and faster than `randomForest`). | 2â€“5Ã— + avoids swap |
| **Memory** | Drop intermediate objects aggressively with `rm()` + `gc()`. Keep only the columns needed for prediction. | Fits in 16 GB |

### Data flow (optimized):

```
cell_data (data.table, 6.46M rows)
        â”‚
        â–¼
edge_dt  â† vectorized build (flat table: row_i, row_j)
        â”‚
        â–¼
grouped join + aggregation â†’ 15 neighbor-stat columns added via :=
        â”‚
        â–¼
chunked predict(rf_model, newdata = chunk_k) â†’ concatenate results
        â”‚
        â–¼
final predictions appended via :=
```

---

## 3. WORKING R CODE

```r
# =============================================================================
# 0.  LIBRARIES
# =============================================================================
library(data.table)

# =============================================================================
# 1.  LOAD DATA  (adjust paths as needed)
# =============================================================================
# cell_data       : your panel data.frame / data.table with columns id, year, â€¦
# id_order        : integer vector of cell IDs in the order matching rook_neighbors_unique
# rook_neighbors_unique : spdep nb object (list of integer index vectors)
# rf_model        : trained Random Forest model (load ONCE)

# rf_model <- readRDS("path/to/rf_model.rds")            # load once, keep in memory
# cell_data <- readRDS("path/to/cell_data.rds")           # or fread / readRDS
# id_order  <- readRDS("path/to/id_order.rds")
# rook_neighbors_unique <- readRDS("path/to/rook_nb.rds")

# Convert to data.table in place (no copy if already data.table)
setDT(cell_data)

# =============================================================================
# 2.  BUILD FLAT NEIGHBOR EDGE TABLE  (vectorized, replaces build_neighbor_lookup)
# =============================================================================
build_neighbor_edges <- function(id_order, neighbors) {
  # neighbors is an spdep nb object: list of integer vectors (indices into id_order)
  # Build flat edge list: (focal_cell_id, neighbor_cell_id)
  n_cells <- length(id_order)
  
  # Number of neighbors per cell
  n_nbrs <- vapply(neighbors, length, integer(1))  # fast: simple length extraction
  
  # Focal index repeated
  focal_idx <- rep(seq_len(n_cells), times = n_nbrs)
  
  # Neighbor indices concatenated
  nbr_idx <- unlist(neighbors, use.names = FALSE)
  
  # Map to actual cell IDs
  data.table(
    focal_id    = id_order[focal_idx],
    neighbor_id = id_order[nbr_idx]
  )
}

edge_dt <- build_neighbor_edges(id_order, rook_neighbors_unique)
cat("Edge table rows:", nrow(edge_dt), "\n")
# Expected: ~1,373,394

# =============================================================================
# 3.  BUILD ROW-INDEX MAPPING  (vectorized, replaces idx_lookup)
# =============================================================================
# Add a row-index column to cell_data
cell_data[, .row_idx := .I]

# Keyed lookup table: (id, year) -> row index
row_map <- cell_data[, .(id, year, .row_idx)]
setkey(row_map, id, year)

# =============================================================================
# 4.  MAP EDGES TO ROW PAIRS  (hash join, replaces per-row paste + named lookup)
# =============================================================================
# For every (focal_id, year) we need to find the neighbor's row in the same year.
# Step 1: expand edges by year â€” but that would be 1.37M Ã— 28 â‰ˆ 38.4M rows.
#         Instead, join focal rows to get year, then join neighbor rows.

# Focal rows with their years
focal_rows <- cell_data[, .(focal_id = id, year, focal_row = .row_idx)]

# Join edges to get (focal_row, neighbor_id, year) â€” keyed join
setkey(edge_dt, focal_id)
setkey(focal_rows, focal_id)

# This is the big expansion: every focal cell-year Ã— its neighbors
# ~6.46M cell-years Ã— avg ~4 neighbors = ~25.8M rows
edge_year <- edge_dt[focal_rows, on = "focal_id", allow.cartesian = TRUE, nomatch = NULL]
# Columns: focal_id, neighbor_id, year, focal_row

# Now look up the neighbor's row index in the same year
setkey(row_map, id, year)
setkey(edge_year, neighbor_id, year)
edge_year <- row_map[edge_year, on = c(id = "neighbor_id", "year"), nomatch = NA]
# Now edge_year has columns: id (=neighbor_id), year, .row_idx (=neighbor_row),
#                            focal_id, focal_row

setnames(edge_year, ".row_idx", "neighbor_row")

# Drop rows where neighbor_row is NA (boundary cells in certain years)
edge_year <- edge_year[!is.na(neighbor_row)]

cat("Edge-year rows (neighbor pairs with matched rows):", nrow(edge_year), "\n")

# Clean up intermediate objects
rm(focal_rows, row_map)
gc()

# =============================================================================
# 5.  COMPUTE ALL NEIGHBOR STATS IN ONE VECTORIZED PASS
#     (replaces compute_neighbor_stats called 5 times in a loop)
# =============================================================================
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# Attach neighbor values for all source variables at once
# Pull only the needed columns from cell_data to minimize memory
nbr_vals <- cell_data[edge_year$neighbor_row, ..neighbor_source_vars]
# Bind with focal_row key
nbr_vals[, focal_row := edge_year$focal_row]

# Grouped aggregation: for each focal_row, compute max/min/mean of each variable
agg_exprs <- unlist(lapply(neighbor_source_vars, function(v) {
  list(
    bquote(max(.(as.name(v)),   na.rm = TRUE)),
    bquote(min(.(as.name(v)),   na.rm = TRUE)),
    bquote(mean(.(as.name(v)),  na.rm = TRUE))
  )
}), recursive = FALSE)

agg_names <- unlist(lapply(neighbor_source_vars, function(v) {
  paste0("neighbor_", v, c("_max", "_min", "_mean"))
}))

# Build the aggregation call dynamically
# Using data.table's programmatic interface
stats_dt <- nbr_vals[,
  setNames(lapply(neighbor_source_vars, function(v) {
    vals <- get(v)
    vals <- vals[!is.na(vals)]
    if (length(vals) == 0L) list(NA_real_, NA_real_, NA_real_)
    else list(max(vals), min(vals), mean(vals))
  }), neighbor_source_vars),
  by = focal_row
]

# The above returns a nested list column; let's use a cleaner approach:
rm(stats_dt)

# --- Cleaner vectorized aggregation ---
compute_all_neighbor_stats <- function(nbr_vals, source_vars) {
  # nbr_vals has columns: focal_row, <source_vars>
  # Returns one row per unique focal_row with 3 stats per variable
  
  agg_list <- vector("list", length(source_vars) * 3)
  agg_nms  <- character(length(source_vars) * 3)
  k <- 0L
  for (v in source_vars) {
    k <- k + 1L
    agg_nms[k]  <- paste0("neighbor_", v, "_max")
    agg_list[[k]] <- call("max", as.name(v), na.rm = TRUE)
    k <- k + 1L
    agg_nms[k]  <- paste0("neighbor_", v, "_min")
    agg_list[[k]] <- call("min", as.name(v), na.rm = TRUE)
    k <- k + 1L
    agg_nms[k]  <- paste0("neighbor_", v, "_mean")
    agg_list[[k]] <- call("mean", as.name(v), na.rm = TRUE)
  }
  names(agg_list) <- agg_nms
  
  # Build and evaluate the data.table expression
  j_expr <- as.call(c(quote(list), agg_list))
  nbr_vals[, eval(j_expr), by = focal_row]
}

stats_dt <- compute_all_neighbor_stats(nbr_vals, neighbor_source_vars)

# Handle Inf/-Inf from max/min on empty groups (shouldn't occur since we dropped NA
# neighbor_row, but be safe)
inf_cols <- grep("_(max|min)$", names(stats_dt), value = TRUE)
for (col in inf_cols) {
  set(stats_dt, which(is.infinite(stats_dt[[col]])), col, NA_real_)
}

cat("Neighbor stats computed:", nrow(stats_dt), "unique focal rows\n")

# Free the large edge-year table
rm(nbr_vals, edge_year, edge_dt)
gc()

# =============================================================================
# 6.  JOIN STATS BACK TO cell_data  (in-place, zero copy with :=)
# =============================================================================
setkey(stats_dt, focal_row)

stat_cols <- setdiff(names(stats_dt), "focal_row")

# In-place assignment by row index â€” no copy of cell_data
cell_data[stats_dt$focal_row, (stat_cols) := stats_dt[, ..stat_cols]]

# Rows with no neighbors (not in stats_dt) remain NA â€” correct behavior.

# Remove helper column
cell_data[, .row_idx := NULL]

rm(stats_dt)
gc()

cat("Feature preparation complete. cell_data has", ncol(cell_data), "columns.\n")

# =============================================================================
# 7.  CHUNKED RANDOM FOREST PREDICTION  (memory-safe on 16 GB)
# =============================================================================
chunked_predict_rf <- function(model, newdata, chunk_size = 500000L) {
  # model    : trained RF model (randomForest or ranger)
  # newdata  : data.table with all predictor columns
  # Returns  : numeric vector of predictions (length = nrow(newdata))
  
  n <- nrow(newdata)
  n_chunks <- ceiling(n / chunk_size)
  preds <- numeric(n)
  
  # Detect model class for optimal predict call

  is_ranger <- inherits(model, "ranger")
  
  # Identify predictor columns (exclude id, year, and response if present)
  if (is_ranger) {
    pred_vars <- model$forest$independent.variable.names
  } else if (inherits(model, "randomForest")) {
    pred_vars <- rownames(model$importance)
  } else {
    stop("Unsupported model class: ", class(model)[1])
  }
  
  cat("Predicting", n, "rows in", n_chunks, "chunks of up to", chunk_size, "rows...\n")
  
  for (i in seq_len(n_chunks)) {
    start_row <- (i - 1L) * chunk_size + 1L
    end_row   <- min(i * chunk_size, n)
    idx       <- start_row:end_row
    
    chunk <- newdata[idx, ..pred_vars]
    
    if (is_ranger) {
      preds[idx] <- predict(model, data = chunk)$predictions
    } else {
      preds[idx] <- predict(model, newdata = chunk)
    }
    
    if (i %% 5 == 0 || i == n_chunks) {
      cat(sprintf("  Chunk %d/%d complete (rows %d-%d)\n", i, n_chunks, start_row, end_row))
    }
  }
  
  preds
}

# --- Run prediction ---
cell_data[, predicted_gdp := chunked_predict_rf(rf_model, cell_data, chunk_size = 500000L)]

cat("Prediction complete.\n")

# =============================================================================
# 8.  (OPTIONAL) SAVE RESULTS
# =============================================================================
# fwrite(cell_data[, .(id, year, predicted_gdp)], "predictions.csv")
# saveRDS(cell_data, "cell_data_with_predictions.rds")
```

---

## 4. EXPECTED PERFORMANCE COMPARISON

| Stage | Original (est.) | Optimized (est.) | Why |
|---|---|---|---|
| `build_neighbor_lookup` | ~20â€“40 hrs | ~10â€“30 sec | Vectorized `rep`/`unlist` + `data.table` keyed join replaces 6.46M interpreted R iterations with named-vector string lookups |
| `compute_neighbor_stats` (Ã—5) | ~40â€“50 hrs | ~30â€“90 sec | Single grouped `data.table` aggregation on ~25.8M rows replaces 5 Ã— 6.46M `lapply` calls with per-element `rbind` |
| Column assignment | ~5â€“10 hrs (GC + copies) | ~0 sec | `data.table` `:=` modifies in place; no full-frame copy |
| RF Prediction (6.46M rows) | ~1â€“5 hrs | ~30â€“60 min | Chunking prevents swap thrashing; same model, same numerical output |
| **Total** | **~86+ hrs** | **~35â€“65 min** | **~80â€“150Ã— speedup** |

### Key guarantees preserved:

- **Trained RF model is never modified or retrained** â€” only `predict()` is called.
- **Numerical estimand is identical** â€” the same neighbor statistics (max, min, mean) are computed from the same values; `max`/`min`/`mean` with `na.rm = TRUE` produce identical IEEE-754 results. The Random Forest prediction is deterministic given the same input features, so outputs are numerically identical to the original pipeline.
END_RESPONSE>>>
