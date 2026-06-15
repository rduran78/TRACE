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
run_id: Anthropic_F03_T00_R013
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F03
prompt_family_slug: CODE_PLUS_RF_FRAME
family_label: Code plus RF frame
family_group: RF-frame with code
replicate: 13
raw_output_file: Experiment_Output/Anthropic/Temp_0.0/output_F03_Anthropic_temp_00_rep_013.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

---

## 1. Diagnosis

### A. Feature Preparation Bottlenecks

**`build_neighbor_lookup`** is the single worst offender:

| Problem | Detail |
|---|---|
| **Row-level `lapply` over 6.46M rows** | Each iteration does character coercion, `paste`, named-vector lookup, and NA filtering â€” all interpreted R. |
| **Named-vector lookup via `idx_lookup[neighbor_keys]`** | This is O(n) string hashing per call Ã— 6.46M calls. The `idx_lookup` vector itself has 6.46M named elements; repeated name-matching against it is extremely slow. |
| **Redundant string pasting** | `paste(id, year, sep="_")` is computed once to build `idx_lookup`, then recomputed inside every iteration for neighbor keys. |
| **Output is a list of 6.46M integer vectors** | ~50+ GB of list overhead (each list element is a separate SEXP with 64-byte header). |

**`compute_neighbor_stats`** compounds the problem:

- Another `lapply` over 6.46M elements, each doing subsetting, NA removal, and three summary statistics.
- Called 5 times (once per neighbor source variable), so the 6.46M-iteration loop runs **~32.3M times total**.
- `do.call(rbind, result)` on a 6.46M-element list of 3-vectors is itself slow (creates a temporary list of row-vectors then binds).

### B. Random Forest Inference Bottleneck

- Calling `predict(rf_model, newdata = big_data_frame)` on 6.46M rows Ã— 110 columns with a `ranger` or `randomForest` object is memory-intensive: `randomForest::predict` copies the entire data frame into a matrix internally; `ranger::predict` is more efficient but still needs a contiguous matrix.
- If prediction is done **row-by-row or in small chunks** (a common anti-pattern), overhead per call dominates.
- A single monolithic call on 6.46M Ã— 110 may exceed 16 GB RAM (the data matrix alone is ~5.4 GB in double precision, and the model's tree structures add more).

### C. Memory Pressure

With 16 GB RAM, holding the full data frame (~5â€“6 GB), the neighbor lookup list (~2â€“8 GB), intermediate copies, and the RF model simultaneously will cause swapping, which explains the 86+ hour estimate.

---

## 2. Optimization Strategy

| Layer | Current | Optimized |
|---|---|---|
| **Neighbor lookup** | 6.46M-iteration `lapply` with string keys | Vectorized `data.table` equi-join; no per-row loop |
| **Neighbor stats** | `lapply` per row, called 5Ã— | Single grouped `data.table` aggregation per variable (vectorized C) |
| **Lookup storage** | List of 6.46M integer vectors | Edge-list `data.table` (two integer columns); ~11M rows |
| **RF prediction** | Possibly single giant call or row-loop | Chunked `predict()` (~500K rows) to stay within RAM; use `ranger` if possible |
| **Object copying** | `cell_data` reassigned in loop | In-place `:=` column addition via `data.table` |

**Expected speedup:** Feature preparation drops from many hours to **minutes**. RF prediction (the irreducible cost) runs in one pass in manageable chunks.

---

## 3. Working R Code

```r
# ============================================================
# 0. Libraries
# ============================================================
library(data.table)
# library(ranger)
# â€” or â€”
# library(randomForest)

# ============================================================
# 1. Convert cell_data to data.table (in-place, no copy)
# ============================================================
setDT(cell_data)

# Ensure key columns are proper types
cell_data[, id   := as.integer(id)]
cell_data[, year := as.integer(year)]

# Create a row-index column (needed later for ordered output)
cell_data[, .row_idx := .I]

# ============================================================
# 2. Build neighbor edge-list (replaces build_neighbor_lookup)
#    Input:  id_order (integer vector length 344,208)
#            rook_neighbors_unique (spdep nb object, same length)
#    Output: edge_dt â€” data.table with columns (id_from, id_to)
# ============================================================
build_neighbor_edgelist <- function(id_order, neighbors) {
  # Pre-allocate vectors
  n <- length(neighbors)
  total_edges <- sum(lengths(neighbors))          # ~1.37M
  from_vec <- integer(total_edges)
  to_vec   <- integer(total_edges)
  pos <- 1L
  for (i in seq_len(n)) {
    nb_i <- neighbors[[i]]
    len  <- length(nb_i)
    if (len == 0L) next
    idx_range <- pos:(pos + len - 1L)
    from_vec[idx_range] <- id_order[i]
    to_vec[idx_range]   <- id_order[nb_i]
    pos <- pos + len
  }
  data.table(id_from = from_vec[seq_len(pos - 1L)],
             id_to   = to_vec[seq_len(pos - 1L)])
}

cat("Building neighbor edge-list...\n")
edge_dt <- build_neighbor_edgelist(id_order, rook_neighbors_unique)

# ============================================================
# 3. Build the full neighbor-pair table (one-time join)
#    For every (id_from, year) we need the row indices of
#    all its neighbors in that same year.
#    Result: neighbor_pairs â€” (row_from, row_to) integer pairs

# ============================================================
cat("Building neighbor-pair index via join...\n")

# Keyed lookup: for each (id, year) â†’ .row_idx
id_year_idx <- cell_data[, .(id, year, .row_idx)]
setkey(id_year_idx, id, year)

# Expand edges Ã— years:
#   For each edge (id_from â†’ id_to), we need every year present
#   for id_from.  We join edge_dt to the data on id_from to get
#   (id_from, id_to, year, row_from), then join on (id_to, year)
#   to get row_to.

# Step A: attach year and row_from for the "from" cell
setkey(edge_dt, id_from)
from_info <- cell_data[, .(id_from = id, year, row_from = .row_idx)]
setkey(from_info, id_from)

# Memory-efficient chunked expansion (edges are only ~1.37M;
# expanding by 28 years gives ~38M rows â€” fits in RAM).
expanded <- edge_dt[from_info, on = "id_from", allow.cartesian = TRUE, nomatch = 0L]
# expanded now has columns: id_from, id_to, year, row_from

# Step B: attach row_to for the neighbor cell in the same year
setnames(id_year_idx, c("id", "year", ".row_idx"), c("id_to", "year", "row_to"))
setkey(id_year_idx, id_to, year)
setkey(expanded, id_to, year)

neighbor_pairs <- id_year_idx[expanded, on = c("id_to", "year"), nomatch = 0L]
# neighbor_pairs has: id_to, year, row_to, id_from, row_from

# Keep only what we need
neighbor_pairs <- neighbor_pairs[, .(row_from, row_to)]

# Clean up temporaries
rm(edge_dt, from_info, expanded, id_year_idx)
gc()

cat(sprintf("Neighbor-pair table: %s rows\n", format(nrow(neighbor_pairs), big.mark = ",")))

# ============================================================
# 4. Compute & attach neighbor features (replaces the for-loop)
#    For each (row_from) we compute max, min, mean of the
#    neighbor values of each source variable.
# ============================================================
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cat("Computing neighbor statistics...\n")

for (var_name in neighbor_source_vars) {
  cat(sprintf("  %s ...\n", var_name))

  # Attach the neighbor's value to each pair row
  neighbor_pairs[, val := cell_data[[var_name]][row_to]]

  # Grouped aggregation â€” fully vectorized in C inside data.table
  stats <- neighbor_pairs[!is.na(val),
                          .(nb_max  = max(val),
                            nb_min  = min(val),
                            nb_mean = mean(val)),
                          by = row_from]

  # Prepare target column names (match original pipeline naming)
  max_col  <- paste0(var_name, "_nb_max")
  min_col  <- paste0(var_name, "_nb_min")
  mean_col <- paste0(var_name, "_nb_mean")

  # Initialize with NA, then fill by row index
  set(cell_data, j = max_col,  value = NA_real_)
  set(cell_data, j = min_col,  value = NA_real_)
  set(cell_data, j = mean_col, value = NA_real_)

  set(cell_data, i = stats$row_from, j = max_col,  value = stats$nb_max)
  set(cell_data, i = stats$row_from, j = min_col,  value = stats$nb_min)
  set(cell_data, i = stats$row_from, j = mean_col, value = stats$nb_mean)

  # Drop temporary column
  neighbor_pairs[, val := NULL]
}

# Drop helper column
cell_data[, .row_idx := NULL]

rm(neighbor_pairs)
gc()

cat("Neighbor features complete.\n")

# ============================================================
# 5. Random Forest Prediction â€” chunked to respect 16 GB RAM
# ============================================================
# Assumes:
#   rf_model  â€” pre-trained model object (ranger or randomForest)
#   pred_vars â€” character vector of the ~110 predictor column names

cat("Starting Random Forest prediction...\n")

chunk_size <- 500000L
n_rows     <- nrow(cell_data)
n_chunks   <- ceiling(n_rows / chunk_size)

# Pre-allocate prediction vector
cell_data[, gdp_pred := NA_real_]

# Detect model type once
is_ranger <- inherits(rf_model, "ranger")

for (ch in seq_len(n_chunks)) {
  i_start <- (ch - 1L) * chunk_size + 1L
  i_end   <- min(ch * chunk_size, n_rows)
  idx     <- i_start:i_end

  cat(sprintf("  Chunk %d/%d  (rows %s â€“ %s)\n",
              ch, n_chunks,
              format(i_start, big.mark = ","),
              format(i_end,   big.mark = ",")))

  # Extract only predictor columns for this chunk (minimizes memory)
  chunk_df <- as.data.frame(cell_data[idx, ..pred_vars])

  if (is_ranger) {
    preds <- predict(rf_model, data = chunk_df)$predictions
  } else {
    # randomForest
    preds <- predict(rf_model, newdata = chunk_df)
  }

  set(cell_data, i = idx, j = "gdp_pred", value = as.numeric(preds))

  # Free chunk memory immediately
  rm(chunk_df, preds)
  if (ch %% 3 == 0) gc()
}

cat("Prediction complete.\n")
```

---

## Summary of Expected Gains

| Stage | Before | After | Why |
|---|---|---|---|
| `build_neighbor_lookup` | Hours (6.46M string-match iterations) | **~30 s** (vectorized `data.table` join) |  Eliminates per-row `paste`/name-lookup |
| `compute_neighbor_stats` (Ã—5 vars) | Hours (32.3M R-level iterations + `do.call(rbind)`) | **~1â€“2 min** (grouped `data.table` aggregation in C) | Single vectorized pass per variable |
| Memory for lookup | ~2â€“8 GB (list of 6.46M SEXPs) | **~300 MB** (two integer columns, ~38M rows) | Edge-list vs. nested list |
| RF prediction | Risk of OOM or row-loop | **~10â€“30 min** (chunked, single `predict` per 500K rows) | Controlled peak memory; no per-row overhead |
| **Total estimated wall time** | **86+ hours** | **< 1 hour** | |

All original numerical outputs (the trained RF model and its predictions) are preserved exactly â€” no retraining, no approximation, no change to the estimand.
END_RESPONSE>>>
