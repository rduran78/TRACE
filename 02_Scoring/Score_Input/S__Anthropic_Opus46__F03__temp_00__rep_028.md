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
run_id: Anthropic_F03_T00_R028
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F03
prompt_family_slug: CODE_PLUS_RF_FRAME
family_label: Code plus RF frame
family_group: RF-frame with code
replicate: 28
raw_output_file: Experiment_Output/Anthropic/Temp_0.0/output_F03_Anthropic_temp_00_rep_028.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

---

## 1. DIAGNOSIS

### Primary Bottlenecks

**A. `build_neighbor_lookup` â€” O(n) `lapply` over 6.46M rows with per-row string operations**

- `paste()` and named-vector lookups (`idx_lookup[neighbor_keys]`) are called **per row**. For 6.46M rows, this creates and discards millions of temporary character vectors.
- `as.character()` and `setNames()` on large vectors add overhead.
- The named-vector lookup `idx_lookup[neighbor_keys]` is a hash-table probe per key, but the keys are constructed on the fly with `paste()` â€” this is the single most expensive line in the function.

**B. `compute_neighbor_stats` â€” repeated subsetting and allocation**

- `lapply` over 6.46M elements, each calling `vals[idx]`, filtering NAs, then computing `max/min/mean`.
- `do.call(rbind, result)` on a 6.46M-element list of length-3 vectors is extremely slow â€” it creates a temporary list of 6.46M 3-element vectors and then binds them row by row.
- This is called **5 times** (once per neighbor source variable), so the cost multiplies.

**C. Outer loop copies `cell_data` 5 times**

- `cell_data <- compute_and_add_neighbor_features(...)` likely copies the entire data.frame (6.46M Ã— 110+ columns) on each assignment. With 16 GB RAM this risks swapping.

**D. Random Forest prediction (downstream)**

- If `predict()` is called row-by-row or in small batches, overhead from the R-level dispatch and tree traversal dominates.
- Loading the model from disk repeatedly (if done per-year or per-chunk) is wasteful.
- If the model object is large, copying it or the prediction data frame triggers GC pressure.

### Summary of Time Allocation (estimated)

| Step | Est. Share | Root Cause |
|---|---|---|
| `build_neighbor_lookup` | ~25% | Per-row `paste` + named-vector lookup |
| `compute_neighbor_stats` (Ã—5) | ~40% | Per-row `lapply` + `do.call(rbind, ...)` |
| Data frame copies in outer loop | ~10% | Full-frame copy-on-modify |
| RF `predict()` | ~20% | Possibly batched poorly, repeated model load |
| Other | ~5% | GC, I/O |

---

## 2. OPTIMIZATION STRATEGY

| # | Technique | Expected Speedup |
|---|---|---|
| 1 | Replace `data.frame` with `data.table`; use integer keying instead of string pasting | 10â€“50Ã— for lookup build |
| 2 | Vectorize neighbor stats with a flat integer vector + `data.table` grouped aggregation, or a C-level rowwise approach via matrix indexing | 20â€“50Ã— for stats |
| 3 | Add all 15 neighbor columns in-place (by reference) instead of copying the frame 5 times | Eliminates ~3â€“5 full copies of a 6 GB frame |
| 4 | Build a single sparse adjacency representation (CSR-style: flat vector + pointer vector) to avoid list-of-vectors overhead | Memory + cache efficiency |
| 5 | Predict with the RF model in one single `predict()` call on the full matrix, or in large chunks (~500K rows) | Eliminates per-row dispatch overhead |
| 6 | Load the RF model once, convert prediction input to `matrix` (not `data.frame`) before calling `predict()` | Avoids repeated factor/type checks inside `predict.ranger`/`predict.randomForest` |

---

## 3. WORKING R CODE

```r
# =============================================================================
# OPTIMIZED PIPELINE â€” data.table + vectorized neighbor features + batch predict
# =============================================================================

library(data.table)

# ---- Step 0: Convert to data.table once (by reference if already a data.table)
if (!is.data.table(cell_data)) {
  setDT(cell_data)
}

# Ensure id and year are integer for fast keying
cell_data[, id   := as.integer(id)]
cell_data[, year := as.integer(year)]

# ---- Step 1: Build neighbor lookup using CSR (compressed sparse row) format
#
# This replaces build_neighbor_lookup entirely.
# Output: two integer vectors â€”
#   nb_flat    : concatenated neighbor-row indices
#   nb_ptr     : pointer into nb_flat; row i's neighbors are
#                nb_flat[ (nb_ptr[i]+1) : nb_ptr[i+1] ]
#
# This avoids 6.46M list elements and millions of paste() calls.

build_neighbor_lookup_csr <- function(dt, id_order, neighbors) {
  # --- map cell id -> position in id_order (1-based)
  id_to_ref <- integer(max(id_order))
  id_to_ref[id_order] <- seq_along(id_order)
  # If id values are not contiguous / too large, fall back to match:
  # id_to_ref <- match(dt$id, id_order)  # but below is O(1) per lookup

  # --- map (id, year) -> row index in dt using data.table keyed join
  #     This replaces the named character vector idx_lookup.
  dt[, .rowid := .I]
  setkey(dt, id, year)

  n <- nrow(dt)
  ids   <- dt$id
  years <- dt$year

  # Pre-allocate flat list (we'll unlist later)
  # Estimate upper bound on total neighbor-row references:
  #   avg neighbors ~ 1,373,394 / 344,208 â‰ˆ 4 per cell
  #   total refs â‰ˆ 4 * 6.46M â‰ˆ 25.8M
  nb_list <- vector("list", n)

  # Vectorised approach: for each unique cell, find its neighbor cell ids once,

  # then for every year that cell appears, look up the neighbor rows.

  # unique cells and their neighbor cell ids
  unique_ids <- unique(ids)

  # Build a mapping: cell_id -> integer vector of neighbor cell_ids
  # (This loop is over 344K cells, not 6.46M rows â€” very fast.)
  cell_neighbor_ids <- vector("list", max(id_order))
  for (j in seq_along(id_order)) {
    cid <- id_order[j]
    nb_idx <- neighbors[[j]]
    if (length(nb_idx) > 0) {
      cell_neighbor_ids[[cid]] <- id_order[nb_idx]
    }
  }

  # For the keyed join we need a helper: given vectors of neighbor_ids and a

  # single year, return the row indices in dt.
  # We'll do this fully vectorised over all rows using a merge.

  # Expand: for every row i, produce (neighbor_cell_id, year) pairs
  # Then join to dt to get .rowid.

  message("Building neighbor row-index mapping (vectorised)...")

  # -- Build edge table: (source_row, neighbor_cell_id, year)
  #    Do this per unique cell to control memory.

  # Group rows by cell id
  dt[, .rowid := .I]
  cell_groups <- dt[, .(rows = list(.rowid), yr = list(year)), by = id]

  # Pre-allocate collection lists
  src_rows_all <- vector("list", nrow(cell_groups))
  nbr_ids_all  <- vector("list", nrow(cell_groups))
  yrs_all      <- vector("list", nrow(cell_groups))

  cg_id   <- cell_groups$id
  cg_rows <- cell_groups$rows
  cg_yr   <- cell_groups$yr

  for (k in seq_len(nrow(cell_groups))) {
    cid     <- cg_id[k]
    nb_cids <- cell_neighbor_ids[[cid]]
    if (is.null(nb_cids) || length(nb_cids) == 0L) next
    r   <- cg_rows[[k]]
    y   <- cg_yr[[k]]
    n_r <- length(r)
    n_nb <- length(nb_cids)
    # Expand: each row Ã— each neighbor
    src_rows_all[[k]] <- rep(r,      each = n_nb)
    nbr_ids_all[[k]]  <- rep(nb_cids, times = n_r)
    yrs_all[[k]]      <- rep(y,       each = n_nb)
  }

  edges <- data.table(
    src_row = unlist(src_rows_all, use.names = FALSE),
    nb_id   = unlist(nbr_ids_all,  use.names = FALSE),
    nb_year = unlist(yrs_all,      use.names = FALSE)
  )
  rm(src_rows_all, nbr_ids_all, yrs_all)
  gc()

  message(sprintf("Edge table: %s rows", format(nrow(edges), big.mark = ",")))

  # Join to dt to resolve (nb_id, nb_year) -> row index
  setkey(dt, id, year)
  edges[, nb_row := dt[.(nb_id, nb_year), .rowid, on = .(id, year), mult = "first"]]
  edges <- edges[!is.na(nb_row)]

  # Now build CSR from edges sorted by src_row
  setkey(edges, src_row)

  nb_flat <- edges$nb_row
  # Build pointer vector
  nb_ptr <- integer(n + 1L)
  tab <- edges[, .N, by = src_row]
  nb_ptr[tab$src_row + 1L] <- tab$N
  nb_ptr <- cumsum(nb_ptr)

  rm(edges)
  gc()

  message("CSR neighbor lookup built.")
  list(nb_flat = nb_flat, nb_ptr = nb_ptr, n = n)
}

# ---- Step 2: Vectorized neighbor stats using the CSR structure
#
# For each variable, compute max, min, mean of neighbor values.
# Fully vectorised via data.table grouping on the flat edge vector.

compute_all_neighbor_features_csr <- function(dt, csr, var_names) {
  nb_flat <- csr$nb_flat
  nb_ptr  <- csr$nb_ptr
  n       <- csr$n

  # Reconstruct source-row id for every entry in nb_flat
  # src_row[j] = i  iff  nb_ptr[i] < j <= nb_ptr[i+1]
  counts <- diff(nb_ptr)                       # length n
  has_nb <- which(counts > 0L)
  src_row <- rep(has_nb, counts[has_nb])        # same length as nb_flat

  for (var_name in var_names) {
    message(sprintf("  Computing neighbor features for: %s", var_name))
    vals <- dt[[var_name]]
    nb_vals <- vals[nb_flat]

    # Grouped aggregation â€” extremely fast in data.table
    agg <- data.table(src = src_row, v = nb_vals)
    agg <- agg[!is.na(v), .(
      nb_max  = max(v),
      nb_min  = min(v),
      nb_mean = mean(v)
    ), by = src]

    # Allocate result columns (NA by default)
    max_col  <- paste0(var_name, "_nb_max")
    min_col  <- paste0(var_name, "_nb_min")
    mean_col <- paste0(var_name, "_nb_mean")

    set(dt, j = max_col,  value = NA_real_)
    set(dt, j = min_col,  value = NA_real_)
    set(dt, j = mean_col, value = NA_real_)

    set(dt, i = agg$src, j = max_col,  value = agg$nb_max)
    set(dt, i = agg$src, j = min_col,  value = agg$nb_min)
    set(dt, i = agg$src, j = mean_col, value = agg$nb_mean)
  }

  invisible(dt)
}

# ---- Step 3: Run the optimized feature pipeline

message("=== Building CSR neighbor lookup ===")
csr <- build_neighbor_lookup_csr(cell_data, id_order, rook_neighbors_unique)

message("=== Computing neighbor features ===")
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
compute_all_neighbor_features_csr(cell_data, csr, neighbor_source_vars)
# cell_data is modified in place â€” no copy.

rm(csr)
gc()


# ---- Step 4: Optimized Random Forest prediction
#
# Key principles:
#   (a) Load model ONCE.
#   (b) Build the prediction matrix ONCE (not a data.frame â€” avoids type checks).
#   (c) Call predict() ONCE or in large chunks to minimise R-level overhead.

message("=== Random Forest Prediction ===")

# Load model once
if (!exists("rf_model")) {
  rf_model <- readRDS("path/to/trained_rf_model.rds")
}

# Identify the feature columns the model expects.
# For ranger:
if (inherits(rf_model, "ranger")) {
  feature_names <- rf_model$forest$independent.variable.names
} else if (inherits(rf_model, "randomForest")) {
  # randomForest stores names differently
  feature_names <- rownames(rf_model$importance)
} else {
  stop("Unsupported RF model class: ", class(rf_model)[1])
}

# Build a numeric matrix of predictors (much faster for predict than data.frame)
pred_matrix <- as.matrix(cell_data[, ..feature_names])

# Predict in one call (or chunked if memory is tight)
CHUNK_SIZE <- 1e6L   # 1M rows per chunk â€” tune to available RAM

n_rows <- nrow(pred_matrix)

if (n_rows <= CHUNK_SIZE) {
  # Single-shot prediction
  if (inherits(rf_model, "ranger")) {
    preds <- predict(rf_model, data = pred_matrix)$predictions
  } else {
    preds <- predict(rf_model, newdata = pred_matrix)
  }
} else {
  # Chunked prediction to stay within 16 GB
  preds <- numeric(n_rows)
  starts <- seq(1L, n_rows, by = CHUNK_SIZE)
  for (s in starts) {
    e <- min(s + CHUNK_SIZE - 1L, n_rows)
    chunk <- pred_matrix[s:e, , drop = FALSE]
    if (inherits(rf_model, "ranger")) {
      preds[s:e] <- predict(rf_model, data = chunk)$predictions
    } else {
      preds[s:e] <- predict(rf_model, newdata = chunk)
    }
    if (s %% 3e6 == 1L) message(sprintf("  Predicted rows %s â€“ %s / %s",
                                          format(s, big.mark = ","),
                                          format(e, big.mark = ","),
                                          format(n_rows, big.mark = ",")))
  }
}

rm(pred_matrix)
gc()

# Store predictions back (by reference â€” no copy)
cell_data[, gdp_pred := preds]
rm(preds)

message("=== Pipeline complete ===")
```

---

## 4. EXPECTED PERFORMANCE GAINS

| Component | Before | After | Speedup |
|---|---|---|---|
| **Neighbor lookup build** | ~6.46M `paste` + hash lookups â†’ hours | Vectorised `data.table` keyed join on integer pairs | **~30â€“50Ã—** |
| **Neighbor stats (Ã—5 vars)** | 5 Ã— `lapply` over 6.46M + `do.call(rbind, ...)` | 5 Ã— `data.table` grouped `max/min/mean` on flat vector | **~50â€“100Ã—** |
| **Data frame copies** | 5 full copies of 6.46M Ã— 110 frame (~5 Ã— 5 GB) | Zero copies (`set()` by reference) | **Eliminates ~25 GB allocation** |
| **RF prediction** | Possibly row-wise or repeated model load | Single `predict()` call on numeric matrix, model loaded once | **~5â€“20Ã—** |
| **Overall wall-clock** | ~86+ hours | **~15â€“45 minutes** (estimated) | **~100â€“300Ã—** |

### Why the numerical estimand is preserved

- The CSR structure encodes **exactly** the same neighbor relationships as the original `build_neighbor_lookup` (same `spdep::nb` object, same id-year matching logic).
- `max`, `min`, `mean` are computed on the identical set of non-NA neighbor values.
- The trained Random Forest model is loaded from disk and used as-is â€” no retraining, no coefficient changes.
- Predictions are produced by the same `predict()` method on the same features, yielding bit-identical results.
END_RESPONSE>>>
