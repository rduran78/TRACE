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
run_id: Anthropic_F03_T10_R005
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F03
prompt_family_slug: CODE_PLUS_RF_FRAME
family_label: Code plus RF frame
family_group: RF-frame with code
replicate: 5
raw_output_file: Experiment_Output/Anthropic/Temp_1.0/output_F03_Anthropic_temp_10_rep_005.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

---

## 1. DIAGNOSIS

### Primary Bottlenecks

**A. `build_neighbor_lookup` â€” O(n) `lapply` over 6.46M rows with per-row string operations**

The function iterates over every cell-year row (â‰ˆ6.46M), and for each row:
- Converts an integer ID to character and does a named-vector lookup (`id_to_ref`).
- Builds string keys via `paste()` for every neighbor.
- Looks up those keys in a named character vector (`idx_lookup`).

Named-vector lookup in R is hash-based but carries heavy per-call overhead when done millions of times inside `lapply`. The `paste()`-based key construction allocates millions of small character vectors. This alone is likely **hours** of wall-clock time.

**B. `compute_neighbor_stats` â€” repeated over 5 variables, each iterating 6.46M rows**

Each call to `compute_neighbor_stats` does an `lapply` over 6.46M elements, subsetting a numeric vector and computing `max/min/mean`. The `do.call(rbind, result)` at the end binds 6.46M 3-element vectors â€” a notoriously slow pattern in R. With 5 variables this runs 5 Ã— 6.46M = 32.3M R-level function calls.

**C. Random Forest Prediction**

Predicting 6.46M rows Ã— 110 features through a Random Forest (typically hundreds of trees) is memory- and CPU-intensive. If `predict()` is called row-by-row or in small chunks, overhead dominates. Even a single vectorised call on 6.46M rows may require careful memory management on 16 GB RAM.

**D. Memory and object copying**

Repeatedly modifying `cell_data` inside the `for` loop (`cell_data <- compute_and_add_neighbor_features(...)`) triggers R's copy-on-modify semantics, potentially duplicating a â‰ˆ5.7 GB data.frame multiple times.

---

## 2. OPTIMIZATION STRATEGY

| Bottleneck | Strategy | Expected Speedup |
|---|---|---|
| `build_neighbor_lookup` | Eliminate string keys entirely. Use a `data.table` join on `(id, year)` â†’ row index, then vectorised expansion of the `nb` object. Build lookup once as a pair of integer vectors (row â†’ neighbor-rows). | 50â€“200Ã— |
| `compute_neighbor_stats` | Replace per-row `lapply` + `do.call(rbind,â€¦)` with vectorised `data.table` grouped aggregation over a pre-built edge-list. Compute all 5 variables' stats in one pass. | 20â€“100Ã— |
| Column mutation / copying | Use `data.table` set-by-reference (`:=`) to avoid full-frame copies. | 2â€“5Ã— |
| RF Prediction | Load model once; predict in a single vectorised call (or in ~10 large chunks to stay within RAM). Ensure the prediction input is a plain `matrix` or `data.frame` with no factors/extra columns. | 2â€“10Ã— |
| Overall | Target: **under 30 minutes** end-to-end on a 16 GB laptop. | ~200Ã— vs 86 h |

---

## 3. WORKING R CODE

```r
# ============================================================
# 0. Libraries
# ============================================================
library(data.table)
library(ranger)
# If your model is from 'randomForest', use library(randomForest) instead.

# ============================================================
# 1. Load pre-trained RF model (once)
# ============================================================
rf_model <- readRDS("rf_model.rds")  # adjust path as needed

# ============================================================
# 2. Convert main data to data.table (by reference if possible)
# ============================================================
if (!is.data.table(cell_data)) {
  setDT(cell_data)  # converts in place â€” no copy
}

# ============================================================
# 3. Build integer row-index mapping  (id, year) -> row
# ============================================================
cell_data[, .row_idx := .I]  # add row index column

# Keyed lookup table: given (id, year) return the row index
setkey(cell_data, id, year)

# ============================================================
# 4. Build neighbour edge-list (vectorised, no paste/string ops)
#
#    rook_neighbors_unique is an nb object: a list of length
#    n_cells where element [[i]] gives the indices (into
#    id_order) of cell i's neighbours.
#
#    We expand this to a data.table of directed edges
#    (from_id, to_id) and then cross-join with years to get
#    (from_row, to_row) pairs in the panel.
# ============================================================

# --- 4a. Cell-level edge list ----------------------------------
n_cells <- length(id_order)   # 344,208

# Vectorised expansion of the nb list to an edge data.table
to_lengths <- lengths(rook_neighbors_unique)
edge_dt <- data.table(
  from_cell_pos = rep(seq_len(n_cells), times = to_lengths),
  to_cell_pos   = unlist(rook_neighbors_unique, use.names = FALSE)
)
# Map positional indices to actual cell IDs
edge_dt[, from_id := id_order[from_cell_pos]]
edge_dt[, to_id   := id_order[to_cell_pos]]
edge_dt[, c("from_cell_pos", "to_cell_pos") := NULL]

# --- 4b. Expand to panel edges (from_row, to_row) -------------
#
#  For every (from_id, to_id) pair, and for every year in the
#  data, we need the row indices of both the focal cell-year
#  and the neighbour cell-year.
#
#  Strategy: join edge_dt with the row-index table twice.

years_vec <- sort(unique(cell_data$year))
n_years   <- length(years_vec)

# Cross-join edges Ã— years
panel_edges <- edge_dt[, .(year = years_vec), by = .(from_id, to_id)]

# Join to get from_row
setkey(cell_data, id, year)
panel_edges[cell_data, from_row := i..row_idx,
            on = .(from_id = id, year = year)]

# Join to get to_row (neighbour's row index in that year)
panel_edges[cell_data, to_row := i..row_idx,
            on = .(to_id = id, year = year)]

# Drop edges where either side is missing in the panel
panel_edges <- panel_edges[!is.na(from_row) & !is.na(to_row)]

# Clean up temporaries
rm(edge_dt); gc()

cat("Panel edges:", nrow(panel_edges), "\n")

# ============================================================
# 5. Compute all neighbour stats (5 vars Ã— 3 stats) in one pass
#
#    Grouped aggregation over the edge-list is fully vectorised
#    inside data.table's C backend.
# ============================================================

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# Pre-extract the columns we need into the edge table so that
# the grouped aggregation only touches numeric vectors.

for (v in neighbor_source_vars) {
  # Map each to_row to its value of variable v
  set(panel_edges, j = v, value = cell_data[[v]][panel_edges$to_row])
}

# Grouped aggregation: one group per from_row
# Produces columns like ntl_nb_max, ntl_nb_min, ntl_nb_mean, ...
agg_exprs <- list()
for (v in neighbor_source_vars) {
  agg_exprs[[paste0(v, "_nb_max")]]  <- call("max",  as.name(v), na.rm = TRUE)
  agg_exprs[[paste0(v, "_nb_min")]]  <- call("min",  as.name(v), na.rm = TRUE)
  agg_exprs[[paste0(v, "_nb_mean")]] <- call("mean", as.name(v), na.rm = TRUE)
}

# Build a single J-expression
j_expr <- as.call(c(as.name("list"), agg_exprs))

cat("Computing neighbour statistics â€¦\n")
nb_stats <- panel_edges[, eval(j_expr), by = .(from_row)]

# Replace infinite values (from max/min on empty sets) with NA
for (col in names(nb_stats)[-1L]) {
  vals <- nb_stats[[col]]
  set(nb_stats, i = which(is.infinite(vals)), j = col, value = NA_real_)
}

# Free memory from panel_edges
rm(panel_edges); gc()

# ============================================================
# 6. Join neighbour stats back to cell_data (by reference)
# ============================================================
# nb_stats is keyed on from_row; cell_data has .row_idx
setkey(nb_stats, from_row)

stat_cols <- setdiff(names(nb_stats), "from_row")
cell_data[nb_stats, (stat_cols) := mget(paste0("i.", stat_cols)),
          on = .(.row_idx = from_row)]

# Clean up helper column
cell_data[, .row_idx := NULL]

cat("Feature preparation complete. Columns:", ncol(cell_data), "\n")

# ============================================================
# 7. Random Forest Prediction  â€”  chunked, single-model load
# ============================================================

# Identify the predictor columns the model expects.
# For ranger:
if (inherits(rf_model, "ranger")) {
  pred_vars <- rf_model$forest$independent.variable.names
} else {
  # randomForest package stores the names used in training
  pred_vars <- rownames(rf_model$importance)
}

# Validate that all required columns are present
missing_vars <- setdiff(pred_vars, names(cell_data))
if (length(missing_vars) > 0L) {
  stop("Missing predictor columns: ", paste(missing_vars, collapse = ", "))
}

# Build a prediction matrix (data.table subset is zero-copy friendly)
pred_input <- cell_data[, ..pred_vars]

cat("Starting RF prediction on", nrow(pred_input), "rows â€¦\n")

# --- Chunked prediction to limit peak memory -----------------
#     Each chunk â‰ˆ 650K rows â†’ ~10 chunks.  Adjust chunk_size
#     if memory is tighter or more generous.

chunk_size <- 650000L
n_rows     <- nrow(pred_input)
n_chunks   <- ceiling(n_rows / chunk_size)

preds <- numeric(n_rows)

for (ch in seq_len(n_chunks)) {
  i_start <- (ch - 1L) * chunk_size + 1L
  i_end   <- min(ch * chunk_size, n_rows)
  
  chunk_data <- pred_input[i_start:i_end, ]
  
  if (inherits(rf_model, "ranger")) {
    p <- predict(rf_model, data = chunk_data, num.threads = parallel::detectCores())
    preds[i_start:i_end] <- p$predictions
  } else {
    # randomForest package
    preds[i_start:i_end] <- predict(rf_model, newdata = chunk_data)
  }
  
  cat(sprintf("  Chunk %d/%d done (%dâ€“%d)\n", ch, n_chunks, i_start, i_end))
}

rm(pred_input); gc()

# ============================================================
# 8. Attach predictions to cell_data
# ============================================================
cell_data[, predicted_gdp := preds]

cat("Done. Predictions attached as 'predicted_gdp'.\n")
```

---

## 4. SUMMARY OF KEY CHANGES AND WHY THEY WORK

| Original | Optimised | Why it matters |
|---|---|---|
| `paste(id, year)` string keys looked up in named vector, per row | `data.table` integer join on `(id, year)` | Eliminates â‰ˆ6.46M `paste()` calls and hash lookups on character strings. `data.table` binary search on integer keys is orders of magnitude faster. |
| `lapply` over 6.46M rows in `build_neighbor_lookup` | Vectorised `rep()`/`unlist()` expansion + `data.table` cross-join | Replaces 6.46M R function calls with three vectorised C-level operations. |
| `lapply` + `do.call(rbind,â€¦)` over 6.46M rows Ã— 5 variables | Single grouped `data.table` aggregation (`by = from_row`) | `data.table`'s GForce optimisation runs `max`, `min`, `mean` in compiled C per group. Binding is internal, not via `do.call(rbind, â€¦)` (which allocates 6.46M tiny matrices). |
| `cell_data <- compute_and_add_â€¦(cell_data, â€¦)` in a loop (copies ~5.7 GB frame 5 times) | `:=` set-by-reference | Zero copies; columns are added in-place. |
| (Assumed) single `predict()` on 6.46M rows | Chunked prediction (~650K rows/chunk, auto-threaded) | Keeps peak memory under 16 GB while still giving `predict()` large batches (low per-call overhead). `num.threads` exploits all cores. |
| Model potentially re-loaded or converted | Loaded once as `rf_model`; never retrained | Preserves the trained model exactly. |

**The numerical estimand is preserved**: no transformations, rounding, or approximations are applied to either the features or the predictions. All `max`, `min`, `mean` operations are numerically identical to the originals (same `na.rm = TRUE` semantics; infinities from empty groups are mapped to `NA` exactly as the original `length(idx) == 0` guard did).
END_RESPONSE>>>
