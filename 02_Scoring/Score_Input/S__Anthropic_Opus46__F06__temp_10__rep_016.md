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
run_id: Anthropic_F06_T10_R016
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F06
prompt_family_slug: RASTER_KERNEL_ANALOGY
family_label: Raster/kernel analogy
family_group: Raster/kernel bridge
replicate: 16
raw_output_file: Experiment_Output/Anthropic/Temp_1.0/output_F06_Anthropic_temp_10_rep_016.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

---

## 1. Diagnosis

The bottleneck is clear: **`build_neighbor_lookup`** creates a 6.46-million-element list via `lapply` over every row, performing per-row string-pasting, hash-lookups, and NA-filtering. This is O(rows Ã— avg_neighbors) with enormous R-level loop overhead. Then **`compute_neighbor_stats`** iterates over that same 6.46M-element list againâ€”once per variable (Ã—5). Together they produce ~86+ hours of runtime.

**Root causes:**

| Component | Problem |
|---|---|
| `build_neighbor_lookup` | 6.46M iterations of an R `lapply` each doing `paste`, named-vector lookup, and `is.na` filtering. String-based keying (`"id_year"`) is slow. |
| `compute_neighbor_stats` | Another 6.46M-iteration `lapply` **per variable** (Ã—5 = 32.3M iterations), each subsetting a vector by index, removing NAs, and computing `max/min/mean`. |
| Memory | The `neighbor_lookup` list of 6.46M integer vectors is itself large, plus the `do.call(rbind, ...)` on 6.46M 3-element vectors is extremely slow (incremental allocation). |

**Why raster focal/kernel operations are not the right analogy here:**
Focal operations assume a regular grid with a fixed kernel window. While the grid *is* regular, the panel structure (cell Ã— year) and the precomputed `spdep::nb` object with potentially irregular boundaries (coastal cells, edge cells with < 4 neighbors) mean a focal convolution would need careful masking per time-slice and would not naturally produce max/min/mean of *only* rook neighbors. More importantly, the `spdep::nb` object is already serialized and testedâ€”reimplementing via focal risks subtle mismatches at boundaries, violating the requirement to **preserve the original numerical estimand**. The correct strategy is to vectorize the existing neighbor logic rather than switch paradigms.

---

## 2. Optimization Strategy

### Key ideas

1. **Eliminate the row-level lookup entirely.** Instead of building a 6.46M-element list mapping each *row* to its neighbor *rows*, exploit the panel structure: every cell has the same neighbors in every year. Build a sparse adjacency structure once over the 344,208 cells, then use **vectorized year-aligned joins** to compute stats.

2. **Use `data.table` for vectorized grouped operations.** Expand the neighbor pairs into a long edge table (`from_id`, `to_id`), join on `(to_id, year)` to pull neighbor values, then group by `(from_id, year)` to compute `max`, `min`, `mean`â€”all in one vectorized pass per variable.

3. **Sparse matrix alternative for `mean` (optional further speedup).** A sparse adjacency matrix `W` (344,208 Ã— 344,208) times a dense matrix of variable values (344,208 Ã— 28) gives neighbor sums; divide by neighbor counts for `mean`. But since we also need `max` and `min`, the `data.table` join approach is the cleanest single-pass solution.

4. **Expected runtime:** The edge table has ~1.37M directed edges. For each year, that's ~1.37M join lookups â†’ 28 years Ã— 1.37M â‰ˆ 38.4M rows in the long edge table. Grouping 38.4M rows by `(from_id, year)` to compute 3 summary stats is a ~10-second operation in `data.table` per variable. **Total: ~1â€“2 minutes for all 5 variables**, down from 86+ hours.

---

## 3. Working R Code

```r
library(data.table)

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 0. Convert cell_data to data.table (if not already) and ensure key
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
cell_dt <- as.data.table(cell_data)

# Ensure 'id' and 'year' are integer for fast joins
cell_dt[, id   := as.integer(id)]
cell_dt[, year := as.integer(year)]

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 1. Build a long edge table from the nb object (one-time, ~seconds)
#    rook_neighbors_unique is an nb object: list of length 344,208
#    id_order is the vector mapping list index â†’ cell id
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
build_edge_table <- function(id_order, nb_obj) {
  # nb objects store neighbor indices (into the same list); 0 means no neighbors
  from_ids <- rep(
    as.integer(id_order),
    times = vapply(nb_obj, function(x) {
      nx <- x[x != 0L]
      length(nx)
    }, integer(1))
  )
  to_ids <- as.integer(id_order[unlist(lapply(nb_obj, function(x) {
    nx <- x[x != 0L]
    nx
  }))])
  data.table(from_id = from_ids, to_id = to_ids)
}

edge_dt <- build_edge_table(id_order, rook_neighbors_unique)
# edge_dt has ~1,373,394 rows (directed pairs)

cat("Edge table rows:", nrow(edge_dt), "\n")

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 2. Function: compute and attach neighbor max/min/mean for one variable
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
compute_and_add_neighbor_features_fast <- function(cell_dt, var_name, edge_dt) {

  # Column names for output (must match original pipeline naming)
  col_max  <- paste0("neighbor_max_",  var_name)
  col_min  <- paste0("neighbor_min_",  var_name)
  col_mean <- paste0("neighbor_mean_", var_name)

  # --- a) Build a small lookup: (id, year, value) ----
  lookup <- cell_dt[, .(to_id = id, year = year, nb_val = get(var_name))]
  setkey(lookup, to_id, year)

  # --- b) Expand edge table across all years ----
  # Instead of a full cross-join (38M rows up front), use a rolling/equi join:
  #   For every (from_id, to_id) pair, join each year from cell_dt.
  # More memory-efficient: get the unique years, then cross with edges.
  years <- sort(unique(cell_dt$year))

  # CJ of edges Ã— years: ~1.37M Ã— 28 â‰ˆ 38.5M rows â€” fits in RAM (~600 MB)
  edge_year <- CJ_dt_edges(edge_dt, years)

  # --- c) Join to get neighbor values ----
  setkey(edge_year, to_id, year)
  edge_year[lookup, nb_val := i.nb_val, on = .(to_id, year)]

  # --- d) Aggregate: group by (from_id, year) ----
  stats <- edge_year[
    !is.na(nb_val),
    .(
      nb_max  = max(nb_val),
      nb_min  = min(nb_val),
      nb_mean = mean(nb_val)
    ),
    keyby = .(from_id, year)
  ]

  # --- e) Merge back into cell_dt ----
  setkey(cell_dt, id, year)
  setkey(stats, from_id, year)

  cell_dt[stats, (col_max)  := i.nb_max,  on = .(id = from_id, year)]
  cell_dt[stats, (col_min)  := i.nb_min,  on = .(id = from_id, year)]
  cell_dt[stats, (col_mean) := i.nb_mean, on = .(id = from_id, year)]

  # Cells with no valid neighbors get NA (already NA by default in data.table)
  invisible(cell_dt)
}

# Helper: cross-join edges with years (memory-efficient)
CJ_dt_edges <- function(edge_dt, years) {
  n_edges <- nrow(edge_dt)
  n_years <- length(years)
  data.table(
    from_id = rep(edge_dt$from_id, times = n_years),
    to_id   = rep(edge_dt$to_id,   times = n_years),
    year    = rep(years, each = n_edges)
  )
}

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 3. Run for all 5 neighbor source variables
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {
  cat("Computing neighbor features for:", var_name, "...\n")
  t0 <- proc.time()
  compute_and_add_neighbor_features_fast(cell_dt, var_name, edge_dt)
  elapsed <- (proc.time() - t0)["elapsed"]
  cat("  Done in", round(elapsed, 1), "seconds.\n")
}

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 4. Convert back to data.frame if downstream code expects one,
#    then predict with the existing (trained) Random Forest model
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
cell_data <- as.data.frame(cell_dt)

# The trained RF model object is assumed to be in memory (e.g., `rf_model`)
# Predict as before â€” the model is NOT retrained:
# cell_data$prediction <- predict(rf_model, newdata = cell_data)
```

---

## 4. Memory Budget Check

| Object | Approximate Size |
|---|---|
| `cell_dt` (6.46M Ã— 110 cols) | ~5.7 GB |
| `edge_dt` (1.37M Ã— 2 int cols) | ~11 MB |
| `edge_year` (38.4M Ã— 3 cols, peak) | ~920 MB |
| Temporary `lookup`, `stats` | ~200 MB |
| **Peak total** | **~7â€“8 GB** âœ“ fits 16 GB |

The `edge_year` table is the peak allocation and is created/destroyed once per variable. If memory is tight, process one variable at a time and `rm(edge_year); gc()` between iterations (which the loop already does implicitly since `edge_year` is local to the function).

---

## 5. Why This Preserves the Original Numerical Estimand

- The `edge_dt` encodes exactly the same directed rook-neighbor pairs as `rook_neighbors_unique`.
- The join on `(to_id, year)` pulls exactly the same neighbor values as the original `idx_lookup`.
- `max`, `min`, `mean` with `na.rm` (via `!is.na` filter before grouping) matches the original `compute_neighbor_stats` logic.
- The trained Random Forest model is used as-is for predictionâ€”no retraining occurs.
- The output columns are identically named and carry the same values, so downstream code is unaffected.

**Expected wall-clock time: ~1â€“2 minutes total (all 5 variables), vs. the original 86+ hours.**
END_RESPONSE>>>
