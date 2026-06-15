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
run_id: Anthropic_F06_T03_R008
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.3
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F06
prompt_family_slug: RASTER_KERNEL_ANALOGY
family_label: Raster/kernel analogy
family_group: Raster/kernel bridge
replicate: 8
raw_output_file: Experiment_Output/Anthropic/Temp_0.3/output_F06_Anthropic_temp_03_rep_008.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

---

## 1. Diagnosis

The current implementation has two critical bottlenecks:

### Bottleneck A: `build_neighbor_lookup` â€” O(n) `lapply` over ~6.46 million rows
For each of the ~6.46M cell-year rows, the function:
1. Looks up the cell's rook neighbors via character key matching.
2. Constructs `paste(..., sep="_")` keys for every neighbor Ã— year combination.
3. Performs named-vector lookups (`idx_lookup[neighbor_keys]`), which are **hash-table lookups on character strings** repeated billions of times.

This produces a **list of 6.46 million integer vectors** â€” an enormous, allocation-heavy R object.

### Bottleneck B: `compute_neighbor_stats` â€” O(n) `lapply` over the same 6.46M rows
For each row, it subsets a numeric vector by the neighbor indices, removes NAs, and computes `max`, `min`, `mean`. This is called **5 times** (once per source variable), so ~32.3 million R-level function calls with repeated allocation.

### Why it takes 86+ hours
- **Character hashing** at scale: `paste()` and named-vector lookups on 6.46M Ã— ~4 neighbors â‰ˆ 25.8 billion character operations.
- **R-level loops** (`lapply`) with per-element allocation: R's interpreter overhead dominates.
- **Redundant recomputation**: The neighbor lookup is the same for all 5 variables but the stats loop re-traverses the same index structure 5 times.

### Why raster focal/kernel operations don't directly apply
Raster focal operations (e.g., `terra::focal`) assume a **complete regular grid with uniform time dimension**. Here, the panel is cell Ã— year with potentially missing cell-years and an irregular neighbor structure (stored as an `nb` object). Forcing this into a 3D raster stack would require padding, reindexing, and would risk altering the numerical results for boundary/missing cells. The correct approach is to **vectorize the sparse-neighbor computation using data.table joins**, which preserves the exact numerical estimand.

---

## 2. Optimization Strategy

| Step | What changes | Speedup mechanism |
|------|-------------|-------------------|
| 1 | Replace character-key lookup with **integer join** via `data.table` | Eliminates all `paste()` and character hashing |
| 2 | Build a **long-form edge table** (from_row â†’ to_row) once | One vectorized operation instead of 6.46M `lapply` calls |
| 3 | Compute all 5 variables' neighbor stats in a **single grouped aggregation** | `data.table` grouped `j` computes max/min/mean in C |
| 4 | Join results back to the main table in one merge | No per-row allocation |

**Expected runtime**: ~2â€“5 minutes on a 16 GB laptop.

**Numerical equivalence**: The grouped `max`, `min`, `mean` on the exact same neighbor sets produce identical results. No model retraining is needed.

---

## 3. Working R Code

```r
library(data.table)

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 0.  Convert to data.table (if not already) and create an integer row key
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
setDT(cell_data)

# Ensure id_order is the vector of unique cell IDs in the same order
# as rook_neighbors_unique (the nb object).
# id_order and rook_neighbors_unique are already in memory.

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 1.  Build a long-form edge table:  (cell_id, neighbor_cell_id)
#     from the nb object â€” done once, ~1.37M edges
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
build_edge_table <- function(id_order, nb_obj) {
  from_ids <- rep(
    id_order,
    times = vapply(nb_obj, length, integer(1))
  )
  to_idx <- unlist(nb_obj, use.names = FALSE)
  # nb objects use 0 to denote "no neighbors"; remove those
  valid <- to_idx > 0L
  data.table(
    cell_id          = from_ids[valid],
    neighbor_cell_id = id_order[to_idx[valid]]
  )
}

edges <- build_edge_table(id_order, rook_neighbors_unique)

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 2.  Cross edges with years to get the full (row â†’ neighbor_row) map
#     This is the equivalent of build_neighbor_lookup but fully vectorized.
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Create a lean lookup:  (id, year) â†’ row index in cell_data
cell_data[, .row_idx := .I]

# Key the main table for fast joins
setkey(cell_data, id, year)

years <- sort(unique(cell_data$year))  # 1992:2019

# Expand edges Ã— years  (~1.37M edges Ã— 28 years â‰ˆ 38.5M rows)
# This fits comfortably in 16 GB (38.5M Ã— 3 int cols â‰ˆ 0.9 GB).
edge_year <- CJ_dt_edges(edges, years)

# Helper: cross join edges with years efficiently
# (CJ from data.table is for single-table cross join; we do it manually)
edge_year <- edges[, .(cell_id, neighbor_cell_id, year = list(years)),
                   by = .I][
  , .(cell_id, neighbor_cell_id, year = unlist(year)),
  by = .I][, .I := NULL]

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 2b. Attach the source-row index (the row of the *neighbor* cell-year)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Join to get neighbor's row index
setkey(edge_year, neighbor_cell_id, year)

neighbor_rows <- cell_data[, .(neighbor_cell_id = id, year, nb_row_idx = .row_idx)]
setkey(neighbor_rows, neighbor_cell_id, year)

edge_year <- neighbor_rows[edge_year, nomatch = 0L]
# Now edge_year has columns: neighbor_cell_id, year, nb_row_idx, cell_id

# Also attach the focal cell's row index
focal_rows <- cell_data[, .(cell_id = id, year, focal_row_idx = .row_idx)]
setkey(focal_rows, cell_id, year)
setkey(edge_year, cell_id, year)
edge_year <- focal_rows[edge_year, nomatch = 0L]

# edge_year now has: cell_id, year, focal_row_idx, neighbor_cell_id, nb_row_idx

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 3.  Compute neighbor stats for all 5 variables in one pass
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# Pull neighbor values into the edge table for each variable
for (v in neighbor_source_vars) {
  set(edge_year, j = v, value = cell_data[[v]][edge_year$nb_row_idx])
}

# Grouped aggregation: one group per focal_row_idx
# Compute max, min, mean for each variable, dropping NAs
agg_exprs <- unlist(lapply(neighbor_source_vars, function(v) {
  list(
    bquote(as.numeric(max(.(as.name(v)), na.rm = TRUE))),
    bquote(as.numeric(min(.(as.name(v)), na.rm = TRUE))),
    bquote(mean(.(as.name(v)), na.rm = TRUE))
  )
}))

agg_names <- unlist(lapply(neighbor_source_vars, function(v) {
  paste0("neighbor_", c("max_", "min_", "mean_"), v)
}))

# Build the aggregation call programmatically
stats <- edge_year[,
  setNames(lapply(neighbor_source_vars, function(v) {
    nv <- get(v)
    nv <- nv[!is.na(nv)]
    if (length(nv) == 0L) {
      list(NA_real_, NA_real_, NA_real_)
    } else {
      list(max(nv), min(nv), mean(nv))
    }
  }), neighbor_source_vars),
  by = focal_row_idx
]

# â”€â”€â”€ Cleaner and faster approach using direct expressions â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Build j expression dynamically
make_j_expr <- function(vars) {
  parts <- lapply(vars, function(v) {
    sprintf(
      'neighbor_max_%1$s = fifelse(all(is.na(%1$s)), NA_real_, max(%1$s, na.rm=TRUE)),
       neighbor_min_%1$s = fifelse(all(is.na(%1$s)), NA_real_, min(%1$s, na.rm=TRUE)),
       neighbor_mean_%1$s = fifelse(all(is.na(%1$s)), NA_real_, mean(%1$s, na.rm=TRUE))',
      v
    )
  })
  parse(text = paste0("list(", paste(parts, collapse = ",\n"), ")"))
}

j_expr <- make_j_expr(neighbor_source_vars)

stats <- edge_year[, eval(j_expr), by = focal_row_idx]

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 4.  Join stats back to cell_data
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# stats is keyed by focal_row_idx (one row per cell-year that has neighbors)
setkey(stats, focal_row_idx)

# Remove any prior neighbor columns if re-running
old_cols <- grep("^neighbor_(max|min|mean)_", names(cell_data), value = TRUE)
if (length(old_cols)) cell_data[, (old_cols) := NULL]

# Merge by row index
stat_cols <- setdiff(names(stats), "focal_row_idx")
cell_data[stats, (stat_cols) := mget(stat_cols), on = .(.row_idx = focal_row_idx)]

# Clean up helper column
cell_data[, .row_idx := NULL]

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 5.  Predict with the pre-trained Random Forest (unchanged)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# The model object (e.g., `rf_model`) is loaded from disk as-is.
# cell_data now contains the same 15 neighbor feature columns
# (5 vars Ã— 3 stats) with identical numerical values.
# Prediction proceeds exactly as before:
#
#   cell_data$predicted <- predict(rf_model, newdata = cell_data)
#
# No retraining is required.
```

---

## 4. Simplified, Self-Contained Drop-In Version

If the above feels complex, here is a **minimal, self-contained replacement** that is easier to audit for correctness:

```r
library(data.table)
setDT(cell_data)

# --- Step 1: Edge list from nb object ---
from_id <- rep(id_order, times = lengths(rook_neighbors_unique))
to_idx  <- unlist(rook_neighbors_unique)
keep    <- to_idx > 0L
edges   <- data.table(from_id = from_id[keep],
                       to_id   = id_order[to_idx[keep]])

# --- Step 2: Expand to cell-year level and attach neighbor values ---
cell_data[, row_idx := .I]
setkey(cell_data, id, year)

# One join per variable (avoids materializing a 38M Ã— 8 table all at once)
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# Prepare a slim focal table
focal <- cell_data[, .(focal_row = row_idx, from_id = id, year)]

# Merge focal with edges to get neighbor IDs per focal cell-year
setkey(focal, from_id)
setkey(edges, from_id)
focal_edges <- edges[focal, allow.cartesian = TRUE, nomatch = 0L]
# Columns: from_id, to_id, focal_row, year

# Attach neighbor row indices
nb_key <- cell_data[, .(to_id = id, year, nb_row = row_idx)]
setkey(nb_key, to_id, year)
setkey(focal_edges, to_id, year)
focal_edges <- nb_key[focal_edges, nomatch = 0L]
# Columns: to_id, year, nb_row, from_id, focal_row

# --- Step 3: Compute stats per variable and join back ---
for (v in neighbor_source_vars) {
  # Pull neighbor values
  focal_edges[, val := cell_data[[v]][nb_row]]

  # Aggregate
  agg <- focal_edges[!is.na(val),
    .(vmax  = max(val),
      vmin  = min(val),
      vmean = mean(val)),
    by = focal_row]

  # Name columns
  max_col  <- paste0("neighbor_max_", v)
  min_col  <- paste0("neighbor_min_", v)
  mean_col <- paste0("neighbor_mean_", v)
  setnames(agg, c("vmax","vmin","vmean"), c(max_col, min_col, mean_col))

  # Join back
  cell_data[agg, (c(max_col, min_col, mean_col)) :=
    mget(c(max_col, min_col, mean_col)), on = .(row_idx = focal_row)]
}

cell_data[, row_idx := NULL]

# --- Step 4: Predict (unchanged) ---
# cell_data$predicted <- predict(rf_model, newdata = cell_data)
```

---

## 5. Summary

| Aspect | Original | Optimized |
|--------|----------|-----------|
| **Neighbor lookup** | 6.46M `lapply` calls with `paste()` + character hash | One vectorized `data.table` join (~38.5M rows) |
| **Stats computation** | 6.46M Ã— 5 = 32.3M R-level function calls | 5 grouped `data.table` aggregations (C-level) |
| **Memory** | ~6.46M-element list of integer vectors | ~38.5M-row `data.table` (~1.5 GB peak) |
| **Estimated time** | 86+ hours | **2â€“5 minutes** |
| **Numerical results** | Baseline | **Identical** (same max/min/mean on same neighbor sets) |
| **RF model** | Pre-trained | **Unchanged, no retraining** |
END_RESPONSE>>>
