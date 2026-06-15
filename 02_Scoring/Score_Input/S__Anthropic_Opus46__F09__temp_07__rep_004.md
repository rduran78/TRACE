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
run_id: Anthropic_F09_T07_R004
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F09
prompt_family_slug: ADJACENCY_TABLE_OPTION
family_label: Adjacency-table option
family_group: Representation cue
replicate: 4
raw_output_file: Experiment_Output/Anthropic/Temp_0.7/output_F09_Anthropic_temp_07_rep_004.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The current implementation is slow (~86+ hours) for three interconnected reasons:

### 1. `build_neighbor_lookup` is O(NÂ²)-like in practice
It iterates over all **6.46 million cell-year rows** with `lapply`, and for each row it:
- Performs a character-based hash lookup (`id_to_ref`) to find the spatial cell reference index.
- Retrieves that cell's neighbor IDs from the `nb` object.
- Constructs `paste(..., sep="_")` keys for every neighbor Ã— the current year.
- Looks those keys up in `idx_lookup` (a named character vector of length 6.46M).

This means ~6.46M iterations, each doing string concatenation, named-vector lookup into a 6.46M-length vector, and `NA` filtering. Named-vector lookup in R is **O(n)** in the worst case (hash collisions on large vectors), and `paste()` allocates new strings each time. This alone accounts for the bulk of the 86-hour runtime.

### 2. The neighbor lookup is **year-redundant**
The spatial neighbor structure is **identical across all 28 years** â€” cell *i*'s rook neighbors are always the same cells. Yet the current code rebuilds the lookup per cell-year row, essentially duplicating the same spatial topology 28 times and entangling it with the year dimension.

### 3. `compute_neighbor_stats` uses row-level `lapply`
Even after the lookup is built, computing stats iterates over 6.46M list elements in R-level `lapply`, with per-element subsetting, `NA` removal, and `max/min/mean` calls. This is slow due to R's interpreter overhead on millions of small operations.

---

## Optimization Strategy

**Core insight:** Separate the *spatial topology* (which is static) from the *temporal attributes* (which vary by year). Build the adjacency table **once** as a two-column edge list of `(cell_id, neighbor_id)`, then use a vectorized `data.table` join-and-aggregate to compute neighbor stats for all cell-years simultaneously.

### Steps:

1. **Build a static edge table** from `rook_neighbors_unique` (the `nb` object): a `data.table` with columns `(cell_id, neighbor_id)` â€” ~1.37M rows. Do this **once**.

2. **Join yearly attributes onto the edge table.** For each year, each cell's neighbors' attribute values are obtained by joining `cell_data` onto the edge table by `(neighbor_id, year)`. Because `data.table` joins are vectorized and hash-based, this is orders of magnitude faster than named-vector lookups in a loop.

3. **Aggregate** (`max`, `min`, `mean`) grouped by `(cell_id, year)` in a single `data.table` operation â€” fully vectorized, no R-level `lapply`.

4. **Join the aggregated stats back** onto `cell_data`.

5. **Predict** with the existing trained Random Forest model (unchanged).

**Expected speedup:** From ~86 hours to **minutes** (typically 2â€“10 minutes on a 16 GB laptop), because:
- The edge table has only ~1.37M rows (not 6.46M).
- After joining with 28 years, the expanded edge-year table is ~1.37M Ã— 28 â‰ˆ 38.4M rows, but `data.table` handles this efficiently in RAM (~2â€“4 GB for 5 variables).
- All operations are vectorized C-level code (no R-level loops over millions of elements).

---

## Working R Code

```r
library(data.table)

# ---------------------------------------------------------------
# STEP 0: Ensure cell_data is a data.table with columns: id, year,
#         ntl, ec, pop_density, def, usd_est_n2, ... (110 predictors)
# ---------------------------------------------------------------
cell_data <- as.data.table(cell_data)
setkey(cell_data, id, year)

# ---------------------------------------------------------------
# STEP 1: Build static spatial edge table ONCE from nb object
#
#   rook_neighbors_unique : spdep nb object (list of integer vectors)
#   id_order              : vector mapping positional index -> cell id
# ---------------------------------------------------------------
build_edge_table <- function(id_order, neighbors) {
  # Pre-allocate: count total edges
  n_edges <- sum(lengths(neighbors))
  
  from_id <- integer(n_edges)
  to_id   <- integer(n_edges)
  
  pos <- 1L
  for (i in seq_along(neighbors)) {
    nb_i <- neighbors[[i]]
    # spdep nb objects use 0L to indicate no neighbors
    nb_i <- nb_i[nb_i != 0L]
    n_i  <- length(nb_i)
    if (n_i > 0L) {
      from_id[pos:(pos + n_i - 1L)] <- id_order[i]
      to_id[pos:(pos + n_i - 1L)]   <- id_order[nb_i]
      pos <- pos + n_i
    }
  }
  
  # Trim if any 0-neighbor cells caused over-allocation
  data.table(cell_id = from_id[1:(pos - 1L)],
             neighbor_id = to_id[1:(pos - 1L)])
}

edge_dt <- build_edge_table(id_order, rook_neighbors_unique)
# ~1.37M rows, two integer columns â€” tiny in memory

# ---------------------------------------------------------------
# STEP 2: For each neighbor source variable, compute neighbor
#         max, min, mean via vectorized join + grouped aggregation
# ---------------------------------------------------------------
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# Prepare a slim lookup: only id, year, and the 5 source variables
lookup_cols <- c("id", "year", neighbor_source_vars)
attr_lookup <- cell_data[, ..lookup_cols]
setnames(attr_lookup, "id", "neighbor_id")
setkey(attr_lookup, neighbor_id, year)

# Cross join edge table with all years to get edge-year table
all_years <- sort(unique(cell_data$year))
edge_year <- edge_dt[, .(year = all_years), by = .(cell_id, neighbor_id)]
# This is ~1.37M * 28 â‰ˆ 38.4M rows

setkey(edge_year, neighbor_id, year)

# Join neighbor attributes onto edge-year table
edge_year <- attr_lookup[edge_year, on = .(neighbor_id, year), nomatch = NA]
# Now edge_year has: neighbor_id, year, ntl, ec, pop_density, def, usd_est_n2, cell_id

# ---------------------------------------------------------------
# STEP 3: Aggregate neighbor stats grouped by (cell_id, year)
# ---------------------------------------------------------------
# Build aggregation expressions dynamically
agg_exprs <- unlist(lapply(neighbor_source_vars, function(v) {
  list(
    bquote(as.numeric(max(.(as.name(v)), na.rm = TRUE))),
    bquote(as.numeric(min(.(as.name(v)), na.rm = TRUE))),
    bquote(as.numeric(mean(.(as.name(v)), na.rm = TRUE)))
  )
}))

agg_names <- unlist(lapply(neighbor_source_vars, function(v) {
  paste0("neighbor_", c("max_", "min_", "mean_"), v)
}))

names(agg_exprs) <- agg_names

# Evaluate aggregation in one pass
neighbor_stats <- edge_year[,
  lapply(agg_exprs, eval, envir = .SD),
  by = .(cell_id, year),
  .SDcols = neighbor_source_vars
]

# Handle Inf/-Inf from max/min on all-NA groups -> convert to NA
inf_to_na <- function(x) { x[is.infinite(x)] <- NA_real_; x }
stat_cols <- names(neighbor_stats)[-(1:2)]
neighbor_stats[, (stat_cols) := lapply(.SD, inf_to_na), .SDcols = stat_cols]

setnames(neighbor_stats, "cell_id", "id")
setkey(neighbor_stats, id, year)

# ---------------------------------------------------------------
# STEP 4: Join neighbor stats back onto cell_data
# ---------------------------------------------------------------
# Remove any old neighbor columns if they exist (idempotency)
old_cols <- intersect(names(cell_data), stat_cols)
if (length(old_cols) > 0) cell_data[, (old_cols) := NULL]

cell_data <- neighbor_stats[cell_data, on = .(id, year)]

# ---------------------------------------------------------------
# STEP 5: Predict with the existing trained Random Forest model
#         (model object is unchanged; no retraining)
# ---------------------------------------------------------------
# Ensure column order/types match what the model expects
# rf_model is your pre-trained randomForest / ranger object

# Example with ranger:
# pred <- predict(rf_model, data = cell_data)
# cell_data[, prediction := pred$predictions]

# Example with randomForest:
# pred <- predict(rf_model, newdata = as.data.frame(cell_data))
# cell_data[, prediction := pred]
```

### Simpler alternative for Step 3 (if the dynamic `bquote` approach is unclear)

```r
# ---------------------------------------------------------------
# STEP 3 (alternative): Loop over variables, compute stats, merge
# ---------------------------------------------------------------
neighbor_stats_list <- vector("list", length(neighbor_source_vars))

for (k in seq_along(neighbor_source_vars)) {
  v <- neighbor_source_vars[k]
  
  # Slim edge-year table with only the current variable
  ey_slim <- edge_year[, .(cell_id, year, val = get(v))]
  
  stats_k <- ey_slim[!is.na(val),
    .(nmax  = max(val),
      nmin  = min(val),
      nmean = mean(val)),
    by = .(cell_id, year)
  ]
  
  setnames(stats_k, c("nmax", "nmin", "nmean"),
           paste0("neighbor_", c("max_", "min_", "mean_"), v))
  
  neighbor_stats_list[[k]] <- stats_k
}

# Merge all variable stats together
neighbor_stats <- neighbor_stats_list[[1]]
for (k in 2:length(neighbor_stats_list)) {
  neighbor_stats <- merge(neighbor_stats, neighbor_stats_list[[k]],
                          by = c("cell_id", "year"), all = TRUE)
}

setnames(neighbor_stats, "cell_id", "id")
setkey(neighbor_stats, id, year)

# Remove old columns if present, then join
old_cols <- intersect(names(cell_data),
                      names(neighbor_stats)[-(1:2)])
if (length(old_cols) > 0) cell_data[, (old_cols) := NULL]

cell_data <- neighbor_stats[cell_data, on = .(id, year)]
```

---

## Summary

| Aspect | Original | Optimized |
|---|---|---|
| **Neighbor lookup** | 6.46M R-level iterations with `paste` + named-vector lookup | Static edge table built once (~1.37M rows) |
| **Attribute retrieval** | Per-row string key lookup into 6.46M-length named vector | Vectorized `data.table` keyed join |
| **Stat computation** | `lapply` over 6.46M list elements | Grouped `data.table` aggregation (C-level) |
| **Estimated runtime** | ~86+ hours | ~2â€“10 minutes |
| **RAM usage** | Moderate (6.46M-element list of integer vectors) | ~4â€“6 GB peak (38.4M-row edge-year table) |
| **RF model** | Preserved âœ“ | Preserved âœ“ |
| **Numerical estimand** | Preserved âœ“ | Preserved âœ“ (same max/min/mean, same column names) |
END_RESPONSE>>>
