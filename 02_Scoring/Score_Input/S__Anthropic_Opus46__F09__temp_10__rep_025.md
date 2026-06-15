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
run_id: Anthropic_F09_T10_R025
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F09
prompt_family_slug: ADJACENCY_TABLE_OPTION
family_label: Adjacency-table option
family_group: Representation cue
replicate: 25
raw_output_file: Experiment_Output/Anthropic/Temp_1.0/output_F09_Anthropic_temp_10_rep_025.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The current pipeline is extremely slow (estimated 86+ hours) for two fundamental reasons:

### 1. `build_neighbor_lookup` is O(NÂ²)-like in practice
The function iterates over **every cell-year row** (~6.46 million rows) with `lapply`, and for each row it:
- Looks up the cell's spatial neighbors.
- Constructs character keys by pasting cell IDs and years.
- Performs named-vector lookups (`idx_lookup[neighbor_keys]`), which in R are hash-table lookups on a **6.46-million-entry named character vector** â€” slow due to repeated hashing and memory access patterns.

Because the spatial neighbor structure is **identical across all 28 years**, this work is redundantly repeated 28 times per cell. For 344,208 cells Ã— 28 years = 9,637,824 neighbor-set resolutions, each touching ~4 neighbors on average, this produces tens of millions of character-paste and hash-lookup operations.

### 2. `compute_neighbor_stats` uses row-level `lapply`
For each of the 5 variables, the function loops over 6.46 million rows in R-level `lapply`, subsetting a numeric vector with small index vectors and computing `max/min/mean`. The per-element R interpreter overhead on 6.46M iterations is enormous.

### 3. The neighbor topology is time-invariant but never exploited
The rook-neighbor structure is purely spatial. It does not change year to year. Yet the current code rebuilds a full row-level lookup that embeds year information, missing the opportunity to separate the **static spatial topology** from the **dynamic yearly attributes**.

---

## Optimization Strategy

**Core insight:** Build the adjacency table **once** at the cell level (344K cells, not 6.46M cell-years), then for each year, use vectorized joins and grouped operations to compute neighbor statistics.

### Step-by-step plan:

1. **Build a static edge table** from `rook_neighbors_unique`: a two-column `data.table` of `(cell_id, neighbor_cell_id)` â€” created once, ~1.37M rows.

2. **For each year**, join the yearly cell attributes onto the edge table by `neighbor_cell_id`, then group by `cell_id` to compute `max`, `min`, `mean` â€” all vectorized inside `data.table`.

3. **Join the results back** onto the main `cell_data` table.

This eliminates all `lapply` over 6.46M rows, all character-key pasting, and all named-vector hash lookups. The `data.table` approach uses binary-search joins and columnar grouped aggregation, which are orders of magnitude faster.

**Expected speedup:** From 86+ hours to roughly **2â€“10 minutes** depending on disk I/O and RAM pressure.

**Preservation guarantees:**
- The trained Random Forest model is not retouched.
- The numerical output (neighbor max, min, mean per variable per cell-year) is identical to the original.

---

## Working R Code

```r
library(data.table)

# =============================================================================
# STEP 1: Build the static spatial edge table ONCE
# =============================================================================
# rook_neighbors_unique is an nb object (list of integer vectors).
# id_order is the vector of cell IDs in the same order as the nb object.
# Each element rook_neighbors_unique[[i]] contains integer indices into id_order
# for the neighbors of cell id_order[i].

build_edge_table <- function(id_order, neighbors) {
  # Pre-allocate: count total edges
  n_edges <- sum(lengths(neighbors))
  
  from_id <- integer(n_edges)
  to_id   <- integer(n_edges)
  
  pos <- 1L
  for (i in seq_along(neighbors)) {
    nb <- neighbors[[i]]
    # spdep::nb encodes "no neighbors" as a single 0L; skip those
    if (length(nb) == 1L && nb[1] == 0L) next
    n <- length(nb)
    from_id[pos:(pos + n - 1L)] <- id_order[i]
    to_id[pos:(pos + n - 1L)]   <- id_order[nb]
    pos <- pos + n
  }
  
  # Trim if any 0-neighbor cells caused over-allocation
  if (pos - 1L < n_edges) {
    from_id <- from_id[1:(pos - 1L)]
    to_id   <- to_id[1:(pos - 1L)]
  }
  
  data.table(cell_id = from_id, neighbor_cell_id = to_id)
}

edge_dt <- build_edge_table(id_order, rook_neighbors_unique)

# Set key for fast joins on neighbor_cell_id
setkey(edge_dt, neighbor_cell_id)

# =============================================================================
# STEP 2: Convert cell_data to data.table (if not already)
# =============================================================================
if (!is.data.table(cell_data)) {
  cell_data <- as.data.table(cell_data)
}

# Ensure original row order is preserved for later reassembly
cell_data[, .row_order := .I]

# =============================================================================
# STEP 3: Compute neighbor stats for all variables â€” vectorized
# =============================================================================
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# We process one variable at a time to control peak RAM.
# For each variable, we:
#   (a) Extract (id, year, variable) from cell_data
#   (b) Join onto edge_dt by neighbor_cell_id to get each edge's neighbor value
#   (c) Group by (cell_id, year) to get max, min, mean
#   (d) Join results back onto cell_data

compute_neighbor_features_dt <- function(cell_dt, edge_dt, var_name) {
  
  # Column names for output (must match original pipeline naming)
  col_max  <- paste0("neighbor_max_", var_name)
  col_min  <- paste0("neighbor_min_", var_name)
  col_mean <- paste0("neighbor_mean_", var_name)
  
  # (a) Subset: only the columns we need for the join
  #     'id' is the cell identifier in cell_data matching id_order values
  attr_dt <- cell_dt[, .(id, year, val = get(var_name))]
  setnames(attr_dt, "id", "neighbor_cell_id")
  setkey(attr_dt, neighbor_cell_id)
  
  # (b) Join edge table with neighbor attributes
  #     For each directed edge (cell_id -> neighbor_cell_id) and each year,
  #     attach the neighbor's value.
  #     We need the Cartesian product of edges Ã— years, but it's more efficient
  #     to join edges onto the attribute table keyed by (neighbor_cell_id, year).
  setkey(attr_dt, neighbor_cell_id, year)
  
  # Expand edge_dt with year from attr_dt via a rolling/equi join:
  # For each edge, for each year that the neighbor has data, get the value.
  edge_with_val <- edge_dt[attr_dt,
                           .(cell_id     = x.cell_id,
                             year        = i.year,
                             neighbor_val = i.val),
                           on = .(neighbor_cell_id),
                           nomatch = 0L,
                           allow.cartesian = TRUE]
  
  # (c) Aggregate by (cell_id, year)
  stats <- edge_with_val[!is.na(neighbor_val),
                         .(nmax  = max(neighbor_val),
                           nmin  = min(neighbor_val),
                           nmean = mean(neighbor_val)),
                         keyby = .(cell_id, year)]
  
  setnames(stats,
           c("nmax", "nmin", "nmean"),
           c(col_max, col_min, col_mean))
  
  # (d) Join back onto cell_data
  # Remove old columns if they exist (idempotent re-runs)
  for (cc in c(col_max, col_min, col_mean)) {
    if (cc %in% names(cell_dt)) cell_dt[, (cc) := NULL]
  }
  
  cell_dt <- merge(cell_dt, stats,
                   by.x = c("id", "year"),
                   by.y = c("cell_id", "year"),
                   all.x = TRUE,
                   sort = FALSE)
  
  cell_dt
}

# --- Main loop (now fast) ---
for (var_name in neighbor_source_vars) {
  message(Sys.time(), " | Computing neighbor stats for: ", var_name)
  cell_data <- compute_neighbor_features_dt(cell_data, edge_dt, var_name)
  gc()
}

# Restore original row order (merge may shuffle)
setorder(cell_data, .row_order)
cell_data[, .row_order := NULL]

message(Sys.time(), " | Done. Neighbor features appended for all variables.")

# =============================================================================
# STEP 4: Predict with the EXISTING trained Random Forest (unchanged)
# =============================================================================
# The model object (e.g., rf_model) is already in memory or loaded from disk.
# Prediction proceeds exactly as before:
#
#   cell_data$prediction <- predict(rf_model, newdata = cell_data)
#
# The trained model is NOT retrained. The numerical estimand is preserved
# because the neighbor max, min, and mean values are computed identically
# (same set of neighbors, same aggregation functions, same variable values).
```

---

## Why This Is Correct and Equivalent

| Concern | Guarantee |
|---|---|
| **Same neighbor sets** | `edge_dt` is built from the same `rook_neighbors_unique` nb object and `id_order`. Each cell's directed neighbor list is identical. |
| **Same per-year scoping** | The join on `(neighbor_cell_id, year)` ensures that only same-year neighbor values are aggregated â€” exactly as the original `paste(id, year)` key did. |
| **Same aggregation** | `max`, `min`, `mean` with `NA` removal (`!is.na(neighbor_val)`) matches the original `compute_neighbor_stats` logic. Cells with no valid neighbors get `NA` via the `all.x = TRUE` merge. |
| **Trained RF preserved** | No model code is touched. The prediction input columns are numerically identical. |

## Expected Performance

| Component | Original | Optimized |
|---|---|---|
| Neighbor lookup construction | ~hours (6.46M `lapply` + char hashing) | ~2 seconds (vectorized edge table) |
| Neighbor stats per variable | ~17 hours (6.46M `lapply`) | ~15â€“30 seconds (`data.table` grouped agg) |
| **Total for 5 variables** | **86+ hours** | **~2â€“5 minutes** |
| Peak RAM | Moderate (large named vectors) | ~2â€“3 GB for largest intermediate join |
END_RESPONSE>>>
