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
run_id: Anthropic_F09_T10_R024
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F09
prompt_family_slug: ADJACENCY_TABLE_OPTION
family_label: Adjacency-table option
family_group: Representation cue
replicate: 24
raw_output_file: Experiment_Output/Anthropic/Temp_1.0/output_F09_Anthropic_temp_10_rep_024.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The current implementation is catastrophically slow (~86+ hours) for three compounding reasons:

### 1. Row-level `lapply` over 6.46 million rows in `build_neighbor_lookup`
For every single cell-year row, the function performs:
- A named-vector lookup (`id_to_ref`) â€” O(1) amortized but with overhead per call.
- A `paste`/key construction for every neighbor of that row.
- A named-vector lookup (`idx_lookup`) on a 6.46M-entry character vector â€” this is extremely slow because R's named vector lookup is O(n) or at best O(log n) per access, and you do it **for every neighbor of every row**.
- The result is a list of 6.46 million integer vectors.

**Memory:** That list alone consumes enormous RAM (each list element has overhead â‰ˆ 100+ bytes even when small, so ~600 MB+ just in list overhead, plus the integer vectors).

### 2. `compute_neighbor_stats` iterates row-by-row again
Another `lapply` over 6.46M elements, extracting subsets of a numeric vector via index vectors. This is inherently serial and R-loop-bound.

### 3. The neighbor topology is **year-invariant** but is recomputed per cell-year
The rook-neighbor structure is purely spatial. Cell ð‘–'s neighbors are the same in 1992 as in 2019. Yet `build_neighbor_lookup` re-indexes everything at the cell-year level, blowing up the problem from ~344K cells to ~6.46M rows.

**Key insight:** The adjacency structure involves only **344,208 cells** with ~1.37M directed edges. The yearly attribute values should be **joined onto** this small, fixed graph, not used to rebuild a lookup for every row.

---

## Optimization Strategy

1. **Build the adjacency table once** â€” a two-column `data.table` of `(cell_id, neighbor_id)` with ~1.37M rows, derived from `rook_neighbors_unique` and `id_order`. This is year-invariant and tiny.

2. **Join yearly attributes onto the edge table** â€” For each year and each variable, join the cell-year attribute onto the `neighbor_id` column. This gives you the neighbor's value for that variable in that year.

3. **Aggregate** â€” Group by `(cell_id, year)` and compute `max`, `min`, `mean` of the neighbor values in one vectorized `data.table` operation.

4. **Join back** â€” Merge the resulting neighbor-stats columns back onto the main `cell_data` table.

This replaces ~6.46M R-level list operations with a handful of **vectorized `data.table` joins and grouped aggregations** on a ~1.37M-row edge table Ã— 28 years â‰ˆ 38.4M rows (which `data.table` handles in seconds).

**Expected speedup:** From ~86 hours to **< 5 minutes** on a 16 GB laptop.

**Numerical equivalence:** The max, min, and mean computations are identical â€” same neighbors, same values, same aggregation functions. The trained Random Forest model is never touched.

---

## Working R Code

```r
library(data.table)

# ===========================================================
# STEP 1: Build a year-invariant adjacency table (once)
# ===========================================================
# Inputs:
#   id_order             â€” integer/numeric vector of cell IDs (length 344,208)
#   rook_neighbors_unique â€” spdep nb object (list of integer index vectors)
#
# Output:
#   adj_dt â€” data.table with columns: cell_id, neighbor_id
#            (~1,373,394 rows â€” one per directed edge)

build_adjacency_table <- function(id_order, neighbors) {
  # Pre-allocate by computing total number of edges
  n_cells <- length(id_order)
  n_edges <- sum(vapply(neighbors, length, integer(1)))
  
  from_id <- integer(n_edges)
  to_id   <- integer(n_edges)
  
  pos <- 1L
  for (i in seq_len(n_cells)) {
    nb_idx <- neighbors[[i]]
    n_nb   <- length(nb_idx)
    if (n_nb > 0L) {
      from_id[pos:(pos + n_nb - 1L)] <- id_order[i]
      to_id[pos:(pos + n_nb - 1L)]   <- id_order[nb_idx]
      pos <- pos + n_nb
    }
  }
  
  data.table(cell_id = from_id, neighbor_id = to_id)
}

adj_dt <- build_adjacency_table(id_order, rook_neighbors_unique)

cat(sprintf("Adjacency table: %d directed edges among %d cells\n",
            nrow(adj_dt), length(id_order)))

# ===========================================================
# STEP 2: Convert cell_data to data.table (if not already)
# ===========================================================
if (!is.data.table(cell_data)) {
  cell_data <- as.data.table(cell_data)
}

# Ensure keyed for fast joins
setkey(cell_data, id, year)

# ===========================================================
# STEP 3: For each variable, join â†’ aggregate â†’ merge back
# ===========================================================
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# Expand adjacency table by year (all 28 years)
#   This creates ~1.37M Ã— 28 â‰ˆ 38.4M rows, manageable in RAM (~1 GB)
years <- sort(unique(cell_data$year))

# Cross join adjacency edges with years
adj_year_dt <- CJ(edge_idx = seq_len(nrow(adj_dt)), year = years)
adj_year_dt[, `:=`(
  cell_id     = adj_dt$cell_id[edge_idx],
  neighbor_id = adj_dt$neighbor_id[edge_idx]
)]
adj_year_dt[, edge_idx := NULL]

cat(sprintf("Expanded edge-year table: %d rows\n", nrow(adj_year_dt)))

# Key for joining neighbor attributes
setkey(adj_year_dt, neighbor_id, year)

for (var_name in neighbor_source_vars) {
  cat(sprintf("Processing neighbor stats for: %s\n", var_name))
  
  # Extract only the columns we need for the join
  # (neighbor's attribute value, keyed by id and year)
  val_dt <- cell_data[, .(id, year, val = get(var_name))]
  setkey(val_dt, id, year)
  
  # Join: attach the neighbor cell's value to each edge-year row
  # The join is on neighbor_id == id, year == year
  adj_year_dt[val_dt, nb_val := i.val, on = .(neighbor_id = id, year = year)]
  
  # Aggregate: for each (cell_id, year), compute max/min/mean of neighbor values
  stats <- adj_year_dt[
    !is.na(nb_val),
    .(
      nb_max  = max(nb_val),
      nb_min  = min(nb_val),
      nb_mean = mean(nb_val)
    ),
    by = .(cell_id, year)
  ]
  
  # Name the output columns to match the original pipeline's naming convention
  # (adjust these names if your compute_and_add_neighbor_features used different names)
  max_col  <- paste0(var_name, "_nb_max")
  min_col  <- paste0(var_name, "_nb_min")
  mean_col <- paste0(var_name, "_nb_mean")
  
  setnames(stats, c("nb_max", "nb_min", "nb_mean"),
                  c(max_col,  min_col,  mean_col))
  
  # Remove old columns if they exist (idempotent re-runs)
  for (col in c(max_col, min_col, mean_col)) {
    if (col %in% names(cell_data)) {
      cell_data[, (col) := NULL]
    }
  }
  
  # Merge back onto cell_data
  cell_data <- merge(cell_data, stats,
                     by.x = c("id", "year"),
                     by.y = c("cell_id", "year"),
                     all.x = TRUE)
  
  # Clean up the temporary column
  adj_year_dt[, nb_val := NULL]
  
  cat(sprintf("  Done. Added columns: %s, %s, %s\n", max_col, min_col, mean_col))
}

# Re-key for any downstream operations
setkey(cell_data, id, year)

cat("All neighbor features computed.\n")

# ===========================================================
# STEP 4: Predict with existing Random Forest (unchanged)
# ===========================================================
# The trained model object (e.g., `rf_model`) is used as-is.
# cell_data now has the same neighbor-stat columns with the
# same numerical values, so predictions are identical.
#
# Example (adjust to your actual model/variable names):
#   cell_data$rf_prediction <- predict(rf_model, newdata = cell_data)
```

---

## Why This Is Numerically Equivalent

| Aspect | Original | Optimized |
|---|---|---|
| **Neighbor set** | `rook_neighbors_unique[[ref_idx]]` mapped to `id_order` | Same indices, same `id_order`, pre-materialized in `adj_dt` |
| **max/min/mean** | Computed via `max()`, `min()`, `mean()` on neighbor values | Identical functions in `data.table` `j` expression |
| **NA handling** | Drops `NA` neighbor values before aggregating | `!is.na(nb_val)` filter before grouping â€” identical |
| **Missing neighbors** | Returns `c(NA, NA, NA)` â†’ columns get `NA` | `all.x = TRUE` merge leaves unmatched rows as `NA` |
| **RF model** | Not modified | Not modified â€” same object, same `predict()` call |

## Memory Budget (16 GB laptop)

| Object | Approximate size |
|---|---|
| `cell_data` (6.46M Ã— 110 cols) | ~5.7 GB |
| `adj_dt` (1.37M Ã— 2 int cols) | ~11 MB |
| `adj_year_dt` (38.4M Ã— 3 cols + 1 temp) | ~1.2 GB |
| Temporary aggregation tables | ~300 MB peak |
| **Total peak** | **~7.2 GB** âœ“ |

This fits comfortably within 16 GB with room for the R session overhead and the Random Forest model object.
END_RESPONSE>>>
