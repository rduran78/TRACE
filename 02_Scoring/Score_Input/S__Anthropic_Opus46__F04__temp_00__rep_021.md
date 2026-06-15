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
run_id: Anthropic_F04_T00_R021
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F04
prompt_family_slug: NEIGHBOR_BOTTLENECK_HINT
family_label: Neighbor bottleneck hint
family_group: Spatial-neighbor cue
replicate: 21
raw_output_file: Experiment_Output/Anthropic/Temp_0.0/output_F04_Anthropic_temp_00_rep_021.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The bottleneck is **not** the Random Forest inference. It is the two spatial neighbor feature construction functions, and the root causes are:

### 1. `build_neighbor_lookup` â€” Row-level `lapply` over 6.46 million rows

Each iteration performs:
- A character coercion and named-vector lookup (`id_to_ref[as.character(data$id[i])]`).
- A `paste()` call to build neighbor keys.
- A named-vector lookup into `idx_lookup` (which is a named character vector of length 6.46M â€” each lookup is **O(n)** in the worst case for R's hashed named vectors, but the sheer repetition across 6.46M rows is devastating).
- An `is.na` filter.

This produces ~6.46 million list elements. The dominant cost is the **per-row string construction and hash lookup repeated millions of times inside an interpreted R loop**.

### 2. `compute_neighbor_stats` â€” Another row-level `lapply` over 6.46M rows

For each row, it subsets a numeric vector, removes NAs, and computes `max`, `min`, `mean`. This is called **5 times** (once per source variable), so it executes ~32.3 million interpreted iterations total.

### 3. Combined cost estimate

With ~6.46M rows and ~5 variables, the pipeline executes roughly:
- 6.46M iterations for the lookup build.
- 5 Ã— 6.46M = 32.3M iterations for neighbor stats.
- Total: ~38.8 million R-level interpreted loop iterations, each doing string operations and subsetting.

At even ~8 ms per iteration (conservative for the lookup build), `build_neighbor_lookup` alone takes ~14 hours. The full 86+ hour estimate is consistent.

---

## Optimization Strategy

**Eliminate all row-level R loops. Replace with vectorized `data.table` joins and grouped aggregations.**

The key insight: the neighbor relationship is a **cell-to-cell** mapping (not a cell-year-to-cell-year mapping). For any given year, a cell's neighbors are the same cells. So we can:

1. **Expand the neighbor list into an edge table once** â€” a two-column `data.table` of `(id, neighbor_id)` with ~1.37M rows.
2. **Join the edge table to the panel data by `(neighbor_id, year)`** to pull neighbor values â€” this is a single keyed `data.table` merge (~1.37M Ã— 28 â‰ˆ 38.4M rows, manageable in 16 GB).
3. **Group by `(id, year)` and compute `max`, `min`, `mean`** in one vectorized pass per variable.

This replaces ~38.8 million interpreted R iterations with a handful of vectorized `data.table` operations that run in **minutes, not days**.

**The trained Random Forest model is untouched. The numerical results (max, min, mean of non-NA neighbor values) are identical.**

---

## Working R Code

```r
library(data.table)

# ---------------------------------------------------------------
# STEP 1: Build the edge table from the spdep nb object (once)
# ---------------------------------------------------------------
# rook_neighbors_unique is a list of integer vectors (spdep nb object).
# id_order is the vector mapping position -> cell id.

build_edge_table <- function(id_order, neighbors) {
  # Pre-allocate by computing total edges
  n_edges <- sum(lengths(neighbors))
  
  from_id <- integer(n_edges)
  to_id   <- integer(n_edges)
  
  pos <- 1L
  for (i in seq_along(neighbors)) {
    nb <- neighbors[[i]]
    if (length(nb) > 0 && !(length(nb) == 1L && nb[1L] == 0L)) {
      n <- length(nb)
      from_id[pos:(pos + n - 1L)] <- id_order[i]
      to_id[pos:(pos + n - 1L)]   <- id_order[nb]
      pos <- pos + n
    }
  }
  
  # Trim if any nb entries were empty / zero (spdep convention)
  if (pos <= n_edges) {
    from_id <- from_id[1:(pos - 1L)]
    to_id   <- to_id[1:(pos - 1L)]
  }
  
  data.table(id = from_id, neighbor_id = to_id)
}

edge_dt <- build_edge_table(id_order, rook_neighbors_unique)
# ~1.37 M rows, two integer columns â€” trivial memory

# ---------------------------------------------------------------
# STEP 2: Convert panel data to data.table (in-place if possible)
# ---------------------------------------------------------------
if (!is.data.table(cell_data)) {
  setDT(cell_data)
}

# Ensure key columns are proper types
cell_data[, id   := as.integer(id)]
cell_data[, year := as.integer(year)]

# ---------------------------------------------------------------
# STEP 3: Compute neighbor features for all source variables
# ---------------------------------------------------------------
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# We join edge_dt to cell_data on (neighbor_id == id, year == year)
# to get neighbor values, then aggregate by (id, year).

# Prepare a lookup table with only the columns we need
lookup_cols <- c("id", "year", neighbor_source_vars)
lookup_dt   <- cell_data[, ..lookup_cols]
setnames(lookup_dt, "id", "neighbor_id")
setkeyv(lookup_dt, c("neighbor_id", "year"))

# Add year to edge table for the cross-join with years
# Instead of a full cross join (which would be huge), we merge stepwise.

# Keyed join: edge_dt + year from cell_data
# Strategy: join edge_dt to lookup_dt by neighbor_id and year.
# We need (id, year) on the left side. We get year from cell_data's own rows.

# Build left side: (id, year, neighbor_id) by joining edge_dt to the
# distinct (id, year) pairs â€” but that's 6.46M Ã— avg_degree â‰ˆ 25.8M rows.
# More efficient: just add year via a merge.

# Most memory-efficient approach: loop over variables, not rows.

setkeyv(edge_dt, "neighbor_id")

for (var_name in neighbor_source_vars) {
  cat("Processing neighbor features for:", var_name, "\n")
  
  # Subset lookup to just this variable
  var_lookup <- cell_data[, .(neighbor_id = id, year, val = get(var_name))]
  setkeyv(var_lookup, c("neighbor_id", "year"))
  
  # Join: for each edge (id, neighbor_id), and for each year,
  # get the neighbor's value.
  # We need to bring 'year' from the focal cell. 
  # Approach: merge edge_dt with cell_data's (id, year) to get
  # (id, year, neighbor_id), then merge with var_lookup.
  
  # Get distinct (id, year) from cell_data
  focal <- cell_data[, .(id, year)]
  
  # Merge focal with edge_dt on 'id' to get (id, year, neighbor_id)
  # This is the expensive step memory-wise: ~6.46M * avg_neighbors
  # avg neighbors â‰ˆ 1,373,394 * 2 / 344,208 â‰ˆ ~4 (rook), so ~25.8M rows
  setkeyv(focal, "id")
  setkeyv(edge_dt, "id")
  
  expanded <- edge_dt[focal, on = "id", allow.cartesian = TRUE, nomatch = 0L]
  # expanded has columns: id, neighbor_id, year
  # ~25.8M rows â€” fits in memory easily
  
  # Now join to get neighbor values
  setkeyv(expanded, c("neighbor_id", "year"))
  expanded[var_lookup, val := i.val, on = c("neighbor_id", "year")]
  
  # Aggregate by (id, year), removing NAs as in the original code
  agg <- expanded[!is.na(val), 
                  .(nb_max  = max(val),
                    nb_min  = min(val),
                    nb_mean = mean(val)),
                  by = .(id, year)]
  
  # Name columns to match original pipeline expectations
  max_col  <- paste0("nb_max_",  var_name)
  min_col  <- paste0("nb_min_",  var_name)
  mean_col <- paste0("nb_mean_", var_name)
  setnames(agg, c("nb_max", "nb_min", "nb_mean"), c(max_col, min_col, mean_col))
  
  # Merge back into cell_data
  setkeyv(agg, c("id", "year"))
  setkeyv(cell_data, c("id", "year"))
  
  # Remove old columns if they exist (idempotent reruns)
  for (col in c(max_col, min_col, mean_col)) {
    if (col %in% names(cell_data)) cell_data[, (col) := NULL]
  }
  
  cell_data <- agg[cell_data, on = c("id", "year")]
  
  # Clean up to free memory within the loop
  rm(var_lookup, focal, expanded, agg)
  gc()
}

cat("Neighbor feature construction complete.\n")

# ---------------------------------------------------------------
# STEP 4: Predict with the existing trained Random Forest (unchanged)
# ---------------------------------------------------------------
# The trained RF model object is used as-is. Example:
# cell_data$predicted_gdp <- predict(trained_rf_model, newdata = cell_data)
```

---

## Expected Performance Improvement

| Component | Original | Optimized |
|---|---|---|
| `build_neighbor_lookup` | ~14 hours (6.46M interpreted iterations with string ops) | **Replaced** by `build_edge_table`: ~2 seconds (vectorized) |
| `compute_neighbor_stats` (Ã—5 vars) | ~72 hours (32.3M interpreted iterations) | **Replaced** by keyed `data.table` joins + grouped aggregation: ~5â€“15 minutes total |
| Peak memory | Moderate (6.46M-element list of integer vectors) | ~2â€“3 GB for the largest intermediate `expanded` table (~25.8M rows Ã— 3 cols) â€” fits in 16 GB |
| **Total** | **86+ hours** | **~5â€“20 minutes** |

## Why Numerical Results Are Identical

The original code computes, for each `(id, year)` pair, the `max`, `min`, and `mean` of `var_name` across all rook neighbors present in the data for that year, after removing `NA` values. The optimized code performs exactly the same operation: it expands the same neighbor relationships, joins the same values by `(neighbor_id, year)`, filters `!is.na(val)`, and computes the same three aggregation functions grouped by `(id, year)`. Rows with no valid neighbors receive `NA` via the final right join (`agg[cell_data, ...]`), matching the original `c(NA, NA, NA)` return. The trained Random Forest model is never modified.
END_RESPONSE>>>
