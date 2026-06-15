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
run_id: Anthropic_F07_T07_R027
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F07
prompt_family_slug: RASTER_INVALID_IRREGULAR
family_label: Raster invalid irregular topology
family_group: Raster/kernel bridge
replicate: 27
raw_output_file: Experiment_Output/Anthropic/Temp_0.7/output_F07_Anthropic_temp_07_rep_027.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

## Diagnosis

The bottleneck is clear: **`build_neighbor_lookup` creates a list of 6.46 million elements, each produced by an `lapply` call that performs character-based key lookups (`paste` + named-vector indexing) per row.** This is O(n) string operations on ~6.46M rows, with each iteration doing string concatenation, named-vector lookups, and NA filtering. The `compute_neighbor_stats` function then iterates over the same 6.46M-element list again for each of the 5 variables, doing subsetting and summary statistics in pure R loops.

**Specific problems:**

1. **`build_neighbor_lookup`:** The `paste(..., sep="_")` key construction and `idx_lookup[neighbor_keys]` named-vector lookup is extremely slow at scale. Named vector lookup in R is hash-based but the overhead of 6.46M `lapply` iterations, each creating character keys and subsetting, is enormous.

2. **`compute_neighbor_stats`:** An `lapply` over 6.46M elements, each calling `max`, `min`, `mean` on small vectors, has massive per-call overhead. The `do.call(rbind, result)` on a 6.46M-element list of 3-element vectors is also very slow.

3. **Memory:** Building a 6.46M-element list of integer vectors (the neighbor lookup) plus intermediate character vectors consumes significant RAM on a 16 GB machine.

**Estimated complexity of current approach:** ~6.46M Ã— (string ops + hash lookups + stats) Ã— 5 variables â‰ˆ 86+ hours.

## Optimization Strategy

**Core idea:** Replace all per-row R-level loops with vectorized operations using `data.table` joins and grouped aggregations.

1. **Vectorized neighbor lookup:** Instead of building a per-row list, create a **long-format edge table** (`data.table`) mapping each `(id, year)` to its neighbor `(neighbor_id, year)`. This is a single merge/join operation.

2. **Vectorized neighbor stats:** Join the edge table to the data to get neighbor values, then compute `max`, `min`, `mean` as a grouped `data.table` aggregation â€” a single pass per variable, fully vectorized in C.

3. **Memory management:** The edge table has ~1.37M directed neighbor pairs Ã— 28 years â‰ˆ 38.5M rows of integer pairs â€” about 600 MB, well within 16 GB. We reuse it for all 5 variables.

4. **Preserve the trained RF model:** We produce columns with identical names and identical numerical values (same neighbor topology, same aggregation functions), so the trained model remains valid with no retraining.

**Expected speedup:** From 86+ hours to **~5â€“15 minutes**.

## Working R Code

```r
library(data.table)

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# STEP 0: Ensure cell_data is a data.table with columns: id, year, ...
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
cell_dt <- as.data.table(cell_data)

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# STEP 1: Build a long-format directed edge table from the nb object
#
# rook_neighbors_unique is an nb object (list of integer index vectors)
# id_order is the vector mapping list position -> cell id
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
build_edge_table <- function(id_order, neighbors) {
  # Each element neighbors[[i]] contains the *positions* (indices into
  # id_order) of the rook neighbors of cell id_order[i].
  # A zero-length element or a single 0L means no neighbors (spdep convention).
  
  n <- length(neighbors)
  from_idx <- rep.int(seq_len(n), lengths(neighbors))
  to_idx   <- unlist(neighbors, use.names = FALSE)
  
  # Remove spdep's "no-neighbor" sentinel (0)
  valid <- to_idx != 0L
  from_idx <- from_idx[valid]
  to_idx   <- to_idx[valid]
  
  data.table(
    id          = id_order[from_idx],
    neighbor_id = id_order[to_idx]
  )
}

edge_dt <- build_edge_table(id_order, rook_neighbors_unique)

cat(sprintf("Edge table: %d directed neighbor pairs\n", nrow(edge_dt)))

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# STEP 2: Expand edges across all years (cross join edges Ã— years)
#
# Instead of a huge cross join, we join edges into the panel directly.
# For each row (id, year) we need neighbor values at the same year.
# Strategy: join cell_dt to edge_dt on id, then join again on
# (neighbor_id, year) to fetch the neighbor's value.
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

# Create a unique year vector
years <- sort(unique(cell_dt$year))

# Expand edge_dt by year: every edge exists in every year
# ~1.37M edges Ã— 28 years â‰ˆ 38.5M rows â€” manageable
edge_year_dt <- CJ_dt_edge(edge_dt, years)

# Efficient cross-join helper
CJ_dt_edge <- function(edge_dt, years) {
  # Cartesian product of edges and years
  edge_dt[, .(neighbor_id = neighbor_id, year = rep(years, each = .N)),
          by = .(id)]
  # The above is tricky; simpler and faster:
}

# Actually, the cleanest approach:
edge_year_dt <- edge_dt[, .(year = years), by = .(id, neighbor_id)]

cat(sprintf("Edge-year table: %d rows\n", nrow(edge_year_dt)))

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# STEP 3: For each source variable, join neighbor values and aggregate
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# Set keys for fast joins
setkey(cell_dt, id, year)
setkey(edge_year_dt, neighbor_id, year)

for (var_name in neighbor_source_vars) {
  cat(sprintf("Processing neighbor stats for: %s\n", var_name))
  
  # Subset only needed columns from cell_dt for the join
  val_dt <- cell_dt[, .(neighbor_id = id, year, val = get(var_name))]
  setkey(val_dt, neighbor_id, year)
  
  # Join: for each (id, neighbor_id, year) row, attach the neighbor's value
  merged <- edge_year_dt[val_dt, on = .(neighbor_id, year), nomatch = 0L]
  # merged now has columns: id, neighbor_id, year, val
  # where val is the neighbor's value of var_name in that year
  
  # Aggregate by (id, year) â€” compute max, min, mean of neighbor values
  stats <- merged[!is.na(val),
                  .(nbr_max  = max(val),
                    nbr_min  = min(val),
                    nbr_mean = mean(val)),
                  by = .(id, year)]
  
  # Name the new columns to match the original pipeline's naming convention
  max_col  <- paste0(var_name, "_nbr_max")
  min_col  <- paste0(var_name, "_nbr_min")
  mean_col <- paste0(var_name, "_nbr_mean")
  
  setnames(stats, c("nbr_max", "nbr_min", "nbr_mean"),
           c(max_col, min_col, mean_col))
  
  # Join stats back to cell_dt
  setkey(stats, id, year)
  
  # Remove old columns if they exist (idempotent re-runs)
  for (col in c(max_col, min_col, mean_col)) {
    if (col %in% names(cell_dt)) cell_dt[, (col) := NULL]
  }
  
  cell_dt <- stats[cell_dt, on = .(id, year)]
  setkey(cell_dt, id, year)
  
  cat(sprintf("  Done. Added: %s, %s, %s\n", max_col, min_col, mean_col))
}

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# STEP 4: Convert back to data.frame if downstream code expects it
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
cell_data <- as.data.frame(cell_dt)

cat("All neighbor features computed.\n")
```

**However**, the `edge_dt[, .(year = years), by = .(id, neighbor_id)]` step above can itself be memory-heavy if done naively. Here is a cleaner, self-contained, production-ready version:

```r
# ======================================================================
# PRODUCTION VERSION â€” Vectorized neighbor feature engineering
# ======================================================================
library(data.table)

cell_dt <- as.data.table(cell_data)
setkey(cell_dt, id, year)

# --- Step 1: Build directed edge list from nb object ---
n_cells <- length(rook_neighbors_unique)
from_pos <- rep.int(seq_len(n_cells), lengths(rook_neighbors_unique))
to_pos   <- unlist(rook_neighbors_unique, use.names = FALSE)
valid    <- to_pos != 0L
edge_dt  <- data.table(
  id          = id_order[from_pos[valid]],
  neighbor_id = id_order[to_pos[valid]]
)
rm(from_pos, to_pos, valid)

cat(sprintf("Directed edges: %d\n", nrow(edge_dt)))  # ~1,373,394

# --- Step 2: Compute neighbor stats per variable ---
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {
  cat(sprintf("  %s ... ", var_name))
  
  # Prepare a lookup table: for each (cell, year), what is the value?
  val_dt <- cell_dt[, .(id, year, val = get(var_name))]
  setnames(val_dt, "id", "neighbor_id")
  setkey(val_dt, neighbor_id)
  
  # Join edge list to values: for each edge, get all (year, val) combos
  # This is a many-to-many join: each edge Ã— each year the neighbor appears
  merged <- val_dt[edge_dt, on = .(neighbor_id), allow.cartesian = TRUE,
                   nomatch = 0L]
  # Result columns: neighbor_id, year, val, id
  
  # Aggregate: for each (id, year), compute stats over neighbor values
  stats <- merged[!is.na(val),
                  .(nbr_max  = max(val),
                    nbr_min  = min(val),
                    nbr_mean = mean(val)),
                  keyby = .(id, year)]
  
  # Assign new columns
  max_col  <- paste0(var_name, "_nbr_max")
  min_col  <- paste0(var_name, "_nbr_min")
  mean_col <- paste0(var_name, "_nbr_mean")
  
  # Left-join stats into cell_dt (preserves rows with no neighbors as NA)
  cell_dt[stats, on = .(id, year),
          `:=`(TMP_MAX = i.nbr_max,
               TMP_MIN = i.nbr_min,
               TMP_MEAN = i.nbr_mean)]
  
  setnames(cell_dt,
           c("TMP_MAX", "TMP_MIN", "TMP_MEAN"),
           c(max_col, min_col, mean_col))
  
  rm(val_dt, merged, stats)
  cat("done\n")
}

# --- Step 3: Return as data.frame if needed ---
cell_data <- as.data.frame(cell_dt)

cat("All neighbor features computed successfully.\n")
```

## Why This Preserves Correctness

| Concern | Guarantee |
|---|---|
| **Same neighbor topology** | We use the identical `rook_neighbors_unique` nb object and `id_order` mapping. No neighbors are added or removed. |
| **Same aggregation functions** | `max`, `min`, `mean` â€” identical to the original `compute_neighbor_stats`. |
| **Same NA handling** | `!is.na(val)` filtering before aggregation matches the original `neighbor_vals <- neighbor_vals[!is.na(neighbor_vals)]`. Rows with zero valid neighbors get `NA` (from the left-join non-match). |
| **Same column names** | The naming convention `{var}_nbr_max`, `{var}_nbr_min`, `{var}_nbr_mean` is preserved (adjust the pattern if the original `compute_and_add_neighbor_features` used a different convention). |
| **Trained RF model untouched** | No model object is modified. The feature matrix produced is numerically identical, so predictions are unchanged. |

## Performance Estimate

| Component | Original | Optimized |
|---|---|---|
| Neighbor lookup construction | ~hours (6.46M `lapply` with string ops) | ~5 seconds (vectorized `unlist` + `data.table` construction) |
| Stats per variable | ~17 hours (6.46M `lapply` with R-level loops) | ~1â€“2 minutes (`data.table` keyed join + grouped aggregation in C) |
| **Total (5 variables)** | **86+ hours** | **~5â€“15 minutes** |
END_RESPONSE>>>
