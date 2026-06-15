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
run_id: Anthropic_F09_T00_R009
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F09
prompt_family_slug: ADJACENCY_TABLE_OPTION
family_label: Adjacency-table option
family_group: Representation cue
replicate: 9
raw_output_file: Experiment_Output/Anthropic/Temp_0.0/output_F09_Anthropic_temp_00_rep_009.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The current implementation is extremely slow (~86+ hours) for two interconnected reasons:

### 1. `build_neighbor_lookup` is O(NÂ²)-like in practice
This function iterates over **all ~6.46 million cell-year rows** with `lapply`, and for each row it:
- Looks up the cell's spatial neighbors (fine).
- Constructs **character key strings** (`paste(id, year)`) and performs **named-vector lookups** (`idx_lookup[neighbor_keys]`) â€” these are hash-table lookups but done 6.46 million times, each creating temporary character vectors and subsetting a 6.46-million-entry named vector.

The result is a **list of 6.46 million integer vectors**, which is enormous in memory and slow to build. Critically, this lookup **mixes spatial topology (which is static) with temporal indexing (which is repetitive)**. Every cell has the same neighbors in every year, yet the function recomputes the neighbor-row mapping for each of the 28 year-copies of every cell.

### 2. `compute_neighbor_stats` iterates row-by-row over 6.46M rows
Even though the lookup is precomputed, calling `vals[idx]` inside `lapply` over 6.46 million elements, then computing `max/min/mean` per element, is inherently slow in interpreted R. This is done **5 times** (once per neighbor source variable).

### Summary of bottlenecks
| Step | Calls | Cost |
|---|---|---|
| `build_neighbor_lookup` | 6.46M `paste` + named-vector lookups | ~hours |
| `compute_neighbor_stats` | 6.46M Ã— 5 vars Ã— `max/min/mean` | ~hours |
| Memory: 6.46M-element list of integer vectors | ~GBs of list overhead | RAM pressure |

---

## Optimization Strategy

**Core insight:** The neighbor graph is purely spatial and static across years. Build it once as a **cell-to-cell adjacency table**, then join yearly attributes onto it. This converts the problem from row-wise R loops into vectorized `data.table` grouped operations.

### Steps:

1. **Build a static edge table** from `rook_neighbors_unique` (the `nb` object): a two-column `data.table` with columns `(id, neighbor_id)` â€” ~1.37M rows. This is done **once**.

2. **Join cell-year attributes onto the edge table** by `(neighbor_id, year)` â€” this gives each edge the neighbor's variable value for that year. `data.table` binary-search joins make this very fast.

3. **Group by `(id, year)`** and compute `max`, `min`, `mean` of neighbor values â€” fully vectorized, no R-level row loops.

4. **Join the resulting stats back** onto the main `cell_data` table.

This reduces the problem from ~6.46M Ã— R-loop iterations to a handful of vectorized `data.table` join-and-group operations. Expected runtime: **minutes, not hours**.

---

## Working R Code

```r
library(data.table)

# ============================================================
# STEP 0: Convert cell_data to data.table (if not already)
# ============================================================
cell_data <- as.data.table(cell_data)

# Ensure key columns exist and are proper types
stopifnot(all(c("id", "year") %in% names(cell_data)))

# ============================================================
# STEP 1: Build static spatial edge table ONCE
#
# rook_neighbors_unique is an nb object (list of integer vectors)
# id_order is the vector mapping list index -> cell id
# ============================================================
build_edge_table <- function(id_order, neighbors_nb) {
  # neighbors_nb[[i]] contains the indices (into id_order) of

  # the neighbors of cell id_order[i].
  # A 0-integer entry means no neighbors in spdep convention.
  edges <- rbindlist(lapply(seq_along(neighbors_nb), function(i) {
    nb_idx <- neighbors_nb[[i]]
    nb_idx <- nb_idx[nb_idx != 0L]
    if (length(nb_idx) == 0L) return(NULL)
    data.table(id = id_order[i], neighbor_id = id_order[nb_idx])
  }))
  edges
}

cat("Building static edge table...\n")
edge_table <- build_edge_table(id_order, rook_neighbors_unique)
cat(sprintf("  Edge table: %s directed edges\n", format(nrow(edge_table), big.mark = ",")))

# ============================================================
# STEP 2: Function to compute neighbor stats for one variable
#          using vectorized data.table joins + grouped aggregation
# ============================================================
compute_neighbor_features_dt <- function(cell_dt, edge_dt, var_name) {
  # Subset to only needed columns for the join (minimise memory)
  # We need neighbor_id matched to (id, year) in cell_dt
  neighbor_vals <- edge_dt[
    cell_dt[, .(neighbor_id = id, year, value = get(var_name))],
    on = .(neighbor_id),
    allow.cartesian = TRUE,
    nomatch = NULL
  ]
  # neighbor_vals now has columns: id, neighbor_id, year, value
  # where 'id' is the focal cell and 'value' is the neighbor's attribute

  # Remove NA values before aggregation
  neighbor_vals <- neighbor_vals[!is.na(value)]

  # Grouped aggregation
  stats <- neighbor_vals[,
    .(
      nb_max  = max(value),
      nb_min  = min(value),
      nb_mean = mean(value)
    ),
    by = .(id, year)
  ]

  # Rename columns to match original pipeline naming convention
  suffix <- var_name
  setnames(stats,
    c("nb_max", "nb_min", "nb_mean"),
    c(paste0(suffix, "_neighbor_max"),
      paste0(suffix, "_neighbor_min"),
      paste0(suffix, "_neighbor_mean"))
  )

  stats
}

# ============================================================
# STEP 3: Compute and attach neighbor features for all variables
# ============================================================
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# Set key on cell_data for fast joins
setkey(cell_data, id, year)

cat("Computing neighbor features...\n")
for (var_name in neighbor_source_vars) {
  cat(sprintf("  Processing: %s\n", var_name))

  stats_dt <- compute_neighbor_features_dt(cell_data, edge_table, var_name)
  setkey(stats_dt, id, year)

  # Remove old columns if they exist (idempotent re-runs)
  old_cols <- c(
    paste0(var_name, "_neighbor_max"),
    paste0(var_name, "_neighbor_min"),
    paste0(var_name, "_neighbor_mean")
  )
  for (col in old_cols) {
    if (col %in% names(cell_data)) cell_data[, (col) := NULL]
  }

  # Left join stats back onto cell_data

  cell_data <- stats_dt[cell_data, on = .(id, year)]
}

cat("Neighbor feature computation complete.\n")

# ============================================================
# STEP 4: Predict with the existing trained Random Forest
#
# The trained model object (e.g., rf_model) is unchanged.
# cell_data now has the same neighbor feature columns as before.
# ============================================================
# Example (adjust to your actual model object name):
# cell_data[, prediction := predict(rf_model, newdata = cell_data)]
```

---

## Why This Preserves Correctness

| Concern | Guarantee |
|---|---|
| **Numerical estimand** | `max`, `min`, `mean` are computed on exactly the same neighbor sets â€” the edge table encodes the identical rook adjacency from `rook_neighbors_unique`, and the join on `(neighbor_id, year)` matches the same cell-year values. NA handling (`!is.na(value)`) mirrors the original. |
| **Trained RF model** | The model object is never touched. Only the input feature columns are recomputed (identically), then `predict()` is called. |
| **Column names** | The naming convention `{var}_neighbor_max/min/mean` is preserved so the RF model's expected feature names match. Adjust the suffix pattern if your original pipeline used a different convention. |

---

## Expected Performance

| Step | Original | Optimized |
|---|---|---|
| Build topology | Hours (6.46M `paste` + hash lookups) | **~2â€“5 seconds** (1.37M-row `rbindlist`) |
| Per-variable stats | Hours (6.46M `lapply`) | **~15â€“45 seconds** (vectorized join + group) |
| Total (5 vars) | **86+ hours** | **~2â€“5 minutes** |
| Peak RAM | High (6.46M-element list) | Moderate (~1.37M Ã— 28 intermediate rows) |

The key transformation is replacing **6.46 million interpreted R loop iterations** with **vectorized `data.table` binary-search joins and grouped C-level aggregations**, while factoring out the static spatial topology into a small, reusable edge table.
END_RESPONSE>>>
