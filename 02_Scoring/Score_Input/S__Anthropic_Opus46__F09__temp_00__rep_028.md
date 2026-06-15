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
run_id: Anthropic_F09_T00_R028
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F09
prompt_family_slug: ADJACENCY_TABLE_OPTION
family_label: Adjacency-table option
family_group: Representation cue
replicate: 28
raw_output_file: Experiment_Output/Anthropic/Temp_0.0/output_F09_Anthropic_temp_00_rep_028.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The current implementation is extremely slow (~86+ hours) for two interconnected reasons:

### 1. `build_neighbor_lookup` is O(NÂ²)-like in practice
This function iterates over **all ~6.46 million cell-year rows** with `lapply`, and for each row it:
- Looks up the cell's spatial neighbors (fine).
- Constructs **year-specific string keys** by pasting cell ID and year (expensive string allocation Ã— 6.46M).
- Performs **named-vector lookups** (`idx_lookup[neighbor_keys]`) â€” named vector lookup in R is O(n) per query against a character-named vector, not O(1) like a hash. With ~6.46M entries in `idx_lookup` and ~6.46M queries each touching ~4 neighbors on average, this is catastrophically slow.

The fundamental flaw: **the neighbor topology is purely spatial and time-invariant, yet the lookup is rebuilt per cell-year row**, fusing spatial structure with temporal indexing in the most expensive possible way.

### 2. `compute_neighbor_stats` uses per-row `lapply`
Even after the lookup is built, computing stats via `lapply` over 6.46M list elements with R-level anonymous functions is slow due to R's interpreter overhead.

### 3. Memory pressure
A list of 6.46M integer vectors (the neighbor lookup) consumes substantial RAM and creates GC pressure on a 16 GB machine.

---

## Optimization Strategy

**Core insight:** The neighbor graph is **time-invariant**. There are only 344,208 cells and ~1.37M directed rook-neighbor pairs. Build a **spatial-only edge table once**, then use vectorized joins and grouped aggregations per year to compute neighbor stats. This eliminates the 6.46M-row list entirely.

**Steps:**

1. **Build a spatial edge table** (`data.table` with columns `id`, `neighbor_id`) from the `spdep::nb` object â€” only ~1.37M rows.
2. **For each variable**, join the cell-year attribute table onto the edge table by `(neighbor_id, year)`, then compute grouped `max`, `min`, `mean` by `(id, year)` â€” fully vectorized via `data.table`.
3. **Join results back** to the main dataset.

This replaces 6.46M R-level iterations with a handful of vectorized `data.table` joins and group-bys, reducing runtime from ~86 hours to **minutes**.

---

## Working R Code

```r
library(data.table)

# ---------------------------------------------------------------
# STEP 1 â€” Build the time-invariant spatial edge table (once)
# ---------------------------------------------------------------
# rook_neighbors_unique : spdep nb object (list of integer vectors)
# id_order              : vector of cell IDs in the same order as the nb object

build_edge_table <- function(id_order, neighbors) {
  # neighbors[[i]] gives the indices (into id_order) of cell i's rook neighbors
  from_idx <- rep(seq_along(neighbors), lengths(neighbors))
  to_idx   <- unlist(neighbors, use.names = FALSE)

  # Remove zero-length / no-neighbor entries (spdep uses 0L for "no neighbors")
  valid <- to_idx > 0L
  from_idx <- from_idx[valid]
  to_idx   <- to_idx[valid]

  data.table(
    id          = id_order[from_idx],
    neighbor_id = id_order[to_idx]
  )
}

edge_dt <- build_edge_table(id_order, rook_neighbors_unique)
# ~1.37 M rows, two integer columns â€” trivial memory

cat("Edge table rows:", nrow(edge_dt), "\n")

# ---------------------------------------------------------------
# STEP 2 â€” Convert main data to data.table (if not already)
# ---------------------------------------------------------------
setDT(cell_data)
setkey(cell_data, id, year)          # index for fast joins

# ---------------------------------------------------------------
# STEP 3 â€” Vectorized neighbor-stat computation
# ---------------------------------------------------------------
compute_neighbor_features_fast <- function(cell_dt, edge_dt, var_name) {
  # Subset to only the columns we need for the join
  # (id, year, <var_name>)
  val_dt <- cell_dt[, .(id, year, val = get(var_name))]
  setnames(val_dt, "id", "neighbor_id")
  setkey(val_dt, neighbor_id, year)

  # Join: for every (id, neighbor_id) edge, attach the neighbor's value
  # in each year.  We add year from the focal cell.
  # Approach: expand edges Ã— years via join on neighbor_id + year.
  #
  # edge_dt has (id, neighbor_id).
  # val_dt  has (neighbor_id, year, val).
  # Merge on neighbor_id â†’ gives (id, neighbor_id, year, val).

  merged <- merge(edge_dt, val_dt, by = "neighbor_id",
                  allow.cartesian = TRUE, sort = FALSE)
  # merged now has columns: neighbor_id, id, year, val
  # Each row = one directed edge in one year with the neighbor's value.

  # Aggregate by (id, year)
  stats <- merged[!is.na(val),
                  .(nmax  = max(val),
                    nmin  = min(val),
                    nmean = mean(val)),
                  keyby = .(id, year)]

  # Name the output columns to match the original pipeline's convention
  max_col  <- paste0("neighbor_max_",  var_name)
  min_col  <- paste0("neighbor_min_",  var_name)
  mean_col <- paste0("neighbor_mean_", var_name)
  setnames(stats, c("nmax", "nmin", "nmean"),
                  c(max_col,  min_col,  mean_col))

  stats
}

# ---------------------------------------------------------------
# STEP 4 â€” Loop over the 5 source variables and join back
# ---------------------------------------------------------------
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {
  cat("Computing neighbor features for:", var_name, "...\n")
  stats_dt <- compute_neighbor_features_fast(cell_data, edge_dt, var_name)

  # Left-join the three new columns onto cell_data
  cell_data <- merge(cell_data, stats_dt, by = c("id", "year"), all.x = TRUE)
  cat("  Done.\n")
}

# ---------------------------------------------------------------
# STEP 5 â€” Predict with the existing (already-trained) RF model
# ---------------------------------------------------------------
# The trained model object (e.g., `rf_model`) is unchanged.
# All original predictor columns plus the new neighbor_* columns
# are now present in cell_data.

cell_data[, prediction := predict(rf_model, newdata = .SD)]

cat("Pipeline complete.\n")
```

---

## Why This Preserves Correctness

| Concern | Guarantee |
|---|---|
| **Numerical estimand** | `max`, `min`, `mean` are computed on exactly the same neighbor sets and the same variable values. The edge table encodes the identical rook-neighbor topology. No approximation is introduced. |
| **Trained RF model** | The model object is never modified or retrained. Only `predict()` is called. |
| **Missing-data handling** | `!is.na(val)` in the aggregation mirrors the original `neighbor_vals[!is.na(neighbor_vals)]` filter. Cells with no valid neighbors receive `NA` via the `all.x = TRUE` left join, matching the original `c(NA, NA, NA)` return. |

---

## Expected Performance

| Stage | Old | New |
|---|---|---|
| Build lookup / edge table | ~hours (6.46M string ops) | **< 1 second** (1.37M integer pairs) |
| Compute stats (5 vars) | ~80+ hours (6.46M Ã— 5 `lapply`) | **~2â€“5 minutes** (vectorized `data.table` join + group-by) |
| Total | **~86+ hours** | **~5â€“10 minutes** on a 16 GB laptop |

The speedup is roughly **500â€“1000Ã—**, achieved entirely by separating the time-invariant spatial topology from the time-varying attributes and replacing R-level iteration with vectorized `data.table` operations.
END_RESPONSE>>>
