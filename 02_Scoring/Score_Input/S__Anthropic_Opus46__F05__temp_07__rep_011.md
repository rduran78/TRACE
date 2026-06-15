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
run_id: Anthropic_F05_T07_R011
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F05
prompt_family_slug: STRING_KEY_REDUNDANCY
family_label: String-key probe
family_group: Line-level probe
replicate: 11
raw_output_file: Experiment_Output/Anthropic/Temp_0.7/output_F05_Anthropic_temp_07_rep_011.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis and Optimized Solution

## Diagnosis

The repeated string-key work is **a symptom of a larger repeated-lookup pattern**, not merely a local inefficiency. Here's the full picture:

### Local Inefficiency
- `paste(data$id, data$year, sep = "_")` creates ~6.46M strings to build `idx_lookup` â€” this happens once, so it's O(n) and tolerable in isolation.
- But inside the `lapply` over all 6.46M rows, `paste(neighbor_cell_ids, data$year[i], sep = "_")` is called per row, generating ~1.37 billion string constructions and hash lookups across all rows. This is the dominant cost.

### Broader Algorithmic Problem
The entire string-keyed lookup is an **unnecessary indirection**. The fundamental operation is: *for each cell-year row, find the rows that share the same year and belong to a neighboring cell*. This is a **join** â€” specifically a relational equi-join on `(year, neighbor_id)` â€” being implemented as a per-row hash-table probe via string keys. The correct reformulation is:

1. **Build the neighbor lookup once as an integer-indexed structure** (cell index â†’ neighbor cell indices), which is already available from `rook_neighbors_unique`.
2. **Build a year-indexed structure** (year â†’ row indices per cell), so that finding "the row for cell `j` in year `t`" is an O(1) integer lookup, not a string-hash probe.
3. **Vectorize the neighbor-stat computation** using `data.table` grouped operations or a single sparse-matrix multiplication, eliminating the per-row `lapply` entirely.

The best reformulation recognizes that the neighbor mean/max/min over a variable is equivalent to a **sparse matrix operation**: if `W` is the row-normalized (or raw) neighbor adjacency matrix in cell-year space, then `neighbor_mean = W %*% x`. Max and min require a grouped operation but can still be vectorized.

---

## Optimization Strategy

| Aspect | Current | Proposed |
|---|---|---|
| Neighbor resolution | Per-row string paste + hash lookup (6.46M Ã— ~4 neighbors) | One-time integer expansion via `data.table` join |
| Stats computation | `lapply` over 6.46M rows, R-level loop | Vectorized `data.table` grouped aggregation |
| Repetition across vars | Neighbor lookup reused, but stats loop is R-level | Same neighbor index structure, fully vectorized stats |
| Estimated time | 86+ hours | **Minutes** (dominated by `data.table` grouped ops) |
| RAM | String vector ~6.46M Ã— 20 bytes + hash table | Integer edge list ~50M rows Ã— 3 cols â‰ˆ ~1.2 GB |

---

## Working R Code

```r
library(data.table)

#' Build a fully vectorized neighbor feature pipeline.
#' Preserves the exact numerical estimand (max, min, mean of neighbor values).
#'
#' @param cell_data       data.frame or data.table with columns: id, year, and all neighbor_source_vars
#' @param id_order        integer vector of cell IDs in the order matching rook_neighbors_unique
#' @param rook_neighbors  spdep::nb object (list of integer index vectors into id_order)
#' @param neighbor_source_vars character vector of variable names
#' @return data.table with original columns plus neighbor feature columns appended

build_all_neighbor_features <- function(cell_data,
                                        id_order,
                                        rook_neighbors,
                                        neighbor_source_vars) {

  dt <- as.data.table(cell_data)

  # ---- Step 1: Build directed edge list (focal_cell_idx -> neighbor_cell_id) ----
  # rook_neighbors[[k]] gives integer indices into id_order for the neighbors of id_order[k]
  message("Building edge list...")

  # Pre-allocate edge list vectors
  n_edges <- sum(lengths(rook_neighbors))
  focal_ids   <- integer(n_edges)
  neighbor_ids <- integer(n_edges)

  pos <- 1L
  for (k in seq_along(rook_neighbors)) {
    nb_idx <- rook_neighbors[[k]]
    if (length(nb_idx) == 0L) next
    n <- length(nb_idx)
    focal_ids[pos:(pos + n - 1L)]    <- id_order[k]
    neighbor_ids[pos:(pos + n - 1L)] <- id_order[nb_idx]
    pos <- pos + n
  }

  edges <- data.table(focal_id = focal_ids, neighbor_id = neighbor_ids)
  rm(focal_ids, neighbor_ids)

  # ---- Step 2: Assign a row index to each cell-year observation ----
  dt[, row_idx := .I]

  # ---- Step 3: For each focal row, find all neighbor rows (same year) via join ----
  message("Joining edges to panel on year...")

  # Keyed lookup: for a given (id, year), what is the row_idx?
  id_year_key <- dt[, .(id, year, row_idx)]

  # Join edges with focal rows to get (focal_row_idx, neighbor_id, year)
  # Then join with id_year_key to get neighbor_row_idx
  # This replaces the entire per-row lapply + string-key lookup.

  # Focal side: get year for each focal cell-year
  setkey(id_year_key, id, year)

  # Expand edges Ã— years:
  # For every edge (focal_id, neighbor_id) and every year the focal_id appears,
  # find the neighbor_id's row in that same year.

  # Get unique (focal_id, year, focal_row_idx)
  focal_rows <- dt[, .(focal_id = id, year, focal_row_idx = row_idx)]

  # Join: for each focal row, attach its neighbor cell IDs

  setkey(edges, focal_id)
  setkey(focal_rows, focal_id)

  # This is the big expansion: ~6.46M rows Ã— ~4 neighbors = ~25.8M rows
  message("Expanding focal-neighbor-year combinations...")
  expanded <- edges[focal_rows, on = .(focal_id), allow.cartesian = TRUE, nomatch = NULL]
  # Columns: focal_id, neighbor_id, year, focal_row_idx

  # Now resolve neighbor_id + year -> neighbor_row_idx
  setnames(id_year_key, c("id", "year", "row_idx"), c("neighbor_id", "year", "neighbor_row_idx"))
  setkey(id_year_key, neighbor_id, year)
  setkey(expanded, neighbor_id, year)

  message("Resolving neighbor row indices...")
  expanded <- id_year_key[expanded, on = .(neighbor_id, year), nomatch = NA]
  # Drop rows where neighbor has no observation in that year
  expanded <- expanded[!is.na(neighbor_row_idx)]

  # ---- Step 4: Compute neighbor stats vectorized ----
  message("Computing neighbor statistics...")

  for (var_name in neighbor_source_vars) {
    message("  Processing: ", var_name)

    # Pull the variable values aligned to neighbor rows
    expanded[, nval := dt[[var_name]][neighbor_row_idx]]

    # Compute grouped stats per focal row
    stats <- expanded[!is.na(nval),
                      .(
                        nb_max  = max(nval),
                        nb_min  = min(nval),
                        nb_mean = mean(nval)
                      ),
                      by = .(focal_row_idx)]

    # Initialize columns with NA
    max_col  <- paste0(var_name, "_neighbor_max")
    min_col  <- paste0(var_name, "_neighbor_min")
    mean_col <- paste0(var_name, "_neighbor_mean")

    dt[, (max_col)  := NA_real_]
    dt[, (min_col)  := NA_real_]
    dt[, (mean_col) := NA_real_]

    # Assign results back by row index
    dt[stats$focal_row_idx, (max_col)  := stats$nb_max]
    dt[stats$focal_row_idx, (min_col)  := stats$nb_min]
    dt[stats$focal_row_idx, (mean_col) := stats$nb_mean]

    expanded[, nval := NULL]
  }

  # Clean up
  dt[, row_idx := NULL]

  message("Done.")
  return(dt)
}
```

### Drop-in Replacement for the Outer Loop

```r
# ---- BEFORE (86+ hours) ----
# neighbor_lookup <- build_neighbor_lookup(cell_data, id_order, rook_neighbors_unique)
# neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
# for (var_name in neighbor_source_vars) {
#   cell_data <- compute_and_add_neighbor_features(cell_data, var_name, neighbor_lookup)
# }

# ---- AFTER (minutes) ----
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cell_data <- build_all_neighbor_features(
  cell_data            = cell_data,
  id_order             = id_order,
  rook_neighbors       = rook_neighbors_unique,
  neighbor_source_vars = neighbor_source_vars
)

# Convert back to data.frame if downstream code expects it
cell_data <- as.data.frame(cell_data)

# The trained Random Forest model is unchanged â€” only the feature table is rebuilt.
# Column names (e.g., ntl_neighbor_max, ntl_neighbor_min, ntl_neighbor_mean) must
# match what the RF model expects. Adjust the naming pattern above if needed.
```

---

## Why This Preserves Correctness

| Property | Guarantee |
|---|---|
| **Same neighbor set** | Edge list is built from the identical `rook_neighbors_unique` nb object and `id_order` mapping. |
| **Same year matching** | The `data.table` join on `(neighbor_id, year)` is the exact equivalent of `paste(neighbor_cell_ids, data$year[i], sep = "_")` lookup. |
| **Same NA handling** | `nomatch = NA` + `!is.na(neighbor_row_idx)` mirrors `result[!is.na(result)]`; `!is.na(nval)` mirrors `neighbor_vals[!is.na(neighbor_vals)]`. |
| **Same aggregates** | `max`, `min`, `mean` computed on the identical filtered neighbor value vectors. |
| **RF model untouched** | No retraining. The feature columns are numerically identical, so predictions are identical. |

## Complexity Comparison

| | Current | Proposed |
|---|---|---|
| String constructions | ~6.46M (lookup) + ~25.8B (per-row paste) | 0 |
| Hash probes | ~25.8B | 0 |
| Core operation | R-level `lapply` over 6.46M rows | `data.table` binary-search join + grouped aggregation (C-level) |
| Expected wall time | 86+ hours | **5â€“15 minutes** on 16 GB laptop |
| Peak RAM | Moderate (but slow) | ~2â€“4 GB for the expanded edge table |
END_RESPONSE>>>
