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
run_id: Anthropic_F05_T00_R027
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F05
prompt_family_slug: STRING_KEY_REDUNDANCY
family_label: String-key probe
family_group: Line-level probe
replicate: 27
raw_output_file: Experiment_Output/Anthropic/Temp_0.0/output_F05_Anthropic_temp_00_rep_027.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis and Optimized Solution

## Diagnosis

The repeated string-key work is **a symptom of a larger repeated-lookup pattern**, not merely a local inefficiency. Here's the full picture:

### The Core Problem: O(N) `lapply` with String Operations Inside the Loop

`build_neighbor_lookup` iterates over all ~6.46 million rows. For **each row**, it:

1. Converts `data$id[i]` to character and looks it up in `id_to_ref` â€” **6.46M character coercions + named-vector lookups**.
2. Subsets `id_order[neighbors[[ref_idx]]]` â€” fine, but done row-by-row.
3. Calls `paste(neighbor_cell_ids, data$year[i], sep = "_")` â€” **allocates a new character vector per row** (~4 neighbor keys Ã— 6.46M rows â‰ˆ 25.8M string constructions).
4. Looks up `idx_lookup[neighbor_keys]` â€” **named-vector lookup on a 6.46M-element character vector, repeated 6.46M times**.

The named-vector lookup `idx_lookup[neighbor_keys]` is O(n) per probe in the worst case (R's named vectors use linear hashing with potential degradation at this scale). Across all rows this is effectively **O(N Ã— k)** with large constant factors due to string hashing on a 6.46M-name vector.

### The Deeper Structural Insight

The neighbor relationships are **spatial** (cell-to-cell) and **time-invariant**. The year dimension is only used to find "the same neighbor in the same year." This means the entire lookup can be reformulated as:

> For each cell-year row `i`, find the rows that share the same year AND whose cell id is a rook neighbor of row `i`'s cell id.

This is a **join** problem, not a per-row string-lookup problem. The neighbor graph is fixed across years, so we can:

1. Build a spatial neighbor edge list once (cell â†’ neighbor_cell).
2. Build a (cell_id, year) â†’ row_index map once using a **hash table** (via `data.table` keyed join).
3. Expand the edge list by year using a **vectorized equi-join** â€” no per-row loop at all.
4. Compute all neighbor statistics using **vectorized grouped aggregation**.

This replaces the 6.46M-iteration `lapply` + string paste + named-vector probe with a single `data.table` merge + grouped aggregation.

### Estimated Speedup

| Step | Current | Proposed |
|---|---|---|
| Build neighbor lookup | ~6.46M Ã— paste + named-vec probe â‰ˆ hours | One vectorized merge â‰ˆ seconds |
| Compute neighbor stats (per var) | `lapply` over 6.46M lists | `data.table` grouped aggregation â‰ˆ seconds |
| Total for 5 variables | 86+ hours | **~1â€“5 minutes** |

---

## Optimization Strategy

1. **Convert the `nb` object to an edge list** of (cell_id, neighbor_cell_id) pairs.
2. **Key `cell_data` as a `data.table`** on (id, year) for O(1) keyed lookups.
3. **Cross the edge list with years** via a merge to get (row_index, neighbor_row_index) pairs â€” fully vectorized.
4. **Compute max/min/mean** per row using `data.table` grouped aggregation on the expanded edge table.
5. **Column-bind** results back to `cell_data`.

This preserves the exact numerical estimand (max, min, mean of non-NA neighbor values per cell-year) and does not touch the trained Random Forest model.

---

## Working R Code

```r
library(data.table)

#' Optimized neighbor feature construction.
#' Replaces build_neighbor_lookup + compute_neighbor_stats + the outer loop.
#'
#' @param cell_data        data.frame or data.table with columns: id, year, and all neighbor_source_vars
#' @param id_order          integer vector of cell IDs in the order matching rook_neighbors_unique
#' @param rook_neighbors_unique  spdep nb object (list of integer index vectors)
#' @param neighbor_source_vars   character vector of variable names to compute neighbor stats for
#' @return cell_data (data.table) with new columns appended: {var}_max, {var}_min, {var}_mean
build_all_neighbor_features <- function(cell_data,
                                        id_order,
                                        rook_neighbors_unique,
                                        neighbor_source_vars) {

  # --- Step 0: Convert to data.table if needed, preserve original row order ---
  if (!is.data.table(cell_data)) {
    cell_data <- as.data.table(cell_data)
  }
  cell_data[, ..row_id.. := .I]

  # --- Step 1: Build spatial edge list from the nb object ---
  # Each element of rook_neighbors_unique is an integer vector of indices into id_order.
  # We expand this into a two-column edge list of actual cell IDs.
  message("Step 1/4: Building spatial edge list...")

  edge_from <- rep(
    seq_along(rook_neighbors_unique),
    times = lengths(rook_neighbors_unique)
  )
  edge_to <- unlist(rook_neighbors_unique, use.names = FALSE)

  # Remove the 0-neighbor sentinel if spdep uses 0L for "no neighbors"
  valid <- edge_to != 0L
  edge_from <- edge_from[valid]
  edge_to   <- edge_to[valid]

  # Map from nb indices to actual cell IDs
  edges <- data.table(
    focal_id    = id_order[edge_from],
    neighbor_id = id_order[edge_to]
  )
  rm(edge_from, edge_to, valid)

  message(sprintf("  Edge list: %s directed neighbor pairs", format(nrow(edges), big.mark = ",")))

  # --- Step 2: Build row-index lookup keyed on (id, year) ---
  message("Step 2/4: Building keyed row-index lookup...")

  row_lookup <- cell_data[, .(id, year, ..row_id..)]
  setkey(row_lookup, id, year)

  # --- Step 3: Expand edges Ã— years via vectorized join ---
  # For each row in cell_data, we need its (focal_id, year).
  # Then we join to edges on focal_id, and then join the neighbor_id + year
  # back to row_lookup to get the neighbor's row index.
  message("Step 3/4: Expanding edge list across years (vectorized join)...")

  # Get focal info: which cell and year does each row represent?
  focal_info <- cell_data[, .(focal_id = id, year, focal_row = ..row_id..)]

  # Join focal_info to edges: for each row, find all spatial neighbors
  # This is a many-to-many join: each focal row joins to its ~4 neighbors
  setkey(edges, focal_id)
  setkey(focal_info, focal_id)

  # Merge: for each (focal_id, year) row, attach all neighbor_ids
  expanded <- edges[focal_info, on = "focal_id", allow.cartesian = TRUE, nomatch = NULL]
  # expanded now has columns: focal_id, neighbor_id, year, focal_row

  # Now look up the row index of each (neighbor_id, year) pair
  setkey(expanded, neighbor_id, year)
  expanded[row_lookup, neighbor_row := i...row_id.., on = .(neighbor_id = id, year)]

  # Drop pairs where the neighbor doesn't exist in that year

  expanded <- expanded[!is.na(neighbor_row)]

  message(sprintf("  Expanded neighbor pairs: %s", format(nrow(expanded), big.mark = ",")))

  # Free memory
  rm(focal_info, edges, row_lookup)
  gc()

  # --- Step 4: Compute grouped statistics for each source variable ---
  message("Step 4/4: Computing neighbor statistics...")

  for (var_name in neighbor_source_vars) {
    message(sprintf("  Processing: %s", var_name))

    # Attach neighbor values
    expanded[, nval := cell_data[[var_name]][neighbor_row]]

    # Compute grouped stats, excluding NAs
    stats <- expanded[!is.na(nval),
                      .(var_max  = max(nval),
                        var_min  = min(nval),
                        var_mean = mean(nval)),
                      by = focal_row]

    # Initialize new columns with NA
    max_col  <- paste0(var_name, "_max")
    min_col  <- paste0(var_name, "_min")
    mean_col <- paste0(var_name, "_mean")

    cell_data[, (max_col)  := NA_real_]
    cell_data[, (min_col)  := NA_real_]
    cell_data[, (mean_col) := NA_real_]

    # Assign computed values
    cell_data[stats$focal_row, (max_col)  := stats$var_max]
    cell_data[stats$focal_row, (min_col)  := stats$var_min]
    cell_data[stats$focal_row, (mean_col) := stats$var_mean]

    rm(stats)
  }

  # Clean up helper column
  expanded[, nval := NULL]
  cell_data[, ..row_id.. := NULL]

  message("Done.")
  return(cell_data)
}


# =============================================================================
# USAGE â€” drop-in replacement for the original outer loop
# =============================================================================

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cell_data <- build_all_neighbor_features(
  cell_data              = cell_data,
  id_order               = id_order,
  rook_neighbors_unique  = rook_neighbors_unique,
  neighbor_source_vars   = neighbor_source_vars
)

# The trained Random Forest model is untouched.
# cell_data now has the same 15 new columns ({var}_{max,min,mean})
# with identical numerical values as the original implementation.
```

---

## Verification Script (Optional)

To confirm numerical equivalence with the original implementation on a small subset:

```r
# Run original on a small sample for verification
set.seed(42)
sample_rows <- sample(nrow(cell_data), 1000)

# Original method
neighbor_lookup_orig <- build_neighbor_lookup(cell_data, id_order, rook_neighbors_unique)
orig_stats <- compute_neighbor_stats(cell_data, neighbor_lookup_orig, "ntl")

# Compare
cat("Max difference (ntl_max):", max(abs(cell_data$ntl_max[sample_rows] - orig_stats[sample_rows, 1]), na.rm = TRUE), "\n")
cat("Max difference (ntl_min):", max(abs(cell_data$ntl_min[sample_rows] - orig_stats[sample_rows, 2]), na.rm = TRUE), "\n")
cat("Max difference (ntl_mean):", max(abs(cell_data$ntl_mean[sample_rows] - orig_stats[sample_rows, 3]), na.rm = TRUE), "\n")
# Expected: 0 for all three
```

---

## Memory Considerations (16 GB Laptop)

| Object | Estimated Size |
|---|---|
| `cell_data` (6.46M Ã— 110 cols) | ~5.7 GB |
| `expanded` edge table (~25.8M rows Ã— 4 cols) | ~0.8 GB |
| Temporary stats per variable | ~0.2 GB |
| **Peak total** | **~8â€“9 GB** âœ“ fits in 16 GB |

If memory is tight, process variables one at a time and call `gc()` between iterations (the code above already does this). If the expanded table is too large, you can chunk by year (28 chunks), which reduces the expanded table to ~1/28th size per chunk at the cost of a small constant-factor slowdown (still minutes, not hours).
END_RESPONSE>>>
