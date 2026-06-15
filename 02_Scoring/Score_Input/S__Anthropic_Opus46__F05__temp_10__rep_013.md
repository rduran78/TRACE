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
run_id: Anthropic_F05_T10_R013
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F05
prompt_family_slug: STRING_KEY_REDUNDANCY
family_label: String-key probe
family_group: Line-level probe
replicate: 13
raw_output_file: Experiment_Output/Anthropic/Temp_1.0/output_F05_Anthropic_temp_10_rep_013.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis and Optimized Solution

## Diagnosis

The repeated string-key work is **a symptom of a larger repeated-lookup pattern**, not merely a local inefficiency. Here's why:

### The Core Problem: O(N) `lapply` with String Operations over 6.46M Rows

`build_neighbor_lookup` calls `lapply` over every row (6.46M iterations). Inside each iteration it:

1. **Pastes** `neighbor_cell_ids` with `data$year[i]` to create string keys â€” **per row**.
2. **Looks up** those keys in a named character vector (`idx_lookup`) â€” which is itself a hash-table lookup, but the paste/allocation cost dominates.

This means ~6.46M calls to `paste()` and named-vector subset, each producing small character vectors. The string allocation and hashing alone, across billions of neighbor pairs, accounts for the bulk of the 86+ hour estimate.

### But the Deeper Issue Is Architectural

The neighbor structure is **time-invariant** â€” rook contiguity doesn't change across years. Yet the code rebuilds the mapping from cell-to-cell neighbors into row-to-row neighbors by string-matching `(id, year)` pairs. This conflates two orthogonal dimensions:

- **Spatial neighbor topology** (fixed): cell `i` â†’ cells `{jâ‚, jâ‚‚, ...}`
- **Temporal alignment** (regular panel): every cell appears exactly once per year in a balanced panel

Because the panel is balanced (344,208 cells Ã— 28 years = 9,637,824 potential rows, ~6.46M observed), you can exploit **integer arithmetic** to convert a cell-level neighbor list into a row-level neighbor list without any string operations at all.

### Summary of Inefficiencies

| Layer | Problem | Magnitude |
|---|---|---|
| String paste in inner loop | O(N Ã— avg_neighbors) string allocations | ~6.46M Ã— ~4 = ~26M paste calls |
| Named-vector lookup | Hash-table lookup on character keys | ~26M lookups |
| `lapply` over rows | R-level loop, no vectorisation | 6.46M iterations |
| Redundant across years | Same spatial topology re-derived 28 times | 28Ã— redundant work |
| `compute_neighbor_stats` is fine | Already vectorized via list indexing | Not the bottleneck |

---

## Optimization Strategy

### Key Insight: Separate Space from Time

1. **Build a cell-index â†’ row-indices map** (one integer lookup table): For each cell index `c` (1..344,208), store the row numbers where that cell appears, keyed by year. In a balanced panel sorted by `(id, year)`, this is trivially an arithmetic offset. Even in an unbalanced panel, a single pass with `data.table` grouping suffices.

2. **Build the row-level neighbor list using only integer indexing**: For each unique `(cell, year)` combination, look up the cell's spatial neighbors (from `rook_neighbors_unique`), then for each neighbor find its row in the same year â€” all via integer vectors, no strings.

3. **Vectorize the statistics computation** using `data.table` for the final aggregation step, avoiding even the `lapply` in `compute_neighbor_stats`.

### Expected Speedup

- Eliminates all `paste()` and character hashing: **~100â€“500Ã— faster** for the lookup build.
- The neighbor lookup build should drop from ~86 hours to **minutes**.
- Stats computation was already tolerable but we vectorize it further for safety.

---

## Working R Code

```r
# =============================================================================
# OPTIMIZED NEIGHBOR FEATURE CONSTRUCTION
# =============================================================================
# Prerequisites: data.table
# Preserves: all original numerical outputs (max, min, mean of neighbor values)
# Preserves: the trained Random Forest model (no retraining needed)
# =============================================================================

library(data.table)

#' Build row-level neighbor lookup using integer arithmetic only.
#'
#' @param data         data.frame/data.table with columns `id` and `year`
#' @param id_order     integer vector of cell IDs in the order used by the nb object
#' @param neighbors    spdep nb object (list of integer vectors; indices into id_order)
#' @return             list of length nrow(data); each element is an integer vector
#'                     of row indices of that row's spatial neighbors in the same year.
build_neighbor_lookup_fast <- function(data, id_order, neighbors) {
  
  dt <- as.data.table(data[, c("id", "year")])
  dt[, row_idx := .I]
  
  # --- Step 1: Map cell ID -> position in id_order (spatial index) -----------
  id_to_spatial <- setNames(seq_along(id_order), as.character(id_order))
  dt[, spatial_idx := id_to_spatial[as.character(id)]]
  
  # --- Step 2: Create a lookup matrix: spatial_idx Ã— year -> row_idx ---------
  # This avoids all string operations. We use a keyed data.table for the join.
  setkey(dt, spatial_idx, year)
  
  # --- Step 3: Expand the neighbor list into an edge table (spatial level) ---
  # neighbors[[s]] gives spatial indices of neighbors of spatial cell s
  n_spatial <- length(id_order)
  
  # Build edge list: from_spatial -> to_spatial
  from_spatial <- rep(seq_len(n_spatial), lengths(neighbors))
  to_spatial   <- unlist(neighbors, use.names = FALSE)
  
  # Remove any 0-length or NA entries (defensive)
  valid <- !is.na(to_spatial)
  edge_dt <- data.table(from_spatial = from_spatial[valid],
                        to_spatial   = to_spatial[valid])
  
  # --- Step 4: Join edges with the panel to get row-level mappings -----------
  # For each row in dt, we need: its spatial_idx and year.
  # Then for each (spatial_idx, year), join to edge_dt to get neighbor spatial indices,
  # then join back to dt to get neighbor row indices in the same year.
  
  # Start from the data rows
  row_info <- dt[, .(row_idx, spatial_idx, year)]
  
  # Join to get neighbor spatial indices
  # row_info joins edge_dt on from_spatial = spatial_idx
  setkey(edge_dt, from_spatial)
  setkey(row_info, spatial_idx)
  
  # This is the key join: for each row, get all its neighbor spatial indices
  expanded <- edge_dt[row_info,
                      .(row_idx = i.row_idx,
                        year = i.year,
                        neighbor_spatial = x.to_spatial),
                      on = .(from_spatial = spatial_idx),
                      allow.cartesian = TRUE,
                      nomatch = NA]
  
  # Drop rows with no neighbors (NA from nomatch)
  expanded <- expanded[!is.na(neighbor_spatial)]
  
  # Now join to dt again to find the row index of the neighbor in the same year
  setkey(dt, spatial_idx, year)
  expanded[, neighbor_row := dt[.(neighbor_spatial, year),
                                 row_idx,
                                 nomatch = NA,
                                 mult = "first",
                                 on = .(spatial_idx, year)]
  ]
  # The above is not efficient for large joins; use a proper keyed join instead:
  expanded[, c("neighbor_row") := NULL]  
  
  neighbor_rows <- dt[expanded,
                      .(row_idx = i.row_idx, neighbor_row = x.row_idx),
                      on = .(spatial_idx = neighbor_spatial, year = year),
                      nomatch = NA]
  
  # Remove NAs (neighbor cell not present in that year)
  neighbor_rows <- neighbor_rows[!is.na(neighbor_row)]
  
  # --- Step 5: Collapse into a list indexed by row_idx -----------------------
  setkey(neighbor_rows, row_idx)
  
  n_rows <- nrow(data)
  
  # Split neighbor_row by row_idx
  result_list <- vector("list", n_rows)
  
  # Efficient split using data.table
  split_dt <- neighbor_rows[, .(neighbors = list(neighbor_row)), by = row_idx]
  
  # Fill the result list
  result_list[split_dt$row_idx] <- split_dt$neighbors
  
  # Fill remaining (no neighbors) with integer(0)
  empty <- which(vapply(result_list, is.null, logical(1)))
  if (length(empty) > 0) {
    result_list[empty] <- list(integer(0))
  }
  
  result_list
}


#' Compute neighbor stats (max, min, mean) for one variable â€” vectorized.
#'
#' @param data            data.frame with the source variable
#' @param neighbor_lookup list of integer row-index vectors (from build_neighbor_lookup_fast)
#' @param var_name        character: column name
#' @return                3-column matrix (max, min, mean), one row per data row
compute_neighbor_stats_fast <- function(data, neighbor_lookup, var_name) {
  
  vals <- data[[var_name]]
  n <- length(neighbor_lookup)
  
  # Build an edge table: (row_idx, neighbor_row)
  row_idx_rep <- rep(seq_len(n), lengths(neighbor_lookup))
  neighbor_row <- unlist(neighbor_lookup, use.names = FALSE)
  
  if (length(neighbor_row) == 0) {
    # No neighbors at all â€” return all NA
    return(matrix(NA_real_, nrow = n, ncol = 3,
                  dimnames = list(NULL, c("max", "min", "mean"))))
  }
  
  dt <- data.table(row_idx = row_idx_rep,
                   neighbor_val = vals[neighbor_row])
  
  # Remove NAs in neighbor values
  dt <- dt[!is.na(neighbor_val)]
  
  # Aggregate
  agg <- dt[, .(nmax  = max(neighbor_val),
                nmin  = min(neighbor_val),
                nmean = mean(neighbor_val)),
            by = row_idx]
  
  # Allocate result matrix
  result <- matrix(NA_real_, nrow = n, ncol = 3)
  colnames(result) <- c("max", "min", "mean")
  result[agg$row_idx, 1] <- agg$nmax
  result[agg$row_idx, 2] <- agg$nmin
  result[agg$row_idx, 3] <- agg$nmean
  
  result
}


#' Add neighbor features for one variable to the data.
#'
#' Drop-in replacement for compute_and_add_neighbor_features.
#' Column naming convention preserved: {var_name}_neighbor_{max|min|mean}
compute_and_add_neighbor_features_fast <- function(data, var_name, neighbor_lookup) {
  
  stats <- compute_neighbor_stats_fast(data, neighbor_lookup, var_name)
  
  data[[paste0(var_name, "_neighbor_max")]]  <- stats[, 1]
  data[[paste0(var_name, "_neighbor_min")]]  <- stats[, 2]
  data[[paste0(var_name, "_neighbor_mean")]] <- stats[, 3]
  
  data
}


# =============================================================================
# MAIN EXECUTION (drop-in replacement for the original outer loop)
# =============================================================================

# --- Build neighbor lookup ONCE (fast) ---
message("Building row-level neighbor lookup (integer-only)...")
t0 <- Sys.time()
neighbor_lookup <- build_neighbor_lookup_fast(
  cell_data, id_order, rook_neighbors_unique
)
message("Neighbor lookup built in ", round(difftime(Sys.time(), t0, units = "mins"), 1), " min")

# --- Compute and attach neighbor features ---
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {
  message("Computing neighbor stats for: ", var_name)
  t1 <- Sys.time()
  cell_data <- compute_and_add_neighbor_features_fast(
    cell_data, var_name, neighbor_lookup
  )
  message("  Done in ", round(difftime(Sys.time(), t1, units = "secs"), 1), " sec")
}

message("All neighbor features computed. Proceeding to prediction with existing RF model.")

# The trained Random Forest model is unchanged â€” use it directly:
# predictions <- predict(rf_model, newdata = cell_data)
```

---

## Why This Preserves Correctness

| Property | Original | Optimized |
|---|---|---|
| Neighbor definition | Rook contiguity from `spdep::nb` | Same `spdep::nb` object used |
| Year matching | String paste `id_year` â†’ lookup | Keyed join on `(spatial_idx, year)` â€” identical semantics |
| Missing handling | `!is.na(result)` filters missing neighbors | `nomatch = NA` then `!is.na()` â€” identical |
| Stats computed | `max`, `min`, `mean` of non-NA neighbor values | Identical aggregation |
| Column names | `{var}_neighbor_{max,min,mean}` | Identical naming |
| RF model | Not touched | Not touched |

## Expected Performance

| Step | Original Estimate | Optimized Estimate |
|---|---|---|
| `build_neighbor_lookup` | ~70â€“86 hours (string ops in R loop) | ~2â€“5 minutes (data.table keyed joins) |
| `compute_neighbor_stats` (Ã—5 vars) | ~minutes | ~30â€“60 seconds total |
| **Peak RAM** | ~8â€“10 GB (string vectors) | ~6â€“8 GB (integer edge tables) |
| **Total** | **~86+ hours** | **~5â€“10 minutes** |

The bottleneck was never the algorithm's *correctness* â€” it was the choice of string-keyed lookups inside an R-level loop over 6.46M rows instead of a single vectorized integer join over the Cartesian product of edges Ã— years.
END_RESPONSE>>>
