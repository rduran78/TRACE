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
run_id: Anthropic_F12_T00_R006
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F12
prompt_family_slug: FALSE_RBIND_DIAGNOSIS
family_label: False rbind diagnosis
family_group: Adversarial diagnosis
replicate: 6
raw_output_file: Experiment_Output/Anthropic/Temp_0.0/output_F12_Anthropic_temp_00_rep_006.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

## Diagnosis: Rejecting the Colleague's Claim

The colleague's claim that `do.call(rbind, result)` and repeated list binding inside `compute_neighbor_stats()` are the main bottleneck is **incorrect**. While `do.call(rbind, ...)` on a list of 6.46 million small vectors is not free, it is a single matrix assembly step per variable â€” only 5 calls total. Each inner function returns a fixed-length-3 numeric vector, so there is no "repeated list binding" inside `compute_neighbor_stats()`; it uses `lapply`, which pre-allocates its output list. That cost is modest.

**The true, deeper bottleneck is `build_neighbor_lookup()`.**

Specifically:

1. **`paste()` key construction and named-vector lookup (`idx_lookup[neighbor_keys]`)** is called once per row (6.46M times) inside an `lapply`. Each call does string concatenation and character-based hash lookup for every neighbor of every row. With ~1.37M directed neighbor relationships spread across 6.46M cell-years, this means tens of millions of `paste` + name-match operations inside an interpreted R loop.

2. **`as.character(data$id[i])` and `id_to_ref[...]`** are called row-by-row (6.46M scalar lookups), each involving a character conversion and named-vector lookup.

3. **The entire lookup is rebuilt as a list of length 6.46M**, where each element is an integer vector of row indices. Storing and iterating over this structure is memory-heavy (~hundreds of MB to GBs of integer vectors plus list overhead).

In summary, `build_neighbor_lookup` performs **O(N Ã— average_neighbors)** string operations inside an interpreted loop over 6.46 million rows. This dwarfs the cost of `do.call(rbind, ...)` on 5 variables. The 86+ hour runtime is dominated by this function.

## Optimization Strategy

1. **Eliminate per-row string pasting and named-vector lookups entirely.** Replace the character-key approach with direct integer indexing via a merge/join.
2. **Vectorize the neighbor lookup construction** using `data.table` joins: expand the neighbor graph into a full edge list (cell_id, neighbor_id), join with the data on (id, year) to get row indices for neighbors, then group-by to form the lookup or â€” better â€” compute the stats directly.
3. **Compute neighbor stats directly from the joined edge table** using `data.table` grouped aggregation, eliminating both the 6.46M-element list and the `lapply` loop in `compute_neighbor_stats`.
4. **Process all 5 variables in one pass** over the joined table to avoid redundant joins.

This reduces the entire pipeline from ~86 hours to **minutes**.

## Working R Code

```r
library(data.table)

#' Optimized: build neighbor edge list and compute all neighbor features
#' in a single vectorized pass.
#'
#' @param cell_data       data.frame/data.table with columns: id, year, and all var columns
#' @param id_order        integer vector of cell IDs (index positions correspond to nb object)
#' @param rook_neighbors  spdep::nb object (list of integer index vectors)
#' @param neighbor_source_vars character vector of variable names
#' @return cell_data with neighbor features appended (preserves row order)
compute_all_neighbor_features <- function(cell_data,
                                          id_order,
                                          rook_neighbors,
                                          neighbor_source_vars) {

  dt <- as.data.table(cell_data)
  dt[, ..row_id := .I]
  
  # --- Step 1: Build a full directed edge list from the nb object ----------
  # Each entry rook_neighbors[[i]] gives the neighbor *indices* into id_order
  # for cell id_order[i].
  # We expand this into a two-column data.table: (focal_cell_id, neighbor_cell_id)
  
  n_cells <- length(id_order)
  
  # Efficient expansion: compute lengths, use rep + unlist
  lens <- lengths(rook_neighbors)                       # integer vector, length = n_cells
  focal_idx   <- rep(seq_len(n_cells), times = lens)    # index into id_order
  neighbor_idx <- unlist(rook_neighbors, use.names = FALSE)
  
  edges <- data.table(
    focal_id    = id_order[focal_idx],
    neighbor_id = id_order[neighbor_idx]
  )
  rm(focal_idx, neighbor_idx, lens)
  
  # --- Step 2: Join edges with data to get neighbor values -----------------
  # For every (focal_id, year) row, we need the variable values of its neighbors
  # in the same year.
  
  # Create a slim table for the focal side: (focal_id = id, year, row_id)
  focal <- dt[, .(focal_id = id, year, ..row_id)]
  
  # Cross edges with years: for each edge (focal_id, neighbor_id), 
  # we need all years. But it's cheaper to join focal -> edges -> neighbor_data.
  
  # Join focal rows to edges to get (row_id, neighbor_id, year)
  setkey(edges, focal_id)
  setkey(focal, focal_id)
  
  # This produces one row per (focal_row, neighbor_cell) combination
  joined <- edges[focal, on = "focal_id", allow.cartesian = TRUE, nomatch = NULL]
  # joined has columns: focal_id, neighbor_id, year, ..row_id
  
  rm(focal, edges)
  gc()
  
  # Now join to get the neighbor's variable values in the same year
  # Prepare neighbor data: subset to needed columns only
  neighbor_cols <- c("id", "year", neighbor_source_vars)
  neighbor_data <- dt[, ..neighbor_cols, with = FALSE]
  setnames(neighbor_data, "id", "neighbor_id")
  
  setkey(joined, neighbor_id, year)
  setkey(neighbor_data, neighbor_id, year)
  
  joined <- neighbor_data[joined, on = c("neighbor_id", "year"), nomatch = NA]
  
  rm(neighbor_data)
  gc()
  
  # --- Step 3: Grouped aggregation to compute max, min, mean per focal row -
  # Group by ..row_id (the original row in cell_data)
  
  # Build aggregation expressions for all variables at once
  agg_exprs <- list()
  for (v in neighbor_source_vars) {
    v_sym <- as.name(v)
    prefix <- v
    agg_exprs[[paste0("neighbor_max_", prefix)]] <-
      bquote(as.numeric(max(.(v_sym), na.rm = TRUE)))
    agg_exprs[[paste0("neighbor_min_", prefix)]] <-
      bquote(as.numeric(min(.(v_sym), na.rm = TRUE)))
    agg_exprs[[paste0("neighbor_mean_", prefix)]] <-
      bquote(mean(.(v_sym), na.rm = TRUE))
  }
  
  # Convert to a single call
  agg_call <- as.call(c(as.name("list"), agg_exprs))
  
  stats <- joined[, eval(agg_call), by = ..row_id]
  
  rm(joined)
  gc()
  
  # Replace Inf/-Inf (from max/min on all-NA) with NA
  for (col_name in names(stats)) {
    if (col_name == "..row_id") next
    set(stats, which(is.infinite(stats[[col_name]])), col_name, NA_real_)
  }
  
  # --- Step 4: Merge back to original data in original row order -----------
  setkey(stats, ..row_id)
  
  # For rows with no neighbors (not present in stats), we need NAs
  # Left join dt onto stats
  new_cols <- setdiff(names(stats), "..row_id")
  dt[stats, (new_cols) := mget(new_cols), on = "..row_id"]
  
  # Drop helper column
  dt[, ..row_id := NULL]
  
  # Convert back to data.frame if the input was a data.frame
  if (!is.data.table(cell_data)) {
    return(as.data.frame(dt))
  }
  return(dt)
}

# ---- Usage (drop-in replacement for the original outer loop) ----

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cell_data <- compute_all_neighbor_features(
  cell_data,
  id_order,
  rook_neighbors_unique,
  neighbor_source_vars
)

# The trained Random Forest model is untouched.
# The output columns (neighbor_max_*, neighbor_min_*, neighbor_mean_*) 
# are numerically identical to the originals, preserving the estimand.
```

## Why This Works and What It Preserves

| Concern | Status |
|---|---|
| **Numerical equivalence** | `max`, `min`, `mean` with `na.rm=TRUE` on the same neighbor sets produce identical values. Infâ†’NA handling mirrors the original `length(neighbor_vals)==0` guard. |
| **Trained Random Forest** | No model retraining. The code only reconstructs the feature columns that the model expects as inputs. Column names match the original pattern. |
| **Correctness** | The join on `(neighbor_id, year)` replicates the original `paste(id, year)` key logic exactly, but via integer-indexed equi-joins instead of character hashing. |
| **Performance** | The ~1.37M edges Ã— 28 years â‰ˆ 38M rows in the joined table are handled by `data.table`'s radix-sort joins and grouped C-level aggregation. Expected runtime: **5â€“15 minutes** on the described laptop, down from 86+ hours â€” a ~300â€“1000Ã— speedup. |
| **Memory** | The joined table at ~38M rows Ã— ~8 columns â‰ˆ 2â€“3 GB, well within 16 GB RAM. Intermediate objects are freed with `rm(); gc()`. |
END_RESPONSE>>>
