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
run_id: Anthropic_F12_T10_R027
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F12
prompt_family_slug: FALSE_RBIND_DIAGNOSIS
family_label: False rbind diagnosis
family_group: Adversarial diagnosis
replicate: 27
raw_output_file: Experiment_Output/Anthropic/Temp_1.0/output_F12_Anthropic_temp_10_rep_027.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

## Diagnosis

The colleague is **partially correct but misidentifies the primary bottleneck**. Let me audit each claim:

### Claim 1: `do.call(rbind, result)` is the main bottleneck
`do.call(rbind, result)` on a list of ~6.46 million 3-element vectors is not trivial, but it's a single vectorized C-level operation. With 6.46M rows Ã— 3 columns, this completes in seconds. **This is NOT the main bottleneck.**

### Claim 2: "Repeated list binding inside `compute_neighbor_stats()`"
There is no repeated list binding inside `compute_neighbor_stats()`. It uses `lapply` to build a list in one pass, then `rbind`s once. The colleague is factually wrong about the mechanism. Each iteration returns a length-3 vector â€” there's no incremental growth. **This claim is incorrect.**

### The actual primary bottleneck: `build_neighbor_lookup()`

The function `build_neighbor_lookup()` is the **true deep bottleneck**, and it is catastrophic at this scale:

1. **Per-row string pasting and named-vector lookup**: For each of ~6.46 million rows, it calls `paste()` to construct neighbor keys, then performs **named character vector lookup** (`idx_lookup[neighbor_keys]`). Named vector lookup in R is O(n) per query via linear hashing over a vector of 6.46M names. With ~4 neighbors per cell on average, that's ~25.8 million key lookups against a 6.46M-length named vector.

2. **`as.character()` and `id_to_ref[]` lookups**: Similarly, `id_to_ref[as.character(data$id[i])]` is called 6.46 million times, each time converting to character and doing a named lookup over a 344,208-length vector.

3. **The `lapply` over 6.46M rows**: Each iteration does multiple string allocations (`paste`, `as.character`), named vector subsetting, and `is.na` filtering. The overhead per iteration is small but multiplied by 6.46 million it dominates runtime.

**Estimated cost**: `build_neighbor_lookup()` alone likely accounts for **70â€“85%+ of the 86-hour runtime**. The `compute_neighbor_stats()` function, while called 5 times, does only simple numeric subsetting per row and is comparatively fast.

### Secondary bottleneck: `compute_neighbor_stats()` is mildly inefficient
The `lapply` over 6.46M elements with per-element `max`/`min`/`mean` and `is.na` filtering is suboptimal but not catastrophic â€” each iteration is pure numeric ops on ~4 values. It can still be improved via vectorization.

---

## Optimization Strategy

1. **Replace `build_neighbor_lookup()` entirely** with a vectorized `data.table` merge/join approach. Instead of building a row-index lookup per cell-year, expand the neighbor relationships into a flat edge table, join against the data to get row indices, and group. This replaces 6.46M string-paste-and-lookup iterations with a single equi-join.

2. **Replace `compute_neighbor_stats()` with grouped `data.table` aggregation.** Once we have a flat edge table mapping each row to its neighbor rows, we can extract neighbor values, and compute `max`, `min`, `mean` in one grouped operation â€” fully vectorized.

3. **Process all 5 variables in a single pass** over the neighbor structure rather than looping.

4. **Preserve the trained Random Forest model** â€” we only change feature engineering, producing identical numerical output.

---

## Working R Code

```r
library(data.table)

optimize_neighbor_features <- function(cell_data, id_order, rook_neighbors_unique,
                                       neighbor_source_vars) {
  # Convert to data.table for speed; keep original class to restore later
  was_df <- is.data.frame(cell_data) && !is.data.table(cell_data)
  dt <- as.data.table(cell_data)

  # ---- Step 1: Assign row indices ----
  dt[, .row_idx := .I]

  # ---- Step 2: Build a flat edge table (focal_id -> neighbor_id) ----
  # rook_neighbors_unique is an nb object: list of integer vectors

  # id_order[i] is the cell id for the i-th element in the nb list
  # neighbors[[i]] contains indices (into id_order) of neighbors of cell i

  edge_list <- rbindlist(lapply(seq_along(rook_neighbors_unique), function(i) {
    nb <- rook_neighbors_unique[[i]]
    if (length(nb) == 0L || (length(nb) == 1L && nb[1] == 0L)) {
      return(NULL)
    }
    data.table(focal_id = id_order[i], neighbor_id = id_order[nb])
  }))

  # ---- Step 3: Expand edges across years via join on (focal_id, year) -> row_idx ----
  # Create a lookup from (id, year) -> .row_idx for focal rows
  focal_key <- dt[, .(focal_id = id, year, focal_row = .row_idx)]

  # Create a lookup from (id, year) -> .row_idx for neighbor rows
  neighbor_key <- dt[, .(neighbor_id = id, year, neighbor_row = .row_idx)]

  # Join: for each directed edge (focal_id, neighbor_id), cross with each year

  # First join edges to focal rows to get (focal_row, neighbor_id, year)
  setkey(edge_list, focal_id)
  setkey(focal_key, focal_id)
  expanded <- edge_list[focal_key, on = "focal_id", allow.cartesian = TRUE, nomatch = 0L]
  # expanded now has: focal_id, neighbor_id, year, focal_row

  # Now join to get neighbor_row for each (neighbor_id, year)
  setkey(expanded, neighbor_id, year)
  setkey(neighbor_key, neighbor_id, year)
  expanded <- neighbor_key[expanded, on = c("neighbor_id", "year"), nomatch = NA]
  # expanded now has: neighbor_id, year, neighbor_row, focal_id, focal_row

  # Drop rows where neighbor_row is NA (neighbor cell has no data for that year)
  expanded <- expanded[!is.na(neighbor_row)]

  # ---- Step 4: Compute stats for all variables at once ----
  # Extract neighbor values for all source vars at once
  # Build a matrix of neighbor values
  for (var_name in neighbor_source_vars) {
    # Get the neighbor values via vectorized subsetting
    expanded[, paste0("nval_", var_name) := dt[[var_name]][neighbor_row]]
  }

  # Group by focal_row and compute max, min, mean for each variable
  agg_exprs <- list()
  for (var_name in neighbor_source_vars) {
    nval_col <- paste0("nval_", var_name)
    agg_exprs[[paste0(var_name, "_neighbor_max")]]  <-
      substitute(max(x[!is.na(x)], na.rm = FALSE),
                 list(x = as.name(nval_col)))
    agg_exprs[[paste0(var_name, "_neighbor_min")]]  <-
      substitute(min(x[!is.na(x)], na.rm = FALSE),
                 list(x = as.name(nval_col)))
    agg_exprs[[paste0(var_name, "_neighbor_mean")]] <-
      substitute(mean(x[!is.na(x)], na.rm = FALSE),
                 list(x = as.name(nval_col)))
  }

  # Safer aggregation: handle all-NA groups
  # We use a custom function approach to match the original logic exactly
  stats <- expanded[, {
    out <- list()
    for (var_name in neighbor_source_vars) {
      nv <- get(paste0("nval_", var_name))
      nv <- nv[!is.na(nv)]
      if (length(nv) == 0L) {
        out[[paste0(var_name, "_neighbor_max")]]  <- NA_real_
        out[[paste0(var_name, "_neighbor_min")]]  <- NA_real_
        out[[paste0(var_name, "_neighbor_mean")]] <- NA_real_
      } else {
        out[[paste0(var_name, "_neighbor_max")]]  <- max(nv)
        out[[paste0(var_name, "_neighbor_min")]]  <- min(nv)
        out[[paste0(var_name, "_neighbor_mean")]] <- mean(nv)
      }
    }
    out
  }, by = focal_row]

  # ---- Step 5: Merge stats back to dt ----
  # First, handle rows with NO neighbors (they won't appear in stats)
  # Set column names for the new features
  new_cols <- unlist(lapply(neighbor_source_vars, function(v)
    paste0(v, c("_neighbor_max", "_neighbor_min", "_neighbor_mean"))))

  # Initialize all new columns to NA
  for (col_name in new_cols) {
    set(dt, j = col_name, value = NA_real_)
  }

  # Fill in computed values
  if (nrow(stats) > 0) {
    rows_with_stats <- stats$focal_row
    for (col_name in new_cols) {
      set(dt, i = rows_with_stats, j = col_name, value = stats[[col_name]])
    }
  }

  # Clean up helper column
  dt[, .row_idx := NULL]

  # ---- Step 6: Return in original format ----
  if (was_df) {
    return(as.data.frame(dt))
  }
  return(dt)
}

# ---- Usage ----
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cell_data <- optimize_neighbor_features(
  cell_data, id_order, rook_neighbors_unique, neighbor_source_vars
)

# The trained Random Forest model is untouched.
# Predictions use the same column names and identical numerical values.
# rf_predictions <- predict(trained_rf_model, newdata = cell_data)
```

---

## Summary

| Aspect | Colleague's Claim | Actual Finding |
|---|---|---|
| `do.call(rbind, result)` | "Main bottleneck" | **Seconds-level cost.** Not the bottleneck. |
| "Repeated list binding" in `compute_neighbor_stats` | Claimed present | **Does not exist.** `lapply` + single `rbind`. |
| `build_neighbor_lookup()` | Not mentioned | **TRUE bottleneck.** 6.46M iterations of string-paste + named-vector lookup (O(n) per query). Dominates the 86-hour runtime. |
| **Verdict** | **Rejected.** | Replace both functions with a single vectorized `data.table` join-and-aggregate pass. |

**Expected speedup**: From ~86+ hours down to **~5â€“20 minutes** on the same hardware, depending on RAM pressure. The `data.table` equi-join replaces millions of named-vector lookups with hash-based joins, and grouped aggregation replaces millions of per-row R function calls.
END_RESPONSE>>>
