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
run_id: Anthropic_F11_T00_R012
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F11
prompt_family_slug: FALSE_RF_DIAGNOSIS
family_label: False RF diagnosis
family_group: Adversarial diagnosis
replicate: 12
raw_output_file: Experiment_Output/Anthropic/Temp_0.0/output_F11_Anthropic_temp_00_rep_012.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

## Diagnosis: Rejecting the Colleague's Claim

The colleague's claim that Random Forest inference is the main bottleneck is **incorrect**. The code evidence points clearly to the **spatial neighbor feature engineering** â€” specifically `build_neighbor_lookup()` and `compute_neighbor_stats()` â€” as the true bottleneck.

Here's why:

1. **`build_neighbor_lookup()`** iterates over **~6.46 million rows** with `lapply`, and for each row performs: a character coercion (`as.character`), a named-vector lookup (`id_to_ref[...]`), a `paste()` to construct neighbor keys, another named-vector lookup (`idx_lookup[neighbor_keys]`), and NA filtering. Named vector lookups in R are hash-based but carry significant per-call overhead at this scale. With ~6.46M rows and an average of ~4 rook neighbors per cell, this generates roughly **25.8 million string-paste and hash-lookup operations** inside a sequential `lapply`.

2. **`compute_neighbor_stats()`** is called **5 times** (once per neighbor source variable), each time iterating over all ~6.46M rows with `lapply`, subsetting a numeric vector, removing NAs, and computing `max`, `min`, and `mean`. That's another **~32.3 million** R-level function-call iterations total.

3. **Combined**: The neighbor pipeline performs on the order of **~38+ million R-interpreted loop iterations** with per-element string operations and small-vector statistics. This is the classic R anti-pattern that produces the estimated 86+ hour runtime.

4. **Random Forest inference**, by contrast, is a single call to `predict()` on a pre-trained model over a matrix of ~6.46M Ã— 110 features. The `ranger` or `randomForest` predict methods are implemented in C/C++ and are highly optimized. Even on a laptop, prediction on this scale typically completes in **seconds to a few minutes** â€” orders of magnitude faster than the neighbor feature loop.

**Verdict**: The bottleneck is the row-level R `lapply` loops performing string construction, named-vector lookups, and per-row summary statistics across millions of rows. The Random Forest inference is negligible by comparison.

---

## Optimization Strategy

1. **Replace `build_neighbor_lookup()`** with a vectorized `data.table` merge/join approach. Instead of looping row-by-row to paste keys and look up indices, we:
   - Expand the `nb` object into an edge list (cell_id â†’ neighbor_id) once.
   - Join this edge list with the data on (neighbor_id, year) to get row indices of neighbors directly.
   - This replaces millions of `paste` + named-vector lookups with a single keyed `data.table` join.

2. **Replace `compute_neighbor_stats()`** with a **grouped `data.table` aggregation**. Once we have an edge table mapping each row to its neighbor rows, we can gather neighbor values and compute `max`, `min`, `mean` in one vectorized grouped operation per variable â€” no `lapply` over 6.46M elements.

3. **Process all 5 variables** in the same join pass or with minimal repeated joins.

This reduces the runtime from ~86+ hours to an estimated **minutes** (typically 5â€“20 minutes depending on RAM pressure on a 16 GB laptop).

---

## Working R Code

```r
library(data.table)

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# STEP 0: Convert cell_data to data.table and assign a row index
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
setDT(cell_data)
cell_data[, row_idx := .I]

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# STEP 1: Build the edge list from the nb object (one-time cost)
#
# rook_neighbors_unique is an nb object: a list of length
# length(id_order), where element i contains integer indices into
# id_order of the neighbors of id_order[i].
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
build_edge_list <- function(id_order, nb_obj) {
  # Pre-allocate: count total edges
  n_edges <- sum(vapply(nb_obj, function(x) {
    if (length(x) == 1L && x[1] == 0L) 0L else length(x)
  }, integer(1)))

  from_id <- integer(n_edges)
  to_id   <- integer(n_edges)
  pos <- 1L

  for (i in seq_along(nb_obj)) {
    nbrs <- nb_obj[[i]]
    # spdep::nb encodes "no neighbors" as a single 0
    if (length(nbrs) == 1L && nbrs[1] == 0L) next
    n <- length(nbrs)
    from_id[pos:(pos + n - 1L)] <- id_order[i]
    to_id[pos:(pos + n - 1L)]   <- id_order[nbrs]
    pos <- pos + n
  }

  data.table(focal_id = from_id, neighbor_id = to_id)
}

edge_dt <- build_edge_list(id_order, rook_neighbors_unique)

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# STEP 2: Map each (focal_id, year) â†’ its row_idx, and each
#         (neighbor_id, year) â†’ the neighbor's row_idx, via keyed joins
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

# Lookup table: id + year â†’ row_idx
id_year_idx <- cell_data[, .(id, year, row_idx)]
setkey(id_year_idx, id, year)

# Get all unique years
all_years <- sort(unique(cell_data$year))

# Cross-join edges Ã— years, then join to get focal and neighbor row indices
# To manage memory on a 16 GB laptop, we process year-by-year.

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# Pre-allocate result columns in cell_data
for (var_name in neighbor_source_vars) {
  cell_data[, paste0("max_", var_name, "_neighbor") := NA_real_]
  cell_data[, paste0("min_", var_name, "_neighbor") := NA_real_]
  cell_data[, paste0("mean_", var_name, "_neighbor") := NA_real_]
}

# Process year by year to limit peak memory (~230K edges Ã— 1 year at a time)
setkey(edge_dt, focal_id)

for (yr in all_years) {

  # Subset rows for this year
  yr_data <- cell_data[year == yr, c("id", "row_idx", ..neighbor_source_vars)]
  setkey(yr_data, id)

  # Join edges to get focal row_idx
  # edge_dt: focal_id, neighbor_id
  # Join focal side
  edges_yr <- merge(edge_dt, yr_data[, .(id, focal_row_idx = row_idx)],
                    by.x = "focal_id", by.y = "id", allow.cartesian = TRUE)

  # Join neighbor side to get neighbor variable values
  neighbor_vals <- yr_data[, c("id", neighbor_source_vars), with = FALSE]
  setnames(neighbor_vals, "id", "neighbor_id")
  setkey(neighbor_vals, neighbor_id)
  setkey(edges_yr, neighbor_id)

  edges_yr <- merge(edges_yr, neighbor_vals, by = "neighbor_id",
                    allow.cartesian = FALSE)

  # Now edges_yr has columns:
  #   neighbor_id, focal_id, focal_row_idx, ntl, ec, pop_density, def, usd_est_n2
  # where the variable columns are the NEIGHBOR's values.

  # Group by focal_row_idx and compute stats for each variable
  agg_exprs <- list()
  for (var_name in neighbor_source_vars) {
    max_nm  <- paste0("max_", var_name, "_neighbor")
    min_nm  <- paste0("min_", var_name, "_neighbor")
    mean_nm <- paste0("mean_", var_name, "_neighbor")
    agg_exprs[[max_nm]]  <- call("max",  as.name(var_name), na.rm = TRUE)
    agg_exprs[[min_nm]]  <- call("min",  as.name(var_name), na.rm = TRUE)
    agg_exprs[[mean_nm]] <- call("mean", as.name(var_name), na.rm = TRUE)
  }

  # Build the aggregation call
  agg_call <- as.call(c(as.name("list"),
                         lapply(names(agg_exprs), function(nm) {
                           call("<-", as.name(nm), agg_exprs[[nm]])
                         })))

  # Simpler approach: use .SDcols
  stats_yr <- edges_yr[, {
    res <- list()
    for (vn in neighbor_source_vars) {
      v <- get(vn)
      v <- v[!is.na(v)]
      if (length(v) == 0L) {
        res[[paste0("max_", vn, "_neighbor")]]  <- NA_real_
        res[[paste0("min_", vn, "_neighbor")]]  <- NA_real_
        res[[paste0("mean_", vn, "_neighbor")]] <- NA_real_
      } else {
        res[[paste0("max_", vn, "_neighbor")]]  <- max(v)
        res[[paste0("min_", vn, "_neighbor")]]  <- min(v)
        res[[paste0("mean_", vn, "_neighbor")]] <- mean(v)
      }
    }
    res
  }, by = focal_row_idx]

  # Write results back into cell_data using row indices
  stat_cols <- setdiff(names(stats_yr), "focal_row_idx")
  cell_data[stats_yr$focal_row_idx, (stat_cols) := stats_yr[, ..stat_cols]]
}

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# STEP 3: Clean up helper column
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
cell_data[, row_idx := NULL]

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# STEP 4: Random Forest prediction (unchanged â€” not the bottleneck)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# The pre-trained RF model is loaded and used as-is.
# Example (preserving the original estimand):
#   predictions <- predict(rf_model, data = cell_data)
#   cell_data[, predicted := predictions$predictions]  # ranger
#   # or
#   cell_data[, predicted := predict(rf_model, newdata = cell_data)]  # randomForest
```

---

## Summary

| Aspect | Original | Optimized |
|---|---|---|
| **`build_neighbor_lookup`** | ~6.46M `lapply` iterations with `paste` + named-vector hash lookups | One-time edge-list construction + keyed `data.table` joins |
| **`compute_neighbor_stats`** | 5 Ã— ~6.46M `lapply` iterations computing per-row `max/min/mean` | Grouped `data.table` aggregation per year (~344K groups Ã— 28 years) |
| **String operations** | ~25.8M `paste()` calls | Zero â€” joins on integer keys |
| **Estimated runtime** | 86+ hours | ~5â€“20 minutes |
| **RF model** | Preserved (not retrained) | Preserved (not retrained) |
| **Numerical results** | Original `max/min/mean` neighbor stats | Identical `max/min/mean` neighbor stats |

The true bottleneck is the **row-level R-interpreted neighbor feature engineering**, not Random Forest inference. The fix is to vectorize the spatial lookups and aggregations using `data.table` keyed joins and grouped operations.
END_RESPONSE>>>
