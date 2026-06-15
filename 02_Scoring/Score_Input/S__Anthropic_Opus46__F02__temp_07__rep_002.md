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
run_id: Anthropic_F02_T07_R002
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F02
prompt_family_slug: CODE_ONLY_NO_HINT
family_label: Code only, no hint
family_group: Code-visible neutral
replicate: 2
raw_output_file: Experiment_Output/Anthropic/Temp_0.7/output_F02_Anthropic_temp_07_rep_002.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The core bottleneck is in **`build_neighbor_lookup`**, which uses `lapply` over ~6.46 million rows, performing per-row string pasting and named-vector lookups. Named vector lookups in R (`idx_lookup[neighbor_keys]`) are hash-table lookups, but doing millions of `paste` + lookup operations inside a sequential `lapply` is extremely slow. The second bottleneck is **`compute_neighbor_stats`**, which iterates over the same 6.46 million rows, subsetting and computing `max/min/mean` per row. Together, for 5 variables, this is ~32.3 million row-level R function calls. The `do.call(rbind, result)` on a 6.46-million-element list is also costly.

**Summary of problems:**

| Component | Issue |
|---|---|
| `build_neighbor_lookup` | 6.46M iterations of `paste` + named-vector hash lookup; builds a 6.46M-element list of integer vectors |
| `compute_neighbor_stats` | 6.46M iterations Ã— 5 variables = 32.3M R-level subsetting + aggregation calls |
| `do.call(rbind, ...)` | Binding 6.46M small vectors into a matrix is slow |
| Memory | The neighbor lookup list alone (6.46M elements, each a variable-length integer vector) consumes several GB |

---

## Optimization Strategy

**Replace row-level R loops with vectorized `data.table` joins and group-by aggregations.**

The key insight: the neighbor lookup is conceptually a **join**. Each `(cell, year)` row needs to find its neighbors' `(neighbor_cell, year)` rows and aggregate their values. This is a classic equi-join + grouped aggregation, which `data.table` handles in highly optimized C code.

**Steps:**

1. **Build an edge table** (a two-column `data.table` of `id â†’ neighbor_id`) from the `nb` object â€” done once, ~1.37M rows.
2. **Join** the edge table to the panel data on `(neighbor_id, year)` to get neighbor values â€” a single keyed join, no R-level loop.
3. **Aggregate** `max`, `min`, `mean` grouped by the origin row â€” a single `data.table` group-by.
4. **Repeat** for each of the 5 variables (or do all at once).

This eliminates all 6.46M-iteration `lapply` calls and the giant list-of-vectors lookup structure.

**Expected improvement:** From ~86+ hours to roughly **5â€“20 minutes** on the same laptop, with peak RAM well within 16 GB.

---

## Working R Code

```r
library(data.table)

# ==============================================================
# STEP 1: Convert the nb object to a data.table edge list (once)
# ==============================================================
build_edge_table <- function(id_order, neighbors) {
  # id_order: vector of cell IDs in the order matching the nb object
  # neighbors: spdep nb object (list of integer index vectors)
  from_idx <- rep(seq_along(neighbors), lengths(neighbors))
  to_idx   <- unlist(neighbors, use.names = FALSE)
  data.table(
    id          = id_order[from_idx],
    neighbor_id = id_order[to_idx]
  )
}

edge_dt <- build_edge_table(id_order, rook_neighbors_unique)
# ~1.37M rows, two integer columns â€” trivial memory

# ==============================================================
# STEP 2: Convert cell_data to data.table (if not already)
# ==============================================================
if (!is.data.table(cell_data)) {
  cell_data <- as.data.table(cell_data)
}

# Create a row index to preserve original order and merge results back
cell_data[, .row_id := .I]

# ==============================================================
# STEP 3: Compute neighbor features for all variables at once
# ==============================================================
compute_all_neighbor_features <- function(cell_data, edge_dt, var_names) {
  # Subset to only the columns we need for the join
  # This keeps memory low during the join
  join_cols <- c("id", "year", var_names)
  dt_slim   <- cell_data[, ..join_cols]

  # Key the slim table for fast join on (id, year)
  setkey(dt_slim, id, year)

  # Join: for every (id, year) row, find its neighbors' values

  # edge_dt has (id, neighbor_id). We join edge_dt to dt_slim twice:
  #   - first to get the origin row's year
  #   - then to get the neighbor's values for that year

  # Approach: build a long table of (origin_id, year, neighbor_id),
  # then join to get neighbor values, then aggregate.

  # Get unique (id, year) with row id
  origin <- cell_data[, .(id, year, .row_id)]

  # Merge origin with edge table to get (origin_row_id, year, neighbor_id)
  # This is the most memory-intensive step: ~1.37M edges Ã— 28 years â‰ˆ 38.4M rows
  # But each row is just 3 integers â‰ˆ 38.4M Ã— 12 bytes â‰ˆ 461 MB â€” fits in 16 GB
  setkey(edge_dt, id)
  setkey(origin, id)
  expanded <- edge_dt[origin, on = "id", allow.cartesian = TRUE,
                      nomatch = NULL,
                      .(.row_id, year, neighbor_id)]

  # Now join to get the neighbor's variable values
  # We join expanded to dt_slim on (neighbor_id == id, year == year)
  setnames(expanded, "neighbor_id", "nb_id")
  setkey(expanded, nb_id, year)
  setkey(dt_slim, id, year)

  neighbor_vals <- dt_slim[expanded, on = c(id = "nb_id", "year"),
                           nomatch = NA]
  # neighbor_vals now has columns: id (=neighbor), year, var_names..., .row_id

  # Aggregate per origin row
  agg_exprs <- list()
  for (v in var_names) {
    v_sym <- as.name(v)
    agg_exprs[[paste0("neighbor_max_", v)]]  <- substitute(
      suppressWarnings(max(V, na.rm = TRUE)), list(V = v_sym))
    agg_exprs[[paste0("neighbor_min_", v)]]  <- substitute(
      suppressWarnings(min(V, na.rm = TRUE)), list(V = v_sym))
    agg_exprs[[paste0("neighbor_mean_", v)]] <- substitute(
      mean(V, na.rm = TRUE), list(V = v_sym))
  }

  agg_result <- neighbor_vals[, eval(as.call(c(as.name("list"),
                                                agg_exprs))),
                              by = .row_id]

  # Replace Inf/-Inf (from max/min on all-NA groups) with NA
  inf_cols <- grep("^neighbor_(max|min)_", names(agg_result), value = TRUE)
  for (col in inf_cols) {
    set(agg_result, which(is.infinite(agg_result[[col]])), col, NA_real_)
  }

  return(agg_result)
}

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

agg_result <- compute_all_neighbor_features(cell_data, edge_dt,
                                            neighbor_source_vars)

# ==============================================================
# STEP 4: Merge back to cell_data
# ==============================================================
setkey(agg_result, .row_id)
setkey(cell_data, .row_id)
cell_data <- agg_result[cell_data, on = ".row_id"]

# Clean up helper column
cell_data[, .row_id := NULL]

# ==============================================================
# STEP 5: (Optional) Convert back to data.frame if downstream
#          code or the trained RF model expects one
# ==============================================================
# cell_data <- as.data.frame(cell_data)
```

---

## Memory Management Variant (If 16 GB Is Tight)

If the ~38M-row expanded join table strains RAM, process one variable at a time and drop intermediates:

```r
compute_neighbor_features_lowmem <- function(cell_data, edge_dt, var_names) {
  origin <- cell_data[, .(id, year, .row_id)]
  setkey(edge_dt, id)
  setkey(origin, id)

  # Expand once â€” this is the big table (~38M rows, 3 int cols â‰ˆ 460 MB)
  expanded <- edge_dt[origin, on = "id", allow.cartesian = TRUE,
                      nomatch = NULL,
                      .(.row_id, year, neighbor_id)]
  setnames(expanded, "neighbor_id", "nb_id")
  setkey(expanded, nb_id, year)

  all_agg <- cell_data[, .(.row_id)]  # anchor table

  for (v in var_names) {
    message("Processing neighbor stats for: ", v)
    dt_v <- cell_data[, .(id, year, val = get(v))]
    setkey(dt_v, id, year)

    joined <- dt_v[expanded, on = c(id = "nb_id", "year"), nomatch = NA]

    agg_v <- joined[, .(
      nmax  = suppressWarnings(max(val, na.rm = TRUE)),
      nmin  = suppressWarnings(min(val, na.rm = TRUE)),
      nmean = mean(val, na.rm = TRUE)
    ), by = .row_id]

    # Fix Inf
    agg_v[is.infinite(nmax), nmax := NA_real_]
    agg_v[is.infinite(nmin), nmin := NA_real_]

    setnames(agg_v,
             c("nmax", "nmin", "nmean"),
             paste0("neighbor_", c("max_", "min_", "mean_"), v))

    all_agg <- agg_v[all_agg, on = ".row_id"]
    rm(joined, agg_v, dt_v); gc()
  }

  return(all_agg)
}

cell_data[, .row_id := .I]
agg_result <- compute_neighbor_features_lowmem(cell_data, edge_dt,
                                               neighbor_source_vars)
setkey(agg_result, .row_id)
setkey(cell_data, .row_id)
cell_data <- agg_result[cell_data, on = ".row_id"]
cell_data[, .row_id := NULL]
```

---

## Why This Preserves Correctness

| Concern | Guarantee |
|---|---|
| **Trained RF model** | No model code is touched. Only the feature-engineering step is optimized. The output columns have the same names and identical numerical values. |
| **Numerical estimand** | `max`, `min`, `mean` with `na.rm = TRUE` are the same operations as the original code. `Inf`/`-Inf` from empty groups are converted to `NA`, matching the original `c(NA, NA, NA)` return. |
| **Row order** | The `.row_id` key ensures results merge back in the original row order. |

**Expected runtime:** ~5â€“20 minutes on a 16 GB laptop (dominated by the single 38M-row keyed join and grouped aggregation, both executed in `data.table`'s parallel C backend) versus the original 86+ hours.
END_RESPONSE>>>
