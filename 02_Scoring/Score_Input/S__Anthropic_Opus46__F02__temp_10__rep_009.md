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
run_id: Anthropic_F02_T10_R009
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F02
prompt_family_slug: CODE_ONLY_NO_HINT
family_label: Code only, no hint
family_group: Code-visible neutral
replicate: 9
raw_output_file: Experiment_Output/Anthropic/Temp_1.0/output_F02_Anthropic_temp_10_rep_009.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

---

## 1. Diagnosis

The bottleneck has two root causes:

### A. `build_neighbor_lookup` â€” O(n) `lapply` over 6.46 million rows with per-row string pasting and named-vector lookups

* `paste(neighbor_cell_ids, data$year[i], sep = "_")` and `idx_lookup[neighbor_keys]` are called **once per row**. Named character vector lookup in R is hash-based but the constant factor is high when called millions of times.
* Each call also allocates a new character vector, hits the global string cache, and creates intermediate objects. Across 6.46M rows this dominates wall-clock time.

### B. `compute_neighbor_stats` â€” R-level loop with per-row subsetting and aggregation

* For each of the 5 variables, another `lapply` over 6.46M rows computes `max`, `min`, `mean` on small vectors. The overhead is the R interpreter loop, not the arithmetic. This is called 5 times, producing ~32.3 million R-level function invocations.

### Memory pressure

* The `neighbor_lookup` list itself stores ~6.46M integer vectors. With an average of ~4 rook neighbors per cell and R's 40-byte minimum vector overhead, this alone is â‰¥ 6.46M Ã— 72 bytes â‰ˆ **0.46 GB**. Combined with the 6.46M Ã— 110-column data frame (~5.7 GB at 8 bytes/double), 16 GB is tight but workable **only if intermediate copies are eliminated**.

---

## 2. Optimization Strategy

| Principle | Technique |
|---|---|
| **Eliminate per-row R loops** | Convert the neighbor lookup into a flat edge list (two integer columns: `from_row`, `to_row`) and use **vectorised grouped operations** via `data.table`. |
| **Build the edge list vectorised** | Use `data.table` joins instead of `paste`/named-vector lookup. One equi-join replaces 6.46M `paste` + hash probes. |
| **Compute stats vectorised** | Join the edge list to the variable column, then `group by from_row` to get `max`, `min`, `mean` â€” all in C-level `data.table` code. |
| **Process all 5 variables in one pass** | A single join + grouped aggregation over all 5 source variables avoids rebuilding intermediate structures. |
| **Keep memory bounded** | No giant list-of-vectors; the edge list is two integer columns (â‰ˆ 2 Ã— 6.46M Ã— 4 neighbors Ã— 4 bytes â‰ˆ 206 MB). |

Expected speedup: from 86+ hours to **minutes** (the dominant cost becomes a handful of `data.table` joins and grouped aggregations on ~26M edge rows).

---

## 3. Working R Code

```r
library(data.table)

# â”€â”€ Step 0: Ensure cell_data is a data.table with an integer row index â”€â”€â”€â”€â”€â”€
setDT(cell_data)
cell_data[, row_idx := .I]

# â”€â”€ Step 1: Build a vectorised edge list (replaces build_neighbor_lookup) â”€â”€â”€â”€
build_edge_list <- function(cell_dt, id_order, neighbors) {
  # Map each cell id to its position in id_order
  id_to_ref <- data.table(
    id  = id_order,
    ref = seq_along(id_order)
  )


  # Expand the nb object into a flat edge list of (from_id, to_id)
  from_ref <- rep(
    seq_along(neighbors),
    lengths(neighbors)
  )
  to_ref <- unlist(neighbors, use.names = FALSE)

  edge_ids <- data.table(
    from_id = id_order[from_ref],
    to_id   = id_order[to_ref]
  )

  # Create a lookup from (id, year) â†’ row_idx
  id_year_lookup <- cell_dt[, .(id, year, row_idx)]

  # Join: for every (from_id, year) find the from_row
  # First cross edge_ids with every year present for each from_id
  # Efficient approach: join edge_ids to the data on from_id, then
  # join to_id + year to get to_row.


  # from side: get (from_id, year, from_row)
  setkey(id_year_lookup, id)
  from_dt <- id_year_lookup[edge_ids, on = .(id = from_id),
                            .(from_row = row_idx,
                              to_id    = i.to_id,
                              year     = year),
                            nomatch = NULL,
                            allow.cartesian = TRUE]

  # to side: get to_row by joining (to_id, year)
  setkey(id_year_lookup, id, year)
  from_dt[id_year_lookup,
          to_row := i.row_idx,
          on = .(to_id = id, year = year),
          nomatch = NA]

  # Drop edges where the neighbor has no observation in that year

  from_dt <- from_dt[!is.na(to_row)]

  from_dt[, .(from_row = as.integer(from_row),
              to_row   = as.integer(to_row))]
}

edge_list <- build_edge_list(cell_data, id_order, rook_neighbors_unique)
setkey(edge_list, from_row)


# â”€â”€ Step 2: Compute all neighbor stats in one vectorised pass â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

compute_all_neighbor_features <- function(cell_dt, edge_dt, vars) {
  # Attach the source variable values to every edge via to_row
  # We only select the columns we need to keep memory low.
  val_cols  <- vars
  to_vals   <- cell_dt[edge_dt$to_row, ..val_cols]
  to_vals[, from_row := edge_dt$from_row]

  # Grouped aggregation: max, min, mean per from_row, per variable

  agg_exprs <- unlist(lapply(val_cols, function(v) {
    list(
      bquote(max(.(as.name(v)),   na.rm = TRUE)),
      bquote(min(.(as.name(v)),   na.rm = TRUE)),
      bquote(mean(.(as.name(v)),  na.rm = TRUE))
    )
  }), recursive = FALSE)

  agg_names <- unlist(lapply(val_cols, function(v) {
    paste0("neighbor_", c("max_", "min_", "mean_"), v)
  }))

  names(agg_exprs) <- agg_names

  stats <- to_vals[, lapply(agg_exprs, eval, envir = .SD), by = from_row]

  # Replace -Inf/Inf (from max/min on all-NA groups) with NA
  for (col in agg_names) {
    set(stats, which(is.infinite(stats[[col]])), col, NA_real_)
  }

  stats
}

neighbor_stats <- compute_all_neighbor_features(cell_data, edge_list, neighbor_source_vars)


# â”€â”€ Step 3: Left-join the new features back onto cell_data â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Rows with no neighbors will get NA automatically.
cell_data[neighbor_stats, on = .(row_idx = from_row),
          (names(neighbor_stats)[names(neighbor_stats) != "from_row"]) :=
            mget(names(neighbor_stats)[names(neighbor_stats) != "from_row"])]

# Clean up helper column
cell_data[, row_idx := NULL]

# â”€â”€ cell_data now contains the 15 new neighbor feature columns â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# (neighbor_max_ntl, neighbor_min_ntl, neighbor_mean_ntl, â€¦ for all 5 vars)
# The trained Random Forest model can be applied directly to this table.
```

> **Note on the grouped aggregation:** If you prefer an even simpler (and sometimes faster) formulation that avoids `bquote`, here is a drop-in alternative for Step 2:

```r
compute_all_neighbor_features_v2 <- function(cell_dt, edge_dt, vars) {
  to_vals        <- cell_dt[edge_dt$to_row, ..vars]
  to_vals[, from_row := edge_dt$from_row]

  result_list <- vector("list", length(vars))

  for (i in seq_along(vars)) {
    v   <- vars[i]
    tmp <- to_vals[, .(
      vmax  = max(.SD[[1L]],  na.rm = TRUE),
      vmin  = min(.SD[[1L]],  na.rm = TRUE),
      vmean = mean(.SD[[1L]], na.rm = TRUE)
    ), by = from_row, .SDcols = v]

    # Fix Inf from all-NA groups
    tmp[is.infinite(vmax),  vmax  := NA_real_]
    tmp[is.infinite(vmin),  vmin  := NA_real_]

    setnames(tmp,
             c("vmax", "vmin", "vmean"),
             paste0("neighbor_", c("max_", "min_", "mean_"), v))

    result_list[[i]] <- tmp
  }

  # Merge all variable-specific results
  out <- result_list[[1L]]
  for (i in seq_along(result_list)[-1L]) {
    out <- merge(out, result_list[[i]], by = "from_row", all = TRUE)
  }
  out
}
```

---

## 4. Summary

| Aspect | Original | Optimised |
|---|---|---|
| Neighbor lookup | 6.46M `paste` + named-vector probes in R loop | One `data.table` equi-join (~seconds) |
| Stat computation | 5 Ã— 6.46M R-level `lapply` calls | 5 grouped `data.table` aggregations (C-level) |
| Memory pattern | 6.46M-element list of int vectors (~0.5 GB overhead) | Flat 2-column integer edge table (~200 MB) |
| Estimated time | 86+ hours | **< 10 minutes** on 16 GB laptop |
| Model impact | None â€” columns produced are numerically identical | Identical `max`, `min`, `mean` per row |

The trained Random Forest model requires no changes because the output columns are numerically identical to the originals â€” only the computational pathway has changed.
END_RESPONSE>>>
