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
run_id: Anthropic_F07_T10_R027
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F07
prompt_family_slug: RASTER_INVALID_IRREGULAR
family_label: Raster invalid irregular topology
family_group: Raster/kernel bridge
replicate: 27
raw_output_file: Experiment_Output/Anthropic/Temp_1.0/output_F07_Anthropic_temp_10_rep_027.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The current implementation has two critical bottlenecks:

### Bottleneck 1: `build_neighbor_lookup` â€” O(nÂ²) list construction via row-by-row `lapply`

`build_neighbor_lookup` iterates over **~6.46 million rows** one at a time. For each row, it:
1. Looks up the cell's position in `id_order`.
2. Extracts neighbor cell IDs from the `nb` object.
3. Pastes string keys (`id_year`) and matches them into `idx_lookup`.

The `paste` + named-vector lookup for 6.46M rows is extremely slow. The character hashing on millions of keys, done row by row in an `lapply`, is the dominant cost.

### Bottleneck 2: `compute_neighbor_stats` â€” row-by-row `lapply` over 6.46M entries

Even after the lookup is built, computing `max`, `min`, `mean` one row at a time in a 6.46M-length `lapply` is slow. Each iteration subsets a numeric vector, removes NAs, and computes three summary statistics. The overhead of millions of R function calls is enormous.

### Why 86+ hours?

~6.46M iterations Ã— 2 passes (build + compute) Ã— 5 variables = tens of millions of R-level loops with string operations and small-vector allocations. This is a classic "death by a million small R operations" problem.

---

## Optimization Strategy

1. **Vectorize `build_neighbor_lookup` entirely** using `data.table` joins instead of character-key lookups. Convert the `nb` object to a flat edge list `(id_from, id_to)` once. Then join on `(id_to, year)` to get the row index of each neighbor in each year. This replaces millions of `paste` + named-vector lookups with a single equi-join.

2. **Vectorize `compute_neighbor_stats`** using `data.table` grouped aggregation on the edge list. For each `(from_row, variable)`, compute `max`, `min`, `mean` of the neighbor values in one grouped operation â€” no R-level loop at all.

3. **Process all 5 variables in a single pass** over the edge table, avoiding rebuilding or re-traversing the neighbor structure.

4. **Memory management**: The flat edge list for ~6.46M rows Ã— ~4 neighbors per cell â‰ˆ ~25.8M edges per year, but since the edge list is just integer pairs + year, it fits comfortably in RAM (~1â€“2 GB).

5. **Preserve the numerical estimand exactly**: `max`, `min`, `mean` are computed on the identical set of non-NA rook neighbors, producing bit-identical results.

6. **Do not touch the trained Random Forest model.**

Expected speedup: from 86+ hours to **~2â€“10 minutes**.

---

## Working R Code

```r
library(data.table)

# ==============================================================
# STEP 1: Convert the nb object to a flat directed edge list
#         (cell-level, time-invariant)
# ==============================================================
build_edge_list_from_nb <- function(id_order, nb_obj) {
  # nb_obj is a list of length = number of cells

  # nb_obj[[i]] contains integer indices (into id_order) of neighbors of cell i
  # id_order[i] gives the actual cell id for position i

  from_ref <- rep(seq_along(nb_obj), lengths(nb_obj))
  to_ref   <- unlist(nb_obj, use.names = FALSE)

  # Remove the spdep convention where 0L means "no neighbors"
  valid <- to_ref > 0L
  from_ref <- from_ref[valid]
  to_ref   <- to_ref[valid]

  data.table(
    id_from = id_order[from_ref],
    id_to   = id_order[to_ref]
  )
}

# ==============================================================
# STEP 2: Build the full (from_row, to_row) edge table
#         by joining on (id, year), entirely vectorized
# ==============================================================
build_neighbor_edge_table <- function(cell_dt, edges_cell) {
  # cell_dt must have columns: .ROW_ID, id, year
  # edges_cell has columns: id_from, id_to

  # Key the data for fast join
  keyed <- cell_dt[, .(.ROW_ID, id, year)]
  setkey(keyed, id, year)

  # For each edge (id_from -> id_to), replicate across all years
  # that id_from appears in, then look up whether id_to exists in that year.
  #
  # Efficient approach: join edges_cell to keyed on id_from = id to get
  # (id_from, id_to, year, from_row), then join on (id_to, year) to get to_row.

  # Step A: Get all (id_from, year, from_row) combinations
  setnames(keyed, c(".ROW_ID", "id", "year"), c("from_row", "id_from", "year"))
  setkey(keyed, id_from)

  # Join edges onto the "from" side
  edge_years <- edges_cell[keyed, on = "id_from", allow.cartesian = TRUE,
                           nomatch = NULL]
  # edge_years now has: id_from, id_to, from_row, year

  # Step B: Look up to_row for (id_to, year)
  to_lookup <- cell_dt[, .(to_row = .ROW_ID, id_to = id, year)]
  setkey(to_lookup, id_to, year)
  setkey(edge_years, id_to, year)

  edge_full <- to_lookup[edge_years, on = c("id_to", "year"), nomatch = NA]
  # Keep only edges where the neighbor actually exists in that year
  edge_full <- edge_full[!is.na(to_row)]

  edge_full[, .(from_row, to_row)]
}

# ==============================================================
# STEP 3: Compute neighbor stats for all variables at once
#         using grouped data.table aggregation
# ==============================================================
compute_all_neighbor_stats <- function(cell_dt, edge_table, var_names) {
  # edge_table: data.table with columns from_row, to_row
  # var_names: character vector of variable names

  n <- nrow(cell_dt)

  # Pre-allocate result columns
  for (v in var_names) {
    cell_dt[, paste0("n_max_", v) := NA_real_]
    cell_dt[, paste0("n_min_", v) := NA_real_]
    cell_dt[, paste0("n_mean_", v) := NA_real_]
  }

  # Attach neighbor values to edge table
  # We pull all needed variable columns for the "to" rows at once
  to_vals <- cell_dt[edge_table$to_row, ..var_names]
  work <- cbind(edge_table, to_vals)

  # Group by from_row and compute stats for each variable
  agg_exprs <- unlist(lapply(var_names, function(v) {
    list(
      bquote(max(.(as.name(v)), na.rm = TRUE)),
      bquote(min(.(as.name(v)), na.rm = TRUE)),
      bquote(mean(.(as.name(v)), na.rm = TRUE))
    )
  }), recursive = FALSE)

  agg_names <- unlist(lapply(var_names, function(v) {
    paste0(c("n_max_", "n_min_", "n_mean_"), v)
  }))

  # Build the aggregation call
  # For large data, a single grouped aggregation is fastest
  stats <- work[, {
    out <- vector("list", length(agg_names))
    k <- 1L
    for (v in var_names) {
      vals <- get(v)
      vals <- vals[!is.na(vals)]
      if (length(vals) == 0L) {
        out[[k]]     <- NA_real_
        out[[k + 1L]] <- NA_real_
        out[[k + 2L]] <- NA_real_
      } else {
        out[[k]]     <- max(vals)
        out[[k + 1L]] <- min(vals)
        out[[k + 2L]] <- mean(vals)
      }
      k <- k + 3L
    }
    names(out) <- agg_names
    out
  }, by = from_row]

  # Assign back into cell_dt by row index
  setkey(stats, from_row)
  for (col_name in agg_names) {
    cell_dt[stats$from_row, (col_name) := stats[[col_name]]]
  }

  cell_dt
}

# ==============================================================
# MAIN PIPELINE
# ==============================================================

# Convert cell_data to data.table if not already
cell_data <- as.data.table(cell_data)
cell_data[, .ROW_ID := .I]

# Step 1: Build cell-level edge list from the nb object (~344K cells)
edges_cell <- build_edge_list_from_nb(id_order, rook_neighbors_unique)
cat("Cell-level edges:", nrow(edges_cell), "\n")

# Step 2: Expand to row-level edge table across all years (~25M edges)
edge_table <- build_neighbor_edge_table(cell_data, edges_cell)
cat("Row-level edges:", nrow(edge_table), "\n")

# Step 3: Compute all neighbor features in one pass
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
cell_data <- compute_all_neighbor_stats(cell_data, edge_table, neighbor_source_vars)

# Clean up helper column
cell_data[, .ROW_ID := NULL]

# Convert back to data.frame if downstream code expects it
# cell_data <- as.data.frame(cell_data)

cat("Done. Neighbor features added for:", paste(neighbor_source_vars, collapse = ", "), "\n")
```

---

## Summary

| Aspect | Original | Optimized |
|---|---|---|
| `build_neighbor_lookup` | 6.46M R-level iterations with `paste`/string matching | Single `data.table` equi-join on `(id, year)` |
| `compute_neighbor_stats` | 6.46M Ã— 5 = 32.3M R-level `lapply` iterations | One grouped `data.table` aggregation over the flat edge table |
| Estimated runtime | 86+ hours | **~2â€“10 minutes** |
| Numerical output | `max`, `min`, `mean` of non-NA rook neighbors | **Identical** â€” same neighbor set, same functions, same NA handling |
| Trained RF model | Untouched | **Untouched** |
| Memory | Millions of small R lists | ~25M-row integer edge table (~200 MB) + grouped aggregation â€” fits in 16 GB |

The key insight: the `nb` object defines a **sparse graph**. Representing it as a flat edge list and using `data.table` joins transforms millions of R-level string lookups into a single vectorized operation, eliminating the bottleneck entirely.
END_RESPONSE>>>
