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
run_id: Anthropic_F02_T10_R029
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F02
prompt_family_slug: CODE_ONLY_NO_HINT
family_label: Code only, no hint
family_group: Code-visible neutral
replicate: 29
raw_output_file: Experiment_Output/Anthropic/Temp_1.0/output_F02_Anthropic_temp_10_rep_029.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The two critical bottlenecks are:

### 1. `build_neighbor_lookup` â€” O(n) `lapply` over 6.46 million rows with per-row string pasting and named-vector lookups
- Each of the ~6.46M rows constructs character keys via `paste()`, then does named-vector subsetting (`idx_lookup[neighbor_keys]`). Named-vector lookup in R is effectively a hash-table probe *per element*, but creating ~6.46M Ã— k character keys and performing millions of individual hash lookups is extremely slow.
- The output is a list of 6.46M integer vectors â€” a large, ragged structure that is both slow to build and memory-hungry.

### 2. `compute_neighbor_stats` â€” another O(n) `lapply` over the ragged list, repeated 5 times
- For each of the 5 variables, it traverses the entire 6.46M-element list, extracts neighbor values, and computes `max/min/mean`. This is called sequentially for every variable, so the total work is ~32.3M list-element accesses.
- `do.call(rbind, result)` on a 6.46M-element list of length-3 vectors is itself slow (repeated memory allocation).

### Memory
- The ragged neighbor lookup list alone, at an average of ~4 neighbors per cell-year, stores ~25.8M integers plus R list overhead â€” roughly 1â€“2 GB.
- Intermediate character vectors from `paste()` add another temporary spike.

### Summary
| Component | Root Cause | Impact |
|---|---|---|
| `build_neighbor_lookup` | Per-row `paste` + named-vector lookup across 6.46M rows | ~70â€“80 % of total time |
| `compute_neighbor_stats` | Per-row `lapply` Ã— 5 variables, `do.call(rbind,â€¦)` | ~15â€“25 % of total time |
| Memory | Ragged list of 6.46M elements + character key intermediaries | Approaches 16 GB limit |

---

## Optimization Strategy

The key insight is: **eliminate the ragged per-row list entirely**. Replace it with a flat, vectorized sparse-matrix representation (CSR-style) built once via `data.table` joins, then compute all neighbor statistics with grouped vectorized operations â€” no `lapply` at all.

### Step-by-step

1. **Flatten the `nb` object into an edge-list** of `(cell_id, neighbor_cell_id)` pairs â€” ~1.37M rows.
2. **Cross-join with years** using `data.table` to produce a `(cell_id, year, neighbor_cell_id)` edge-year table (~1.37M Ã— 28 â‰ˆ 38.5M rows, but only those that exist in the data).
3. **Inner-join** the edge-year table back to the data to attach each neighbor's variable values â€” one join per variable, or all at once.
4. **Group-by `(cell_id, year)`** and compute `max`, `min`, `mean` â€” fully vectorized in `data.table`, no R-level loop.
5. **Left-join** results back to the main data.

This replaces two nested R-level loops (6.46M iterations each) with a handful of `data.table` joins and grouped aggregations that run in C and complete in minutes.

### Complexity comparison

| | Original | Optimized |
|---|---|---|
| Neighbor lookup | 6.46M `paste` + hash lookups | One `data.table` join (~38M rows) |
| Stats computation | 5 Ã— 6.46M `lapply` iterations | 5 grouped aggregations (vectorized C) |
| Peak memory | ~10â€“14 GB (ragged list + copies) | ~4â€“6 GB (flat tables) |
| Estimated time | 86+ hours | **10â€“30 minutes** |

---

## Working R Code

```r
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Optimized neighbor-feature pipeline using data.table
# Preserves the trained RF model (no retraining) and the original numerical
# estimand (max, min, mean of each neighbor variable).
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

library(data.table)

#' Flatten an spdep nb object into a two-column data.table of directed edges.
#'
#' @param nb_obj   An nb object (list of integer neighbor vectors).
#' @param id_order Character or integer vector mapping list position -> cell id.
#' @return A data.table with columns \code{id} and \code{neighbor_id}.
nb_to_edge_dt <- function(nb_obj, id_order) {
    # Pre-allocate vectors
    n_edges <- sum(lengths(nb_obj))
    from_id <- integer(n_edges)
    to_id   <- integer(n_edges)
    pos <- 1L
    for (i in seq_along(nb_obj)) {
        nbrs <- nb_obj[[i]]
        if (length(nbrs) == 0L || (length(nbrs) == 1L && nbrs[1L] == 0L)) next
        n <- length(nbrs)
        from_id[pos:(pos + n - 1L)] <- id_order[i]
        to_id[pos:(pos + n - 1L)]   <- id_order[nbrs]
        pos <- pos + n
    }
    data.table(id = from_id[seq_len(pos - 1L)],
               neighbor_id = to_id[seq_len(pos - 1L)])
}

#' Compute neighbor summary statistics for multiple variables at once.
#'
#' @param cell_dt           A data.table with at least columns: id, year, and
#'                          every name in \code{var_names}.
#' @param edge_dt           A data.table from \code{nb_to_edge_dt}.
#' @param var_names         Character vector of column names to summarize.
#' @return \code{cell_dt} with new columns appended:
#'         \code{<var>_neighbor_max}, \code{<var>_neighbor_min},
#'         \code{<var>_neighbor_mean} for each var.
compute_all_neighbor_features <- function(cell_dt, edge_dt, var_names) {

    # Ensure data.table
    if (!is.data.table(cell_dt)) cell_dt <- as.data.table(cell_dt)
    setDT(cell_dt)

    # --- 1. Build the edge-year table by joining edges to the data twice ------
    #
    # We need, for every (id, year), the variable values of its neighbors in
    # the same year.  Strategy:
    #   a) Join edge_dt to the unique (id, year) pairs to get
    #      (id, year, neighbor_id) â€” call this "edge_year".
    #   b) Join edge_year to cell_dt on (neighbor_id == id, year) to attach
    #      the neighbor's values.

    # Columns we need from the neighbor rows
    neighbor_cols <- var_names

    # a) Expand edges by year -------------------------------------------------
    #    For every row in cell_dt that has id âˆˆ edge_dt$id, attach neighbors.
    #    This is an inner join: cell_dt[edge_dt, on = "id", allow.cartesian = TRUE]
    #    but we only need id + year from cell_dt and neighbor_id from edge_dt.

    # Minimal keys from data
    keys_dt <- unique(cell_dt[, .(id, year)])
    setkey(keys_dt, id)
    setkey(edge_dt, id)

    # Join: one row per (id, year, neighbor_id)
    edge_year <- edge_dt[keys_dt, on = "id", allow.cartesian = TRUE, nomatch = 0L]
    # Columns now: id, neighbor_id, year

    # b) Attach neighbor values -----------------------------------------------
    #    We need to look up cell_dt rows by (neighbor_id, year).
    #    Rename for the join.
    setkey(cell_dt, id, year)

    # Create a lookup with only the columns we need (saves memory)
    lookup_cols <- c("id", "year", neighbor_cols)
    nbr_vals <- cell_dt[, ..lookup_cols]
    setnames(nbr_vals, "id", "neighbor_id")
    setkey(nbr_vals, neighbor_id, year)
    setkey(edge_year, neighbor_id, year)

    # Inner join â€” attaches neighbor variable values
    edge_year <- nbr_vals[edge_year, on = .(neighbor_id, year), nomatch = 0L]
    # Columns: neighbor_id, year, <var_names...>, id

    # --- 2. Grouped aggregation -----------------------------------------------
    #    Group by (id, year), compute max/min/mean for each variable.

    # Build aggregation expressions programmatically
    agg_exprs <- unlist(lapply(var_names, function(v) {
        list(
            bquote(max(.(as.name(v)), na.rm = TRUE)),
            bquote(min(.(as.name(v)), na.rm = TRUE)),
            bquote(mean(.(as.name(v)), na.rm = TRUE))
        )
    }), recursive = FALSE)

    agg_names <- unlist(lapply(var_names, function(v) {
        paste0(v, c("_neighbor_max", "_neighbor_min", "_neighbor_mean"))
    }))

    # Construct the call:  edge_year[, .(expr1, expr2, ...), by = .(id, year)]
    agg_call <- as.call(c(
        as.name("list"),
        setNames(agg_exprs, agg_names)
    ))

    stats_dt <- edge_year[, eval(agg_call), by = .(id, year)]

    # Replace Inf / -Inf (from max/min on all-NA groups) with NA
    for (col in agg_names) {
        set(stats_dt, which(is.infinite(stats_dt[[col]])), col, NA_real_)
    }

    # --- 3. Left-join back to the main table ----------------------------------
    setkey(stats_dt, id, year)
    setkey(cell_dt, id, year)

    # Remove any pre-existing neighbor columns to avoid duplication
    existing <- intersect(agg_names, names(cell_dt))
    if (length(existing) > 0L) {
        cell_dt[, (existing) := NULL]
    }

    cell_dt <- stats_dt[cell_dt, on = .(id, year)]

    return(cell_dt)
}


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Usage â€” drop-in replacement for the original outer loop
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

# Convert cell_data to data.table (zero-copy if already one)
cell_data <- as.data.table(cell_data)

# 1. Flatten the nb object once (~1.37 M rows)
edge_dt <- nb_to_edge_dt(rook_neighbors_unique, id_order)

# 2. Compute all 5 Ã— 3 = 15 neighbor features in one pass
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cell_data <- compute_all_neighbor_features(
    cell_dt   = cell_data,
    edge_dt   = edge_dt,
    var_names = neighbor_source_vars
)

# cell_data now has the same 15 new columns that the original code produced:
#   ntl_neighbor_max, ntl_neighbor_min, ntl_neighbor_mean,
#   ec_neighbor_max,  ec_neighbor_min,  ec_neighbor_mean,
#   ...
# with identical numerical values (max, min, mean preserve the original
# estimand).  The trained Random Forest model can be applied directly.
```

---

## Why This Works and Is Safe

| Concern | Explanation |
|---|---|
| **Numerical equivalence** | `max`, `min`, `mean` with `na.rm = TRUE` in `data.table` produce the same IEEE-754 results as the original R code. `Inf`/`-Inf` edge cases (all-NA neighbors) are mapped back to `NA`, matching the original `return(c(NA, NA, NA))` path. |
| **Trained RF model unchanged** | No model retraining occurs. The output columns have the same names and values, so `predict(rf_model, cell_data)` works identically. |
| **Memory** | The largest intermediate object (`edge_year`) is ~38.5M rows Ã— 7 numeric columns â‰ˆ 2.1 GB. Combined with `cell_data` (~5.7 GB for 6.46M Ã— 110 columns), total peak is ~8â€“10 GB â€” well within 16 GB. The original code's ragged list + character vectors approached 14 GB. |
| **Speed** | `data.table` joins and grouped aggregations run in parallel C threads. Expected wall-clock: **10â€“30 minutes** vs. 86+ hours. |
| **No external dependencies beyond `data.table`** | `data.table` is a standard, well-tested CRAN package already common in spatial-econometric pipelines. |
END_RESPONSE>>>
