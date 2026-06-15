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
run_id: Anthropic_F08_T03_R020
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.3
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F08
prompt_family_slug: TOPOLOGY_INVARIANCE
family_label: Topology invariance
family_group: Topology cue
replicate: 20
raw_output_file: Experiment_Output/Anthropic/Temp_0.3/output_F08_Anthropic_temp_03_rep_020.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

## Diagnosis

The current implementation is extremely slow (~86+ hours) because of a fundamental design flaw: **it conflates static topology with dynamic data**.

### Root Cause Analysis

1. **`build_neighbor_lookup` rebuilds per-row (cellÃ—year) neighbor indices.** It creates a list of ~6.46 million entries, each requiring string-pasting, hash lookups, and NA filtering. But the neighbor *relationships* are purely spatial â€” cell A neighbors cell B regardless of year. The only thing that changes across years is the *values* attached to cells. This means the lookup is doing 28Ã— redundant work (once per year per cell, when the topology is year-invariant).

2. **`compute_neighbor_stats` iterates over ~6.46 million rows** with `lapply`, indexing into a flat cell-year vector. Each call pulls values from scattered positions in a 6.46M-length vector. This is cache-unfriendly and R-loop-heavy.

3. **String key construction** (`paste(id, year, sep="_")`) and named-vector lookups (`idx_lookup[neighbor_keys]`) are O(n) hash operations repeated millions of times â€” enormously wasteful.

### The Key Insight

- **Static:** The neighbor graph (which cell IDs neighbor which cell IDs) â€” 344,208 cells, ~1.37M directed edges. This never changes.
- **Dynamic:** The variable values attached to each cell â€” these change by year.

The redesign should: build the neighbor structure **once over cells only** (344K entries, not 6.46M), then for each year, slice the relevant variable column, compute neighbor max/min/mean using vectorized operations over the static cell-indexed neighbor list.

---

## Optimization Strategy

1. **Build a cell-level neighbor lookup once** â€” a list of length 344,208 where each element contains integer indices (1-based positions in `id_order`) of that cell's neighbors. This is just a cleaned version of `rook_neighbors_unique` and costs essentially nothing.

2. **For each variable and each year**, subset the data to that year (or index into a cell-indexed vector), pull neighbor values using the static cell-level lookup, and compute max/min/mean with vectorized R or a fast compiled helper.

3. **Use `data.table`** for efficient subsetting, column assignment, and join-free indexing by cell position.

4. **Vectorize the inner loop** using `vapply` over 344K cells (not 6.46M rows) per year, and parallelize across years or variables if needed.

### Expected Speedup

- Lookup construction: 6.46M â†’ 344K entries = **~19Ã— faster**.
- Stats computation: operating on 344K cells Ã— 28 years with vectorized column access instead of 6.46M string-keyed lookups = **~20-50Ã— faster**.
- Overall: from ~86 hours to **~1-3 hours** on a 16 GB laptop, potentially under 1 hour with `data.table` and careful memory management.

---

## Working R Code

```r
library(data.table)

# ============================================================
# STEP 0: Convert to data.table if not already
# ============================================================
if (!is.data.table(cell_data)) {
  cell_data <- as.data.table(cell_data)
}

# ============================================================
# STEP 1: Build STATIC cell-level neighbor lookup (once)
#
# rook_neighbors_unique is an nb object (list of integer vectors)
# indexed by position in id_order. We just need to clean it:
# remove 0L entries (spdep uses 0L for "no neighbors").
# ============================================================
build_cell_neighbor_lookup <- function(neighbors) {
  # neighbors is an nb object: list of integer index vectors

  # Each element i contains the positional indices (into id_order)
  # of the neighbors of cell id_order[i].
  # spdep encodes "no neighbors" as integer(0) or 0L.
  lapply(neighbors, function(nb_idx) {
    nb_idx <- nb_idx[nb_idx > 0L]
    as.integer(nb_idx)
  })
}

cell_neighbor_lookup <- build_cell_neighbor_lookup(rook_neighbors_unique)
# cell_neighbor_lookup[[i]] = integer vector of positional indices
# into id_order for the neighbors of cell id_order[i].

# ============================================================
# STEP 2: Build a mapping from cell id -> position in id_order
# ============================================================
id_to_pos <- setNames(seq_along(id_order), as.character(id_order))

# ============================================================
# STEP 3: Ensure cell_data is keyed and has a cell position column
# ============================================================
cell_data[, cell_pos := id_to_pos[as.character(id)]]

# Verify no NAs (every id in cell_data must be in id_order)
stopifnot(!anyNA(cell_data$cell_pos))

# Key by year and cell_pos for fast subsetting
setkey(cell_data, year, cell_pos)

# ============================================================
# STEP 4: Compute neighbor stats â€” static topology, dynamic values
# ============================================================
compute_neighbor_features <- function(dt, cell_nb_lookup, var_name,
                                      id_order_vec) {

  n_cells <- length(id_order_vec)
  max_col <- paste0("neighbor_max_", var_name)
  min_col <- paste0("neighbor_min_", var_name)
  mean_col <- paste0("neighbor_mean_", var_name)

  # Pre-allocate output columns
  dt[, (max_col) := NA_real_]
  dt[, (min_col) := NA_real_]
  dt[, (mean_col) := NA_real_]

  years <- sort(unique(dt$year))

  for (yr in years) {
    # --- Extract a cell-position-indexed vector of values for this year ---
    # Because we keyed by (year, cell_pos), subset is fast
    yr_rows <- dt[.(yr)]  # subset by year via key

    # Build a dense vector: position i -> value for cell at position i
    # Some cells may be missing for a year; those stay NA.
    vals_by_pos <- rep(NA_real_, n_cells)
    vals_by_pos[yr_rows$cell_pos] <- yr_rows[[var_name]]

    # --- Compute neighbor stats for each cell present this year ---
    cell_positions <- yr_rows$cell_pos

    # Vectorized computation over cells present this year
    stats <- vapply(cell_positions, function(cp) {
      nb_pos <- cell_nb_lookup[[cp]]
      if (length(nb_pos) == 0L) return(c(NA_real_, NA_real_, NA_real_))
      nv <- vals_by_pos[nb_pos]
      nv <- nv[!is.na(nv)]
      if (length(nv) == 0L) return(c(NA_real_, NA_real_, NA_real_))
      c(max(nv), min(nv), mean(nv))
    }, numeric(3))
    # stats is 3 x length(cell_positions)

    # --- Write results back into the data.table ---
    # We need the row indices in the *original* dt, not in yr_rows.
    # Since dt is keyed by (year, cell_pos), we can use a join to assign.
    # But more directly, we can find the row indices:
    row_idx <- dt[.(yr), which = TRUE]

    set(dt, i = row_idx, j = max_col,  value = stats[1, ])
    set(dt, i = row_idx, j = min_col,  value = stats[2, ])
    set(dt, i = row_idx, j = mean_col, value = stats[3, ])

    if (interactive()) {
      cat(sprintf("  %s | year %d done\n", var_name, yr))
    }
  }

  invisible(dt)
}

# ============================================================
# STEP 5: Run for all neighbor source variables
# ============================================================
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {
  cat(sprintf("Computing neighbor features for: %s\n", var_name))
  compute_neighbor_features(cell_data, cell_neighbor_lookup,
                            var_name, id_order)
}

# ============================================================
# STEP 6: Clean up helper column, restore original class if needed
# ============================================================
cell_data[, cell_pos := NULL]

# ============================================================
# STEP 7: Predict with the pre-trained Random Forest (unchanged)
# ============================================================
# The trained RF model object and prediction code remain exactly
# as they were. The output columns have identical names and
# identical numerical values (same estimand), so predictions
# are fully preserved.
#
# Example (unchanged from original pipeline):
# predictions <- predict(rf_model, newdata = cell_data)
```

---

## Why This Preserves the Original Numerical Estimand

| Aspect | Original | Redesigned |
|---|---|---|
| Neighbor definition | Rook contiguity from `rook_neighbors_unique` | Same `rook_neighbors_unique`, same positional indices |
| Neighbor values source | `data[[var_name]][idx]` where `idx` matched by `(id, year)` | `vals_by_pos[nb_pos]` where `vals_by_pos` is indexed by cell position for the same year |
| Aggregation functions | `max()`, `min()`, `mean()` with NA removal | Identical `max()`, `min()`, `mean()` with identical NA removal |
| Output column names | `neighbor_max_<var>`, `neighbor_min_<var>`, `neighbor_mean_<var>` | Identical |
| RF model | Pre-trained, not retrained | Pre-trained, not retrained â€” receives identical feature columns |

The computation is mathematically identical; only the execution strategy differs â€” exploiting the static-vs-dynamic decomposition to eliminate ~95% of redundant work.

### Optional Further Speedup: Rcpp Inner Loop

If the `vapply` over 344K cells is still a bottleneck (~15 min per variableÃ—year), the inner loop can be pushed to C++ via Rcpp:

```r
Rcpp::cppFunction('
NumericMatrix neighbor_stats_cpp(NumericVector vals_by_pos,
                                 List cell_nb_lookup,
                                 IntegerVector cell_positions) {
  int n = cell_positions.size();
  NumericMatrix out(n, 3); // max, min, mean

  for (int i = 0; i < n; i++) {
    int cp = cell_positions[i] - 1; // 0-based
    IntegerVector nb = cell_nb_lookup[cp];
    int count = 0;
    double vmax = R_NegInf, vmin = R_PosInf, vsum = 0.0;

    for (int j = 0; j < nb.size(); j++) {
      double v = vals_by_pos[nb[j] - 1]; // 0-based
      if (!NumericVector::is_na(v)) {
        if (v > vmax) vmax = v;
        if (v < vmin) vmin = v;
        vsum += v;
        count++;
      }
    }
    if (count == 0) {
      out(i, 0) = NA_REAL;
      out(i, 1) = NA_REAL;
      out(i, 2) = NA_REAL;
    } else {
      out(i, 0) = vmax;
      out(i, 1) = vmin;
      out(i, 2) = vsum / count;
    }
  }
  return out;
}
')
```

This would replace the `vapply` call inside `compute_neighbor_features` and bring per-year-per-variable time down to seconds, reducing total runtime to **under 10 minutes**.
END_RESPONSE>>>
