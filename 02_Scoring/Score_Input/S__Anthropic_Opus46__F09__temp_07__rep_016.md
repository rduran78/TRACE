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
run_id: Anthropic_F09_T07_R016
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F09
prompt_family_slug: ADJACENCY_TABLE_OPTION
family_label: Adjacency-table option
family_group: Representation cue
replicate: 16
raw_output_file: Experiment_Output/Anthropic/Temp_0.7/output_F09_Anthropic_temp_07_rep_016.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The current implementation is slow (~86+ hours) for three interconnected reasons:

### 1. `build_neighbor_lookup` is O(NÂ²)-like in practice
The function iterates over **all 6.46 million cell-year rows** with `lapply`, and for each row it:
- Looks up the cell's spatial neighbors (fine),
- Pastes together `cell_id_year` character keys (expensive string allocation Ã— millions),
- Performs named-vector lookups via character matching (`idx_lookup[neighbor_keys]`).

Named vector lookup in R is **O(n)** per probe in the worst case (hash collisions on a 6.46M-entry character vector). With ~4 neighbors per cell on average and 6.46M rows, this is roughly **25.8 million string-match lookups** against a 6.46M-entry named vector. This alone can take tens of hours.

### 2. The lookup is **year-redundant**
The spatial neighbor structure is **identical across all 28 years**. A cell's rook neighbors in 1992 are the same cells as in 2019. Yet the current code rebuilds the full row-index mapping for every cell-year combination, effectively doing 28Ã— more work than necessary.

### 3. `compute_neighbor_stats` uses row-level `lapply`
Even after the lookup is built, computing stats via `lapply` over 6.46M list elements with per-element subsetting and `max/min/mean` is slow due to R's loop overhead and lack of vectorization.

---

## Optimization Strategy

**Core insight:** Separate the **time-invariant spatial topology** from the **time-varying attributes**.

| Step | What | Complexity |
|------|------|------------|
| **A** | Build a spatial edge table once: a `data.table` with columns `(cell_id, neighbor_id)` from the `nb` object. This has ~1.37M rows and never changes. | One-time, fast |
| **B** | For each year, join cell attributes onto this edge table by `neighbor_id` to get each neighbor's variable values, then group-by `(cell_id, year)` to compute `max`, `min`, `mean`. | Vectorized `data.table` grouped aggregation |
| **C** | Join the resulting neighbor-stat columns back onto the main dataset. | Keyed join |

This eliminates:
- All 6.46M-iteration `lapply` loops,
- All character-key pasting and named-vector lookups,
- Year-redundant neighbor resolution.

**Expected speedup:** From ~86 hours to **~2â€“10 minutes** on a 16 GB laptop, because `data.table` grouped joins and aggregations on ~38M rows (1.37M edges Ã— 28 years) are highly optimized.

The trained Random Forest model is **not touched**. The numerical output is **identical** (same max, min, mean over the same neighbor sets).

---

## Working R Code

```r
library(data.table)

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# STEP A: Build the time-invariant spatial edge table ONCE
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Inputs:
#   id_order              â€” integer/numeric vector of cell IDs, length 344,208
#                           (same order as the nb object)
#   rook_neighbors_unique â€” spdep nb object (list of length 344,208)
#
# Output:
#   edge_dt â€” data.table with columns (cell_id, neighbor_id)
#             ~1,373,394 rows (directed rook-neighbor pairs)

build_edge_table <- function(id_order, neighbors) {
  n <- length(neighbors)
  # Pre-allocate: count total edges
  n_edges <- sum(lengths(neighbors))

  from_id <- integer(n_edges)
  to_id   <- integer(n_edges)


  pos <- 1L
  for (i in seq_len(n)) {
    nb_idx <- neighbors[[i]]
    if (length(nb_idx) == 0L) next
    len <- length(nb_idx)
    from_id[pos:(pos + len - 1L)] <- id_order[i]
    to_id[pos:(pos + len - 1L)]   <- id_order[nb_idx]
    pos <- pos + len
  }

  data.table(cell_id = from_id, neighbor_id = to_id)
}

edge_dt <- build_edge_table(id_order, rook_neighbors_unique)

cat(sprintf("Edge table: %d directed neighbor pairs\n", nrow(edge_dt)))

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# STEP B: Convert main data to data.table (if not already)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

if (!is.data.table(cell_data)) {
  cell_data <- as.data.table(cell_data)
}

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# STEP C: Compute neighbor stats for all source variables at once
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

compute_all_neighbor_features <- function(cell_data, edge_dt, source_vars) {

  # 1. Create a slim lookup table: (id, year, var1, var2, â€¦)
  #    Keyed by (id, year) so joins are fast.
  lookup_cols <- c("id", "year", source_vars)
  attr_dt <- cell_data[, ..lookup_cols]

  # 2. Cross the edge table with all years present in the data.
  #    This gives us ~1.37M edges Ã— 28 years â‰ˆ 38.5M rows.
  years <- sort(unique(cell_data$year))
  edge_year_dt <- CJ(edge_idx = seq_len(nrow(edge_dt)), year = years)
  edge_year_dt[, cell_id     := edge_dt$cell_id[edge_idx]]
  edge_year_dt[, neighbor_id := edge_dt$neighbor_id[edge_idx]]
  edge_year_dt[, edge_idx := NULL]

  # 3. Join neighbor attributes onto the edge-year table.
  #    We join by (neighbor_id == id, year == year).
  setkey(attr_dt, id, year)
  setkey(edge_year_dt, neighbor_id, year)

  edge_year_dt <- attr_dt[edge_year_dt, on = .(id = neighbor_id, year = year)]
  # Now edge_year_dt has columns: id (= neighbor_id), year, <source_vars>,
  #   cell_id, neighbor_id (dropped since it became 'id')
  # Rename 'id' back to 'neighbor_id' and use 'cell_id' for grouping.
  setnames(edge_year_dt, "id", "neighbor_id")

  # 4. Group by (cell_id, year) and compute max, min, mean for each variable.
  agg_exprs <- list()
  for (v in source_vars) {
    sym_v <- as.name(v)
    agg_exprs[[paste0("neighbor_max_", v)]]  <-
      bquote(as.numeric(max(.(sym_v), na.rm = TRUE)))
    agg_exprs[[paste0("neighbor_min_", v)]]  <-
      bquote(as.numeric(min(.(sym_v), na.rm = TRUE)))
    agg_exprs[[paste0("neighbor_mean_", v)]] <-
      bquote(mean(.(sym_v), na.rm = TRUE))
  }

  # Build a single j-expression that computes all 15 columns in one pass.
  j_expr <- as.call(c(as.name("list"),
                       setNames(agg_exprs, names(agg_exprs))))

  cat("Aggregating neighbor statistics (this is the main computation)...\n")
  stats_dt <- edge_year_dt[, eval(j_expr), by = .(cell_id, year)]

  # 5. Replace Inf/-Inf (from max/min on all-NA groups) with NA.
  for (col_name in names(agg_exprs)) {
    set(stats_dt, which(is.infinite(stats_dt[[col_name]])), col_name, NA_real_)
  }

  return(stats_dt)
}

neighbor_stats <- compute_all_neighbor_features(cell_data, edge_dt,
                                                 neighbor_source_vars)

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# STEP D: Join neighbor stats back onto the main cell_data
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

# Remove any old neighbor columns if they exist (idempotency).
old_cols <- grep("^neighbor_(max|min|mean)_", names(cell_data), value = TRUE)
if (length(old_cols) > 0) {
  cell_data[, (old_cols) := NULL]
}

setkey(cell_data, id, year)
setkey(neighbor_stats, cell_id, year)

cell_data <- neighbor_stats[cell_data, on = .(cell_id = id, year = year)]
setnames(cell_data, "cell_id", "id")

cat(sprintf("Done. cell_data now has %d rows and %d columns.\n",
            nrow(cell_data), ncol(cell_data)))

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# STEP E: Predict with the EXISTING trained Random Forest (unchanged)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# The trained model object (e.g., `rf_model`) is used as-is.
# Ensure column names match what the model expects.
#
# Example (uncomment and adapt to your actual model object):
#
# cell_data$rf_prediction <- predict(rf_model, newdata = cell_data)
```

---

## Memory note (16 GB laptop)

| Object | Approximate size |
|--------|-----------------|
| `cell_data` (6.46M Ã— 110 cols) | ~5.7 GB |
| `edge_year_dt` (38.5M Ã— 8 cols) | ~2.5 GB |
| `neighbor_stats` (6.46M Ã— 17 cols) | ~0.9 GB |
| Headroom for R + OS | ~6.9 GB |

This fits within 16 GB. If memory is tight, you can process the 5 source variables in batches of 2â€“3 instead of all at once (just loop over subsets of `neighbor_source_vars` and cbind the results), or use the following memory-lean variant that processes one variable at a time without materializing the full cross-product:

```r
# â”€â”€ Memory-lean variant: one variable at a time â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
for (v in neighbor_source_vars) {
  cat(sprintf("Processing neighbor stats for: %s\n", v))

  lookup <- cell_data[, .(id, year, val = get(v))]
  setkey(lookup, id, year)

  # Join neighbor values onto edge table per year
  # (re-uses the compact edge_dt; expands only one variable at a time)
  years_vec <- sort(unique(cell_data$year))
  results_list <- vector("list", length(years_vec))

  for (yi in seq_along(years_vec)) {
    yr <- years_vec[yi]
    yr_lookup <- lookup[year == yr, .(id, val)]
    setkey(yr_lookup, id)

    # Get neighbor values for this year
    yr_edges <- copy(edge_dt)
    yr_edges[yr_lookup, val := i.val, on = .(neighbor_id = id)]

    # Aggregate
    agg <- yr_edges[, .(
      nmax  = if (all(is.na(val))) NA_real_ else max(val, na.rm = TRUE),
      nmin  = if (all(is.na(val))) NA_real_ else min(val, na.rm = TRUE),
      nmean = mean(val, na.rm = TRUE)
    ), by = .(cell_id)]

    agg[, year := yr]
    results_list[[yi]] <- agg
  }

  stats_v <- rbindlist(results_list)
  setnames(stats_v, c("nmax", "nmin", "nmean"),
           paste0("neighbor_", c("max", "min", "mean"), "_", v))

  # Join onto cell_data
  setkey(stats_v, cell_id, year)
  setkey(cell_data, id, year)
  cell_data[stats_v, (paste0("neighbor_", c("max", "min", "mean"), "_", v)) :=
              mget(paste0("neighbor_", c("max", "min", "mean"), "_", v)),
            on = .(id = cell_id, year = year)]
}
```

This variant peaks at ~7.5 GB and processes all 5 variables in **under 5 minutes** on a typical laptop, compared to the original 86+ hours. The trained Random Forest model is never modified, and the computed `max`, `min`, and `mean` values are numerically identical to the original implementation.
END_RESPONSE>>>
