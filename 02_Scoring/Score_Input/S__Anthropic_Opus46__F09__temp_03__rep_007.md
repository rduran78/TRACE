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
run_id: Anthropic_F09_T03_R007
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.3
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F09
prompt_family_slug: ADJACENCY_TABLE_OPTION
family_label: Adjacency-table option
family_group: Representation cue
replicate: 7
raw_output_file: Experiment_Output/Anthropic/Temp_0.3/output_F09_Anthropic_temp_03_rep_007.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The current implementation is extremely slow (~86+ hours) for three compounding reasons:

### 1. `build_neighbor_lookup` is O(NÂ²)-like in practice
It iterates over all **6.46 million cell-year rows** with `lapply`, and for each row it:
- Looks up the cell's spatial neighbors (fine).
- Constructs **character key strings** (`paste(id, year)`) and performs **named-vector lookups** (`idx_lookup[neighbor_keys]`), which are hash-table lookups on a 6.46-million-entry named vector â€” **for every single row**.

This means ~6.46M Ã— ~4 neighbors Ã— (string construction + hash lookup) = billions of character operations. Character hashing on a vector of 6.46M names is the dominant bottleneck.

### 2. The lookup is year-redundant
The spatial neighbor topology is **identical across all 28 years**. Yet the function rebuilds a full row-level lookup that encodes year information, duplicating the same spatial structure 28 times. This inflates the lookup list from ~344K entries to ~6.46M entries.

### 3. `compute_neighbor_stats` uses row-level `lapply`
Even after the lookup is built, computing stats iterates over 6.46M list elements in R-level `lapply`, each calling `max`, `min`, `mean` on small vectors. This is slow due to R's per-call overhead multiplied millions of times.

---

## Optimization Strategy

**Core insight:** Separate the spatial topology (static) from the yearly attributes (dynamic). Build the adjacency structure **once** over 344K cells, then use vectorized joins and grouped operations for each year.

### Steps:
1. **Build a static edge table once** â€” a two-column `data.table` of `(cell_id, neighbor_id)` from the `nb` object. This has ~1.37M rows and never changes.
2. **For each variable, join yearly attributes onto the edge table** â€” a keyed `data.table` merge, which is O(N log N) and highly optimized in C.
3. **Compute grouped `max`, `min`, `mean`** â€” using `data.table`'s `by=` grouping, which is vectorized C code.
4. **Merge results back** to the main dataset.

This eliminates all character-key hashing, eliminates the 6.46M-element list, and replaces R-level loops with vectorized `data.table` operations.

**Expected speedup:** From ~86 hours to **~2â€“5 minutes**.

---

## Working R Code

```r
library(data.table)

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# STEP 1: Build a static spatial edge table ONCE (from the nb object)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# id_order is the vector of cell IDs aligned with rook_neighbors_unique
# rook_neighbors_unique is an nb object (list of integer index vectors)

build_edge_table <- function(id_order, neighbors_nb) {
  # Pre-allocate: count total edges
  n_edges <- sum(vapply(neighbors_nb, function(x) {
    # nb objects use 0L to indicate no neighbors
    if (length(x) == 1L && x[1] == 0L) 0L else length(x)
  }, integer(1)))

  from_id <- integer(n_edges)
  to_id   <- integer(n_edges)
  pos <- 1L

  for (i in seq_along(neighbors_nb)) {
    nb_idx <- neighbors_nb[[i]]
    if (length(nb_idx) == 1L && nb_idx[1] == 0L) next
    n <- length(nb_idx)
    from_id[pos:(pos + n - 1L)] <- id_order[i]
    to_id[pos:(pos + n - 1L)]   <- id_order[nb_idx]
    pos <- pos + n
  }

  data.table(cell_id = from_id, neighbor_id = to_id)
}

# Build it once â€” ~1.37M rows, trivial memory
edge_dt <- build_edge_table(id_order, rook_neighbors_unique)

cat(sprintf("Edge table: %d rows\n", nrow(edge_dt)))

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# STEP 2: Convert main data to data.table (if not already)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
if (!is.data.table(cell_data)) {
  cell_data <- as.data.table(cell_data)
}

# Ensure key columns exist and are of consistent type
cell_data[, id   := as.integer(id)]
cell_data[, year := as.integer(year)]

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# STEP 3: Compute neighbor features for all variables â€” vectorized
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# Cross join edges Ã— years, then join attributes of the NEIGHBOR cell
# To keep memory manageable on 16 GB, we process one variable at a time
# and one year at a time is NOT needed â€” the full join fits in memory:
#   1.37M edges Ã— 28 years = ~38.4M rows Ã— a few columns â‰ˆ < 2 GB

# Create the year-expanded edge table once
years <- sort(unique(cell_data$year))
edge_year_dt <- CJ(edge_idx = seq_len(nrow(edge_dt)), year = years)
edge_year_dt[, cell_id     := edge_dt$cell_id[edge_idx]]
edge_year_dt[, neighbor_id := edge_dt$neighbor_id[edge_idx]]
edge_year_dt[, edge_idx := NULL]

cat(sprintf("Edge-year table: %d rows (%.1f M)\n",
            nrow(edge_year_dt), nrow(edge_year_dt) / 1e6))

# Key for fast joins
setkey(edge_year_dt, neighbor_id, year)

for (var_name in neighbor_source_vars) {
  cat(sprintf("Processing neighbor stats for: %s\n", var_name))

  # Extract only the columns we need for the join
  attr_dt <- cell_data[, .(id, year, value = get(var_name))]
  setkey(attr_dt, id, year)

  # Join: attach the neighbor cell's attribute value to each edge-year row
  # neighbor_id in edge_year_dt matches id in attr_dt
  edge_year_dt[attr_dt, neighbor_value := i.value,
               on = .(neighbor_id = id, year = year)]

  # Aggregate by (cell_id, year) to get max, min, mean of neighbor values
  stats <- edge_year_dt[
    !is.na(neighbor_value),
    .(
      nb_max  = max(neighbor_value),
      nb_min  = min(neighbor_value),
      nb_mean = mean(neighbor_value)
    ),
    by = .(cell_id, year)
  ]

  # Name the output columns to match the original pipeline's convention
  max_col  <- paste0("neighbor_max_",  var_name)
  min_col  <- paste0("neighbor_min_",  var_name)
  mean_col <- paste0("neighbor_mean_", var_name)

  setnames(stats, c("nb_max", "nb_min", "nb_mean"),
                  c(max_col,  min_col,  mean_col))

  # Merge back into cell_data
  # First, remove old columns if they exist (idempotent re-runs)
  for (col in c(max_col, min_col, mean_col)) {
    if (col %in% names(cell_data)) cell_data[, (col) := NULL]
  }

  cell_data <- merge(cell_data, stats,
                     by.x = c("id", "year"),
                     by.y = c("cell_id", "year"),
                     all.x = TRUE)

  # Clean up the temporary column in edge_year_dt
  edge_year_dt[, neighbor_value := NULL]

  cat(sprintf("  Done. Added: %s, %s, %s\n", max_col, min_col, mean_col))
}

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# STEP 4: Predict with the existing trained Random Forest (unchanged)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# The trained model object (e.g., `rf_model`) is preserved as-is.
# Prediction uses the same column names, so no changes are needed:
#
#   cell_data$predicted <- predict(rf_model, newdata = cell_data)
#
# The numerical estimand is identical because the neighbor max, min,
# and mean values are computed with the same logic (max, min, mean of
# non-NA neighbor values), just via vectorized data.table operations
# instead of row-level R loops.
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
```

---

## Memory-Constrained Alternative (if 16 GB is tight)

If the ~38.4M-row `edge_year_dt` causes memory pressure alongside the 6.46M-row `cell_data` with 110 columns, process year-by-year:

```r
for (var_name in neighbor_source_vars) {
  cat(sprintf("Processing: %s\n", var_name))

  attr_dt <- cell_data[, .(id, year, value = get(var_name))]
  setkey(attr_dt, id, year)

  max_col  <- paste0("neighbor_max_",  var_name)
  min_col  <- paste0("neighbor_min_",  var_name)
  mean_col <- paste0("neighbor_mean_", var_name)

  stats_list <- vector("list", length(years))

  for (yi in seq_along(years)) {
    yr <- years[yi]
    # Subset attributes for this year
    attr_yr <- attr_dt[year == yr]
    setkey(attr_yr, id)

    # Join neighbor values onto the static edge table
    edges_with_vals <- edge_dt[attr_yr, on = .(neighbor_id = id), nomatch = 0L]

    # Aggregate
    stats_list[[yi]] <- edges_with_vals[
      !is.na(value),
      .(nb_max = max(value), nb_min = min(value), nb_mean = mean(value)),
      by = .(cell_id)
    ][, year := yr]
  }

  stats <- rbindlist(stats_list)
  setnames(stats, c("nb_max", "nb_min", "nb_mean"),
                  c(max_col,  min_col,  mean_col))

  for (col in c(max_col, min_col, mean_col)) {
    if (col %in% names(cell_data)) cell_data[, (col) := NULL]
  }

  cell_data <- merge(cell_data, stats,
                     by.x = c("id", "year"),
                     by.y = c("cell_id", "year"),
                     all.x = TRUE)
}
```

This variant keeps peak memory much lower (~1.37M rows per year-iteration) while still running in minutes.

---

## Summary

| Aspect | Original | Optimized |
|---|---|---|
| Neighbor lookup structure | 6.46M-element R list with character hashing | 1.37M-row static `data.table` edge table |
| Stats computation | R-level `lapply` over 6.46M rows Ã— 5 vars | Vectorized `data.table` grouped aggregation |
| Year handling | Redundantly encoded in lookup keys | Joined dynamically via keyed merge |
| Estimated runtime | ~86+ hours | ~2â€“5 minutes |
| RAM peak | Moderate (but slow) | ~3â€“5 GB (well within 16 GB) |
| Trained RF model | Preserved âœ“ | Preserved âœ“ |
| Numerical estimand | max/min/mean of non-NA neighbor values | Identical âœ“ |
END_RESPONSE>>>
