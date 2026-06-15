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
run_id: Anthropic_F08_T00_R017
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F08
prompt_family_slug: TOPOLOGY_INVARIANCE
family_label: Topology invariance
family_group: Topology cue
replicate: 17
raw_output_file: Experiment_Output/Anthropic/Temp_0.0/output_F08_Anthropic_temp_00_rep_017.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

## Diagnosis

The core performance problem is that `build_neighbor_lookup` constructs a **per-row** (per cell-year) lookup, producing ~6.46 million list entries, when the neighbor topology is **identical across all 28 years**. The function pastes cell IDs and years into string keys, performs hash lookups for every neighbor of every row, and repeats all of this redundantly for every year. This means:

1. **Redundant topology computation**: The neighbor graph is spatial and static. There are only 344,208 unique cells, but the lookup is built for 6,460,000+ rows. That's a ~19Ã— blowup.
2. **String-key hashing at scale**: `paste(id, year)` key construction and named-vector lookup for ~6.46M rows Ã— ~4 neighbors each â‰ˆ 25.8M string operations.
3. **Per-row R-level `lapply`**: 6.46M iterations of an R closure is inherently slow.
4. **`compute_neighbor_stats` also uses per-row `lapply`**: Another 6.46M R-level iterations per variable, times 5 variables = 32.3M iterations.

The fundamental insight: **neighbor relationships are between cells, not between cell-years**. The topology needs to be computed only once over 344,208 cells. Then for each year, we simply slice the variable values by cell and apply the static topology.

## Optimization Strategy

1. **Build the neighbor lookup once over cells, not cell-years.** Convert `rook_neighbors_unique` (an `nb` object) into a simple integer-index mapping from cell position â†’ neighbor positions. This is O(344K) and trivial.

2. **Organize data so that for each year, variable values are in a vector indexed by cell position.** Use a matrix (cells Ã— years) or split-by-year approach.

3. **Vectorize neighbor stat computation using `data.table` and matrix operations.** For each variable, build a cellÃ—year matrix, then compute neighbor max/min/mean using the static adjacency list â€” iterating over 344K cells (not 6.46M rows) and leveraging vectorized column operations.

4. **Use `data.table` for fast joins** to merge results back.

5. **Preserve the trained RF model and numerical outputs exactly** â€” we only change how neighbor features are computed, not what they are.

Expected speedup: from ~86 hours to **minutes**.

## Working R Code

```r
library(data.table)

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# STEP 0: Convert cell_data to data.table if not already
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
if (!is.data.table(cell_data)) {
  cell_data <- as.data.table(cell_data)
}

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# STEP 1: Build STATIC neighbor lookup (once, over cells only)
#
# rook_neighbors_unique is an nb object of length = length(id_order).
# id_order[i] is the cell id for position i.
# rook_neighbors_unique[[i]] gives integer positions of neighbors of cell i.
# This is ALREADY the static lookup we need â€” no transformation required
# beyond ensuring 0-neighbor entries are handled.
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

n_cells <- length(id_order)

# Precompute: for each cell position, which positions are its neighbors?
# nb objects store 0L for no-neighbor cases; normalise to integer(0).
static_neighbors <- lapply(seq_len(n_cells), function(i) {

  nb_i <- rook_neighbors_unique[[i]]
  if (length(nb_i) == 1L && nb_i[1] == 0L) integer(0) else as.integer(nb_i)
})

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# STEP 2: Establish a consistent cell-position index in the data
#
# We need each row's cell to map to a position in 1..n_cells matching
# the order in id_order (which matches the nb object).
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

id_to_pos <- setNames(seq_along(id_order), as.character(id_order))
cell_data[, cell_pos := id_to_pos[as.character(id)]]

# Get sorted unique years
years <- sort(unique(cell_data$year))
n_years <- length(years)
year_to_col <- setNames(seq_along(years), as.character(years))
cell_data[, year_idx := year_to_col[as.character(year)]]

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# STEP 3: Function to compute neighbor stats for one variable
#
# Strategy: build a matrix [n_cells x n_years] of variable values,
# then for each cell, pull neighbor rows and compute column-wise
# (i.e., year-wise) max, min, mean across neighbors.
#
# To avoid a slow R loop over 344K cells, we use a "sparse expansion"
# approach: create a long table of (cell_pos, neighbor_pos), join
# variable values, and aggregate with data.table.
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

compute_neighbor_features_fast <- function(dt, var_name, static_neighbors, years) {

  cat("Computing neighbor features for:", var_name, "\n")

  # --- Build edge list (static, computed once but passed in; we build here

  #     for clarity; in practice, factor this out) ---
  # Edge list: data.table with columns (cell_pos, neighbor_pos)
  # We'll build this once outside and reuse â€” see below.


  # --- Build cell_pos Ã— year_idx value table ---
  val_dt <- dt[, .(cell_pos, year_idx, val = get(var_name))]
  setkey(val_dt, cell_pos, year_idx)

  # --- Join neighbor values via edge list ---
  # For each (cell_pos, neighbor_pos) pair and each year_idx,
  # get the neighbor's value, then aggregate.
  # edge_dt is (cell_pos, neighbor_pos) â€” see below, we use the
  # pre-built one.

  # Join: for each edge (cell_pos, neighbor_pos), for each year,
  # get neighbor's value.
  neighbor_vals <- edge_dt[val_dt,
    on = .(neighbor_pos = cell_pos),
    .(cell_pos = x.cell_pos, year_idx = i.year_idx, val = i.val),
    allow.cartesian = TRUE,
    nomatch = NULL
  ]

  # Aggregate by (cell_pos, year_idx)
  stats <- neighbor_vals[,
    .(
      nb_max  = max(val, na.rm = TRUE),
      nb_min  = min(val, na.rm = TRUE),
      nb_mean = mean(val, na.rm = TRUE)
    ),
    by = .(cell_pos, year_idx)
  ]

  # Fix Inf/-Inf from all-NA groups
  stats[is.infinite(nb_max), nb_max := NA_real_]
  stats[is.infinite(nb_min), nb_min := NA_real_]
  stats[is.nan(nb_mean), nb_mean := NA_real_]

  # Rename columns to match original feature names
  max_col  <- paste0("neighbor_max_", var_name)
  min_col  <- paste0("neighbor_min_", var_name)
  mean_col <- paste0("neighbor_mean_", var_name)
  setnames(stats, c("nb_max", "nb_min", "nb_mean"), c(max_col, min_col, mean_col))

  return(stats)
}

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# STEP 4: Build the STATIC edge list ONCE (reused for all variables)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

cat("Building static edge list...\n")

edge_list <- rbindlist(lapply(seq_len(n_cells), function(i) {
  nb <- static_neighbors[[i]]
  if (length(nb) == 0L) return(NULL)
  data.table(cell_pos = i, neighbor_pos = nb)
}))

setkey(edge_list, neighbor_pos)

# Make it available to the function (or pass explicitly)
edge_dt <- edge_list

cat("Edge list built:", nrow(edge_dt), "directed edges\n")

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# STEP 5: Compute and attach neighbor features for all variables
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# Create a join key in cell_data
setkey(cell_data, cell_pos, year_idx)

for (var_name in neighbor_source_vars) {

  # --- Build cell_pos Ã— year_idx value table ---
  val_dt <- cell_data[, .(cell_pos, year_idx, val = get(var_name))]
  setkey(val_dt, cell_pos, year_idx)

  # --- For each edge, look up the NEIGHBOR's value in each year ---
  # edge_dt has (cell_pos, neighbor_pos). We want, for each year,
  # the value of the neighbor cell.
  # Join val_dt onto edge_dt by neighbor_pos == cell_pos
  neighbor_vals <- val_dt[edge_dt,
    on = .(cell_pos = neighbor_pos),
    .(cell_pos = i.cell_pos,   # the focal cell
      year_idx = x.year_idx,
      val      = x.val),
    allow.cartesian = TRUE,
    nomatch = NULL
  ]

  # Aggregate by (focal cell, year)
  stats <- neighbor_vals[!is.na(val),
    .(
      nb_max  = max(val),
      nb_min  = min(val),
      nb_mean = mean(val)
    ),
    by = .(cell_pos, year_idx)
  ]

  # Name columns to match original pipeline
  max_col  <- paste0("neighbor_max_", var_name)
  min_col  <- paste0("neighbor_min_", var_name)
  mean_col <- paste0("neighbor_mean_", var_name)
  setnames(stats, c("nb_max", "nb_min", "nb_mean"), c(max_col, min_col, mean_col))
  setkey(stats, cell_pos, year_idx)

  # Merge into cell_data
  # Remove old columns if they exist (idempotency)
  for (col in c(max_col, min_col, mean_col)) {
    if (col %in% names(cell_data)) cell_data[, (col) := NULL]
  }

  cell_data <- stats[cell_data, on = .(cell_pos, year_idx)]

  cat("Done:", var_name, "\n")
}

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# STEP 6: Clean up helper columns
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
cell_data[, c("cell_pos", "year_idx") := NULL]

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# STEP 7: Predict with the pre-trained Random Forest (UNCHANGED)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# predictions <- predict(rf_model, newdata = cell_data)
# (This step remains exactly as in the original pipeline.)

cat("Neighbor feature computation complete.\n")
```

## Summary of Changes and Guarantees

| Aspect | Original | Redesigned |
|---|---|---|
| **Neighbor lookup granularity** | Per cell-year (~6.46M entries) | Per cell (~344K entries, static) |
| **String key construction** | ~25.8M `paste()` + hash lookups | Zero â€” integer positions only |
| **Stat computation loop** | `lapply` over 6.46M rows Ã— 5 vars | `data.table` grouped aggregation |
| **Edge list** | Rebuilt implicitly per row | Built once (1.37M edges), reused |
| **Numerical result** | max/min/mean of neighbor values | Identical â€” same neighbors, same values, same aggregation functions |
| **RF model** | Not touched | Not touched â€” only `newdata` is prepared |
| **Expected runtime** | ~86+ hours | ~5â€“15 minutes |

**Key invariant preserved**: For every (cell, year) row, the neighbor max, min, and mean of each variable are computed over exactly the same set of rook-contiguous neighbors with exactly the same values as the original code. The static topology is separated from the year-varying values, eliminating all redundant computation.
END_RESPONSE>>>
