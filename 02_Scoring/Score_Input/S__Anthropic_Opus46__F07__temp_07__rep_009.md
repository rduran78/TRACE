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
run_id: Anthropic_F07_T07_R009
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F07
prompt_family_slug: RASTER_INVALID_IRREGULAR
family_label: Raster invalid irregular topology
family_group: Raster/kernel bridge
replicate: 9
raw_output_file: Experiment_Output/Anthropic/Temp_0.7/output_F07_Anthropic_temp_07_rep_009.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

## Diagnosis

The bottleneck is **`build_neighbor_lookup`**. It creates a list of length ~6.46 million, where each element is built by:

1. **Character key construction and lookup** (`paste` + named-vector indexing) â€” for every single row. Named-vector lookup in R is O(n) in the worst case per query because it uses linear hashing with potential collisions, and doing this ~6.46M times with a lookup vector of ~6.46M entries is catastrophic.
2. **`lapply` over 6.46M rows** â€” each iteration does string pasting, named-vector subsetting, and NA filtering. The per-element overhead of `lapply` at this scale is significant.
3. **`compute_neighbor_stats`** then does *another* `lapply` over 6.46M elements, 5 times (once per variable). That's 32.3M R-level function calls with vector subsetting inside each.

**Root cause summary:**
- Named character vector indexing is used as a hash map but is extremely slow at scale in R.
- The entire approach is row-wise in R (no vectorization).
- The neighbor lookup is ~6.46M list elements, each constructed via string operations â€” this alone likely accounts for 80+ hours.

## Optimization Strategy

1. **Replace character-key lookups with integer-arithmetic joins.** Since years are contiguous (1992â€“2019, 28 years), we can compute the row index of any (cell, year) pair arithmetically if the data is sorted by (id, year). Row index = `(cell_position - 1) * 28 + (year - 1992) + 1`. This eliminates all `paste`/string operations and named-vector lookups.

2. **Vectorize neighbor stat computation using `data.table`.** Expand the neighbor list into an edge list (from_row, to_row), join the variable values, and compute grouped `max`, `min`, `mean` in one vectorized pass per variable.

3. **Avoid creating the 6.46M-element list entirely.** The edge-list approach replaces it with a two-column integer matrix (~1.37M edges Ã— 28 years â‰ˆ 38.5M rows), which `data.table` handles in seconds.

**Expected speedup:** From 86+ hours to **~2â€“5 minutes**.

## Working R Code

```r
library(data.table)

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 0.  Ensure cell_data is a data.table sorted by (id, year)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
cell_dt <- as.data.table(cell_data)
setorder(cell_dt, id, year)              # sort in place
cell_dt[, row_idx := .I]                 # row index 1..N

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 1.  Build an integer mapping:  cell id  â†’  position (1-based)
#     and store year range info
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
id_order_vec  <- id_order                          # length = 344,208
n_cells       <- length(id_order_vec)
year_min      <- 1992L
n_years       <- 28L                               # 1992-2019

# Map from cell id to its 1-based position in id_order
id_to_pos <- integer(max(id_order_vec))            # direct-address table
id_to_pos[id_order_vec] <- seq_along(id_order_vec)
# If ids are not contiguous / too large, use data.table or environment:
# But for typical grid-cell integer ids this is fine.
# Fallback for very large / sparse ids:
if (max(id_order_vec) > 5e7) {
  id_to_pos_env <- new.env(hash = TRUE, size = n_cells)
  for (k in seq_along(id_order_vec)) {
    id_to_pos_env[[as.character(id_order_vec[k])]] <- k
  }
  get_pos <- function(ids) {
    vapply(as.character(ids), function(x) id_to_pos_env[[x]], integer(1),
           USE.NAMES = FALSE)
  }
} else {
  get_pos <- function(ids) id_to_pos[ids]
}

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 2.  Build directed edge list  (from_cell_pos, to_cell_pos)
#     from the spdep::nb object  rook_neighbors_unique
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
from_pos <- rep(
  seq_along(rook_neighbors_unique),
  lengths(rook_neighbors_unique)
)
to_pos <- unlist(rook_neighbors_unique, use.names = FALSE)

# Remove the spdep convention where 0 means "no neighbors"
valid <- to_pos != 0L
from_pos <- from_pos[valid]
to_pos   <- to_pos[valid]

edges <- data.table(from_pos = from_pos, to_pos = to_pos)
cat("Directed edges (cell-level):", nrow(edges), "\n")

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 3.  Expand edges across all 28 years to get (from_row, to_row)
#     Row index formula (data sorted by id, year):
#       row_of(cell_pos, year) = (cell_pos - 1) * n_years + (year - year_min) + 1
#
#     We verify the sort assumption:
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Verify mapping is correct for a sample
stopifnot(all(cell_dt$id == id_order_vec[
  rep(seq_len(n_cells), each = n_years)
]))

years_vec <- seq.int(year_min, year_min + n_years - 1L)

# Cross join edges Ã— years  (38.5M rows, 3 integer cols â‰ˆ 460 MB)
edge_years <- CJ(edge_idx = seq_len(nrow(edges)), year = years_vec)
edge_years[, `:=`(
  from_row = (edges$from_pos[edge_idx] - 1L) * n_years + (year - year_min) + 1L,
  to_row   = (edges$to_pos[edge_idx]   - 1L) * n_years + (year - year_min) + 1L
)]
edge_years[, edge_idx := NULL]

cat("Edge-year rows:", nrow(edge_years), "\n")

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 4.  Compute neighbor stats per variable â€” fully vectorized
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {
  cat("Processing neighbor stats for:", var_name, "\n")

  # Attach the neighbor's value to each edge-year row
  edge_years[, nbr_val := cell_dt[[var_name]][to_row]]

  # Compute grouped stats by from_row (= the focal cell-year)
  stats <- edge_years[
    !is.na(nbr_val),
    .(
      nb_max  = max(nbr_val),
      nb_min  = min(nbr_val),
      nb_mean = mean(nbr_val)
    ),
    keyby = from_row
  ]

  # Initialize columns with NA
  max_col  <- paste0(var_name, "_neighbor_max")
  min_col  <- paste0(var_name, "_neighbor_min")
  mean_col <- paste0(var_name, "_neighbor_mean")

  cell_dt[, (max_col)  := NA_real_]
  cell_dt[, (min_col)  := NA_real_]
  cell_dt[, (mean_col) := NA_real_]

  # Assign results
  cell_dt[stats$from_row, (max_col)  := stats$nb_max]
  cell_dt[stats$from_row, (min_col)  := stats$nb_min]
  cell_dt[stats$from_row, (mean_col) := stats$nb_mean]

  # Clean up for next iteration
  edge_years[, nbr_val := NULL]
  rm(stats)
}

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 5.  Convert back to data.frame if needed downstream
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
cell_dt[, row_idx := NULL]
cell_data <- as.data.frame(cell_dt)

# The trained Random Forest model is untouched â€” only input features
# were computed. The numerical estimand is preserved exactly:
#   neighbor_max  = max of non-NA rook-neighbor values
#   neighbor_min  = min of non-NA rook-neighbor values
#   neighbor_mean = mean of non-NA rook-neighbor values
# These are identical to the original compute_neighbor_stats definitions.
```

## Memory Note

The `edge_years` table is the largest object (~38.5M rows Ã— 3 integer columns + 1 double column â‰ˆ 460â€“770 MB). On a 16 GB laptop this is feasible. If memory is tight, process in year-batches:

```r
for (yr in years_vec) {
  ey <- edges[, .(
    from_row = (from_pos - 1L) * n_years + (yr - year_min) + 1L,
    to_row   = (to_pos   - 1L) * n_years + (yr - year_min) + 1L
  )]
  # ... compute stats for this year-slice and rbind
}
```

## Summary

| Aspect | Original | Optimized |
|---|---|---|
| Lookup construction | 6.46M string pastes + named-vector lookups | Integer arithmetic, O(1) per edge |
| Stat computation | 6.46M Ã— 5 `lapply` calls | 5 vectorized `data.table` group-bys |
| Data structure | 6.46M-element R list | Single edge-list `data.table` |
| Estimated runtime | 86+ hours | **2â€“5 minutes** |
| Numerical result | max/min/mean of non-NA rook neighbors | **Identical** |
| RF model | Untouched | **Untouched** |
END_RESPONSE>>>
