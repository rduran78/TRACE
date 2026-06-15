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
run_id: Anthropic_F02_T10_R016
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F02
prompt_family_slug: CODE_ONLY_NO_HINT
family_label: Code only, no hint
family_group: Code-visible neutral
replicate: 16
raw_output_file: Experiment_Output/Anthropic/Temp_1.0/output_F02_Anthropic_temp_10_rep_016.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The performance bottleneck has two root causes:

### 1. `build_neighbor_lookup` â€” O(n) `lapply` over 6.46 million rows with per-row string pasting and named-vector lookups

- **String key construction**: `paste(id, year, sep="_")` is called millions of times inside the loop body (once per neighbor per row), producing enormous intermediate character vectors.
- **Named-vector lookup (`idx_lookup[neighbor_keys]`)**: Named vector indexing in R uses hashing, but creating and resolving ~6.46M Ã— avg-neighbors character keys is extremely slow and memory-hungry.
- **`lapply` over 6.46M elements**: Each closure allocates small vectors; the cumulative overhead is massive.

### 2. `compute_neighbor_stats` â€” repeated `lapply` over 6.46M ragged index lists

- Called once per variable (Ã—5), each time iterating 6.46M list elements.
- `do.call(rbind, result)` on a 6.46M-element list of length-3 vectors is a known R anti-pattern (quadratic copy risk in older R; still slow).

### Memory pressure

- The `neighbor_lookup` list alone: 6.46M list elements, each a small integer vector â†’ hundreds of MB of list overhead (each list element has a 56-byte SEXP header on 64-bit R, so ~360 MB just in headers, plus the integer data).
- Holding all of `cell_data` (~6.46M Ã— 110 columns) simultaneously adds several more GB.

**Net effect**: the combination of character-key gymnastics, per-row R-level iteration, and ragged list structures yields the estimated 86+ hour runtime.

---

## Optimization Strategy

| Technique | What it fixes | Expected speedup |
|---|---|---|
| **Replace the ragged list with a flat edge table (data.table)** | Eliminates per-row `lapply`, string keys, and named-vector lookups | 50â€“200Ã— for `build_neighbor_lookup` |
| **Vectorised join + grouped aggregation (`data.table`)** | Replaces `lapply` in `compute_neighbor_stats` with a single keyed join and `by=` aggregation | 20â€“100Ã— per variable |
| **Process all 5 variables in one pass** | Avoids 5 separate full-table scans | ~5Ã— for the stats phase |
| **Integer keys throughout** | Avoid all `paste`/character work; use composite integer keys or multi-column keys | Large constant-factor improvement |
| **In-place `:=` assignment** | Avoid repeated shallow-copy of the growing `cell_data` frame | Reduces memory spikes |

The strategy preserves every numerical output (max, min, mean of rook neighbors per variable per cell-year) and never touches the trained Random Forest model.

---

## Working R Code

```r
library(data.table)

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 0.  Ensure cell_data is a data.table with an integer row-id column
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
if (!is.data.table(cell_data)) setDT(cell_data)

# Ensure 'id' and 'year' are integer (avoids implicit coercion later)
cell_data[, id   := as.integer(id)]
cell_data[, year := as.integer(year)]

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 1.  Build a flat edge table from the nb object (once)
#
#     rook_neighbors_unique is a list of length N_cells (344,208).
#     id_order is the vector that maps list-position â†’ cell id.
#     We explode it into a two-column data.table: (id, neighbor_id).
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
build_edge_table <- function(id_order, neighbors) {
  # Pre-allocate vectors
  n_edges <- sum(lengths(neighbors))
  from_id <- integer(n_edges)
  to_id   <- integer(n_edges)

  pos <- 1L
  for (i in seq_along(neighbors)) {
    nb_i <- neighbors[[i]]
    if (length(nb_i) == 0L || (length(nb_i) == 1L && nb_i[1] == 0L)) next
    n_i <- length(nb_i)
    from_id[pos:(pos + n_i - 1L)] <- id_order[i]
    to_id[pos:(pos + n_i - 1L)]   <- id_order[nb_i]
    pos <- pos + n_i
  }
  # Trim if any nb entries were empty / zero-neighbor islands
  data.table(id = from_id[1:(pos - 1L)],
             neighbor_id = to_id[1:(pos - 1L)])
}

cat("Building edge table â€¦\n")
edge_dt <- build_edge_table(id_order, rook_neighbors_unique)
# edge_dt now has ~1.37 M rows (directed edges), all integer columns.

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 2.  Vectorised neighbor-stat computation
#
#     For every (id, year) pair and every variable, we need:
#       neighbor_max, neighbor_min, neighbor_mean
#     across all rook neighbors present in the data for that year.
#
#     Approach:
#       a) Join edge_dt onto cell_data to get (id, year, neighbor_id).
#       b) Join again to pull the neighbor's variable values.
#       c) Group-by (id, year) and compute max/min/mean.
#     All of this is one pass per variable (or batched).
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

# Key cell_data for fast joins
setkey(cell_data, id, year)

# We need a lookup from (neighbor_id, year) â†’ variable values.
# Build a small reference table with only the columns we need.
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# Subset only what's needed for the neighbor lookup
ref_cols <- c("id", "year", neighbor_source_vars)
ref_dt   <- cell_data[, ..ref_cols]
setnames(ref_dt, "id", "neighbor_id")
setkey(ref_dt, neighbor_id, year)

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 3.  For each variable, join â†’ aggregate â†’ assign back in-place
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
cat("Computing neighbor features â€¦\n")

for (var in neighbor_source_vars) {

  cat("  â†’", var, "\n")

  # Columns we need from the reference table for this variable
  ref_sub <- ref_dt[, .(neighbor_id, year, val = get(var))]
  setkey(ref_sub, neighbor_id, year)

  # Step A: expand cell_data rows by their neighbors
  #   Start from cell_data's (id, year), join to edge_dt to get neighbor_id,
  #   then join to ref_sub to get the neighbor's value.
  #
  #   To avoid materialising the full 6.46M Ã— avg_neighbors table in memory
  #   we process year-by-year (28 chunks â‰ˆ 230K Ã— 4 neighbors each).

  stats_list <- vector("list", length(unique(cell_data$year)))
  years <- sort(unique(cell_data$year))

  for (yi in seq_along(years)) {
    yr <- years[yi]

    # Rows for this year: their ids
    ids_yr <- cell_data[year == yr, .(id)]

    # Attach neighbor ids via edge table
    #   ids_yr  join  edge_dt  on id  â†’  (id, neighbor_id)
    expanded <- edge_dt[ids_yr, on = "id", nomatch = NULL, allow.cartesian = TRUE]
    # expanded has columns: id, neighbor_id

    # Attach neighbor values
    expanded[, year := yr]
    expanded <- ref_sub[expanded, on = .(neighbor_id, year), nomatch = NA]
    # Now expanded has: neighbor_id, year, val, id

    # Aggregate
    stats_yr <- expanded[!is.na(val),
                         .(nb_max  = max(val),
                           nb_min  = min(val),
                           nb_mean = mean(val)),
                         by = .(id)]
    stats_yr[, year := yr]

    stats_list[[yi]] <- stats_yr
  }

  stats_all <- rbindlist(stats_list)
  setkey(stats_all, id, year)

  # Assign columns back into cell_data in-place
  max_col  <- paste0("neighbor_max_",  var)
  min_col  <- paste0("neighbor_min_",  var)
  mean_col <- paste0("neighbor_mean_", var)

  # Initialize with NA
  cell_data[, (max_col)  := NA_real_]
  cell_data[, (min_col)  := NA_real_]
  cell_data[, (mean_col) := NA_real_]

  # Update join â€” only touches rows that have neighbor data
  cell_data[stats_all,
            `:=`(
              (max_col)  = i.nb_max,
              (min_col)  = i.nb_min,
              (mean_col) = i.nb_mean
            ),
            on = .(id, year)]

  # Free intermediate objects
  rm(ref_sub, stats_list, stats_all, expanded, stats_yr, ids_yr)
  gc()
}

cat("Done. Neighbor features added to cell_data.\n")
```

---

## Why This Preserves Correctness

| Concern | Guarantee |
|---|---|
| **Numerical estimand** | `max()`, `min()`, `mean()` are applied to exactly the same set of non-NA rook-neighbor values per (id, year) as the original code. The join path `cell_data â†’ edge_dt â†’ ref_dt` reproduces the same directed neighbor mapping that `build_neighbor_lookup` constructed via `id_to_ref` and `idx_lookup`. |
| **Trained Random Forest** | The model object is never referenced or modified. Only `cell_data` gains 15 new columns (3 stats Ã— 5 variables) whose names match the originals. |
| **Column names** | Naming convention `neighbor_max_<var>`, `neighbor_min_<var>`, `neighbor_mean_<var>` matches the output of the presumed `compute_and_add_neighbor_features` wrapper. Adjust the prefix if your wrapper uses a different convention. |

---

## Expected Performance

| Phase | Original | Optimised | Notes |
|---|---|---|---|
| `build_neighbor_lookup` (â†’ `build_edge_table`) | ~hours (6.46M `paste` + named lookups) | **< 5 seconds** | Simple integer pre-allocation loop; runs once |
| `compute_neighbor_stats` per variable | ~15+ hours (6.46M `lapply`) | **~1â€“3 minutes** | `data.table` keyed join + `by=` aggregation, processed in 28 year-chunks to cap memory |
| **Total for 5 variables** | **86+ hours** | **~5â€“15 minutes** | Depends on disk I/O and available RAM |
| **Peak RAM** | High (ragged list + growing copies) | **~4â€“6 GB** | Year-chunking keeps the expanded join table small; `:=` avoids copies |

The year-by-year loop (28 iterations) is the key memory-control device: each chunk expands ~230K cells Ã— ~4 neighbors = ~920K rows, trivially small for `data.table`. If your machine has headroom you can remove the year loop and process all years at once for a further ~2Ã— speedup.
END_RESPONSE>>>
