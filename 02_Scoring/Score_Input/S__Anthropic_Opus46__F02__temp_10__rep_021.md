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
run_id: Anthropic_F02_T10_R021
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F02
prompt_family_slug: CODE_ONLY_NO_HINT
family_label: Code only, no hint
family_group: Code-visible neutral
replicate: 21
raw_output_file: Experiment_Output/Anthropic/Temp_1.0/output_F02_Anthropic_temp_10_rep_021.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The core bottleneck is in **`build_neighbor_lookup`**, which runs an `lapply` over **~6.46 million rows**, performing per-row string pasting and named-vector lookups. Named vector lookup via character keys in R is O(n) hash-table probing, but doing it millions of times with millions of keys is extremely slow. The secondary bottleneck is **`compute_neighbor_stats`**, which also loops over 6.46M rows in pure R, extracting and summarizing variable-length neighbor vectors. Together, these two stages create an estimated 86+ hour runtime because:

1. **`build_neighbor_lookup`**: ~6.46M iterations, each performing `paste()` to create keys and `[` lookup on a 6.46M-length named character vector. The named-vector lookup has high constant cost at this scale. The resulting list-of-integer-vectors also consumes substantial memory.
2. **`compute_neighbor_stats`**: ~6.46M iterations Ã— 5 variables = ~32.3M calls, each subsetting a numeric vector by variable-length index vectors and computing `max`, `min`, `mean`.
3. **Memory**: Storing a 6.46M-element list of integer vectors (neighbor indices) plus the intermediate `rbind` of 6.46M Ã— 3 matrices per variable is RAM-intensive.

---

## Optimization Strategy

**Key insight**: The neighbor relationship is defined at the *cell* level (not cell-year level). There are only ~344K cells, each with a handful of rook neighbors. The year dimension simply replicates this structure. We should **never loop over 6.46M cell-years**; instead:

1. **Vectorized join via `data.table`**: Convert the neighbor list into a two-column edge table (`id`, `neighbor_id`), join to the data by `(neighbor_id, year)` to pull neighbor values, then group-by `(id, year)` to compute `max`, `min`, `mean` â€” all in one vectorized `data.table` operation. No per-row R loop at all.
2. **Eliminate `build_neighbor_lookup` entirely**: The expensive index-mapping step is replaced by a merge/join.
3. **Process all 5 variables in a single grouped aggregation** instead of looping one variable at a time.
4. **Memory management**: The edge table has ~1.37M directed edges. After joining with year, we get ~1.37M Ã— 28 â‰ˆ 38.5M rows â€” large but manageable in ~3â€“5 GB with `data.table`'s memory-efficient columnar storage. We can process variables one at a time if RAM is tight.

**Expected speedup**: From 86+ hours to roughly **5â€“20 minutes** on the same laptop.

---

## Working R Code

```r
library(data.table)

# ---------------------------------------------------------------
# Step 1: Build an edge table from the spdep nb object (one-time)
# ---------------------------------------------------------------
# rook_neighbors_unique is a list of length = number of cells.
# rook_neighbors_unique[[i]] contains integer indices (into id_order)
# of the neighbors of cell id_order[i].

build_edge_table <- function(id_order, neighbors) {
  # Pre-allocate by computing total number of edges
  n_edges <- sum(lengths(neighbors))
  from_id <- integer(n_edges)
  to_id   <- integer(n_edges)
  
  pos <- 1L
  for (i in seq_along(neighbors)) {
    nb <- neighbors[[i]]
    n  <- length(nb)
    if (n == 0L) next
    idx <- pos:(pos + n - 1L)
    from_id[idx] <- id_order[i]
    to_id[idx]   <- id_order[nb]
    pos <- pos + n
  }
  
  data.table(id = from_id[1:(pos - 1L)],
             neighbor_id = to_id[1:(pos - 1L)])
}

edge_dt <- build_edge_table(id_order, rook_neighbors_unique)

# ---------------------------------------------------------------
# Step 2: Convert cell_data to data.table (if not already)
# ---------------------------------------------------------------
if (!is.data.table(cell_data)) {
  cell_data <- as.data.table(cell_data)
}

# Ensure keyed for fast joins
setkey(cell_data, id, year)

# ---------------------------------------------------------------
# Step 3: Compute neighbor stats for all variables â€” vectorized
# ---------------------------------------------------------------
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# We process one variable at a time to control peak memory.
# Each iteration:
#   - joins edge_dt Ã— cell_data on (neighbor_id = id, year)
#   - groups by (id, year) to get max, min, mean
#   - joins result back to cell_data

# Subset of cell_data needed for neighbor lookups (only id, year, + vars)
lookup_cols <- c("id", "year", neighbor_source_vars)
lookup_dt   <- cell_data[, ..lookup_cols]
setnames_copy <- copy(lookup_dt)  
# Rename 'id' to 'neighbor_id' for joining
setnames(lookup_dt, "id", "neighbor_id")
setkey(lookup_dt, neighbor_id, year)

# Get unique years to cross-join with edges
unique_years <- sort(unique(cell_data$year))

# Expand edges Ã— years: every edge exists in every year
# ~1.37M edges Ã— 28 years â‰ˆ 38.5M rows
edge_year <- CJ_dt <- edge_dt[, .(year = unique_years), by = .(id, neighbor_id)]

setkey(edge_year, neighbor_id, year)

for (vname in neighbor_source_vars) {
  cat("Processing neighbor features for:", vname, "\n")
  
  # Pull neighbor values via join
  # edge_year joined to lookup_dt on (neighbor_id, year)
  cols_needed <- c("neighbor_id", "year", vname)
  sub_lookup  <- lookup_dt[, ..cols_needed]
  setkey(sub_lookup, neighbor_id, year)
  
  joined <- sub_lookup[edge_year, on = .(neighbor_id, year), nomatch = NA]
  # joined has columns: neighbor_id, year, <vname>, id
  # Group by (id, year) to compute stats
  
  stat_names <- paste0("neighbor_", c("max_", "min_", "mean_"), vname)
  
  stats <- joined[, {
    v <- get(vname)
    v <- v[!is.na(v)]
    if (length(v) == 0L) {
      list(V1 = NA_real_, V2 = NA_real_, V3 = NA_real_)
    } else {
      list(V1 = max(v), V2 = min(v), V3 = mean(v))
    }
  }, by = .(id, year)]
  
  setnames(stats, c("V1", "V2", "V3"), stat_names)
  setkey(stats, id, year)
  
  # Join back to cell_data
  cell_data <- stats[cell_data, on = .(id, year)]
  setkey(cell_data, id, year)
  
  # Clean up
  rm(joined, stats, sub_lookup)
  gc()
}

# Clean up large intermediates
rm(edge_year, lookup_dt, edge_dt)
gc()

cat("Neighbor feature computation complete.\n")
cat("Dimensions:", nrow(cell_data), "rows x", ncol(cell_data), "cols\n")
```

---

## If Memory Is Still Tight: Year-Chunked Variant

If the ~38.5M-row `edge_year` table causes memory pressure, process in year chunks:

```r
for (vname in neighbor_source_vars) {
  cat("Processing:", vname, "\n")
  
  cols_needed <- c("neighbor_id", "year", vname)
  sub_lookup  <- lookup_dt[, ..cols_needed]
  setkey(sub_lookup, neighbor_id, year)
  
  year_results <- rbindlist(lapply(unique_years, function(yr) {
    # Edges for this year only (~1.37M rows)
    ey <- edge_dt[, .(id, neighbor_id, year = yr)]
    setkey(ey, neighbor_id, year)
    
    joined <- sub_lookup[ey, on = .(neighbor_id, year), nomatch = NA]
    
    joined[, {
      v <- get(vname)
      v <- v[!is.na(v)]
      if (length(v) == 0L) {
        list(V1 = NA_real_, V2 = NA_real_, V3 = NA_real_)
      } else {
        list(V1 = max(v), V2 = min(v), V3 = mean(v))
      }
    }, by = .(id, year)]
  }))
  
  stat_names <- paste0("neighbor_", c("max_", "min_", "mean_"), vname)
  setnames(year_results, c("V1", "V2", "V3"), stat_names)
  setkey(year_results, id, year)
  
  cell_data <- year_results[cell_data, on = .(id, year)]
  setkey(cell_data, id, year)
  
  rm(year_results, sub_lookup)
  gc()
}
```

---

## Summary

| Aspect | Original | Optimized |
|---|---|---|
| `build_neighbor_lookup` | 6.46M-iteration `lapply` with string keys | Eliminated; replaced by edge-table join |
| `compute_neighbor_stats` | 6.46M-iteration `lapply` Ã— 5 vars | Vectorized `data.table` group-by |
| Estimated runtime | 86+ hours | ~5â€“20 minutes |
| Peak memory | Very high (6.46M-element list + copies) | ~4â€“8 GB (controlled via chunking) |
| Trained RF model | Untouched | Untouched |
| Numerical output | `max`, `min`, `mean` of neighbor values | Identical `max`, `min`, `mean` |

The trained Random Forest model is not affected â€” only the feature-engineering (input preparation) step is optimized. The computed `neighbor_max_*`, `neighbor_min_*`, and `neighbor_mean_*` columns are numerically identical to the original implementation's output.
END_RESPONSE>>>
