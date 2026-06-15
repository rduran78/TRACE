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
run_id: Anthropic_F12_T07_R027
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F12
prompt_family_slug: FALSE_RBIND_DIAGNOSIS
family_label: False rbind diagnosis
family_group: Adversarial diagnosis
replicate: 27
raw_output_file: Experiment_Output/Anthropic/Temp_0.7/output_F12_Anthropic_temp_07_rep_027.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

## Diagnosis: Rejecting the Colleague's Claim

The colleague's claim that `do.call(rbind, result)` and repeated list binding inside `compute_neighbor_stats()` are the main bottleneck is **incorrect**. While `do.call(rbind, ...)` on a list of 6.46 million small vectors is not ideal, it's a one-shot operation per variable (only 5 calls total) and each call merely stacks pre-allocated length-3 vectors. This is a minor inefficiency, not the dominant bottleneck.

**The true bottleneck is `build_neighbor_lookup()`.**

Here's why:

1. **Per-row string pasting and hash lookups across 6.46 million rows.** The `lapply` inside `build_neighbor_lookup` iterates over every row (~6.46M), and for each row it:
   - Converts `data$id[i]` to character and looks it up in `id_to_ref` (a named vector lookup â€” O(n) construction, but each lookup involves string hashing).
   - Retrieves neighbor cell IDs, then calls `paste(neighbor_cell_ids, data$year[i], sep = "_")` to construct composite keys â€” **this creates ~6.46M Ã— average_neighbors string vectors**.
   - Looks up each composite key in `idx_lookup`, a named vector of length 6.46M â€” **each named-vector lookup is an O(n) hash probe on a massive table**.

2. **Scale of the problem:** With ~1,373,394 directed rook-neighbor relationships spread across 344,208 cells and 28 years, the total number of string constructions and lookups is approximately **6.46M Ã— ~4 neighbors Ã— 2 lookups â‰ˆ 50+ million string hash operations** against a 6.46M-entry named vector. Named vector lookups in R use linear-probing hash tables that degrade significantly at this scale.

3. **This function is called once but dominates runtime.** `compute_neighbor_stats` is called 5 times (once per variable) and does simple numeric indexing â€” fast. `build_neighbor_lookup` is called once but performs the combinatorial string explosion described above â€” this is where the ~86+ hours come from.

**Secondary inefficiency:** `compute_neighbor_stats` returns `c(NA,NA,NA)` or `c(max,min,mean)` per row and then `do.call(rbind, ...)` on 6.46M elements. This is suboptimal but not catastrophic. It can be trivially replaced with a pre-allocated matrix.

## Optimization Strategy

1. **Eliminate per-row string operations in `build_neighbor_lookup`.** Replace the row-level `lapply` with a vectorized merge/join strategy:
   - Pre-expand the neighbor relationships into a full edge list of `(cell_id, neighbor_id)` pairs (only ~1.37M edges).
   - Cross-join with years using `data.table` to get `(cell_id, year, neighbor_id, year)` â€” but smarter: join the data's `(id, year)` pairs against the neighbor edge list to directly obtain row indices.
   - Use `data.table` keyed joins instead of named-vector string lookups.

2. **Replace `compute_neighbor_stats`'s `do.call(rbind, ...)` with a grouped `data.table` aggregation** over the pre-built edge-to-row mapping, computing `max`, `min`, `mean` in one vectorized pass.

3. **Preserve the trained Random Forest model** â€” we only change feature engineering, producing numerically identical columns.

4. **Preserve the original numerical estimand** â€” the optimized code computes the same `max`, `min`, `mean` of neighbor values.

## Working R Code

```r
library(data.table)

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# STEP 1: Build the neighbor edge list (one-time, fast)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# rook_neighbors_unique is an nb object: a list of length = # cells,
# where each element is an integer vector of neighbor indices into id_order.

build_neighbor_edge_dt <- function(id_order, neighbors) {
  # Expand the nb list into a two-column edge list of cell IDs (not row indices)
  from_idx <- rep(seq_along(neighbors), lengths(neighbors))
  to_idx   <- unlist(neighbors, use.names = FALSE)
  
  data.table(
    focal_id    = id_order[from_idx],
    neighbor_id = id_order[to_idx]
  )
}

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# STEP 2: Build a row-index mapping and join to get neighbor row indices
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
build_neighbor_lookup_fast <- function(dt, id_order, neighbors) {
  # dt must be a data.table with columns 'id', 'year' and a row index
  # Add row positions
  dt[, .row_idx := .I]
  
  # Edge list at the cell level (~1.37M rows)
  edges <- build_neighbor_edge_dt(id_order, neighbors)
  
  # Create a keyed lookup: (id, year) -> row_idx
  row_map <- dt[, .(id, year, .row_idx)]
  setkey(row_map, id, year)
  
  # For every (focal_id, year) combination that exists in the data,
  # find the neighbor rows. Strategy:
  #   1. Get distinct (focal_id, year, focal_row_idx) by joining dt with edges.
  #   2. Then join neighbor_id + year against row_map to get neighbor_row_idx.
  
  # Join focal rows to their neighbor cell IDs
  focal <- dt[, .(focal_id = id, year, focal_row = .row_idx)]
  setkey(edges, focal_id)
  setkey(focal, focal_id)
  
  # This is the critical join: ~6.46M focal rows Ã— ~4 avg neighbors = ~26M rows
  # data.table does this in seconds, not hours.
  expanded <- edges[focal, on = .(focal_id), allow.cartesian = TRUE, nomatch = 0L]
  # expanded has columns: focal_id, neighbor_id, year, focal_row
  
  # Now attach the neighbor's row index by joining on (neighbor_id, year)
  setnames(row_map, c("id", "year", ".row_idx"), c("neighbor_id", "year", "neighbor_row"))
  setkey(row_map, neighbor_id, year)
  setkey(expanded, neighbor_id, year)
  
  expanded <- row_map[expanded, on = .(neighbor_id, year), nomatch = NA_integer_]
  # Keep only rows where the neighbor actually exists in that year
  expanded <- expanded[!is.na(neighbor_row)]
  
  # Return the edge table: focal_row <-> neighbor_row
  expanded[, .(focal_row, neighbor_row)]
}

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# STEP 3: Vectorized neighbor stats (replaces compute_neighbor_stats)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
compute_neighbor_stats_fast <- function(dt, edge_dt, var_name) {
  # edge_dt has columns: focal_row, neighbor_row
  # Pull neighbor values in one vectorized operation
  neighbor_vals <- dt[[var_name]][edge_dt$neighbor_row]
  
  work <- data.table(
    focal_row     = edge_dt$focal_row,
    neighbor_val  = neighbor_vals
  )
  
  # Remove NAs in neighbor values
  work <- work[!is.na(neighbor_val)]
  
  # Grouped aggregation â€” data.table does this with radix sort, very fast
  agg <- work[, .(
    nb_max  = max(neighbor_val),
    nb_min  = min(neighbor_val),
    nb_mean = mean(neighbor_val)
  ), by = focal_row]
  
  # Allocate output columns (NA for rows with no valid neighbors)
  n <- nrow(dt)
  out_max  <- rep(NA_real_, n)
  out_min  <- rep(NA_real_, n)
  out_mean <- rep(NA_real_, n)
  
  out_max[agg$focal_row]  <- agg$nb_max
  out_min[agg$focal_row]  <- agg$nb_min
  out_mean[agg$focal_row] <- agg$nb_mean
  
  # Name columns identically to the original pipeline
  suffix_max  <- paste0(var_name, "_neighbor_max")
  suffix_min  <- paste0(var_name, "_neighbor_min")
  suffix_mean <- paste0(var_name, "_neighbor_mean")
  
  dt[, (suffix_max)  := out_max]
  dt[, (suffix_min)  := out_min]
  dt[, (suffix_mean) := out_mean]
  
  invisible(dt)
}

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# STEP 4: Full optimized pipeline
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Convert to data.table if not already
if (!is.data.table(cell_data)) {
  cell_data <- as.data.table(cell_data)
}

# Build the neighbor edge table ONCE (replaces build_neighbor_lookup)
# This runs in seconds, not hours.
edge_dt <- build_neighbor_lookup_fast(cell_data, id_order, rook_neighbors_unique)

# Compute neighbor features for all 5 variables
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {
  compute_neighbor_stats_fast(cell_data, edge_dt, var_name)
  cat("Done:", var_name, "\n")
}

# Clean up helper column
cell_data[, .row_idx := NULL]

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# The trained Random Forest model is untouched.
# The output columns (e.g., ntl_neighbor_max, ntl_neighbor_min, 
# ntl_neighbor_mean, etc.) are numerically identical to the originals.
# Prediction proceeds as before:
# predictions <- predict(trained_rf_model, newdata = cell_data)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
```

## Expected Performance Improvement

| Component | Original | Optimized |
|---|---|---|
| `build_neighbor_lookup` (string hashing, 6.46M Ã— ~4 neighbors) | **~80+ hours** (dominant) | **~30â€“90 seconds** (data.table keyed join) |
| `compute_neighbor_stats` (5 variables Ã— lapply + do.call rbind) | ~1â€“3 hours | **~10â€“30 seconds** (vectorized grouped aggregation) |
| **Total** | **~86+ hours** | **~2â€“5 minutes** |

The ~1,700Ã— speedup comes from replacing **O(n Ã— k) string construction and hash-table probing on R named vectors** with **data.table's radix-sort-based keyed equi-joins**, which are optimized in C and operate on integer/numeric keys without string allocation.
END_RESPONSE>>>
