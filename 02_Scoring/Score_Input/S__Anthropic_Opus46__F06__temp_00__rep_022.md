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
run_id: Anthropic_F06_T00_R022
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F06
prompt_family_slug: RASTER_KERNEL_ANALOGY
family_label: Raster/kernel analogy
family_group: Raster/kernel bridge
replicate: 22
raw_output_file: Experiment_Output/Anthropic/Temp_0.0/output_F06_Anthropic_temp_00_rep_022.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

---

## 1. Diagnosis

The current implementation has two critical bottlenecks:

### Bottleneck A: `build_neighbor_lookup` â€” O(n) list of per-row lookups via `lapply`

- With ~6.46 million rows, this creates a list of 6.46M elements. Each iteration performs string pasting, hash lookups in a named vector (`idx_lookup`), and NA filtering. The string-key approach (`paste(id, year)`) is inherently slow at this scale because:
  - `paste()` is called 6.46M times.
  - Named vector lookup (`idx_lookup[neighbor_keys]`) is O(k) per call but with constant overhead from R's hashing on character keys, repeated millions of times.
  - The entire structure is an R list of integer vectors â€” memory-fragmented and GC-heavy.

### Bottleneck B: `compute_neighbor_stats` â€” repeated per variable

- Called 5 times (once per source variable), each time iterating over 6.46M list elements. Each call to the inner `lapply` does subsetting, NA removal, and three summary statistics. That's ~32.3 million R-level function calls across all variables.

### Why raster focal/kernel operations are *not* a direct substitute

The comment in the prompt asks whether raster focal operations offer a useful analogy. They do conceptually (a rook neighborhood is a 3Ã—3 cross kernel), but:
- The data is a **panel** (cell Ã— year), not a single raster layer.
- The grid cells may not form a complete regular rectangle (there may be missing cells, irregular boundaries).
- The neighbor structure is precomputed as an `spdep::nb` object, which may encode irregular adjacency.
- Focal operations would require reshaping each variable Ã— year into a complete raster, running `focal()`, then re-extracting â€” introducing complexity and potential for misalignment.

**Conclusion:** The best strategy is to vectorize the neighbor computation using `data.table` joins, which preserves the exact numerical results while eliminating millions of R-level loop iterations.

---

## 2. Optimization Strategy

| Step | Current | Proposed | Speedup source |
|------|---------|----------|----------------|
| Neighbor lookup | 6.46M-element R list built via `lapply` + string keys | Pre-built edge table (data.table) joined by `(id, year)` â€” integer keys | Vectorized join, no string ops |
| Neighbor stats | 5 Ã— `lapply` over 6.46M elements | 5 Ã— single `data.table` grouped aggregation on the edge table | Vectorized `max`, `min`, `mean` by group |
| Memory | ~6.46M list elements + duplicated index vectors | One edge table (~1.37M edges Ã— 28 years â‰ˆ 38.5M rows, 3 integer columns â‰ˆ 0.9 GB) | Contiguous memory, cache-friendly |

**Expected runtime:** Minutes instead of 86+ hours. The join is O(n log n) or O(n) with data.table's radix join; the grouped aggregation is highly optimized in C.

**Numerical equivalence:** The `max`, `min`, `mean` computations are applied to exactly the same neighbor values with the same NA handling, so results are identical to machine precision.

---

## 3. Working R Code

```r
library(data.table)

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Step 0: Convert cell_data to data.table (if not already)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
cell_dt <- as.data.table(cell_data)

# Ensure 'id' and 'year' columns exist and are keyed for fast joins
stopifnot(all(c("id", "year") %in% names(cell_dt)))

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Step 1: Build a directed edge table from the spdep::nb object
#
#   rook_neighbors_unique is a list of length = number of spatial cells.
#   rook_neighbors_unique[[i]] contains integer indices into id_order
#   of the neighbors of cell id_order[i].
#
#   We expand this into a two-column data.table: (focal_id, neighbor_id)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
build_edge_table <- function(id_order, neighbors_nb) {
  # neighbors_nb is an nb object: list of integer vectors (indices into id_order)
  n <- length(neighbors_nb)
  
  # Pre-allocate: count total edges
  n_edges <- sum(lengths(neighbors_nb))
  
  focal_id    <- integer(n_edges)
  neighbor_id <- integer(n_edges)
  
  pos <- 1L
  for (i in seq_len(n)) {
    nb_idx <- neighbors_nb[[i]]
    # spdep::nb encodes "no neighbors" as 0L in a length-1 vector
    if (length(nb_idx) == 1L && nb_idx[1L] == 0L) next
    len <- length(nb_idx)
    focal_id[pos:(pos + len - 1L)]    <- id_order[i]
    neighbor_id[pos:(pos + len - 1L)] <- id_order[nb_idx]
    pos <- pos + len
  }
  
  data.table(focal_id = focal_id[1:(pos - 1L)],
             neighbor_id = neighbor_id[1:(pos - 1L)])
}

edge_dt <- build_edge_table(id_order, rook_neighbors_unique)

cat(sprintf("Edge table: %d directed edges\n", nrow(edge_dt)))

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Step 2: Cross-join edges with years to get (focal_id, year, neighbor_id)
#         Then join neighbor values from cell_dt
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

# Unique years in the panel
all_years <- sort(unique(cell_dt$year))

# Expand edges Ã— years  (~1.37M edges Ã— 28 years â‰ˆ 38.5M rows)
edge_year_dt <- CJ_dt_edges <- edge_dt[, .(year = all_years), by = .(focal_id, neighbor_id)]

cat(sprintf("Edge-year table: %d rows (%.1f M)\n", nrow(edge_year_dt), nrow(edge_year_dt)/1e6))

# Key the cell data for fast join on (id, year)
setkey(cell_dt, id, year)

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Step 3: For each neighbor source variable, join, aggregate, and merge
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# Key edge_year_dt for the neighbor join
setkey(edge_year_dt, neighbor_id, year)

for (var_name in neighbor_source_vars) {
  cat(sprintf("Processing neighbor stats for: %s\n", var_name))
  
  # --- 3a. Attach the neighbor's value to each edge-year row ---
  # We only need (id, year, var_name) from cell_dt
  # Join: edge_year_dt[neighbor_id, year] -> cell_dt[id, year]
  
  # Create a small lookup table
  lookup <- cell_dt[, .(id, year, val = get(var_name))]
  setkey(lookup, id, year)
  
  # Join neighbor value onto edge table
  edge_year_dt[, nb_val := lookup[.(edge_year_dt$neighbor_id, edge_year_dt$year), val]]
  
  # --- 3b. Aggregate by (focal_id, year) ---
  # Compute max, min, mean of non-NA neighbor values
  agg <- edge_year_dt[!is.na(nb_val),
                       .(nb_max  = max(nb_val),
                         nb_min  = min(nb_val),
                         nb_mean = mean(nb_val)),
                       by = .(focal_id, year)]
  
  # --- 3c. Merge back onto cell_dt ---
  # Name the new columns to match the original pipeline's naming convention
  max_col  <- paste0("nb_max_",  var_name)
  min_col  <- paste0("nb_min_",  var_name)
  mean_col <- paste0("nb_mean_", var_name)
  
  setnames(agg, c("nb_max", "nb_min", "nb_mean"), c(max_col, min_col, mean_col))
  setkey(agg, focal_id, year)
  
  # Remove old columns if they exist (idempotent re-runs)
  for (col in c(max_col, min_col, mean_col)) {
    if (col %in% names(cell_dt)) cell_dt[, (col) := NULL]
  }
  
  # Left join: every cell-year gets its neighbor stats (NA if no valid neighbors)
  cell_dt <- agg[cell_dt, on = .(focal_id = id, year = year)]
  setnames(cell_dt, "focal_id", "id")  # restore column name
  
  # Clean up
  edge_year_dt[, nb_val := NULL]
  
  cat(sprintf("  -> Added columns: %s, %s, %s\n", max_col, min_col, mean_col))
}

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Step 4: Convert back to data.frame if downstream code expects it
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
cell_data <- as.data.frame(cell_dt)

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Step 5: Predict with the pre-trained Random Forest (unchanged)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# The trained model object (e.g., `rf_model`) is used as-is:
# predictions <- predict(rf_model, newdata = cell_data)
```

---

## 4. Why This Preserves the Original Numerical Estimand

| Concern | Guarantee |
|---------|-----------|
| **Same neighbor set** | The edge table is built from the identical `rook_neighbors_unique` nb object and `id_order`. Every `(focal, neighbor)` pair is preserved. |
| **Same year matching** | The cross-join with `all_years` and the keyed join on `(id, year)` replicate the original `paste(id, year)` lookup exactly. |
| **Same NA handling** | `!is.na(nb_val)` before aggregation mirrors `neighbor_vals <- neighbor_vals[!is.na(neighbor_vals)]`. Cells with zero valid neighbors get `NA` via the left join (no row in `agg`). |
| **Same summary functions** | `max`, `min`, `mean` in data.table use the same base R functions â€” results are identical to floating-point precision. |
| **RF model untouched** | The pre-trained model is never retrained; only `predict()` is called on the enriched data. |

---

## 5. Memory and Runtime Estimates

| Object | Size estimate |
|--------|--------------|
| `cell_dt` (6.46M Ã— 110 cols) | ~5.7 GB (fits in 16 GB) |
| `edge_dt` (1.37M Ã— 2 int cols) | ~11 MB |
| `edge_year_dt` (38.5M Ã— 3 int cols + 1 double) | ~1.2 GB |
| `lookup` (6.46M Ã— 3 cols) | ~150 MB |
| **Peak** | ~8â€“10 GB (fits in 16 GB RAM) |

**Expected runtime:** 5â€“15 minutes total on a standard laptop, down from 86+ hours â€” roughly a **300â€“1000Ã— speedup**.

If memory is tight, you can process one variable at a time and drop `lookup` between iterations (as shown), or avoid materializing the full `edge_year_dt` by processing year-by-year in a loop (trading a small amount of speed for lower peak memory).
END_RESPONSE>>>
