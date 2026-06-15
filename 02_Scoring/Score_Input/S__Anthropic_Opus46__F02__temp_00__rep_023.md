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
run_id: Anthropic_F02_T00_R023
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F02
prompt_family_slug: CODE_ONLY_NO_HINT
family_label: Code only, no hint
family_group: Code-visible neutral
replicate: 23
raw_output_file: Experiment_Output/Anthropic/Temp_0.0/output_F02_Anthropic_temp_00_rep_023.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The core bottleneck is the **`build_neighbor_lookup` function**, which uses `lapply` over ~6.46 million rows, performing per-row string pasting, hash-table lookups, and NA filtering in pure R. This creates millions of small character vectors and triggers repeated hash lookups on a named vector of 6.46 million entries â€” an O(nÂ²)-like workload in practice. The `compute_neighbor_stats` function then iterates over the same 6.46 million entries again per variable, subsetting numeric vectors by index lists. With 5 variables, that's ~32.3 million list iterations total.

**Specific problems:**

1. **`build_neighbor_lookup`:** `paste()` and named-vector lookup inside a per-row `lapply` over 6.46M rows is extremely slow. The named character vector `idx_lookup` of length 6.46M makes each `[` lookup O(n) in the worst case (R's hashed environments are faster, but `setNames` on a plain vector uses linear search or a hash that still has overhead at this scale). This single function likely accounts for >90% of the 86-hour estimate.

2. **`compute_neighbor_stats`:** `lapply` over 6.46M list elements with per-element `max`/`min`/`mean` is slow but less catastrophic. The `do.call(rbind, ...)` on a 6.46M-element list of 3-vectors is also memory-intensive (creates a temporary list of millions of small vectors before binding).

3. **Memory:** 6.46M rows Ã— 110 columns â‰ˆ 5.7 GB as double. The neighbor lookup list (6.46M elements, each a small integer vector) adds ~1â€“2 GB. Intermediate copies from `do.call(rbind, ...)` and repeated `data[[var_name]]` extraction add pressure. 16 GB is tight.

---

## Optimization Strategy

### Principle: Replace per-row R loops with vectorized joins and matrix operations using `data.table`.

**Step 1 â€” Vectorized neighbor lookup via `data.table` equi-join:**
Instead of building a 6.46M-element list, build a **long-format edge table** (`cell_row`, `neighbor_row`) using vectorized operations. Expand the `nb` object into a two-column edge list of (`id`, `neighbor_id`), join with year to get (`id`, `year`, `neighbor_id`), then map to row indices. This replaces the per-row `lapply` with a single merge.

**Step 2 â€” Vectorized neighbor stats via grouped aggregation:**
Instead of iterating over a list, use `data.table` grouped `max`/`min`/`mean` on the long edge table joined with the variable values. One grouped aggregation replaces 6.46M list iterations per variable.

**Step 3 â€” Memory management:**
Process one variable at a time, attach results immediately, and `rm()` intermediates.

**Expected speedup:** From ~86 hours to **~5â€“15 minutes** on the same laptop.

---

## Working R Code

```r
library(data.table)

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# STEP 0: Convert cell_data to data.table (in-place, no copy)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
setDT(cell_data)

# Ensure there is a row-index column for later re-attachment
cell_data[, .row_id := .I]

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# STEP 1: Build a vectorized edge table from the nb object
#
# rook_neighbors_unique is a list of length N_cells (344,208).
# id_order is the vector mapping list position -> cell id.
# We expand this into a long data.table: (id, neighbor_id)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

build_edge_table <- function(id_order, neighbors) {
  # neighbors is an nb object: list of integer vectors (indices into id_order)
  # Remove 0-length entries gracefully
  n_neighbors <- lengths(neighbors)
  
  from_idx <- rep(seq_along(neighbors), times = n_neighbors)
  to_idx   <- unlist(neighbors, use.names = FALSE)
  
  data.table(
    id          = id_order[from_idx],
    neighbor_id = id_order[to_idx]
  )
}

cat("Building edge table...\n")
edge_dt <- build_edge_table(id_order, rook_neighbors_unique)
cat(sprintf("  Edge table: %s rows\n", format(nrow(edge_dt), big.mark = ",")))

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# STEP 2: Cross-join edges with years to get (id, year, neighbor_id),
#          then map each (neighbor_id, year) to its row in cell_data.
#
# Key insight: every edge (A -> B) exists for ALL 28 years.
# So we can do a single cross-join with the year vector, then
# join to cell_data to pick up the neighbor's variable value.
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

# Unique years
years_vec <- sort(unique(cell_data$year))

# Cross-join edges Ã— years  (~1.37M edges Ã— 28 years â‰ˆ 38.5M rows)
# This is the largest object; ~38.5M Ã— 3 cols Ã— 8 bytes â‰ˆ 0.9 GB
cat("Expanding edges Ã— years...\n")
edge_year_dt <- edge_dt[, .(year = years_vec), by = .(id, neighbor_id)]

# Free the compact edge table
rm(edge_dt)
gc()

# Set keys for fast joins
setkey(edge_year_dt, neighbor_id, year)

# We also need a keyed version of cell_data for the neighbor value lookup
# Create a minimal keyed reference (id, year, .row_id) for the focal cell
setkey(cell_data, id, year)

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# STEP 3: For each variable, join, aggregate, and attach
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# Build a small lookup: (id, year) -> variable value, reused per variable
# We key cell_data on (id, year) once.

cat("Computing neighbor features...\n")

for (var_name in neighbor_source_vars) {
  cat(sprintf("  Processing: %s\n", var_name))
  
  # --- 3a. Build a lookup of neighbor values ---
  # Extract only the columns we need from cell_data for the neighbor side
  val_dt <- cell_data[, .(id, year, val = get(var_name))]
  setkey(val_dt, id, year)
  
  # --- 3b. Join neighbor values onto the edge table ---
  # edge_year_dt is keyed on (neighbor_id, year)
  # val_dt is keyed on (id, year)
  # We join: edge_year_dt[val_dt] matching neighbor_id==id, year==year
  
  # Rename for clarity before join
  setnames(val_dt, "id", "neighbor_id")
  # Now val_dt is keyed on (neighbor_id, year)
  setkey(val_dt, neighbor_id, year)
  
  # Join: attach neighbor's value to each edge-year row
  edge_year_dt[val_dt, neighbor_val := i.val, on = .(neighbor_id, year)]
  
  # --- 3c. Aggregate: group by (id, year) to get max, min, mean ---
  agg <- edge_year_dt[
    !is.na(neighbor_val),
    .(
      nb_max  = max(neighbor_val),
      nb_min  = min(neighbor_val),
      nb_mean = mean(neighbor_val)
    ),
    by = .(id, year)
  ]
  
  # Name the output columns to match the original pipeline's naming convention
  # (adjust these names if your compute_and_add_neighbor_features used different names)
  new_names <- paste0(var_name, c("_nb_max", "_nb_min", "_nb_mean"))
  setnames(agg, c("nb_max", "nb_min", "nb_mean"), new_names)
  
  # --- 3d. Merge back into cell_data ---
  # Left join: cell_data keeps all rows; cells with no valid neighbors get NA
  setkey(agg, id, year)
  cell_data <- agg[cell_data, on = .(id, year)]
  setkey(cell_data, id, year)
  
  # --- 3e. Clean up the temporary column on edge_year_dt ---
  edge_year_dt[, neighbor_val := NULL]
  
  rm(val_dt, agg)
  gc()
  
  cat(sprintf("    -> Added: %s\n", paste(new_names, collapse = ", ")))
}

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# STEP 4: Clean up
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
rm(edge_year_dt)
gc()

# Restore original row order if needed
setorder(cell_data, .row_id)
cell_data[, .row_id := NULL]

cat("Done. Neighbor features attached.\n")
```

---

## Why This Preserves Correctness

| Concern | Guarantee |
|---|---|
| **Trained Random Forest model** | No model code is touched. We only reproduce the same feature columns (`*_nb_max`, `*_nb_min`, `*_nb_mean`) that the original code produced. The RF object remains unchanged on disk/memory. |
| **Numerical estimand** | `max`, `min`, `mean` are computed on exactly the same neighbor sets (same `nb` object, same year matching, same NA exclusion logic). The `data.table` grouped aggregation is numerically identical to the original R `max`/`min`/`mean` calls. |
| **Column naming** | Adjust the `new_names` pattern in Step 3c if your original `compute_and_add_neighbor_features` used a different naming convention (e.g., `"ntl_max_neighbor"`). The values will be identical. |

---

## Memory Budget Estimate

| Object | Approximate Size |
|---|---|
| `cell_data` (6.46M Ã— 110 cols) | ~5.7 GB |
| `edge_year_dt` (38.5M Ã— 3 cols + 1 temp) | ~1.2 GB |
| Temporary `val_dt` + `agg` | ~0.5 GB |
| Overhead / R session | ~1.5 GB |
| **Total peak** | **~9 GB** (fits in 16 GB) |

If memory is still tight, you can **chunk by year** (process 4â€“7 years at a time) or by variable, but the above should fit comfortably on a 16 GB machine.

---

## Expected Runtime

- **Edge table construction:** <1 second (vectorized `rep`/`unlist`).
- **Cross-join expansion:** ~10â€“30 seconds.
- **Per-variable join + aggregate:** ~30â€“60 seconds each.
- **Total for 5 variables:** **~3â€“6 minutes** (vs. 86+ hours originally).
END_RESPONSE>>>
