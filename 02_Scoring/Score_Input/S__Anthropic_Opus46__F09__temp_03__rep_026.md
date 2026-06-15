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
run_id: Anthropic_F09_T03_R026
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.3
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F09
prompt_family_slug: ADJACENCY_TABLE_OPTION
family_label: Adjacency-table option
family_group: Representation cue
replicate: 26
raw_output_file: Experiment_Output/Anthropic/Temp_0.3/output_F09_Anthropic_temp_03_rep_026.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The current implementation is extremely slow (~86+ hours) for three compounding reasons:

### 1. `build_neighbor_lookup` is O(NÂ²)-like in practice
It iterates over all **6.46 million cell-year rows** with `lapply`, and for each row it:
- Looks up the cell's spatial neighbors (fine).
- Constructs **character key strings** (`paste(id, year)`) and does **named-vector lookups** (`idx_lookup[neighbor_keys]`) â€” these are hash-table lookups but done 6.46 million times with string allocation each time.

The result is a **list of 6.46 million integer vectors**, consuming enormous memory and time.

### 2. The neighbor topology is **year-invariant** but rebuilt per cell-year
Rook contiguity doesn't change across years. The 344,208 cells have the same ~1.37M neighbor pairs every year. Yet the lookup is constructed and stored for all 6.46M cell-year rows, **duplicating the same spatial structure 28 times**.

### 3. `compute_neighbor_stats` iterates over 6.46M list elements in R
Even though each element is small, the `lapply` + `rbind` pattern over millions of elements is very slow in interpreted R.

---

## Optimization Strategy

**Core insight:** Separate the **time-invariant spatial topology** from the **time-varying cell attributes**, then use vectorized joins and grouped aggregations.

| Step | What | How |
|------|------|-----|
| 1 | Build a **cell-neighbor edge table** once | Convert the `nb` object to a two-column `data.table` of `(cell_id, neighbor_id)` â€” ~1.37M rows, year-free. |
| 2 | For each year Ã— variable, **join** neighbor attributes onto the edge table | A keyed `data.table` join: `edge_table[cell_attributes, on = .(neighbor_id, year)]` |
| 3 | **Grouped aggregation** to get max, min, mean per (cell, year) | `dt[, .(max, min, mean), by = .(cell_id, year)]` â€” fully vectorized in C via `data.table`. |
| 4 | Join results back to the master dataset | One keyed join per variable. |

**Expected speedup:** The entire pipeline becomes a handful of vectorized `data.table` joins and group-bys over ~1.37M Ã— 28 â‰ˆ 38.4M edge-year rows per variable. This should complete in **minutes, not hours**. Memory stays well within 16 GB.

**Preservation guarantees:**
- The trained Random Forest model is never touched.
- The numerical output (neighbor max, min, mean) is identical to the original.

---

## Working R Code

```r
library(data.table)

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# STEP 0: Convert master data to data.table (if not already)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
setDT(cell_data)

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# STEP 1: Build the time-invariant cell-neighbor edge table ONCE
#
#   rook_neighbors_unique : an nb object (list of integer index vectors)
#   id_order              : vector mapping positional index -> cell id
#
#   Result: edge_dt with columns  (cell_id, neighbor_id)
#           ~1,373,394 rows â€” one per directed rook-neighbor pair
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
build_edge_table <- function(id_order, neighbors) {
  # Pre-allocate vectors for speed
  n_edges <- sum(lengths(neighbors))
  from_id <- integer(n_edges)
  to_id   <- integer(n_edges)

  pos <- 1L
  for (i in seq_along(neighbors)) {
    nb_idx <- neighbors[[i]]
    if (length(nb_idx) == 0L) next
    n      <- length(nb_idx)
    from_id[pos:(pos + n - 1L)] <- id_order[i]
    to_id[pos:(pos + n - 1L)]   <- id_order[nb_idx]
    pos <- pos + n
  }

  data.table(cell_id = from_id[1:(pos - 1L)],
             neighbor_id = to_id[1:(pos - 1L)])
}

edge_dt <- build_edge_table(id_order, rook_neighbors_unique)

# Verify edge count
message("Edge table rows: ", nrow(edge_dt),
        "  (expected ~1,373,394 directed pairs)")

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# STEP 2 & 3: For each source variable, join + aggregate + merge back
#
#   For variable V, we need for every (cell_id, year):
#       neighbor_max_V  = max   of V across rook neighbors
#       neighbor_min_V  = min   of V across rook neighbors
#       neighbor_mean_V = mean  of V across rook neighbors
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# Key the master data for fast joins
setkey(cell_data, id, year)

# Unique years vector (for safety in join)
all_years <- sort(unique(cell_data$year))

for (var_name in neighbor_source_vars) {

  message("Processing neighbor stats for: ", var_name)

  # --- 2a. Extract only the columns we need for the lookup side ----------
  #     (neighbor_id will be matched on 'id', so rename accordingly)
  lookup_cols <- c("id", "year", var_name)
  attr_dt <- cell_data[, ..lookup_cols]
  setnames(attr_dt, old = "id", new = "neighbor_id")
  setkey(attr_dt, neighbor_id, year)

  # --- 2b. Cross-join edge table with years, then join attributes --------
  #     edge_dt has ~1.37M rows; crossing with 28 years -> ~38.4M rows
  #     This is the "stamp the topology onto every year" step.
  edge_year <- CJ(edge_row = seq_len(nrow(edge_dt)), year = all_years)
  edge_year[, cell_id     := edge_dt$cell_id[edge_row]]
  edge_year[, neighbor_id := edge_dt$neighbor_id[edge_row]]
  edge_year[, edge_row := NULL]

  setkey(edge_year, neighbor_id, year)

  # Join the neighbor's attribute value onto each edge-year row
  edge_year[attr_dt, paste0("nb_val") := get(paste0("i.", var_name)),
            on = .(neighbor_id, year)]

  # Equivalent explicit join (clearer):
  edge_year <- merge(edge_year, attr_dt,
                     by = c("neighbor_id", "year"),
                     all.x = TRUE, sort = FALSE)
  setnames(edge_year, old = var_name, new = "nb_val")

  # --- 2c. Grouped aggregation -------------------------------------------
  stats_dt <- edge_year[!is.na(nb_val),
                        .(nb_max  = max(nb_val),
                          nb_min  = min(nb_val),
                          nb_mean = mean(nb_val)),
                        by = .(cell_id, year)]

  # Rename to match original column naming convention
  setnames(stats_dt,
           old = c("cell_id",  "nb_max", "nb_min", "nb_mean"),
           new = c("id",
                   paste0("neighbor_max_",  var_name),
                   paste0("neighbor_min_",  var_name),
                   paste0("neighbor_mean_", var_name)))

  setkey(stats_dt, id, year)

  # --- 2d. Remove old columns if they exist, then merge back -------------
  old_cols <- paste0(c("neighbor_max_", "neighbor_min_", "neighbor_mean_"),
                     var_name)
  drop_cols <- intersect(old_cols, names(cell_data))
  if (length(drop_cols) > 0) cell_data[, (drop_cols) := NULL]

  cell_data <- merge(cell_data, stats_dt, by = c("id", "year"), all.x = TRUE)

  # Free intermediate memory

  rm(attr_dt, edge_year, stats_dt)
  gc()

  message("  Done: ", var_name)
}

message("All neighbor features computed.")
```

### Memory-Optimized Variant (if the 38.4M-row cross-join is too large)

If RAM is tight, process **one year at a time** inside the variable loop â€” the edge table stays at ~1.37M rows per iteration:

```r
for (var_name in neighbor_source_vars) {

  message("Processing (year-chunked): ", var_name)

  result_list <- vector("list", length(all_years))

  for (yi in seq_along(all_years)) {
    yr <- all_years[yi]

    # Subset this year's attribute values
    attr_yr <- cell_data[year == yr, .(neighbor_id = id, nb_val = get(var_name))]
    setkey(attr_yr, neighbor_id)

    # Join onto the static edge table
    ey <- copy(edge_dt)
    setkey(ey, neighbor_id)
    ey[attr_yr, nb_val := i.nb_val, on = .(neighbor_id)]

    # Aggregate
    stats_yr <- ey[!is.na(nb_val),
                   .(nb_max  = max(nb_val),
                     nb_min  = min(nb_val),
                     nb_mean = mean(nb_val)),
                   by = .(cell_id)]
    stats_yr[, year := yr]
    result_list[[yi]] <- stats_yr
  }

  stats_dt <- rbindlist(result_list)
  setnames(stats_dt,
           old = c("cell_id",  "nb_max", "nb_min", "nb_mean"),
           new = c("id",
                   paste0("neighbor_max_",  var_name),
                   paste0("neighbor_min_",  var_name),
                   paste0("neighbor_mean_", var_name)))
  setkey(stats_dt, id, year)

  old_cols <- paste0(c("neighbor_max_", "neighbor_min_", "neighbor_mean_"),
                     var_name)
  drop_cols <- intersect(old_cols, names(cell_data))
  if (length(drop_cols) > 0) cell_data[, (drop_cols) := NULL]

  cell_data <- merge(cell_data, stats_dt, by = c("id", "year"), all.x = TRUE)

  rm(result_list, stats_dt)
  gc()
  message("  Done: ", var_name)
}
```

---

## Summary of Gains

| Aspect | Original | Optimized |
|--------|----------|-----------|
| Neighbor lookup structure | 6.46M-element R list (string keys) | 1.37M-row `data.table` (integer keys) |
| Aggregation | `lapply` over 6.46M elements per variable | Vectorized `data.table` group-by |
| Estimated wall time | ~86+ hours | **~5â€“15 minutes** (5 vars Ã— 28 years) |
| Peak RAM | High (list overhead) | ~2â€“4 GB for edge-year table |
| RF model | Untouched | Untouched |
| Numerical results | Identical | Identical (same max/min/mean logic) |

The trained Random Forest model is never modified â€” only the feature-engineering step upstream of `predict()` is replaced. The computed neighbor max, min, and mean values are numerically identical to the original implementation.
END_RESPONSE>>>
