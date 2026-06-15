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
run_id: Anthropic_F09_T03_R020
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.3
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F09
prompt_family_slug: ADJACENCY_TABLE_OPTION
family_label: Adjacency-table option
family_group: Representation cue
replicate: 20
raw_output_file: Experiment_Output/Anthropic/Temp_0.3/output_F09_Anthropic_temp_03_rep_020.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The current implementation is extremely slow (~86+ hours) because of **two fundamental bottlenecks**:

### 1. `build_neighbor_lookup` is O(nÂ²)-like in practice
This function iterates over all **~6.46 million cell-year rows** and, for each row, performs:
- A named-vector lookup (`id_to_ref`) â€” O(1) amortized but with high constant factor for character hashing at scale.
- Construction of `neighbor_keys` by pasting cell IDs with years â€” millions of `paste()` calls.
- A named-vector lookup into `idx_lookup` (a named vector of length 6.46M) â€” extremely expensive repeated hash lookups.

The result is a **list of 6.46 million integer vectors**. This is the single largest bottleneck. The key insight is: **the neighbor topology is static across years**. Cell 42's rook neighbors are the same in 1992 as in 2019. Yet this function redundantly recomputes neighbor row indices for every cell-year combination, doing 28Ã— the necessary topological work and creating a 6.46M-element list when a 344,208-element list would suffice.

### 2. `compute_neighbor_stats` uses an R-level `lapply` over 6.46M elements
Even after the lookup is built, computing max/min/mean via `lapply` over 6.46 million list elements is slow due to R's interpreter overhead per iteration.

---

## Optimization Strategy

**Core idea: Separate topology from attributes. Build the adjacency table once (344K cells), then join yearly attributes and compute stats vectorized using `data.table`.**

### Step-by-step:

1. **Build a static edge table** from `rook_neighbors_unique` (the `nb` object): a two-column `data.table` with columns `(id, neighbor_id)`. This has ~1.37M rows and is built **once**.

2. **For each variable**, join the cell-year attribute values onto the edge table by `(neighbor_id, year)`, then group by `(id, year)` and compute `max`, `min`, `mean` â€” all vectorized in `data.table`.

3. **Join the resulting neighbor stats back** onto the main `cell_data` table.

This eliminates:
- The 6.46M-element list construction entirely.
- All `paste()`-based key construction.
- All R-level `lapply` loops over millions of elements.

**Expected speedup**: From ~86 hours to **minutes** (roughly 2â€“10 minutes depending on disk I/O and RAM pressure).

**Preservation guarantees**:
- The trained Random Forest model is untouched â€” we only modify feature engineering.
- The numerical output is identical: same max, min, mean of the same neighbor values.

---

## Working R Code

```r
library(data.table)

# ==============================================================
# STEP 1: Build the static edge table ONCE from the nb object
# ==============================================================
# rook_neighbors_unique is a spdep::nb object (list of length 344,208).
# id_order is the vector mapping list index -> cell id.

build_edge_table <- function(id_order, neighbors_nb) {
  # Pre-allocate: count total edges
  n_edges <- sum(vapply(neighbors_nb, length, integer(1)))
  
  from_id <- integer(n_edges)
  to_id   <- integer(n_edges)
  
  pos <- 1L
  for (i in seq_along(neighbors_nb)) {
    nb_idx <- neighbors_nb[[i]]
    # spdep::nb encodes "no neighbors" as a single 0L; skip those
    if (length(nb_idx) == 1L && nb_idx == 0L) next
    n_nb <- length(nb_idx)
    from_id[pos:(pos + n_nb - 1L)] <- id_order[i]
    to_id[pos:(pos + n_nb - 1L)]   <- id_order[nb_idx]
    pos <- pos + n_nb
  }
  
  # Trim if any 0-neighbor cells caused over-allocation
  if (pos <= n_edges) {
    from_id <- from_id[1:(pos - 1L)]
    to_id   <- to_id[1:(pos - 1L)]
  }
  
  data.table(id = from_id, neighbor_id = to_id)
}

edge_dt <- build_edge_table(id_order, rook_neighbors_unique)

cat(sprintf("Edge table: %d directed edges\n", nrow(edge_dt)))
# Expected: ~1,373,394 rows

# ==============================================================
# STEP 2: Convert cell_data to data.table (if not already)
# ==============================================================
if (!is.data.table(cell_data)) {
  cell_data <- as.data.table(cell_data)
}

# Set key for fast joins
setkey(cell_data, id, year)

# ==============================================================
# STEP 3: For each variable, compute neighbor stats via join
# ==============================================================
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

compute_neighbor_features_dt <- function(cell_data, edge_dt, var_name) {
  # Columns we need from cell_data for the join: id, year, and the variable
  # We join edge_dt with cell_data on (neighbor_id = id, year) to get
  # each neighbor's value, then aggregate by (id, year).
  
  # Subset to only needed columns for memory efficiency
  val_dt <- cell_data[, .(id, year, val = get(var_name))]
  setkey(val_dt, id, year)
  
  # Add year to edge table by cross-joining with years
  # More efficient: join edge_dt onto cell_data to get (id, year, neighbor_id),
  # then join again to get neighbor values.
  
  # Step A: Get all (id, year) combinations that exist, paired with neighbor_id
  # This is: for each row in cell_data, look up its neighbors from edge_dt
  
  # Create (id, year) from cell_data, join with edge_dt on id
  id_year_dt <- cell_data[, .(id, year)]
  setkey(id_year_dt, id)
  setkey(edge_dt, id)
  
  # Join: for each (id, year), get all neighbor_ids
  # This produces ~1.37M * 28 â‰ˆ 38.5M rows (but many cells don't have all 28 years)
  # Actually: each cell-year row gets its neighbor list, so total rows =
  # sum over all cell-years of (number of neighbors of that cell)
  # â‰ˆ 6.46M rows * avg ~4 neighbors = ~25.8M rows. Fits in 16GB RAM.
  
  expanded <- edge_dt[id_year_dt, .(id, year, neighbor_id), 
                      on = "id", allow.cartesian = TRUE, nomatch = NULL]
  
  # Step B: Join neighbor values: look up val for (neighbor_id, year)
  setkey(expanded, neighbor_id, year)
  expanded[val_dt, neighbor_val := i.val, on = .(neighbor_id = id, year)]
  
  # Step C: Aggregate by (id, year), dropping NAs in neighbor_val
  stats <- expanded[!is.na(neighbor_val), 
                    .(nb_max  = max(neighbor_val),
                      nb_min  = min(neighbor_val),
                      nb_mean = mean(neighbor_val)),
                    by = .(id, year)]
  
  # Rename columns to match original naming convention
  max_col  <- paste0("neighbor_max_", var_name)
  min_col  <- paste0("neighbor_min_", var_name)
  mean_col <- paste0("neighbor_mean_", var_name)
  setnames(stats, c("nb_max", "nb_min", "nb_mean"), c(max_col, min_col, mean_col))
  
  # Step D: Join stats back onto cell_data
  # First, remove old columns if they exist (idempotency)
  for (col in c(max_col, min_col, mean_col)) {
    if (col %in% names(cell_data)) cell_data[, (col) := NULL]
  }
  
  setkey(stats, id, year)
  setkey(cell_data, id, year)
  cell_data[stats, (c(max_col, min_col, mean_col)) := 
              mget(paste0("i.", c(max_col, min_col, mean_col))),
            on = .(id, year)]
  
  # Rows with no valid neighbors remain NA (correct behavior, matches original)
  invisible(cell_data)
}

# ==============================================================
# STEP 4: Run for all neighbor source variables
# ==============================================================
for (var_name in neighbor_source_vars) {
  cat(sprintf("Computing neighbor features for: %s ...\n", var_name))
  t0 <- proc.time()
  cell_data <- compute_neighbor_features_dt(cell_data, edge_dt, var_name)
  elapsed <- (proc.time() - t0)["elapsed"]
  cat(sprintf("  Done in %.1f seconds.\n", elapsed))
}

# ==============================================================
# STEP 5: Predict with the existing trained Random Forest
# ==============================================================
# The trained model object (e.g., `rf_model`) is unchanged.
# Convert back to data.frame if the model's predict method requires it.

# cell_data_df <- as.data.frame(cell_data)  # only if needed
# predictions <- predict(rf_model, newdata = cell_data_df)

cat("Neighbor feature computation complete. Ready for RF prediction.\n")
```

---

## Summary of Improvements

| Aspect | Original | Optimized |
|---|---|---|
| **Topology computation** | Rebuilt for every cell-year (6.46M iterations) | Built once as edge table (344K cells, ~1.37M edges) |
| **Key lookup mechanism** | Named character vector hashing (`paste` + `[`) | `data.table` binary-search joins (keyed integer joins) |
| **Stats computation** | R-level `lapply` over 6.46M elements | Vectorized `data.table` grouped aggregation |
| **Memory pattern** | 6.46M-element list of integer vectors (fragmented) | Contiguous columnar `data.table` (cache-friendly) |
| **Estimated runtime** | ~86+ hours | **~2â€“10 minutes** |
| **Trained RF model** | âœ… Preserved (untouched) | âœ… Preserved (untouched) |
| **Numerical results** | Baseline | **Identical** (same max/min/mean over same neighbor sets) |

The key conceptual fix is **separating the static spatial topology from the time-varying attributes**, building the adjacency structure once, and leveraging `data.table`'s optimized join and group-by machinery instead of R-level loops over millions of list elements.
END_RESPONSE>>>
