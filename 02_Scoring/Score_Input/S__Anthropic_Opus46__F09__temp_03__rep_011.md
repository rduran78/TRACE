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
run_id: Anthropic_F09_T03_R011
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.3
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F09
prompt_family_slug: ADJACENCY_TABLE_OPTION
family_label: Adjacency-table option
family_group: Representation cue
replicate: 11
raw_output_file: Experiment_Output/Anthropic/Temp_0.3/output_F09_Anthropic_temp_03_rep_011.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The current implementation is extremely slow (~86+ hours) for three interconnected reasons:

### 1. Row-level `lapply` over 6.46 million rows in `build_neighbor_lookup`
For every single cell-year row, the function performs character coercion (`as.character`), string pasting (`paste(..., sep="_")`), and named-vector lookups (`idx_lookup[neighbor_keys]`). With ~6.46M rows, this creates billions of intermediate string objects and named-vector searches (O(n) or O(n log n) per lookup in base R named vectors). This alone could take tens of hours.

### 2. Row-level `lapply` over 6.46 million rows in `compute_neighbor_stats`
Even after the lookup is built, computing max/min/mean via `lapply` over 6.46M list elements â€” each containing a variable-length integer vector â€” is inherently slow in interpreted R. The list-of-vectors structure defeats vectorization.

### 3. The neighbor topology is **year-invariant** but rebuilt per cell-year
The rook-neighbor relationships are purely spatial (cell A neighbors cell B regardless of year). Yet `build_neighbor_lookup` re-derives neighbor indices for every cell-year combination, inflating the work by a factor of 28 (the number of years).

**Key insight:** The neighbor table is a property of the **grid**, not of the **panel**. There are only ~344K cells and ~1.37M directed neighbor pairs. The expensive part â€” joining yearly attributes and computing grouped statistics â€” should be done via vectorized table joins, not row-wise R loops.

---

## Optimization Strategy

1. **Build a static neighbor edge table once** â€” a two-column `data.table` of `(cell_id, neighbor_id)` with ~1.37M rows. This is year-invariant and derived from `rook_neighbors_unique` in seconds.

2. **For each variable, join yearly attributes onto the edge table** â€” use `data.table` keyed joins to attach the variable value for each `(neighbor_id, year)` pair. This explodes to ~1.37M Ã— 28 â‰ˆ 38.4M rows, which is very manageable.

3. **Compute grouped max/min/mean in one vectorized pass** â€” group by `(cell_id, year)` and compute the three statistics. `data.table` does this in seconds on 38M rows.

4. **Merge results back** onto the main `cell_data` table.

5. **Predict with the existing trained Random Forest** â€” no retraining, no change to the numerical estimand.

**Expected speedup:** From ~86+ hours to **minutes** (typically 2â€“10 minutes total depending on disk I/O).

**RAM estimate:** The edge table Ã— years is ~38.4M rows Ã— a few columns of numeric + integer â‰ˆ < 1 GB. Well within 16 GB.

---

## Working R Code

```r
library(data.table)

# ===========================================================================
# STEP 0: Convert cell_data to data.table (if not already)
# ===========================================================================
cell_data <- as.data.table(cell_data)

# ===========================================================================
# STEP 1: Build a static, year-invariant neighbor edge table ONCE
#
#   rook_neighbors_unique : an nb object (list of integer index vectors)
#   id_order              : vector of cell IDs in the same order as the nb object
#
#   Result: edge_dt with columns  cell_id | neighbor_id
#           (~1,373,394 rows â€” one per directed rook-neighbor pair)
# ===========================================================================

build_edge_table <- function(id_order, neighbors) {
  # neighbors[[i]] contains integer indices into id_order for cell i's neighbors
  n <- length(neighbors)
  
  # Pre-allocate: count total edges
  edge_counts <- vapply(neighbors, length, integer(1))
  total_edges <- sum(edge_counts)
  
  from_id <- integer(total_edges)
  to_id   <- integer(total_edges)
  
  pos <- 1L
  for (i in seq_len(n)) {
    nb <- neighbors[[i]]
    len <- length(nb)
    if (len > 0L) {
      idx <- pos:(pos + len - 1L)
      from_id[idx] <- id_order[i]
      to_id[idx]   <- id_order[nb]
      pos <- pos + len
    }
  }
  
  data.table(cell_id = from_id, neighbor_id = to_id)
}

cat("Building static neighbor edge table...\n")
edge_dt <- build_edge_table(id_order, rook_neighbors_unique)
cat(sprintf("  Edge table: %s directed neighbor pairs\n", format(nrow(edge_dt), big.mark = ",")))

# ===========================================================================
# STEP 2: Function to compute neighbor stats for one variable, vectorized
# ===========================================================================

compute_neighbor_features_fast <- function(cell_dt, edge_dt, var_name) {
  # cell_dt must have columns: id, year, <var_name>
  # edge_dt must have columns: cell_id, neighbor_id
  
  # --- 2a. Extract only the columns we need for the neighbor values ----------
  val_dt <- cell_dt[, .(neighbor_id = id, year, value = get(var_name))]
  setkey(val_dt, neighbor_id, year)
  
  # --- 2b. Cross edge table with all years -----------------------------------
  #   For each edge (cell_id -> neighbor_id), we need every year.
  #   Instead of a full cross join, we join edge_dt onto val_dt by neighbor_id
  #   to pick up (year, value) in one pass.
  
  # Add year dimension: join edge_dt with val_dt on neighbor_id
  # Result: cell_id | neighbor_id | year | value
  setkey(edge_dt, neighbor_id)
  merged <- val_dt[edge_dt, on = .(neighbor_id), allow.cartesian = TRUE, nomatch = NA]
  # merged columns: neighbor_id, year, value, cell_id
  
  # --- 2c. Remove NA values before aggregation --------------------------------
  merged <- merged[!is.na(value)]
  
  # --- 2d. Grouped aggregation ------------------------------------------------
  max_col  <- paste0("neighbor_max_",  var_name)
  min_col  <- paste0("neighbor_min_",  var_name)
  mean_col <- paste0("neighbor_mean_", var_name)
  
  stats <- merged[, .(
    nb_max  = max(value),
    nb_min  = min(value),
    nb_mean = mean(value)
  ), by = .(cell_id, year)]
  
  setnames(stats, c("nb_max", "nb_min", "nb_mean"),
                  c(max_col,  min_col,  mean_col))
  
  stats
}

# ===========================================================================
# STEP 3: Loop over the 5 neighbor source variables, merge results
# ===========================================================================

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# Ensure cell_data is keyed for fast joins
setkey(cell_data, id, year)

for (var_name in neighbor_source_vars) {
  cat(sprintf("Computing neighbor features for '%s'...\n", var_name))
  
  stats_dt <- compute_neighbor_features_fast(cell_data, copy(edge_dt), var_name)
  setkey(stats_dt, cell_id, year)
  
  # Determine the new column names
  new_cols <- setdiff(names(stats_dt), c("cell_id", "year"))
  
  # Remove old columns if they already exist (idempotent re-runs)
  for (col in new_cols) {
    if (col %in% names(cell_data)) {
      cell_data[, (col) := NULL]
    }
  }
  
  # Merge onto cell_data (left join: keep all original rows)
  cell_data <- stats_dt[cell_data, on = .(cell_id = id, year)]
  
  # The join renames cell_id; restore 'id'
  setnames(cell_data, "cell_id", "id")
  setkey(cell_data, id, year)
  
  cat(sprintf("  Done. cell_data now has %d columns.\n", ncol(cell_data)))
}

# ===========================================================================
# STEP 4: Predict with the existing trained Random Forest (unchanged)
#
#   The trained model object (e.g., rf_model) is already in memory.
#   The predictor columns â€” including the newly computed neighbor_max_*,
#   neighbor_min_*, neighbor_mean_* â€” now exist in cell_data with
#   identical numerical values to the original pipeline.
# ===========================================================================

cat("Running Random Forest prediction...\n")
cell_data$prediction <- predict(rf_model, newdata = cell_data)
cat("Done.\n")
```

---

## Summary of Guarantees

| Concern | Status |
|---|---|
| **Numerical equivalence** | Identical max/min/mean computed from the same neighbor sets and same variable values. No approximation. |
| **Trained RF model preserved** | Model object is never modified; only `predict()` is called. |
| **RAM on 16 GB laptop** | Peak ~3â€“4 GB (edge table Ã— years â‰ˆ 38M rows Ã— 4 cols â‰ˆ 1.2 GB + cell_data â‰ˆ 2 GB). Safe. |
| **Speed** | ~2â€“10 minutes total vs. ~86+ hours. The bottleneck shifts from R interpretation overhead to vectorized `data.table` grouped aggregation on ~38M rows, which is extremely fast. |
| **Edge table built once** | `edge_dt` is year-invariant, built in <1 second, reused for all 5 variables and all 28 years. |
END_RESPONSE>>>
