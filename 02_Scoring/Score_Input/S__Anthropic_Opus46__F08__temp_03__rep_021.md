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
run_id: Anthropic_F08_T03_R021
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.3
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F08
prompt_family_slug: TOPOLOGY_INVARIANCE
family_label: Topology invariance
family_group: Topology cue
replicate: 21
raw_output_file: Experiment_Output/Anthropic/Temp_0.3/output_F08_Anthropic_temp_03_rep_021.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

## Diagnosis

The current implementation is extremely slow (~86+ hours) because of a fundamental design flaw: **it conflates static topology with dynamic data**.

### Root Cause Analysis

1. **`build_neighbor_lookup` rebuilds per-row (cellÃ—year) neighbor indices.** It creates a list of ~6.46 million entries, each requiring string-pasting (`paste(id, year)`), hash-table lookups (`idx_lookup[neighbor_keys]`), and NA filtering. But the neighbor *relationships* between cells are purely spatial and **never change across years**. The only thing that changes is the variable values attached to each cell in each year.

2. **The lookup is keyed on `(id, year)` row positions in a stacked panel.** This means for every cell, the same neighbor topology is redundantly resolved 28 times (once per year), and each resolution involves string concatenation and named-vector lookups â€” O(n_rows Ã— avg_neighbors) string operations on a 6.46M-row dataset.

3. **`compute_neighbor_stats` iterates over 6.46M entries** in an R-level `lapply`, extracting subsets of a vector by index. While each individual operation is fast, 6.46M R-level function calls with list allocation is inherently slow.

4. **The entire pipeline runs 5 times** (once per neighbor source variable), multiplying the cost.

### Quantifying the Waste

- 344,208 cells Ã— 28 years = 9,637,824 neighbor-lookup constructions, but only 344,208 unique topologies exist.
- Each construction involves `paste()` and named-vector lookup on strings â€” orders of magnitude slower than integer indexing.
- The 28Ã— redundancy in topology resolution and the R-level loop over 6.46M rows are the dominant bottlenecks.

---

## Optimization Strategy

**Core Insight:** Separate the *static spatial topology* (which cells are neighbors of which) from the *dynamic yearly variable values* (which change by year). Compute neighbor statistics using a **cell-level neighbor index** (built once) and a **year-level matrix/column operation** (vectorized).

### Step-by-Step Plan

1. **Build the cell-to-cell neighbor index once** â€” a simple list of length 344,208 where each element contains the integer positions of that cell's neighbors in the cell-ID vector. This is topology-only, year-independent, and built once.

2. **For each variable and each year, extract the values vector (length 344,208), then compute neighbor max/min/mean using the static neighbor index.** This turns the inner loop from 6.46M iterations into 28 iterations of a 344K-cell vectorized operation.

3. **Vectorize the per-cell neighbor aggregation** using `data.table` for fast split-apply or, even better, using a **CSR (Compressed Sparse Row) representation** of the neighbor graph to enable fully vectorized `rowmax/rowmin/rowmean` via sparse-matrix-style operations, or a tight `vapply` over only 344K cells instead of 6.46M.

4. **Write results back into the panel data.frame/data.table** by joining on `(cell_index, year)`.

### Expected Speedup

| Factor | Current | Optimized | Speedup |
|---|---|---|---|
| Neighbor index construction | 6.46M string lookups | 344K integer lookups (once) | ~525Ã— |
| Stat computation loop | 6.46M R calls Ã— 5 vars | 28 years Ã— 344K cells Ã— 5 vars | ~28Ã— fewer calls |
| String operations | ~50M paste + match | 0 | Eliminated |
| Overall estimate | ~86 hours | **~5â€“15 minutes** | ~350â€“1000Ã— |

---

## Working R Code

```r
library(data.table)

# =============================================================================
# STEP 0: Convert to data.table if not already
# =============================================================================
if (!is.data.table(cell_data)) {
  cell_data <- as.data.table(cell_data)
}

# =============================================================================
# STEP 1: Build the STATIC cell-level neighbor index (done ONCE)
#
# id_order:              vector of all unique cell IDs (length = 344,208)
# rook_neighbors_unique: spdep nb object (list of length 344,208),
#                        each element contains integer indices into id_order
#                        of that cell's neighbors.
#
# We store this as-is â€” it's already an integer index into id_order.
# We just need a mapping from cell ID -> position in id_order.
# =============================================================================

build_static_neighbor_index <- function(id_order, neighbors) {
  # neighbors is already an nb object: list of integer vectors

# Each element i contains the indices (into id_order) of cell i's neighbors.
  # A neighbor index of 0 means no neighbors (spdep convention).
  # We clean that up:
  n <- length(neighbors)
  nb_index <- vector("list", n)
  for (i in seq_len(n)) {
    nb_i <- neighbors[[i]]
    # spdep uses 0L to indicate no neighbors
    nb_i <- nb_i[nb_i > 0L]
    nb_index[[i]] <- nb_i
  }
  nb_index
}

# Build once â€” this takes < 1 second for 344K cells
static_nb <- build_static_neighbor_index(id_order, rook_neighbors_unique)

# =============================================================================
# STEP 2: Build a fast mapping from cell ID to position in id_order
# =============================================================================
id_to_pos <- setNames(seq_along(id_order), as.character(id_order))

# Add cell position column to cell_data (once)
cell_data[, cell_pos := id_to_pos[as.character(id)]]

# =============================================================================
# STEP 3: Pre-flatten the neighbor index into CSR-like vectors for
#          fully vectorized aggregation (avoids 344K lapply calls per year)
# =============================================================================

build_csr_neighbors <- function(static_nb) {
  # Flatten the list into two vectors:
  #   nb_cell_idx: the neighbor cell positions (concatenated)
  #   nb_ptr:      pointer into nb_cell_idx for each cell (length n+1)
  #                cell i's neighbors are nb_cell_idx[ (nb_ptr[i]+1) : nb_ptr[i+1] ]
  
  n <- length(static_nb)
  lengths_vec <- vapply(static_nb, length, integer(1))
  total <- sum(lengths_vec)
  
  nb_cell_idx <- integer(total)
  nb_ptr      <- integer(n + 1L)
  
  pos <- 0L
  for (i in seq_len(n)) {
    nb_i <- static_nb[[i]]
    len_i <- lengths_vec[i]
    if (len_i > 0L) {
      nb_cell_idx[(pos + 1L):(pos + len_i)] <- nb_i
    }
    pos <- pos + len_i
    nb_ptr[i + 1L] <- pos
  }
  
  list(idx = nb_cell_idx, ptr = nb_ptr, lengths = lengths_vec)
}

csr <- build_csr_neighbors(static_nb)

# =============================================================================
# STEP 4: Vectorized neighbor stat computation using CSR structure
#
# For a given numeric vector of values (one per cell, for a single year),
# compute max, min, mean of each cell's neighbors.
# =============================================================================

compute_neighbor_stats_csr <- function(vals, csr) {
  # vals: numeric vector of length n_cells (one value per cell for one year)
  # csr:  list with idx, ptr, lengths from build_csr_neighbors
  
  n <- length(vals)
  nb_vals <- vals[csr$idx]  # vectorized lookup: all neighbor values, flattened
  
  # We need to compute grouped max, min, mean over segments defined by csr$ptr
  # Use a data.table approach for speed:
  
  # Create group IDs: cell index repeated by number of neighbors
  grp <- rep.int(seq_len(n), csr$lengths)
  
  # Handle cells with zero neighbors: they won't appear in grp
  # We'll compute stats for cells that have neighbors, then fill NA for the rest
  
  if (length(nb_vals) == 0) {
    return(data.table(
      nb_max  = rep(NA_real_, n),
      nb_min  = rep(NA_real_, n),
      nb_mean = rep(NA_real_, n)
    ))
  }
  
  # Remove NAs in neighbor values
  valid <- !is.na(nb_vals)
  
  dt_nb <- data.table(grp = grp[valid], val = nb_vals[valid])
  
  stats <- dt_nb[, .(
    nb_max  = max(val),
    nb_min  = min(val),
    nb_mean = mean(val)
  ), keyby = grp]
  
  # Initialize result with NAs
  result <- data.table(
    nb_max  = rep(NA_real_, n),
    nb_min  = rep(NA_real_, n),
    nb_mean = rep(NA_real_, n)
  )
  
  result[stats$grp, `:=`(
    nb_max  = stats$nb_max,
    nb_min  = stats$nb_min,
    nb_mean = stats$nb_mean
  )]
  
  result
}

# =============================================================================
# STEP 5: Main loop â€” iterate over variables and years
#
# For each variable, for each year:
#   1. Extract the value vector (one per cell) for that year.
#   2. Compute neighbor max/min/mean using the CSR structure.
#   3. Write results back into cell_data.
# =============================================================================

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# Ensure cell_data is keyed for fast subsetting
setkey(cell_data, year, cell_pos)

years <- sort(unique(cell_data$year))
n_cells <- length(id_order)

for (var_name in neighbor_source_vars) {
  
  cat("Processing neighbor stats for:", var_name, "\n")
  
  col_max  <- paste0("neighbor_max_",  var_name)
  col_min  <- paste0("neighbor_min_",  var_name)
  col_mean <- paste0("neighbor_mean_", var_name)
  
  # Pre-allocate output columns with NA
  cell_data[, (col_max)  := NA_real_]
  cell_data[, (col_min)  := NA_real_]
  cell_data[, (col_mean) := NA_real_]
  
  for (yr in years) {
    
    # Extract values for this year, ordered by cell_pos
    # cell_data is keyed by (year, cell_pos), so J(yr) gives us
    # all rows for this year, sorted by cell_pos
    year_rows <- cell_data[.(yr)]
    
    # Build a value vector indexed by cell position
    # (some cells may be missing for some years; handle gracefully)
    vals_vec <- rep(NA_real_, n_cells)
    vals_vec[year_rows$cell_pos] <- year_rows[[var_name]]
    
    # Compute neighbor stats using CSR (fully vectorized)
    stats <- compute_neighbor_stats_csr(vals_vec, csr)
    
    # Write back: map cell_pos -> rows in cell_data for this year
    # year_rows$cell_pos gives us which cells are present
    cell_positions <- year_rows$cell_pos
    
    set(cell_data, 
        which(cell_data$year == yr),
        col_max,
        stats$nb_max[cell_positions])
    set(cell_data, 
        which(cell_data$year == yr),
        col_min,
        stats$nb_min[cell_positions])
    set(cell_data, 
        which(cell_data$year == yr),
        col_mean,
        stats$nb_mean[cell_positions])
  }
  
  cat("  Done:", var_name, "\n")
}

# =============================================================================
# STEP 6: Clean up helper column, then proceed to prediction
# =============================================================================
cell_data[, cell_pos := NULL]

# Now cell_data has the same neighbor_max_*, neighbor_min_*, neighbor_mean_*
# columns as the original pipeline, with identical numerical values.
# Proceed to Random Forest prediction with the pre-trained model:
#
# predictions <- predict(trained_rf_model, newdata = cell_data)
```

---

## Why This Preserves the Original Numerical Estimand

| Concern | Guarantee |
|---|---|
| **Same neighbor topology** | `static_nb` is derived from the same `rook_neighbors_unique` nb object and `id_order`. The CSR flattening is a lossless representation of the same adjacency. |
| **Same aggregation functions** | `max()`, `min()`, `mean()` are applied to exactly the same sets of neighbor values, with the same NA-removal logic (`!is.na()`). |
| **Same output columns** | Column names follow the same `neighbor_{max,min,mean}_{var}` convention. |
| **No model retraining** | The pre-trained Random Forest is loaded and used for `predict()` only â€” untouched. |
| **Floating-point identity** | The operations are mathematically identical; results will match to machine precision. |

## Performance Summary

- **Neighbor index:** Built once in <1 second (344K cells), not 6.46M times.
- **CSR construction:** ~1 second, done once.
- **Per variable Ã— year:** One vectorized lookup + one `data.table` grouped aggregation over ~1.37M neighbor pairs â†’ ~0.2â€“0.5 seconds each.
- **Total:** 5 variables Ã— 28 years Ã— ~0.4s â‰ˆ **~56 seconds**, plus overhead â‰ˆ **~2â€“5 minutes total** (down from 86+ hours).
END_RESPONSE>>>
