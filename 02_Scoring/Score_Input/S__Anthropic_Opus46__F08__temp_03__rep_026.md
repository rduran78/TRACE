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
run_id: Anthropic_F08_T03_R026
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.3
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F08
prompt_family_slug: TOPOLOGY_INVARIANCE
family_label: Topology invariance
family_group: Topology cue
replicate: 26
raw_output_file: Experiment_Output/Anthropic/Temp_0.3/output_F08_Anthropic_temp_03_rep_026.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

## Diagnosis

The current implementation is extremely slow (~86+ hours) because of a fundamental design flaw: **it conflates static topology with dynamic data**.

### Root Cause Analysis

1. **`build_neighbor_lookup` rebuilds per-row (cellÃ—year) neighbor indices.** It creates a list of ~6.46 million entries, each requiring string-pasting cell IDs with years, hash lookups in `idx_lookup`, and NA filtering. But the neighbor *relationships* between cells are purely spatial and **never change across years**. The year dimension is entirely redundant in the topology.

2. **`compute_neighbor_stats` operates on the full 6.46M-row vector.** Because the lookup was built at the cellÃ—year level, every stat computation indexes into a massive vector, and the `lapply` iterates over 6.46M elements instead of 344K elements.

3. **String-based key construction** (`paste(id, year, sep="_")`) and named-vector lookups (`idx_lookup[neighbor_keys]`) are extremely slow at this scale â€” O(n) string allocations and hash lookups repeated millions of times.

### The Key Insight

- **Static:** The rook-neighbor graph (which cells are neighbors of which) is invariant across all 28 years. There are only ~344K cells and ~1.37M directed neighbor edges.
- **Dynamic:** The variable values (`ntl`, `ec`, `pop_density`, `def`, `usd_est_n2`) change by year.

Therefore, the correct architecture is:

1. **Build the neighbor lookup once over 344K cells** (not 6.46M cell-years).
2. **For each year, slice the data, compute neighbor stats over 344K cells, and write back.**

This reduces the core loop from 6.46M iterations to 344K iterations Ã— 28 years, eliminates all string-key construction, and uses simple integer indexing throughout.

---

## Optimization Strategy

| Aspect | Current | Redesigned |
|---|---|---|
| Lookup granularity | cellÃ—year (6.46M entries) | cell only (344K entries) |
| Lookup construction | String paste + named vector hash | Integer position mapping, built once |
| Stats loop iterations | 6.46M per variable | 344K per year per variable (= 9.6M total, but each iteration is trivial integer indexing) |
| Key mechanism | Character keys | Integer row indices within year-slices |
| Memory | 6.46M-element list of integer vectors | 344K-element list (reused across years and variables) |
| Estimated time | 86+ hours | **Minutes** |

### Steps

1. **Build a cell-level neighbor lookup** â€” a list of length 344K where element `i` contains the integer positions of cell `i`'s neighbors within the cell-order vector. This is built once from `rook_neighbors_unique` and `id_order`.

2. **Sort/index data by (year, cell)** so that within each year-slice, row positions correspond directly to the cell-order positions. This makes neighbor indexing a direct integer offset.

3. **For each variable Ã— each year**, extract the year-slice vector, compute max/min/mean over neighbor indices, and write results back.

4. **Feed the augmented `cell_data` to the pre-trained Random Forest** exactly as before â€” the output columns are numerically identical.

---

## Working R Code

```r
# =============================================================================
# OPTIMIZED NEIGHBOR FEATURE COMPUTATION
# Separates static topology from dynamic (year-varying) variable values.
# Produces numerically identical results to the original implementation.
# =============================================================================

library(data.table)

compute_neighbor_features_optimized <- function(cell_data,
                                                 id_order,
                                                 rook_neighbors_unique,
                                                 neighbor_source_vars) {
  # -------------------------------------------------------------------------
  # STEP 1: Build the STATIC cell-level neighbor lookup (done ONCE)
  # -------------------------------------------------------------------------
  # rook_neighbors_unique is an nb object: a list of length = length(id_order),

  # where element i contains integer indices (into id_order) of cell i's neighbors.
  # We store these directly â€” no string keys, no year dimension.
  
  n_cells <- length(id_order)
  
  # cell_neighbor_idx: list of length n_cells

  # Element i = integer vector of positions (in id_order) of neighbors of cell i.
  # spdep::nb objects already use this convention, but we sanitize:
  cell_neighbor_idx <- lapply(seq_len(n_cells), function(i) {
    nb <- rook_neighbors_unique[[i]]
    # spdep uses 0L to indicate no neighbors
    nb <- nb[nb != 0L]
    as.integer(nb)
  })
  
  # -------------------------------------------------------------------------
  # STEP 2: Convert to data.table and ensure consistent cell ordering per year
  # -------------------------------------------------------------------------
  dt <- as.data.table(cell_data)
  
  # Create a mapping from cell id to position in id_order (1-based)
  id_to_pos <- setNames(seq_along(id_order), as.character(id_order))
  
  # Add cell position column
  dt[, cell_pos := id_to_pos[as.character(id)]]
  
  # Pre-allocate output columns for all neighbor features
  for (var_name in neighbor_source_vars) {
    col_max  <- paste0("neighbor_max_",  var_name)
    col_min  <- paste0("neighbor_min_",  var_name)
    col_mean <- paste0("neighbor_mean_", var_name)
    dt[, (col_max)  := NA_real_]
    dt[, (col_min)  := NA_real_]
    dt[, (col_mean) := NA_real_]
  }
  
  # -------------------------------------------------------------------------
  # STEP 3: For each year, compute neighbor stats using cell-level topology
  # -------------------------------------------------------------------------
  years <- sort(unique(dt$year))
  
  for (yr in years) {
    # Get row indices for this year
    year_rows <- which(dt$year == yr)
    
    # Build a vector indexed by cell_pos for this year's rows
    # cell_positions present in this year
    year_cell_pos <- dt$cell_pos[year_rows]
    
    # Create a mapping: cell_pos -> row index in year_rows
    # We need a fast lookup: for a given cell_pos, what is its row in dt?
    # Use a pre-allocated vector of length n_cells
    pos_to_dtrow <- rep(NA_integer_, n_cells)
    pos_to_dtrow[year_cell_pos] <- year_rows
    
    for (var_name in neighbor_source_vars) {
      col_max  <- paste0("neighbor_max_",  var_name)
      col_min  <- paste0("neighbor_min_",  var_name)
      col_mean <- paste0("neighbor_mean_", var_name)
      
      # Extract the full variable vector for this year, indexed by cell_pos
      # Pre-allocate a vector of length n_cells (NA for missing cells)
      vals_by_pos <- rep(NA_real_, n_cells)
      vals_by_pos[year_cell_pos] <- dt[[var_name]][year_rows]
      
      # Now compute neighbor stats for each cell present this year
      # Vectorized approach using the cell_neighbor_idx list
      n_year <- length(year_rows)
      res_max  <- rep(NA_real_, n_year)
      res_min  <- rep(NA_real_, n_year)
      res_mean <- rep(NA_real_, n_year)
      
      for (j in seq_len(n_year)) {
        cp <- year_cell_pos[j]
        nb_positions <- cell_neighbor_idx[[cp]]
        if (length(nb_positions) == 0L) next
        
        nb_vals <- vals_by_pos[nb_positions]
        nb_vals <- nb_vals[!is.na(nb_vals)]
        if (length(nb_vals) == 0L) next
        
        res_max[j]  <- max(nb_vals)
        res_min[j]  <- min(nb_vals)
        res_mean[j] <- mean(nb_vals)
      }
      
      # Write back to dt
      set(dt, i = year_rows, j = col_max,  value = res_max)
      set(dt, i = year_rows, j = col_min,  value = res_min)
      set(dt, i = year_rows, j = col_mean, value = res_mean)
    }
    
    # Progress reporting
    message(sprintf("Year %d complete.", yr))
  }
  
  # -------------------------------------------------------------------------
  # STEP 4: Clean up and return
  # -------------------------------------------------------------------------
  dt[, cell_pos := NULL]
  
  # Return as data.frame to preserve downstream compatibility
  as.data.frame(dt)
}


# =============================================================================
# USAGE â€” drop-in replacement for the original outer loop
# =============================================================================

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cell_data <- compute_neighbor_features_optimized(
  cell_data            = cell_data,
  id_order             = id_order,
  rook_neighbors_unique = rook_neighbors_unique,
  neighbor_source_vars = neighbor_source_vars
)

# The pre-trained Random Forest model is used exactly as before:
# predictions <- predict(rf_model, newdata = cell_data)
```

---

## Further Acceleration (Optional C++ Inner Loop)

The inner `for (j in seq_len(n_year))` loop over ~344K cells is the remaining bottleneck. If even the optimized version is not fast enough, this can be replaced with an Rcpp function:

```r
# Optional: Rcpp acceleration for the inner loop
Rcpp::sourceCpp(code = '
#include <Rcpp.h>
using namespace Rcpp;

// [[Rcpp::export]]
NumericMatrix neighbor_stats_cpp(NumericVector vals_by_pos,
                                  IntegerVector year_cell_pos,
                                  List cell_neighbor_idx) {
  int n = year_cell_pos.size();
  NumericMatrix out(n, 3); // columns: max, min, mean
  
  for (int j = 0; j < n; j++) {
    out(j, 0) = NA_REAL;
    out(j, 1) = NA_REAL;
    out(j, 2) = NA_REAL;
    
    int cp = year_cell_pos[j]; // 1-based
    IntegerVector nb = cell_neighbor_idx[cp - 1]; // 0-based list index
    
    double vmax = R_NegInf, vmin = R_PosInf, vsum = 0.0;
    int count = 0;
    
    for (int k = 0; k < nb.size(); k++) {
      int pos = nb[k]; // 1-based index into vals_by_pos
      if (pos < 1 || pos > vals_by_pos.size()) continue;
      double v = vals_by_pos[pos - 1];
      if (ISNA(v) || ISNAN(v)) continue;
      if (v > vmax) vmax = v;
      if (v < vmin) vmin = v;
      vsum += v;
      count++;
    }
    
    if (count > 0) {
      out(j, 0) = vmax;
      out(j, 1) = vmin;
      out(j, 2) = vsum / count;
    }
  }
  return out;
}
')
```

Then replace the inner R loop with:

```r
res <- neighbor_stats_cpp(vals_by_pos, year_cell_pos, cell_neighbor_idx)
res_max  <- res[, 1]
res_min  <- res[, 2]
res_mean <- res[, 3]
```

---

## Summary of Gains

| Metric | Original | Optimized (R) | Optimized (Rcpp) |
|---|---|---|---|
| Lookup build | ~6.46M string-key entries | 344K integer lists (once) | Same |
| Stats iterations | 6.46M Ã— 5 vars = 32.3M | 344K Ã— 28 yrs Ã— 5 vars = 48.2M (but trivial ops) | Same, compiled |
| String operations | ~billions of paste/hash | **Zero** | Zero |
| Estimated wall time | 86+ hours | ~10â€“30 minutes | ~2â€“5 minutes |
| Numerical output | Baseline | **Identical** | **Identical** |
| RF model | Unchanged | **Unchanged** | **Unchanged** |
END_RESPONSE>>>
