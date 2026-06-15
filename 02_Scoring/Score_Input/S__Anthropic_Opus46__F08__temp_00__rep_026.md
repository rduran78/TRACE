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
run_id: Anthropic_F08_T00_R026
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F08
prompt_family_slug: TOPOLOGY_INVARIANCE
family_label: Topology invariance
family_group: Topology cue
replicate: 26
raw_output_file: Experiment_Output/Anthropic/Temp_0.0/output_F08_Anthropic_temp_00_rep_026.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

## Diagnosis

The core performance problem is that `build_neighbor_lookup` constructs a **per-row** (per cell-year) lookup, producing ~6.46 million list entries, when the neighbor *topology* is identical across all 28 years. The function pastes cell IDs and years into string keys, performs hash lookups for every neighbor of every row, and repeats all of this identically for each of the 28 years a cell appears. This means:

1. **Redundant topology work Ã—28:** The neighbor graph has ~344K cells and ~1.37M directed edges. The lookup rebuilds this 28 times (once per year-copy of each cell), yielding ~38.4M string-paste + hash-lookup operations instead of ~1.37M.
2. **String-key indirection is slow:** Using `paste(id, year)` as a hash key and `match`/`[` on named vectors is far slower than integer indexing.
3. **`compute_neighbor_stats` iterates 6.46M list entries:** Each call to `lapply` over 6.46M elements with R-level anonymous functions is extremely slow, and this is repeated 5 times (once per variable).
4. **No vectorization:** The stats (max, min, mean) are computed one row at a time in pure R rather than via vectorized/matrix operations.

**Net effect:** ~86+ hours on a 16 GB laptop.

## Optimization Strategy

**Key insight:** Separate the *static topology* (which cells are neighbors of which) from the *dynamic values* (which change by year).

1. **Build the neighbor graph once at the cell level (344K cells, not 6.46M rows).** Store it as a sparse adjacency structure using integer cell indices (not string keys).

2. **For each variable, reshape values into a matrix of cells Ã— years.** This allows extracting all neighbor values via integer row-indexing into a matrix column (one column per year), fully vectorized.

3. **Compute neighbor stats using sparse matrix multiplication** (via the `Matrix` package) or, equivalently, using a CSR-style adjacency to do vectorized grouped operations. Specifically:
   - Construct a sparse binary adjacency matrix **W** of dimension 344,208 Ã— 344,208 (only ~1.37M non-zeros â€” tiny in memory).
   - For each year and each variable, the column of values `v` allows: `neighbor_sum = W %*% v`, `neighbor_count = W %*% (!is.na(v))`, `neighbor_mean = neighbor_sum / neighbor_count`.
   - For max and min, use a sparse-matrix trick: iterate over the adjacency list but in compiled C++ via `Matrix` internals, or use a small Rcpp snippet / vectorized R approach with the CSR representation.

4. **Flatten back** to the original cell-year data frame and attach the 15 new columns (5 vars Ã— 3 stats).

5. **Feed into the pre-trained Random Forest** exactly as before â€” column names and numerical values are preserved.

**Expected speedup:** From ~86 hours to **minutes**. The sparse matrixâ€“vector product for mean is O(nnz) â‰ˆ 1.37M per year per variable = 1.37M Ã— 28 Ã— 5 â‰ˆ 192M operations, trivially fast. Max/min via the CSR loop is the same order.

## Working R Code

```r
library(Matrix)
library(data.table)

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 0.  Inputs assumed to exist:
#       cell_data            : data.frame/data.table with columns id, year, ntl, ec, pop_density, def, usd_est_n2, â€¦
#       id_order             : integer/character vector of unique cell IDs (length 344,208)
#       rook_neighbors_unique: spdep nb object (list of length 344,208, integer neighbor indices)
#       rf_model             : pre-trained Random Forest model object
#       neighbor_source_vars : c("ntl", "ec", "pop_density", "def", "usd_est_n2")
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

# Convert to data.table for speed (non-destructive if already data.table)
cell_dt <- as.data.table(cell_data)

n_cells <- length(id_order)
years   <- sort(unique(cell_dt$year))
n_years <- length(years)

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 1.  Build the STATIC sparse binary adjacency matrix  (done ONCE)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
message("Building sparse adjacency matrix â€¦")

# CSR-style vectors
adj_i <- integer(0)
adj_j <- integer(0)

for (k in seq_len(n_cells)) {
  nb <- rook_neighbors_unique[[k]]
  if (length(nb) > 0L && !(length(nb) == 1L && nb[1] == 0L)) {
    adj_i <- c(adj_i, rep.int(k, length(nb)))
    adj_j <- c(adj_j, nb)
  }
}

W <- sparseMatrix(i = adj_i, j = adj_j, x = 1,
                  dims = c(n_cells, n_cells), giveCsparse = TRUE)

# Also store adjacency as a list of integer vectors for max/min
# (reuse rook_neighbors_unique directly â€” it already is this)
adj_list <- rook_neighbors_unique

rm(adj_i, adj_j)

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 2.  Create a cell-index column in cell_dt for fast matrix mapping
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Map each cell id to its position in id_order (1-based integer index)
id_to_cellidx <- setNames(seq_len(n_cells), as.character(id_order))
cell_dt[, cell_idx := id_to_cellidx[as.character(id)]]

# Ensure rows are sorted by (year, cell_idx) for predictable matrix fill
setkey(cell_dt, year, cell_idx)

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 3.  Helper: build a cells Ã— years matrix from a variable column
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
build_cell_year_matrix <- function(dt, var_name, n_cells, years) {
  # Pre-allocate matrix (NA by default handles any missing cell-years)
  mat <- matrix(NA_real_, nrow = n_cells, ncol = length(years))
  colnames(mat) <- as.character(years)
  
  for (yi in seq_along(years)) {
    yr <- years[yi]
    sub <- dt[year == yr, .(cell_idx, val = get(var_name))]
    mat[sub$cell_idx, yi] <- sub$val
  }
  mat
}

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 4.  Compute neighbor stats for one variable (vectorized over cells)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
compute_neighbor_stats_fast <- function(W, adj_list, val_mat, n_cells, n_years) {
  # Output matrices: cells Ã— years
  nb_max  <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
  nb_min  <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
  nb_mean <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
  
  for (yi in seq_len(n_years)) {
    v <- val_mat[, yi]                       # length n_cells
    not_na <- !is.na(v)
    v_zero <- v
    v_zero[is.na(v_zero)] <- 0               # for sparse multiply (NAs â†’ 0)
    
    # --- Neighbor mean via sparse matrix multiply ---
    nb_sum   <- as.numeric(W %*% v_zero)     # sum of neighbor values (NAâ†’0)
    nb_count <- as.numeric(W %*% as.numeric(not_na))  # count of non-NA neighbors
    
    mean_vec <- ifelse(nb_count > 0, nb_sum / nb_count, NA_real_)
    nb_mean[, yi] <- mean_vec
    
    # --- Neighbor max and min via adjacency list ---
    # Vectorized as much as possible; this loop is over 344K cells (fast)
    max_vec <- rep(NA_real_, n_cells)
    min_vec <- rep(NA_real_, n_cells)
    
    for (k in seq_len(n_cells)) {
      nb <- adj_list[[k]]
      if (length(nb) == 0L || (length(nb) == 1L && nb[1] == 0L)) next
      nb_vals <- v[nb]
      nb_vals <- nb_vals[!is.na(nb_vals)]
      if (length(nb_vals) == 0L) next
      max_vec[k] <- max(nb_vals)
      min_vec[k] <- min(nb_vals)
    }
    
    nb_max[, yi] <- max_vec
    nb_min[, yi] <- min_vec
    
    if (yi %% 5 == 0) message("    year ", yi, "/", n_years, " done")
  }
  
  list(nb_max = nb_max, nb_min = nb_min, nb_mean = nb_mean)
}

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 5.  Main loop: iterate over the 5 variables (not 6.46M rows)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
for (var_name in neighbor_source_vars) {
  message("Processing neighbor stats for: ", var_name)
  
  # 5a. Reshape variable to cells Ã— years matrix
  val_mat <- build_cell_year_matrix(cell_dt, var_name, n_cells, years)
  
  # 5b. Compute stats (vectorized / sparse)
  stats <- compute_neighbor_stats_fast(W, adj_list, val_mat, n_cells, n_years)
  
  # 5c. Map results back to cell_dt rows
  col_max  <- paste0(var_name, "_neighbor_max")
  col_min  <- paste0(var_name, "_neighbor_min")
  col_mean <- paste0(var_name, "_neighbor_mean")
  
  # Flatten matrices back to the row order of cell_dt (keyed by year, cell_idx)
  cell_dt[, (col_max)  := NA_real_]
  cell_dt[, (col_min)  := NA_real_]
  cell_dt[, (col_mean) := NA_real_]
  
  for (yi in seq_along(years)) {
    yr <- years[yi]
    rows <- cell_dt[year == yr, which = TRUE]
    cidx <- cell_dt$cell_idx[rows]
    
    set(cell_dt, i = rows, j = col_max,  value = stats$nb_max [cidx, yi])
    set(cell_dt, i = rows, j = col_min,  value = stats$nb_min [cidx, yi])
    set(cell_dt, i = rows, j = col_mean, value = stats$nb_mean[cidx, yi])
  }
  
  rm(val_mat, stats)
  gc()
  message("  âœ“ ", var_name, " complete.")
}

# Clean up helper column
cell_dt[, cell_idx := NULL]

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 6.  Predict with the pre-trained Random Forest (unchanged)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
cell_dt[, prediction := predict(rf_model, newdata = cell_dt)]

# Convert back to data.frame if downstream code expects it
cell_data <- as.data.frame(cell_dt)
```

---

### Optional: Replace the inner max/min R loop with Rcpp for further speedup

The per-year R loop over 344K cells for max/min is the remaining bottleneck (~5â€“10 min total). If desired, this Rcpp function eliminates it:

```cpp
// file: neighbor_maxmin.cpp
#include <Rcpp.h>
using namespace Rcpp;

// [[Rcpp::export]]
NumericMatrix neighbor_maxmin(List adj_list, NumericVector v) {
  int n = adj_list.size();
  NumericMatrix out(n, 2);  // col 0 = max, col 1 = min
  std::fill(out.begin(), out.end(), NA_REAL);

  for (int i = 0; i < n; i++) {
    IntegerVector nb = adj_list[i];
    if (nb.size() == 0 || (nb.size() == 1 && nb[0] == 0)) continue;
    double mx = R_NegInf, mn = R_PosInf;
    bool found = false;
    for (int j = 0; j < nb.size(); j++) {
      double val = v[nb[j] - 1];  // R is 1-indexed
      if (NumericVector::is_na(val)) continue;
      found = true;
      if (val > mx) mx = val;
      if (val < mn) mn = val;
    }
    if (found) {
      out(i, 0) = mx;
      out(i, 1) = mn;
    }
  }
  return out;
}
```

```r
Rcpp::sourceCpp("neighbor_maxmin.cpp")

# Replace the inner for-k loop in step 4 with:
mm <- neighbor_maxmin(adj_list, v)   # returns n_cells Ã— 2 matrix
nb_max[, yi] <- mm[, 1]
nb_min[, yi] <- mm[, 2]
```

---

### Summary of improvements

| Aspect | Original | Redesigned |
|---|---|---|
| Neighbor lookup granularity | 6.46M cell-year rows | 344K cells (built once) |
| Key mechanism | String paste + named-vector hash | Integer index + sparse matrix |
| Mean computation | R `lapply` over 6.46M entries | Sparse matrix multiply (`W %*% v`) |
| Max/Min computation | R `lapply` over 6.46M entries | R loop over 344K cells (or Rcpp) |
| Times repeated | 5 vars Ã— 6.46M = 32.3M R calls | 5 vars Ã— 28 years Ã— 344K cells |
| Estimated runtime | 86+ hours | **~5â€“15 minutes** (pure R) / **~2â€“5 min** (with Rcpp) |
| Numerical results | â€” | **Identical** (same max, min, mean) |
| RF model | Unchanged | Unchanged |
| Peak RAM | Lookup list ~6.46M entries | Sparse matrix ~22 MB + one 344KÃ—28 matrix ~77 MB |
END_RESPONSE>>>
