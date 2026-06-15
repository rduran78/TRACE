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
run_id: Anthropic_F08_T00_R004
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F08
prompt_family_slug: TOPOLOGY_INVARIANCE
family_label: Topology invariance
family_group: Topology cue
replicate: 4
raw_output_file: Experiment_Output/Anthropic/Temp_0.0/output_F08_Anthropic_temp_00_rep_004.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

## Diagnosis

The core performance problem is that `build_neighbor_lookup` constructs a **per-row** (per cell-year) lookup, producing ~6.46 million list entries, when the neighbor topology is **identical across all 28 years**. The function pastes cell IDs and years into string keys, performs hash lookups for every neighbor of every row, and repeats all of this identically for each of the 28 years a cell appears. This means:

1. **Redundant topology computation**: The neighbor graph has ~344K cells. The lookup is rebuilt as if there are ~6.46M distinct nodes. The string-key hashing (`paste(id, year)` â†’ row index) is O(n) in total rows and is done per neighbor per row.
2. **Redundant neighbor-value gathering**: `compute_neighbor_stats` iterates over all ~6.46M rows, indexing into the full data vector each time. Because the lookup was built at the cell-year level, it re-discovers the same neighbor *cells* 28 times (once per year).
3. **Memory bloat**: The `neighbor_lookup` list has 6.46M entries, each containing integer vectors. This is enormous and cache-unfriendly.
4. **R-level looping**: Both `lapply` calls loop at the R interpreter level over millions of elements.

**The key insight**: Neighbor relationships are a property of **cells**, not cell-years. Only the **variable values** change by year. Therefore, we should:
- Build the neighbor lookup **once over 344K cells** (not 6.46M rows).
- For each variable, extract a **cell Ã— year matrix**, compute neighbor stats using the cell-level adjacency, and write results back.

This reduces the topology work by 28Ã—, enables vectorized/matrix operations, and slashes memory use.

---

## Optimization Strategy

### Step 1: Build a cell-level neighbor index (once)

Convert `rook_neighbors_unique` (an `nb` object, already indexed by cell position in `id_order`) into a simple CSR-like structure: two integer vectors (`adj` and `ptr`) so that the neighbors of cell `i` are `adj[ptr[i]:(ptr[i+1]-1)]`. This is compact and fast to traverse.

### Step 2: Reshape data into cell Ã— year matrices

For each neighbor source variable, pivot the long panel into a `344208 Ã— 28` matrix (rows = cells in `id_order` order, columns = years). This is ~77 MB per variable (double precision) â€” very manageable.

### Step 3: Compute neighbor max/min/mean via vectorized C++ (Rcpp) or optimized R

Loop over cells (344K, not 6.46M), gather neighbor rows from the matrix, and compute column-wise (i.e., per-year) max, min, mean. The result is three `344208 Ã— 28` matrices per variable. An Rcpp inner loop makes this very fast (~seconds per variable).

### Step 4: Unpivot and join back

Melt the result matrices back to long format and bind columns to `cell_data`.

### Result

- Topology work: 344K cells instead of 6.46M rows â†’ **28Ã— reduction**.
- Per-variable work: vectorized matrix operations over 344K cells Ã— 28 years.
- Estimated runtime: **minutes** instead of 86+ hours.
- The trained Random Forest model is untouched; the numerical outputs (neighbor max, min, mean) are identical.

---

## Working R Code

```r
# =============================================================================
# Optimized neighbor-stats computation
# Exploits: neighbor topology is static across years; only values change.
# =============================================================================

library(data.table)
library(Rcpp)

# ---- 1. Build cell-level CSR adjacency (once) ------------------------------

build_cell_adjacency_csr <- function(nb_obj) {
  # nb_obj: spdep nb object, length = number of cells
  # nb_obj[[i]] gives integer vector of neighbor indices (into the same ordering)
  # Convert to CSR (compressed sparse row) for fast C++ traversal
  n <- length(nb_obj)
  # Compute pointers
  lengths <- vapply(nb_obj, function(x) {
    if (length(x) == 1L && x[1] == 0L) 0L else length(x)
  }, integer(1))
  ptr <- c(0L, cumsum(lengths))  # 0-based pointers, length n+1
  # Flatten adjacency (convert to 0-based indices for C++)
  adj <- unlist(lapply(nb_obj, function(x) {
    if (length(x) == 1L && x[1] == 0L) integer(0) else as.integer(x) - 1L
  }), use.names = FALSE)
  list(ptr = ptr, adj = adj, n = n)
}

# ---- 2. Rcpp kernel for neighbor stats on a cell x year matrix --------------

cppFunction('
#include <Rcpp.h>
using namespace Rcpp;

// [[Rcpp::export]]
List neighbor_stats_matrix(NumericMatrix vals,
                           IntegerVector ptr,
                           IntegerVector adj) {
  // vals: n_cells x n_years
  // ptr:  length n_cells + 1 (0-based CSR pointers)
  // adj:  flattened neighbor indices (0-based)
  // Returns list of 3 matrices: max, min, mean (each n_cells x n_years)

  int n = vals.nrow();
  int ny = vals.ncol();

  NumericMatrix out_max(n, ny);
  NumericMatrix out_min(n, ny);
  NumericMatrix out_mean(n, ny);

  for (int i = 0; i < n; i++) {
    int start = ptr[i];
    int end   = ptr[i + 1];
    int n_nbr = end - start;

    if (n_nbr == 0) {
      for (int y = 0; y < ny; y++) {
        out_max(i, y)  = NA_REAL;
        out_min(i, y)  = NA_REAL;
        out_mean(i, y) = NA_REAL;
      }
      continue;
    }

    for (int y = 0; y < ny; y++) {
      double vmax  = R_NegInf;
      double vmin  = R_PosInf;
      double vsum  = 0.0;
      int    count = 0;

      for (int k = start; k < end; k++) {
        double v = vals(adj[k], y);
        if (!NumericMatrix::is_na(v)) {
          if (v > vmax) vmax = v;
          if (v < vmin) vmin = v;
          vsum += v;
          count++;
        }
      }

      if (count == 0) {
        out_max(i, y)  = NA_REAL;
        out_min(i, y)  = NA_REAL;
        out_mean(i, y) = NA_REAL;
      } else {
        out_max(i, y)  = vmax;
        out_min(i, y)  = vmin;
        out_mean(i, y) = vsum / count;
      }
    }
  }

  return List::create(Named("nbr_max")  = out_max,
                      Named("nbr_min")  = out_min,
                      Named("nbr_mean") = out_mean);
}
')

# ---- 3. Main pipeline function ---------------------------------------------

add_all_neighbor_features <- function(cell_data, id_order, nb_obj,
                                      neighbor_source_vars) {
  # cell_data: data.frame or data.table with columns id, year, and all vars
  # id_order:  vector of cell IDs in the order matching nb_obj
  # nb_obj:    spdep::nb object (rook_neighbors_unique)
  # neighbor_source_vars: character vector of variable names

  cat("Building cell-level CSR adjacency...\n")
  csr <- build_cell_adjacency_csr(nb_obj)

  # Convert to data.table for fast reshaping
  dt <- as.data.table(cell_data)

  # Map cell id -> row index in id_order (1-based, matching nb_obj)
  id_to_pos <- setNames(seq_along(id_order), as.character(id_order))

  # Get sorted unique years
  years <- sort(unique(dt$year))
  n_years <- length(years)
  n_cells <- length(id_order)
  year_to_col <- setNames(seq_along(years), as.character(years))

  cat(sprintf("Cells: %d | Years: %d | Rows: %d\n", n_cells, n_years, nrow(dt)))

  # Pre-compute cell_pos and year_col for every row (for fast scatter/gather)
  dt[, cell_pos := id_to_pos[as.character(id)]]
  dt[, year_col := year_to_col[as.character(year)]]

  # Validate
  stopifnot(!anyNA(dt$cell_pos), !anyNA(dt$year_col))

  # Row indices for scatter back (same for all variables)
  scatter_idx <- cbind(dt$cell_pos, dt$year_col)  # matrix index into cell x year

  for (var_name in neighbor_source_vars) {
    cat(sprintf("  Processing variable: %s\n", var_name))

    # --- Pivot to cell x year matrix ---
    vals_mat <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
    vals_mat[scatter_idx] <- dt[[var_name]]

    # --- Compute neighbor stats via Rcpp ---
    stats <- neighbor_stats_matrix(vals_mat,
                                   as.integer(csr$ptr),
                                   as.integer(csr$adj))

    # --- Gather back to long format and attach ---
    max_col  <- paste0("nbr_max_", var_name)
    min_col  <- paste0("nbr_min_", var_name)
    mean_col <- paste0("nbr_mean_", var_name)

    dt[, (max_col)  := stats$nbr_max[scatter_idx]]
    dt[, (min_col)  := stats$nbr_min[scatter_idx]]
    dt[, (mean_col) := stats$nbr_mean[scatter_idx]]

    cat(sprintf("    -> Added %s, %s, %s\n", max_col, min_col, mean_col))
  }

  # Clean up helper columns
  dt[, c("cell_pos", "year_col") := NULL]

  cat("Done.\n")
  return(dt)
}

# ---- 4. Usage (drop-in replacement for the original outer loop) -------------

# Original:
#   neighbor_lookup <- build_neighbor_lookup(cell_data, id_order, rook_neighbors_unique)
#   for (var_name in neighbor_source_vars) {
#     cell_data <- compute_and_add_neighbor_features(cell_data, var_name, neighbor_lookup)
#   }

# Replacement:
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cell_data <- add_all_neighbor_features(
  cell_data        = cell_data,
  id_order         = id_order,
  nb_obj           = rook_neighbors_unique,
  neighbor_source_vars = neighbor_source_vars
)

# cell_data now has columns: nbr_max_ntl, nbr_min_ntl, nbr_mean_ntl, etc.
# Proceed with the pre-trained Random Forest prediction as before:
# predictions <- predict(rf_model, newdata = cell_data)
```

---

## Summary of Changes and Guarantees

| Aspect | Original | Optimized |
|---|---|---|
| **Topology built over** | 6.46M cell-year rows | 344K cells (once) |
| **Lookup structure** | 6.46M-element R list with string hashing | CSR (two integer vectors, ~5.5 MB) |
| **Stats computation** | R-level `lapply` over 6.46M rows Ã— 5 vars | Rcpp loop over 344K cells Ã— 28 years Ã— 5 vars |
| **Memory for lookup** | ~1â€“2 GB (list overhead) | ~11 MB (CSR) + ~77 MB per variable matrix |
| **Estimated runtime** | 86+ hours | **~2â€“5 minutes** |
| **Numerical output** | max, min, mean of non-NA neighbor values | **Identical** â€” same aggregation logic |
| **RF model** | **Untouched** â€” not retrained | **Untouched** â€” not retrained |

The key principle: **separate the static graph topology from the time-varying values**. Build the adjacency once over cells, then sweep variable matrices through it. The Rcpp kernel eliminates R interpreter overhead for the inner loop, and the matrix layout ensures cache-friendly memory access.
END_RESPONSE>>>
