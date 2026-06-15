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
run_id: Anthropic_F10_T10_R022
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F10
prompt_family_slug: SPARSE_GRAPH_FRAME
family_label: Sparse graph frame
family_group: Graph cue
replicate: 22
raw_output_file: Experiment_Output/Anthropic/Temp_1.0/output_F10_Anthropic_temp_10_rep_022.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Optimized R Code

## Diagnosis

The current implementation has three critical bottlenecks:

1. **`build_neighbor_lookup` creates ~6.46 million list entries**, each produced by an `lapply` call that performs string-pasting, hash lookups (`idx_lookup[neighbor_keys]`), and per-row NA filtering. With ~6.46M rows, this is extremely slow in pure R.

2. **`compute_neighbor_stats` iterates over ~6.46M list elements** per variable, extracting subsets of a vector by index and computing `max/min/mean`. This is called 5 times (once per variable), yielding ~32.3M list iterations total.

3. **The neighbor topology is year-invariant** (rook neighbors don't change across years), yet the lookup is built at the cell-year level, inflating the graph from ~344K nodes with ~1.37M edges to ~6.46M nodes with ~25.7M edges (1.37M Ã— 19.something average years per cell â‰ˆ massive duplication). The topology is identical per year and should be factored out.

**Root cause:** The code treats each cell-year as an independent node in a giant graph, when in reality the adjacency structure is **fixed across years**. The aggregation can be done **per year** over a much smaller ~344K-node graph, which is 28Ã— smaller.

## Optimization Strategy

1. **Build a sparse adjacency structure once** over the 344,208 cells using a CSR (Compressed Sparse Row) representation â€” specifically, a `dgRMatrix` or equivalent integer vectors â€” from `rook_neighbors_unique`.

2. **Split data by year** (28 groups of ~230Kâ€“344K rows each).

3. **Per year, perform sparse matrixâ€“vector multiplication** to compute neighbor sums and counts, then derive mean. Use element-wise operations with the adjacency matrix for max/min (via `igraph` or custom C++ via `Rcpp`, or clever sparse matrix tricks).

4. **For max and min:** Use an `Rcpp` function that iterates over CSR row pointers â€” this converts the ~6.46M Ã— 5 Ã— 3 R-level list operations into a tight C++ loop.

5. **Bind results** back into the original `data.frame`/`data.table`.

This reduces runtime from 86+ hours to **minutes** (estimated 2â€“10 minutes depending on I/O).

## Optimized R Code

```r
# ============================================================
# OPTIMIZED NEIGHBOR AGGREGATION PIPELINE
# Preserves numerical equivalence with original implementation.
# ============================================================

library(data.table)
library(Matrix)
library(Rcpp)

# ----------------------------------------------------------
# Step 0: Rcpp workhorse for sparse row-wise max, min, mean
# ----------------------------------------------------------
Rcpp::sourceCpp(code = '
#include <Rcpp.h>
#include <cmath>
#include <limits>
using namespace Rcpp;

// [[Rcpp::export]]
NumericMatrix sparse_neighbor_stats_cpp(
    IntegerVector row_ptr,   // length n+1, 0-based CSR row pointers
    IntegerVector col_idx,   // 0-based column indices
    NumericVector vals,      // attribute values aligned to node order (length n)
    int n                    // number of nodes
) {
  // Output: n x 3 matrix [max, min, mean]
  NumericMatrix out(n, 3);

  for (int i = 0; i < n; i++) {
    int start = row_ptr[i];
    int end   = row_ptr[i + 1];

    // Count valid (non-NA) neighbor values
    double vmax = -std::numeric_limits<double>::infinity();
    double vmin =  std::numeric_limits<double>::infinity();
    double vsum = 0.0;
    int    cnt  = 0;

    for (int j = start; j < end; j++) {
      double v = vals[ col_idx[j] ];
      if (!R_IsNA(v)) {
        if (v > vmax) vmax = v;
        if (v < vmin) vmin = v;
        vsum += v;
        cnt++;
      }
    }

    if (cnt == 0) {
      out(i, 0) = NA_REAL;
      out(i, 1) = NA_REAL;
      out(i, 2) = NA_REAL;
    } else {
      out(i, 0) = vmax;
      out(i, 1) = vmin;
      out(i, 2) = vsum / cnt;
    }
  }

  return out;
}
')

# ----------------------------------------------------------
# Step 1: Build CSR adjacency ONCE from the nb object
# ----------------------------------------------------------
build_csr_from_nb <- function(nb_obj, id_order) {
  # nb_obj: spdep nb object (list of integer vectors, 1-based,
  #         0 means no neighbors)
  # id_order: vector of cell IDs corresponding to nb_obj indices
  # Returns: list(row_ptr, col_idx, id_order, id_to_pos)
  
  n <- length(nb_obj)
  stopifnot(n == length(id_order))
  
  # Flatten to CSR
  # row_ptr: integer vector of length n+1 (0-based cumulative counts)
  # col_idx: integer vector of total edges (0-based neighbor positions)
  
  # Pre-count
  degrees <- vapply(nb_obj, function(x) {
    if (length(x) == 1L && x[1] == 0L) 0L else length(x)
  }, integer(1))
  
  row_ptr <- c(0L, cumsum(degrees))
  total_edges <- row_ptr[n + 1L]
  col_idx <- integer(total_edges)
  
  pos <- 1L
  for (i in seq_len(n)) {
    nbrs <- nb_obj[[i]]
    if (!(length(nbrs) == 1L && nbrs[1] == 0L)) {
      k <- length(nbrs)
      col_idx[pos:(pos + k - 1L)] <- nbrs - 1L   # 0-based
      pos <- pos + k
    }
  }
  
  # id_to_pos: named integer vector mapping cell ID -> 1-based position
  id_to_pos <- setNames(seq_len(n), as.character(id_order))
  
  list(
    row_ptr   = row_ptr,
    col_idx   = col_idx,
    n         = n,
    id_order  = id_order,
    id_to_pos = id_to_pos
  )
}

# ----------------------------------------------------------
# Step 2: Main pipeline function
# ----------------------------------------------------------
run_neighbor_aggregation <- function(cell_data, 
                                      rook_neighbors_unique, 
                                      id_order,
                                      neighbor_source_vars,
                                      rf_model = NULL) {
  # Convert to data.table for speed (non-destructive copy)
  if (!is.data.table(cell_data)) {
    cell_data <- as.data.table(cell_data)
  }
  
  cat("Building CSR adjacency structure...\n")
  csr <- build_csr_from_nb(rook_neighbors_unique, id_order)
  
  # Ensure cell_data has a position column aligned to CSR
  # Map each row's cell id to its CSR position (1-based)
  cell_data[, .csr_pos := csr$id_to_pos[as.character(id)]]
  
  # Pre-allocate output columns
  for (var_name in neighbor_source_vars) {
    max_col  <- paste0(var_name, "_max_neighbor")
    min_col  <- paste0(var_name, "_min_neighbor")
    mean_col <- paste0(var_name, "_mean_neighbor")
    set(cell_data, j = max_col,  value = rep(NA_real_, nrow(cell_data)))
    set(cell_data, j = min_col,  value = rep(NA_real_, nrow(cell_data)))
    set(cell_data, j = mean_col, value = rep(NA_real_, nrow(cell_data)))
  }
  
  # Process year by year
  years <- sort(unique(cell_data$year))
  cat(sprintf("Processing %d years x %d variables...\n", 
              length(years), length(neighbor_source_vars)))
  
  for (yr in years) {
    # Row indices in cell_data for this year
    yr_rows <- which(cell_data$year == yr)
    
    # Build a mapping: csr_position -> row index in cell_data for this year
    # Not all 344K cells may be present every year, so we handle that.
    yr_csr_pos <- cell_data$.csr_pos[yr_rows]
    
    for (var_name in neighbor_source_vars) {
      # Build a full-length attribute vector aligned to CSR node order
      # Nodes not present this year get NA (their neighbors will skip them)
      attr_vec <- rep(NA_real_, csr$n)
      attr_vec[yr_csr_pos] <- cell_data[[var_name]][yr_rows]
      
      # Compute stats via Rcpp (returns n x 3 matrix: max, min, mean)
      stats <- sparse_neighbor_stats_cpp(
        csr$row_ptr, csr$col_idx, attr_vec, csr$n
      )
      
      # Write back only for nodes present this year
      max_col  <- paste0(var_name, "_max_neighbor")
      min_col  <- paste0(var_name, "_min_neighbor")
      mean_col <- paste0(var_name, "_mean_neighbor")
      
      set(cell_data, i = yr_rows, j = max_col,  value = stats[yr_csr_pos, 1])
      set(cell_data, i = yr_rows, j = min_col,  value = stats[yr_csr_pos, 2])
      set(cell_data, i = yr_rows, j = mean_col, value = stats[yr_csr_pos, 3])
    }
    
    cat(sprintf("  Year %d done.\n", yr))
  }
  
  # Clean up temp column
  cell_data[, .csr_pos := NULL]
  
  cat("Neighbor aggregation complete.\n")
  return(cell_data)
}

# ----------------------------------------------------------
# Step 3: Execute
# ----------------------------------------------------------

# Define source variables
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# Run the optimized pipeline
cell_data <- run_neighbor_aggregation(
  cell_data              = cell_data,
  rook_neighbors_unique  = rook_neighbors_unique,
  id_order               = id_order,
  neighbor_source_vars   = neighbor_source_vars
)

# ----------------------------------------------------------
# Step 4: Predict with the pre-trained Random Forest
# (Model is NOT retrained â€” used as-is)
# ----------------------------------------------------------
# rf_model was previously loaded, e.g.:
#   rf_model <- readRDS("path/to/trained_rf_model.rds")
#
# Ensure column names match what the model expects:
cell_data$predicted <- predict(rf_model, newdata = cell_data)
```

## Summary of Gains

| Aspect | Original | Optimized |
|---|---|---|
| **Neighbor lookup** | 6.46M string-paste + hash lookups in R | CSR built once over 344K nodes (seconds) |
| **Per-variable aggregation** | 6.46M R-level list iterations Ã— 5 vars | 28 years Ã— 344K nodes in C++ Ã— 5 vars |
| **Total R-level iterations** | ~32.3M `lapply` calls + ~6.46M for lookup | ~140 C++ calls (28 Ã— 5), each a tight loop |
| **Memory** | ~6.46M-element list of integer vectors | Two integer vectors (CSR), ~11 MB |
| **Estimated runtime** | 86+ hours | **2â€“10 minutes** |
| **Numerical equivalence** | Baseline | âœ… Identical max/min/mean, same NA handling |
| **RF model** | Pre-trained | âœ… Unchanged, predict-only |

**Key insight:** The adjacency graph is static across time. By separating topology (344K nodes, 1.37M edges) from temporal attributes (28 yearly snapshots), and using compiled C++ for the inner loop over CSR neighbors, we eliminate ~99.97% of the R-interpreter overhead.
END_RESPONSE>>>
