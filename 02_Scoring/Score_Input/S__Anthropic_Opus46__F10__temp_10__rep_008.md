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
run_id: Anthropic_F10_T10_R008
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F10
prompt_family_slug: SPARSE_GRAPH_FRAME
family_label: Sparse graph frame
family_group: Graph cue
replicate: 8
raw_output_file: Experiment_Output/Anthropic/Temp_1.0/output_F10_Anthropic_temp_10_rep_008.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Optimized R Code

## Diagnosis

The current implementation has three critical bottlenecks:

1. **`build_neighbor_lookup` creates ~6.46 million list entries**, each requiring string-pasting, dictionary lookups, and NA filtering. The `paste(id, year)` key construction and named-vector lookup (`idx_lookup[neighbor_keys]`) is O(n) string hashing over millions of keys. This is the dominant setup cost.

2. **`compute_neighbor_stats` iterates over 6.46M list elements in a pure-R `lapply`**, extracting subsets of a numeric vector per node. This is called 5 times (once per variable), totaling ~32.3 million R-level list iterations with subsetting.

3. **The neighbor topology is year-invariant but the lookup is rebuilt entangling topology and time.** Every cell has the same rook neighbors every year, yet the lookup embeds year-specific row indices, forcing a full rebuild if the data is reordered and preventing vectorized year-parallel operations.

**Root cause:** The implementation treats the problem as a generic graph with arbitrary node identities, when in reality the spatial topology is fixed and the temporal dimension is uniform â€” a structure that permits a sparse-matrix formulation with vectorized aggregation.

---

## Optimization Strategy

1. **Build a sparse adjacency matrix `W` (344,208 Ã— 344,208) once** from `rook_neighbors_unique`. This encodes the directed graph topology. With ~1.37M edges, it's extremely sparse (<0.001% fill).

2. **Reshape each variable into a dense matrix `V` of shape (344,208 cells Ã— 28 years)** where rows are cells (in `id_order`) and columns are years.

3. **Compute neighbor aggregates via sparse matrixâ€“dense matrix multiplication:**
   - `W %*% V` gives neighbor sums.
   - `W %*% (V != NA)` gives neighbor counts (with NA handling).
   - Neighbor mean = sum / count.
   - For max and min: use a custom sparse-row-sweep approach, iterating over the CSR structure in C++ (via `Rcpp`) or use a grouped-max/min strategy.

4. **Avoid all string operations, all per-row `lapply`, and all named-vector lookups.** Everything becomes matrix algebra or compiled C++ loops over sparse structure.

5. **Memory:** The sparse matrix W is ~1.37M entries Ã— 12 bytes â‰ˆ 16 MB. Each dense matrix V is 344,208 Ã— 28 Ã— 8 bytes â‰ˆ 77 MB. Peak memory well within 16 GB.

6. **The Random Forest model is never retouched.** We only recompute the same 15 neighbor features (5 vars Ã— 3 stats) to numerical equivalence, then predict.

---

## Optimized R Code

```r
# =============================================================================
# OPTIMIZED NEIGHBOR AGGREGATION PIPELINE
# Sparse graph topology + dense year-matrix + Rcpp row-wise extrema
# =============================================================================

library(Matrix)
library(data.table)
library(Rcpp)

# â”€â”€ Step 0: Rcpp helper for row-wise max, min over sparse structure â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# This avoids materializing a dense neighbor-value matrix for extrema.

cppFunction('
#include <Rcpp.h>
using namespace Rcpp;

// Computes row-wise max, min, mean of X[neighbors, col] for each row in a CSC sparse matrix.
// W is the adjacency matrix in dgCMatrix (CSC) format.
// We transpose W to get CSC of W^T = CSR of W, so columns of W^T are rows of W.
// Input: Wt (transpose of W, dgCMatrix), Vmat (dense ncells x nyears), na_val (e.g. NaN)
// Output: list of 3 matrices (max, min, mean) each ncells x nyears.

// [[Rcpp::export]]
List neighbor_stats_sparse(IntegerVector Wt_p, IntegerVector Wt_i,
                           NumericMatrix Vmat, int ncells, int nyears) {
  // Wt is the transpose of W in CSC: column j of Wt = row j of W
  // Wt_p: column pointers (length ncells+1)
  // Wt_i: row indices of nonzeros (= neighbor cell indices)

  NumericMatrix out_max(ncells, nyears);
  NumericMatrix out_min(ncells, nyears);
  NumericMatrix out_mean(ncells, nyears);

  for (int j = 0; j < ncells; j++) {
    int start = Wt_p[j];
    int end   = Wt_p[j + 1];
    int n_neighbors = end - start;

    if (n_neighbors == 0) {
      for (int t = 0; t < nyears; t++) {
        out_max(j, t)  = NA_REAL;
        out_min(j, t)  = NA_REAL;
        out_mean(j, t) = NA_REAL;
      }
      continue;
    }

    for (int t = 0; t < nyears; t++) {
      double vmax = R_NegInf;
      double vmin = R_PosInf;
      double vsum = 0.0;
      int    cnt  = 0;

      for (int k = start; k < end; k++) {
        int neighbor = Wt_i[k];  // neighbor cell index (0-based)
        double val = Vmat(neighbor, t);
        if (!R_IsNA(val) && !ISNAN(val)) {
          if (val > vmax) vmax = val;
          if (val < vmin) vmin = val;
          vsum += val;
          cnt++;
        }
      }

      if (cnt == 0) {
        out_max(j, t)  = NA_REAL;
        out_min(j, t)  = NA_REAL;
        out_mean(j, t) = NA_REAL;
      } else {
        out_max(j, t)  = vmax;
        out_min(j, t)  = vmin;
        out_mean(j, t) = vsum / (double)cnt;
      }
    }
  }

  return List::create(Named("max")  = out_max,
                      Named("min")  = out_min,
                      Named("mean") = out_mean);
}
')

# â”€â”€ Step 1: Build sparse adjacency matrix W from nb object â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

build_adjacency_matrix <- function(nb_obj, id_order) {
  # nb_obj: spdep nb object (list of integer vectors, 0 = no neighbors)
  # id_order: vector of cell IDs corresponding to positions in nb_obj
  n <- length(nb_obj)
  stopifnot(n == length(id_order))

  # Build COO triplets
  from_idx <- integer(0)
  to_idx   <- integer(0)

  for (i in seq_len(n)) {
    nb <- nb_obj[[i]]
    # spdep nb: integer(0) or 0L means no neighbors
    nb <- nb[nb > 0L]
    if (length(nb) > 0) {
      from_idx <- c(from_idx, rep(i, length(nb)))
      to_idx   <- c(to_idx, nb)
    }
  }

  # Build sparse matrix (1-indexed): W[i, j] = 1 means j is a neighbor of i
  W <- sparseMatrix(
    i    = from_idx,
    j    = to_idx,
    x    = 1,
    dims = c(n, n),
    repr = "C"   # CSC format
  )
  return(W)
}

# â”€â”€ Step 2: Reshape panel data into cell Ã— year matrix â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

reshape_to_matrix <- function(dt, var_name, cell_idx_map, year_idx_map) {
  # dt: data.table with columns id, year, <var_name>
  # cell_idx_map: named integer vector, names=cell IDs, values=1..ncells
  # year_idx_map: named integer vector, names=years, values=1..nyears
  # Returns: ncells x nyears matrix

  ncells <- length(cell_idx_map)
  nyears <- length(year_idx_map)
  mat <- matrix(NA_real_, nrow = ncells, ncol = nyears)

  row_i <- cell_idx_map[as.character(dt$id)]
  col_j <- year_idx_map[as.character(dt$year)]

  valid <- !is.na(row_i) & !is.na(col_j)
  mat[cbind(row_i[valid], col_j[valid])] <- dt[[var_name]][valid]

  return(mat)
}

# â”€â”€ Step 3: Flatten result matrices back to panel column â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

flatten_matrix_to_panel <- function(mat, dt, cell_idx_map, year_idx_map) {
  row_i <- cell_idx_map[as.character(dt$id)]
  col_j <- year_idx_map[as.character(dt$year)]
  valid <- !is.na(row_i) & !is.na(col_j)
  out <- rep(NA_real_, nrow(dt))
  out[valid] <- mat[cbind(row_i[valid], col_j[valid])]
  return(out)
}

# =============================================================================
# MAIN PIPELINE
# =============================================================================

run_optimized_pipeline <- function(cell_data, id_order, rook_neighbors_unique,
                                   rf_model) {
  # Convert to data.table for speed
  if (!is.data.table(cell_data)) {
    cell_data <- as.data.table(cell_data)
  }

  cat("Step 1: Building sparse adjacency matrix...\n")
  W <- build_adjacency_matrix(rook_neighbors_unique, id_order)
  # Transpose W: columns of Wt correspond to rows of W (i.e., neighbors of each node)
  Wt <- t(W)
  # Ensure dgCMatrix (CSC)
  Wt <- as(Wt, "dgCMatrix")

  ncells <- length(id_order)
  years  <- sort(unique(cell_data$year))
  nyears <- length(years)

  cat(sprintf("  Cells: %d, Years: %d, Edges: %d\n",
              ncells, nyears, length(W@x)))

  # Index maps
  cell_idx_map <- setNames(seq_along(id_order), as.character(id_order))
  year_idx_map <- setNames(seq_along(years), as.character(years))

  # Extract CSC components of Wt (0-indexed for Rcpp)
  Wt_p <- Wt@p
  Wt_i <- Wt@i  # already 0-indexed in dgCMatrix

  neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

  cat("Step 2: Computing neighbor statistics...\n")

  for (var_name in neighbor_source_vars) {
    cat(sprintf("  Processing: %s\n", var_name))

    # Reshape to cell x year matrix
    Vmat <- reshape_to_matrix(cell_data, var_name, cell_idx_map, year_idx_map)

    # Compute all three stats in one C++ pass over sparse structure
    stats <- neighbor_stats_sparse(Wt_p, Wt_i, Vmat, ncells, nyears)

    # Flatten back to panel and assign columns
    # Column naming convention: match original compute_and_add_neighbor_features
    max_col  <- paste0(var_name, "_neighbor_max")
    min_col  <- paste0(var_name, "_neighbor_min")
    mean_col <- paste0(var_name, "_neighbor_mean")

    cell_data[, (max_col)  := flatten_matrix_to_panel(stats$max,  cell_data,
                                                       cell_idx_map, year_idx_map)]
    cell_data[, (min_col)  := flatten_matrix_to_panel(stats$min,  cell_data,
                                                       cell_idx_map, year_idx_map)]
    cell_data[, (mean_col) := flatten_matrix_to_panel(stats$mean, cell_data,
                                                       cell_idx_map, year_idx_map)]

    # Free intermediate memory
    rm(Vmat, stats); gc(verbose = FALSE)
  }

  cat("Step 3: Generating predictions with pre-trained Random Forest...\n")
  # Predict using the existing trained model â€” model is NOT retrained
  cell_data$prediction <- predict(rf_model, newdata = cell_data)

  cat("Done.\n")
  return(cell_data)
}

# =============================================================================
# USAGE (plug into existing pipeline)
# =============================================================================
#
# # Load pre-existing objects
# load("cell_data.RData")            # cell_data data.frame/data.table
# load("id_order.RData")             # id_order vector
# load("rook_neighbors.RData")       # rook_neighbors_unique (spdep nb object)
# load("rf_model.RData")             # trained Random Forest model
#
# result <- run_optimized_pipeline(
#   cell_data              = cell_data,
#   id_order               = id_order,
#   rook_neighbors_unique  = rook_neighbors_unique,
#   rf_model               = rf_model
# )
```

---

## Performance Analysis

| Aspect | Original | Optimized |
|---|---|---|
| **Topology build** | ~6.46M string pastes + named-vector hash lookups | One-time sparse matrix construction (~1.37M entries) |
| **Per-variable aggregation** | 6.46M R-level `lapply` iterations Ã— 5 vars | Single C++ pass over CSR structure Ã— 5 vars (344K rows Ã— 28 cols) |
| **Max/Min computation** | R-level `max()`/`min()` per node-year | In-line C++ comparison in tight loop |
| **String operations** | ~19.4M `paste()` calls in setup | Zero |
| **Memory pattern** | 6.46M-element nested list (heavy GC pressure) | Dense matrices ~77 MB each + sparse matrix ~16 MB |
| **Expected runtime** | 86+ hours | **~5â€“15 minutes** |

**Why numerical equivalence is preserved:** The C++ kernel computes `max`, `min`, and `sum/count` over exactly the same neighbor sets (defined by the same `rook_neighbors_unique` nb object) with the same NA-exclusion logic (`!is.na && !isnan`). The arithmetic is identical â€” IEEE 754 double-precision max, min, and mean. The adjacency matrix encodes the same directed edges. The year-matching is implicit via column alignment (each column = one year), eliminating key-mismatch bugs. The Random Forest model object is passed through and used only for `predict()`, never modified.
END_RESPONSE>>>
