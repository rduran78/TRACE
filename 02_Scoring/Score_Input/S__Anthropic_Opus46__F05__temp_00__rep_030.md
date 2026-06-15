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
run_id: Anthropic_F05_T00_R030
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F05
prompt_family_slug: STRING_KEY_REDUNDANCY
family_label: String-key probe
family_group: Line-level probe
replicate: 30
raw_output_file: Experiment_Output/Anthropic/Temp_0.0/output_F05_Anthropic_temp_00_rep_030.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis and Optimized Solution

## Diagnosis

The repeated string-key work is **a symptom of a larger repeated-lookup pattern**, not merely a local inefficiency. Here's the full picture:

### The Core Problem: O(N) `lapply` with String Operations Inside the Loop

`build_neighbor_lookup` iterates over all ~6.46 million rows. For **each row**, it:

1. Converts `data$id[i]` to character and looks it up in `id_to_ref` â€” **6.46M character coercions + named-vector lookups**.
2. Subsets `id_order[neighbors[[ref_idx]]]` â€” fine, but done row-by-row.
3. Calls `paste(neighbor_cell_ids, data$year[i], sep = "_")` â€” **allocates a new character vector per row** (~4 neighbor keys Ã— 6.46M rows â‰ˆ 25.8M string constructions).
4. Looks up `idx_lookup[neighbor_keys]` â€” **named-vector lookup via string matching on a 6.46M-element named vector**, repeated 6.46M times.

The `idx_lookup` named vector has 6.46M entries. R's named-vector lookup is **O(n)** per query (linear scan or hashed, but with 6.46M names the hash construction is expensive and each probe still has string-comparison overhead). Doing this 6.46M times is **O(NÂ²)** in string-comparison work â€” this is the dominant bottleneck and the reason the pipeline takes 86+ hours.

### The Broader Pattern

The neighbor lookup is **year-invariant**: the spatial neighbor structure doesn't change across years. Yet the code re-discovers "which rows are my neighbors in my year" from scratch for every single row by string-keying `(cell_id, year)`. This is fundamentally redundant â€” the same spatial neighbor structure is repeated 28 times (once per year).

Then `compute_neighbor_stats` is called 5 times (once per variable), each time iterating over the 6.46M-element `neighbor_lookup` list. This is fine in principle but the list-of-integer-vectors representation forces R into slow `lapply` iteration.

## Optimization Strategy

### Key Insight: Separate Space from Time

Since the neighbor structure is purely spatial and the panel is balanced (every cell appears in every year), we can:

1. **Build the neighbor lookup once in cell-space** (344K cells), not in cell-year-space (6.46M rows).
2. **For each year**, extract the variable column, compute neighbor stats using vectorized/matrix operations, and write results back.
3. **Eliminate all string operations entirely** â€” use integer indexing throughout.

### Algorithmic Reformulation

Instead of a 6.46M-element list of neighbor row indices, build:
- A **sparse adjacency structure** (CSR-style: two integer vectors) over the 344K cells.
- For each year Ã— variable combination, subset the variable vector for that year, then use the sparse structure to compute neighbor max/min/mean in a **single vectorized pass** (or via `data.table` grouped operations, or via sparse matrix multiplication for means).

This reduces the work from ~6.46M Ã— (string ops + hash lookups) to 28 Ã— (344K vectorized integer-index operations).

### Complexity Comparison

| | Original | Optimized |
|---|---|---|
| String constructions | ~25.8M | **0** |
| Named-vector lookups on 6.46M keys | ~6.46M | **0** |
| Loop iterations for neighbor lookup | 6.46M | 28 Ã— 344K (same total but vectorized) |
| `compute_neighbor_stats` iterations | 5 Ã— 6.46M (list-lapply) | 5 Ã— 28 Ã— 344K (vectorized) |
| Estimated time | 86+ hours | **~2â€“10 minutes** |

## Working R Code

```r
# =============================================================================
# OPTIMIZED NEIGHBOR FEATURE CONSTRUCTION
# =============================================================================
# Preserves the exact numerical estimand: for each cell-year row, compute
# max, min, and mean of each neighbor source variable across rook neighbors
# present in the same year.
#
# Assumptions (matching the original pipeline):
#   - cell_data is a data.frame with columns: id, year, and the source vars
#   - id_order is the vector of cell IDs in the order matching rook_neighbors_unique
#   - rook_neighbors_unique is an nb object (list of integer index vectors)
#   - The panel is balanced: every cell in id_order appears in every year
# =============================================================================

library(data.table)

build_neighbor_features_optimized <- function(cell_data,
                                               id_order,
                                               rook_neighbors_unique,
                                               neighbor_source_vars) {

  # --------------------------------------------------------------------------
  # Step 1: Convert to data.table for fast grouped operations
  # --------------------------------------------------------------------------
  dt <- as.data.table(cell_data)

  # Ensure id and year are the types we expect

  dt[, id := as.integer(id)]
  dt[, year := as.integer(year)]

  # --------------------------------------------------------------------------
  # Step 2: Build cell-level integer mapping
  # --------------------------------------------------------------------------
  # Map each cell id to its index in id_order (1-based position in the nb object)
  n_cells <- length(id_order)
  id_to_ref <- integer(max(id_order))
  id_to_ref[id_order] <- seq_len(n_cells)
  # If id_order values are not contiguous or are very large, use a hash instead:
  # id_to_ref_env <- new.env(hash = TRUE, size = n_cells)
  # for (k in seq_len(n_cells)) id_to_ref_env[[as.character(id_order[k])]] <- k

  # --------------------------------------------------------------------------
  # Step 3: Build CSR (Compressed Sparse Row) representation of neighbor graph
  # --------------------------------------------------------------------------
  # For each cell index i in 1:n_cells, neighbors[[i]] gives the neighbor
  # indices (into id_order). We need to map these to cell IDs, then later
  # to within-year row positions.
  #
  # But since the panel is balanced and we'll process year-by-year, we need
  # the neighbor structure in terms of cell-index (position in id_order).
  # The nb object already provides this.

  # Flatten the nb list into CSR vectors
  nb_lengths <- lengths(rook_neighbors_unique)
  nb_ptr     <- c(0L, cumsum(nb_lengths))  # length n_cells + 1
  nb_idx     <- unlist(rook_neighbors_unique, use.names = FALSE)  # neighbor cell-indices
  # Handle nb objects where 0 means "no neighbors"
  # spdep::nb uses integer(0) for no neighbors, but just in case:
  # nb_idx[nb_idx == 0L] <- NA_integer_  # shouldn't be needed with proper nb

  # --------------------------------------------------------------------------
  # Step 4: Create a cell-index column in dt for fast alignment
  # --------------------------------------------------------------------------
  # We need each row to know its position in the id_order vector
  dt[, cell_idx := id_to_ref[id]]

  # Sort by year and cell_idx so that within each year, rows are in cell_idx order

  setkey(dt, year, cell_idx)

  # Verify balanced panel
  years <- sort(unique(dt$year))
  n_years <- length(years)
  stopifnot(nrow(dt) == n_cells * n_years)

  # --------------------------------------------------------------------------
  # Step 5: For each year, the rows are now in cell_idx order (1..n_cells).
  #         We can use the CSR neighbor structure directly with integer indexing.
  # --------------------------------------------------------------------------

  # Pre-allocate output columns
  for (var_name in neighbor_source_vars) {
    col_max  <- paste0("neighbor_max_", var_name)
    col_min  <- paste0("neighbor_min_", var_name)
    col_mean <- paste0("neighbor_mean_", var_name)
    dt[, (col_max)  := NA_real_]
    dt[, (col_min)  := NA_real_]
    dt[, (col_mean) := NA_real_]
  }

  # --------------------------------------------------------------------------
  # Step 6: Vectorized neighbor stat computation using C-style loop via Rcpp
  #         or pure-R vectorized approach
  # --------------------------------------------------------------------------
  # Pure R approach: for each year, extract the variable as a vector aligned
  # to cell_idx, then compute neighbor stats using the CSR structure.
  #
  # We use a compiled inner loop for speed. If Rcpp is not available, we fall
  # back to a vectorized R approach.

  # ------ Try Rcpp approach first (much faster) ------
  use_rcpp <- requireNamespace("Rcpp", quietly = TRUE)

  if (use_rcpp) {
    Rcpp::sourceCpp(code = '
    #include <Rcpp.h>
    using namespace Rcpp;

    // [[Rcpp::export]]
    NumericMatrix neighbor_stats_csr(NumericVector vals,
                                     IntegerVector nb_ptr,
                                     IntegerVector nb_idx) {
      int n = vals.size();
      NumericMatrix out(n, 3); // columns: max, min, mean

      for (int i = 0; i < n; i++) {
        int start = nb_ptr[i];
        int end   = nb_ptr[i + 1];
        if (start == end) {
          out(i, 0) = NA_REAL;
          out(i, 1) = NA_REAL;
          out(i, 2) = NA_REAL;
          continue;
        }
        double vmax = R_NegInf;
        double vmin = R_PosInf;
        double vsum = 0.0;
        int    cnt  = 0;
        for (int j = start; j < end; j++) {
          int idx = nb_idx[j] - 1; // R to C indexing
          double v = vals[idx];
          if (!NumericVector::is_na(v)) {
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
    compute_stats <- function(vals_vec) {
      neighbor_stats_csr(vals_vec, nb_ptr, nb_idx)
    }
  } else {
    # ------ Pure R fallback (still much faster than original) ------
    # Vectorized using the CSR structure: expand neighbor values, group, summarize
    # Build a data.table of (cell_idx, neighbor_cell_idx) pairs
    from_cell <- rep(seq_len(n_cells), times = nb_lengths)
    to_cell   <- nb_idx
    edge_dt   <- data.table(from = from_cell, to = to_cell)

    compute_stats <- function(vals_vec) {
      edge_dt[, val := vals_vec[to]]
      stats <- edge_dt[!is.na(val),
                       .(vmax = max(val), vmin = min(val), vmean = mean(val)),
                       by = from]
      # Align back to 1:n_cells
      out <- matrix(NA_real_, nrow = n_cells, ncol = 3)
      out[stats$from, 1] <- stats$vmax
      out[stats$from, 2] <- stats$vmin
      out[stats$from, 3] <- stats$vmean
      out
    }
  }

  # --------------------------------------------------------------------------
  # Step 7: Iterate over years and variables
  # --------------------------------------------------------------------------
  cat("Computing neighbor features for", n_years, "years x",
      length(neighbor_source_vars), "variables\n")

  for (yr in years) {
    # Row indices for this year (dt is keyed by year, cell_idx)
    yr_rows <- which(dt$year == yr)
    # Since dt is sorted by (year, cell_idx), yr_rows should be contiguous
    # and in cell_idx order 1..n_cells
    stopifnot(length(yr_rows) == n_cells)

    for (var_name in neighbor_source_vars) {
      # Extract values in cell_idx order for this year
      vals_vec <- dt[[var_name]][yr_rows]

      # Compute neighbor stats: n_cells x 3 matrix (max, min, mean)
      stats_mat <- compute_stats(vals_vec)

      # Write back
      col_max  <- paste0("neighbor_max_", var_name)
      col_min  <- paste0("neighbor_min_", var_name)
      col_mean <- paste0("neighbor_mean_", var_name)

      set(dt, i = yr_rows, j = col_max,  value = stats_mat[, 1])
      set(dt, i = yr_rows, j = col_min,  value = stats_mat[, 2])
      set(dt, i = yr_rows, j = col_mean, value = stats_mat[, 3])
    }

    if (yr %% 5 == 0) cat("  Completed year", yr, "\n")
  }

  # --------------------------------------------------------------------------
  # Step 8: Restore original row order and return as data.frame
  # --------------------------------------------------------------------------
  # We need to return rows in the same order as the input cell_data
  # Add an original row index before sorting
  # Actually, we should have saved the original order. Let's fix this:
  # We'll merge back by (id, year) or restore order via a saved index.

  # Remove helper column
  dt[, cell_idx := NULL]

  # Restore original row order: match back to input
  # Create a key for the original data
  orig_key <- paste(cell_data$id, cell_data$year, sep = "_")
  new_key  <- paste(dt$id, dt$year, sep = "_")
  reorder  <- match(orig_key, new_key)
  dt <- dt[reorder]

  return(as.data.frame(dt))
}

# =============================================================================
# USAGE (drop-in replacement for the original pipeline)
# =============================================================================
# Instead of:
#   neighbor_lookup <- build_neighbor_lookup(cell_data, id_order, rook_neighbors_unique)
#   for (var_name in neighbor_source_vars) {
#     cell_data <- compute_and_add_neighbor_features(cell_data, var_name, neighbor_lookup)
#   }
#
# Use:
#   neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
#   cell_data <- build_neighbor_features_optimized(
#     cell_data, id_order, rook_neighbors_unique, neighbor_source_vars
#   )
#
# The trained Random Forest model is not affected â€” the output columns have
# the same names and identical numerical values as the original implementation.
# =============================================================================
```

## Summary of Changes

| Aspect | Original | Optimized |
|---|---|---|
| **Data structure** | 6.46M-element named character vector | CSR integer arrays (two vectors, ~2.7M integers) |
| **Lookup method** | String paste + named-vector hash probe per row | Direct integer indexing |
| **Loop granularity** | 6.46M R-level iterations in `lapply` | 28 years Ã— 5 vars = 140 calls to compiled C++ |
| **Memory** | 6.46M-element list of integer vectors (~500MB+) | 3 integer/double vectors (~30MB) |
| **Time complexity** | O(N Ã— k Ã— string_len) where N=6.46M | O(N_year Ã— k) with integer ops, N_year=344K |
| **Estimated runtime** | 86+ hours | **2â€“10 minutes** |
| **Numerical output** | max, min, mean of non-NA neighbor values | **Identical** â€” same estimand, same column names |
| **RF model impact** | N/A | **None** â€” feature columns are unchanged |

The key algorithmic insight is that **the spatial neighbor graph is time-invariant**, so it should be built once over 344K cells (not discovered per-row over 6.46M rows), and the per-year computation should be a vectorized pass over a compact integer structure rather than millions of string-keyed lookups.
END_RESPONSE>>>
