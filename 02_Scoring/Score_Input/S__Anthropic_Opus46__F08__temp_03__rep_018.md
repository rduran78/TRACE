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
run_id: Anthropic_F08_T03_R018
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.3
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F08
prompt_family_slug: TOPOLOGY_INVARIANCE
family_label: Topology invariance
family_group: Topology cue
replicate: 18
raw_output_file: Experiment_Output/Anthropic/Temp_0.3/output_F08_Anthropic_temp_03_rep_018.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The current implementation is catastrophically slow (~86+ hours) because of a fundamental design flaw: **it conflates static topology with dynamic data**.

### Root Cause Analysis

1. **`build_neighbor_lookup` rebuilds per-row (cellÃ—year) neighbor indices.** It produces ~6.46 million list entries, each requiring string-pasting, hash lookups, and NA filtering. But the neighbor *topology* is identical across all 28 years â€” cell A's neighbors are the same in 1992 as in 2019. This means 27/28 â‰ˆ 96.4% of the work is redundant.

2. **`compute_neighbor_stats` iterates over ~6.46 million rows with `lapply`.** Each call indexes into a values vector and computes max/min/mean. This is an R-level loop over millions of elements â€” inherently slow.

3. **String-key lookups (`paste` + named vector indexing)** are used to map cell IDs to row positions. This is O(n) in construction and O(1) amortized in lookup, but the constant factor for 6.46M string keys is enormous.

4. **The outer loop repeats this for 5 variables**, multiplying the cost Ã—5.

### Key Insight

The neighbor relationship is a **static graph property** of the 344,208 cells. The variable values are **dynamic** (change by year). The correct design is:

- **Build the neighbor lookup once over cells (344,208 entries), not over cell-years (6.46M entries).**
- **Compute neighbor stats per year**, slicing the data by year and using the static cell-level neighbor index to gather values.
- **Vectorize** the gather-and-aggregate step using matrix operations instead of R-level `lapply`.

---

## Optimization Strategy

### 1. Static Neighbor Index (build once, 344K entries)

Build a single list mapping each cell's positional index (1..344,208) to its neighbors' positional indices. This uses `rook_neighbors_unique` directly â€” it's already in this form (an `nb` object). No string pasting, no year dimension.

### 2. Year-Sliced Vectorized Aggregation

For each year:
- Extract the variable column as a vector aligned to the cell ordering.
- Use the static neighbor list to gather neighbor values.
- Compute max, min, mean in a vectorized or compiled manner.

### 3. Sparse Matrix Trick for Mean (and adaptable for Min/Max)

The neighbor topology can be encoded as a **sparse adjacency matrix** W (344,208 Ã— 344,208). Then:
- **Neighbor mean** = `(W %*% x) / (W %*% ones)` â€” a single sparse matrix-vector multiply per variable per year.
- **Neighbor max/min** â€” use a compiled C++ loop via `Rcpp`, or use the sparse structure with row-wise operations.

This reduces the core computation from ~6.46M R-level list iterations to 28 sparse matrix-vector multiplies (for mean) plus 28 compiled loops (for max/min), **per variable**.

### 4. Estimated Speedup

| Component | Current | Optimized |
|---|---|---|
| Neighbor lookup construction | ~6.46M string ops | 0 (use nb object directly) |
| Mean computation (per var) | 6.46M lapply calls | 28 sparse mat-vec multiplies |
| Max/Min computation (per var) | 6.46M lapply calls | 28 Rcpp loops over 344K cells |
| **Total estimated time** | **86+ hours** | **~5â€“15 minutes** |

---

## Working R Code

```r
# =============================================================================
# OPTIMIZED NEIGHBOR FEATURE COMPUTATION
# Separates static topology from dynamic (year-varying) cell attributes.
# =============================================================================

library(Matrix)
library(data.table)

# ---- Step 1: Build static cell-level structures (done ONCE) -----------------

build_static_neighbor_structures <- function(id_order, neighbors_nb) {
  # id_order: vector of 344,208 cell IDs in canonical order
  # neighbors_nb: spdep::nb object (list of integer neighbor indices)
  #
  # Returns:
  #   adj_matrix : sparse binary adjacency matrix (344K x 344K)
  #   degree     : integer vector of neighbor counts per cell
  #   nb_list    : the raw nb list (cleaned of 0-entries) for max/min

  n_cells <- length(id_order)
  stopifnot(length(neighbors_nb) == n_cells)

  # Build sparse adjacency matrix from the nb object
  # Each neighbors_nb[[i]] is an integer vector of neighbor indices (1-based)
  # with the convention that a cell with no neighbors has a single element 0.
  from <- rep(seq_len(n_cells),
              times = vapply(neighbors_nb, function(x) sum(x > 0L), integer(1)))
  to   <- unlist(lapply(neighbors_nb, function(x) x[x > 0L]), use.names = FALSE)

  adj_matrix <- sparseMatrix(
    i    = from,
    j    = to,
    x    = 1,
    dims = c(n_cells, n_cells)
  )

  # Clean nb list: replace 0-neighbor entries with integer(0)
  nb_list <- lapply(neighbors_nb, function(x) {
    x_clean <- x[x > 0L]
    if (length(x_clean) == 0L) integer(0) else as.integer(x_clean)
  })

  degree <- diff(adj_matrix@p)  # number of non-zeros per column... 
  # Actually for CSC, diff(@p) gives col counts. We need row counts.
  # Safer:
  degree <- as.integer(rowSums(adj_matrix))

  list(
    adj_matrix = adj_matrix,
    degree     = degree,
    nb_list    = nb_list,
    id_order   = id_order
  )
}


# ---- Step 2: Compute neighbor max, min, mean per variable ------------------
#
# Strategy:
#   - mean: sparse matrix multiply  (W %*% x) / degree
#   - max, min: compiled R loop using the nb_list
#
# We process one variable at a time, all 28 years at once, using a matrix
# layout: rows = cells (344K), columns = years (28).

compute_neighbor_features_optimized <- function(cell_dt, id_order, static,
                                                 neighbor_source_vars) {
  # cell_dt        : data.table with columns id, year, and all source vars
  # id_order       : canonical cell ID ordering (length 344,208)
  # static         : output of build_static_neighbor_structures()
  # neighbor_source_vars : character vector of variable names
  #
  # Returns: cell_dt with new neighbor feature columns added.

  n_cells <- length(id_order)
  years   <- sort(unique(cell_dt$year))
  n_years <- length(years)

  # Create a mapping from cell ID to canonical position index
  id_to_pos <- setNames(seq_along(id_order), as.character(id_order))

  # Create a mapping from (cell_pos, year_pos) to row index in cell_dt
  # First, add canonical position and year position columns
  cell_dt[, cell_pos := id_to_pos[as.character(id)]]
  year_to_ypos <- setNames(seq_along(years), as.character(years))
  cell_dt[, year_pos := year_to_ypos[as.character(year)]]

  # Sort by cell_pos, year_pos so we can reshape efficiently
  setorder(cell_dt, cell_pos, year_pos)

  # Verify completeness: should be n_cells * n_years rows
  # (If panel is unbalanced, we handle NAs below)
  is_balanced <- (nrow(cell_dt) == n_cells * n_years)

  W      <- static$adj_matrix
  degree <- static$degree
  nb     <- static$nb_list

  # Pre-compute degree matrix reciprocal (for mean), handling 0-degree cells
  inv_degree <- ifelse(degree > 0L, 1.0 / degree, NA_real_)

  for (var_name in neighbor_source_vars) {
    message("Processing neighbor features for: ", var_name)

    max_col  <- paste0("neighbor_max_",  var_name)
    min_col  <- paste0("neighbor_min_",  var_name)
    mean_col <- paste0("neighbor_mean_", var_name)

    # Initialize result columns
    cell_dt[, (max_col)  := NA_real_]
    cell_dt[, (min_col)  := NA_real_]
    cell_dt[, (mean_col) := NA_real_]

    # Process year by year
    for (yr in years) {
      yr_rows <- which(cell_dt$year == yr)

      # Build a full-length vector aligned to id_order for this year
      # (handles potential missing cells gracefully)
      x_full <- rep(NA_real_, n_cells)

      # Map the year-slice rows to their cell positions
      pos_this_year <- cell_dt$cell_pos[yr_rows]
      x_full[pos_this_year] <- cell_dt[[var_name]][yr_rows]

      # ---- Neighbor MEAN via sparse matrix multiply ----
      # Replace NAs with 0 for the multiply, track valid counts
      x_nona <- x_full
      x_valid <- as.double(!is.na(x_full))
      x_nona[is.na(x_nona)] <- 0

      neighbor_sum   <- as.numeric(W %*% x_nona)
      neighbor_count <- as.numeric(W %*% x_valid)

      n_mean <- ifelse(neighbor_count > 0, neighbor_sum / neighbor_count, NA_real_)

      # ---- Neighbor MAX and MIN via nb_list loop ----
      n_max <- rep(NA_real_, n_cells)
      n_min <- rep(NA_real_, n_cells)

      for (i in seq_len(n_cells)) {
        nb_i <- nb[[i]]
        if (length(nb_i) == 0L) next
        vals <- x_full[nb_i]
        vals <- vals[!is.na(vals)]
        if (length(vals) == 0L) next
        n_max[i] <- max(vals)
        n_min[i] <- min(vals)
      }

      # Write results back to the data.table for this year's rows
      set(cell_dt, i = yr_rows, j = max_col,  value = n_max[pos_this_year])
      set(cell_dt, i = yr_rows, j = min_col,  value = n_min[pos_this_year])
      set(cell_dt, i = yr_rows, j = mean_col, value = n_mean[pos_this_year])
    }

    message("  Done: ", var_name)
  }

  # Clean up helper columns
  cell_dt[, c("cell_pos", "year_pos") := NULL]

  return(cell_dt)
}


# ---- Step 2b (OPTIONAL): Rcpp-accelerated max/min for further speedup ------
# If Rcpp is available, this replaces the R-level for loop over 344K cells
# and brings max/min computation from ~minutes to ~seconds per year.

use_rcpp_minmax <- requireNamespace("Rcpp", quietly = TRUE)

if (use_rcpp_minmax) {
  Rcpp::cppFunction('
    #include <Rcpp.h>
    using namespace Rcpp;

    // [[Rcpp::export]]
    List neighbor_minmax_cpp(NumericVector x, List nb_list) {
      int n = nb_list.size();
      NumericVector out_max(n, NA_REAL);
      NumericVector out_min(n, NA_REAL);

      for (int i = 0; i < n; i++) {
        IntegerVector nb_i = nb_list[i];
        int m = nb_i.size();
        if (m == 0) continue;

        double cur_max = R_NegInf;
        double cur_min = R_PosInf;
        int valid = 0;

        for (int j = 0; j < m; j++) {
          double val = x[nb_i[j] - 1];  // R is 1-indexed
          if (!NumericVector::is_na(val)) {
            if (val > cur_max) cur_max = val;
            if (val < cur_min) cur_min = val;
            valid++;
          }
        }

        if (valid > 0) {
          out_max[i] = cur_max;
          out_min[i] = cur_min;
        }
      }

      return List::create(
        Named("max") = out_max,
        Named("min") = out_min
      );
    }
  ')
}

# Rcpp-enabled version of the main function
compute_neighbor_features_fast <- function(cell_dt, id_order, static,
                                            neighbor_source_vars) {
  n_cells <- length(id_order)
  years   <- sort(unique(cell_dt$year))

  id_to_pos <- setNames(seq_along(id_order), as.character(id_order))
  cell_dt[, cell_pos := id_to_pos[as.character(id)]]

  setorder(cell_dt, cell_pos, year)

  W  <- static$adj_matrix
  nb <- static$nb_list

  for (var_name in neighbor_source_vars) {
    message("Processing: ", var_name)
    t0 <- Sys.time()

    max_col  <- paste0("neighbor_max_",  var_name)
    min_col  <- paste0("neighbor_min_",  var_name)
    mean_col <- paste0("neighbor_mean_", var_name)

    cell_dt[, (max_col)  := NA_real_]
    cell_dt[, (min_col)  := NA_real_]
    cell_dt[, (mean_col) := NA_real_]

    for (yr in years) {
      yr_rows <- which(cell_dt$year == yr)

      x_full <- rep(NA_real_, n_cells)
      pos_this_year <- cell_dt$cell_pos[yr_rows]
      x_full[pos_this_year] <- cell_dt[[var_name]][yr_rows]

      # Mean via sparse mat-vec
      x_nona <- x_full
      x_valid <- as.double(!is.na(x_full))
      x_nona[is.na(x_nona)] <- 0
      neighbor_sum   <- as.numeric(W %*% x_nona)
      neighbor_count <- as.numeric(W %*% x_valid)
      n_mean <- ifelse(neighbor_count > 0, neighbor_sum / neighbor_count, NA_real_)

      # Max/Min via Rcpp
      if (use_rcpp_minmax) {
        mm <- neighbor_minmax_cpp(x_full, nb)
        n_max <- mm$max
        n_min <- mm$min
      } else {
        # Fallback: pure R
        n_max <- rep(NA_real_, n_cells)
        n_min <- rep(NA_real_, n_cells)
        for (i in seq_len(n_cells)) {
          nb_i <- nb[[i]]
          if (length(nb_i) == 0L) next
          vals <- x_full[nb_i]
          vals <- vals[!is.na(vals)]
          if (length(vals) == 0L) next
          n_max[i] <- max(vals)
          n_min[i] <- min(vals)
        }
      }

      set(cell_dt, i = yr_rows, j = max_col,  value = n_max[pos_this_year])
      set(cell_dt, i = yr_rows, j = min_col,  value = n_min[pos_this_year])
      set(cell_dt, i = yr_rows, j = mean_col, value = n_mean[pos_this_year])
    }

    message("  Done: ", var_name, " (", round(difftime(Sys.time(), t0, units="mins"), 1), " min)")
  }

  cell_dt[, cell_pos := NULL]
  return(cell_dt)
}


# =============================================================================
# MAIN PIPELINE (drop-in replacement for the original outer loop)
# =============================================================================

# Convert to data.table if not already
if (!inherits(cell_data, "data.table")) {
  cell_data <- as.data.table(cell_data)
}

# Step A: Build static structures ONCE (seconds, ~50 MB memory)
message("Building static neighbor structures...")
static <- build_static_neighbor_structures(id_order, rook_neighbors_unique)

# Step B: Compute all neighbor features (~5-15 min total)
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cell_data <- compute_neighbor_features_fast(
  cell_dt              = cell_data,
  id_order             = id_order,
  static               = static,
  neighbor_source_vars = neighbor_source_vars
)

# Step C: Predict with the pre-trained Random Forest (UNCHANGED)
# The model object and predict() call remain exactly as before.
# cell_data now has the same 15 neighbor feature columns
# (neighbor_max_*, neighbor_min_*, neighbor_mean_* for each of 5 vars)
# with numerically identical values to the original implementation.

# predictions <- predict(trained_rf_model, newdata = cell_data)
```

---

## Summary of Changes

| Aspect | Original | Optimized |
|---|---|---|
| **Neighbor lookup scope** | 6.46M cell-year entries | 344K cell entries (static) |
| **Lookup construction** | String paste + named vector | Direct use of `nb` object |
| **Mean computation** | R `lapply` over 6.46M rows | Sparse matrix multiply (`W %*% x`) per year |
| **Max/Min computation** | R `lapply` over 6.46M rows | Rcpp loop over 344K cells per year |
| **Total iterations** | ~6.46M Ã— 5 vars Ã— 3 stats | 344K Ã— 28 years Ã— 5 vars (compiled) |
| **Estimated runtime** | 86+ hours | **5â€“15 minutes** |
| **Memory** | ~6.46M-element list | ~50 MB sparse matrix + 344K-element list |
| **Numerical output** | Original estimand | **Preserved exactly** |
| **RF model** | Pre-trained | **Untouched** |

The key architectural insight: **the adjacency graph is a property of space, not of time**. By factoring it out, we eliminate 96% of redundant work and unlock vectorized/compiled computation paths.
END_RESPONSE>>>
