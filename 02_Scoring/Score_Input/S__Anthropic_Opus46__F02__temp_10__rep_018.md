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
run_id: Anthropic_F02_T10_R018
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F02
prompt_family_slug: CODE_ONLY_NO_HINT
family_label: Code only, no hint
family_group: Code-visible neutral
replicate: 18
raw_output_file: Experiment_Output/Anthropic/Temp_1.0/output_F02_Anthropic_temp_10_rep_018.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

---

## 1. Diagnosis

The bottleneck is clear: **`build_neighbor_lookup`** iterates over ~6.46 million rows with an `lapply`, performing per-row string pasting, named-vector lookups, and NA filtering. This is an O(n Ã— k) operation executed entirely in interpreted R with per-element overhead from character key construction and named-vector matching. The resulting `neighbor_lookup` list (6.46M elements, each a vector of integer indices) then drives `compute_neighbor_stats`, which again loops over 6.46M elements per variable. Combined:

| Component | Calls | Per-call cost | Total |
|---|---|---|---|
| `build_neighbor_lookup` | 6.46M | `paste` + named-vector lookup (~k=4 neighbors) | ~hours |
| `compute_neighbor_stats` | 6.46M Ã— 5 vars | subset + `max/min/mean` | ~hours |
| Memory: `neighbor_lookup` list | 6.46M entries | R list overhead â‰ˆ 56 bytes/entry + int vectors | ~2â€“4 GB |

**Root causes:**
1. **String-keyed lookups** (`paste` + named-vector indexing) are extremely slow at scale in R.
2. **R-level `lapply` over millions of rows** has high interpreted-loop overhead.
3. **Per-row allocation** of small vectors inside `lapply` causes GC pressure.
4. **Stats computed variable-by-variable**, missing the opportunity to vectorize across all 5 variables at once.

---

## 2. Optimization Strategy

### A. Replace string-keyed lookup with integer-arithmetic indexing
Instead of pasting `id_year` strings, encode the lookup as a direct integer mapping: `(cell_index, year_index) â†’ row_index`. With 344,208 cells and 28 years this is a 344,208 Ã— 28 integer matrix (~38 MB) that gives O(1) row lookups.

### B. Flatten the neighbor list into a CSR (Compressed Sparse Row) structure
Convert the `spdep::nb` object into two integer vectors (`ptr`, `nbr_ids`) â€” a CSR representation. This avoids millions of R list accesses and enables vectorized C++-level traversal.

### C. Vectorize the stats computation with `data.table` + Rcpp
Use a single compiled C++ function (via `Rcpp`) that, for every row, walks the CSR neighbor structure, looks up the corresponding rows via the integer matrix, and computes max/min/mean for all 5 variables simultaneously. This reduces 6.46M Ã— 5 R-level loops to a single compiled pass.

### D. Memory budget
| Object | Size |
|---|---|
| `cell_data` (6.46M Ã— 110 cols, numeric) | ~5.3 GB |
| Row-lookup matrix (344K Ã— 28, integer) | ~38 MB |
| CSR neighbor arrays (two int vectors) | ~11 MB |
| Output columns (6.46M Ã— 15, numeric) | ~775 MB |
| **Working total** | **~6.1 GB** (fits in 16 GB) |

### Expected speedup
The Rcpp inner loop replaces ~32.3 billion interpreted R operations with compiled C++ arithmetic. Expected wall-clock: **5â€“20 minutes** (vs. 86+ hours).

---

## 3. Working R Code

```r
# ============================================================
# 0. Prerequisites
# ============================================================
# install.packages(c("data.table", "Rcpp"))
library(data.table)
library(Rcpp)

# ============================================================
# 1. Build integer row-lookup matrix
#    Maps (cell_index, year_index) -> row position in cell_data
# ============================================================
build_row_lookup_matrix <- function(cell_data, id_order, years) {
  # cell_data must have columns: id, year
  # id_order : character or integer vector of unique cell ids (same order as nb object)
  # years    : sorted integer vector of unique years

  n_cells <- length(id_order)
  n_years <- length(years)

  # Map id -> cell_index (1-based, matches nb object indexing)
  id_to_cidx <- setNames(seq_along(id_order), as.character(id_order))
  # Map year -> year_index (1-based)
  year_to_yidx <- setNames(seq_along(years), as.character(years))

  # Pre-allocate matrix filled with NA (NA_integer_ means "no row for this cell-year")
  lookup_mat <- matrix(NA_integer_, nrow = n_cells, ncol = n_years)

  cidx <- id_to_cidx[as.character(cell_data$id)]
  yidx <- year_to_yidx[as.character(cell_data$year)]

  # Assign row positions
  valid <- !is.na(cidx) & !is.na(yidx)
  lookup_mat[cbind(cidx[valid], yidx[valid])] <- which(valid)

  list(mat = lookup_mat, id_to_cidx = id_to_cidx, year_to_yidx = year_to_yidx)
}

# ============================================================
# 2. Convert spdep::nb list to CSR (Compressed Sparse Row)
# ============================================================
nb_to_csr <- function(neighbors) {
  # neighbors: list of integer vectors (spdep nb object), 1-based cell indices
  # Returns ptr (length n_cells+1) and nbr (concatenated neighbor cell indices)
  # Both are 0-based for C++ consumption.

  n <- length(neighbors)
  lens <- vapply(neighbors, function(x) {
    # spdep uses 0L to denote "no neighbors"
    if (length(x) == 1L && x[1L] == 0L) 0L else length(x)
  }, integer(1))

  ptr <- c(0L, cumsum(lens))

  nbr <- unlist(lapply(neighbors, function(x) {
    if (length(x) == 1L && x[1L] == 0L) integer(0) else as.integer(x)
  }), use.names = FALSE)

  # Convert to 0-based indexing for C++
  list(ptr = as.integer(ptr), nbr = nbr - 1L)
}

# ============================================================
# 3. Rcpp kernel â€“ computes neighbor stats for all rows & vars
# ============================================================
Rcpp::sourceCpp(code = '
#include <Rcpp.h>
#include <cmath>
#include <limits>
using namespace Rcpp;

// [[Rcpp::export]]
NumericMatrix compute_neighbor_stats_cpp(
    IntegerVector cell_idx,    // 0-based cell index for each row
    IntegerVector year_idx,    // 0-based year index for each row
    IntegerVector csr_ptr,     // CSR row pointers (length n_cells+1)
    IntegerVector csr_nbr,     // CSR neighbor cell indices (0-based)
    IntegerMatrix lookup_mat,  // (n_cells x n_years) -> 1-based row index, NA = missing
    NumericMatrix var_mat      // (n_rows x n_vars) source variable values
) {
  int n_rows = cell_idx.size();
  int n_vars = var_mat.ncol();
  int n_out  = n_vars * 3; // max, min, mean per variable

  NumericMatrix out(n_rows, n_out); // column order: var0_max, var0_min, var0_mean, var1_max, ...

  // Fill with NA
  std::fill(out.begin(), out.end(), NA_REAL);

  for (int i = 0; i < n_rows; i++) {
    int ci = cell_idx[i];
    int yi = year_idx[i];

    int nb_start = csr_ptr[ci];
    int nb_end   = csr_ptr[ci + 1];
    int n_nb     = nb_end - nb_start;
    if (n_nb == 0) continue;

    // Collect valid neighbor row indices
    std::vector<int> nb_rows;
    nb_rows.reserve(n_nb);
    for (int j = nb_start; j < nb_end; j++) {
      int nb_ci = csr_nbr[j];
      int row1  = lookup_mat(nb_ci, yi); // 1-based or NA
      if (row1 != NA_INTEGER) {
        nb_rows.push_back(row1 - 1); // convert to 0-based
      }
    }
    if (nb_rows.empty()) continue;

    // Compute stats for each variable
    for (int v = 0; v < n_vars; v++) {
      double vmax = -std::numeric_limits<double>::infinity();
      double vmin =  std::numeric_limits<double>::infinity();
      double vsum = 0.0;
      int    vcnt = 0;

      for (size_t k = 0; k < nb_rows.size(); k++) {
        double val = var_mat(nb_rows[k], v);
        if (!R_IsNA(val)) {
          if (val > vmax) vmax = val;
          if (val < vmin) vmin = val;
          vsum += val;
          vcnt++;
        }
      }

      int base = v * 3;
      if (vcnt > 0) {
        out(i, base)     = vmax;
        out(i, base + 1) = vmin;
        out(i, base + 2) = vsum / vcnt;
      }
      // else: stays NA (already filled)
    }
  }

  return out;
}
')

# ============================================================
# 4. Top-level orchestration function
# ============================================================
add_all_neighbor_features <- function(cell_data, id_order, rook_neighbors_unique,
                                      neighbor_source_vars) {
  # Convert to data.table for efficient column operations (non-destructive)
  if (!is.data.table(cell_data)) {
    cell_data <- as.data.table(cell_data)
  }

  years <- sort(unique(cell_data$year))

  # --- Step 1: row-lookup matrix ---
  message("Building row-lookup matrix...")
  rl <- build_row_lookup_matrix(cell_data, id_order, years)
  lookup_mat <- rl$mat  # integer matrix, n_cells x n_years (1-based row indices)

  # --- Step 2: CSR neighbor structure ---
  message("Converting neighbor list to CSR...")
  csr <- nb_to_csr(rook_neighbors_unique)

  # --- Step 3: Prepare inputs for C++ ---
  message("Preparing index vectors...")
  cell_idx_vec <- as.integer(rl$id_to_cidx[as.character(cell_data$id)]) - 1L # 0-based
  year_idx_vec <- as.integer(rl$year_to_yidx[as.character(cell_data$year)]) - 1L # 0-based

  # Build variable matrix (n_rows x n_vars)
  var_mat <- as.matrix(cell_data[, ..neighbor_source_vars])
  storage.mode(var_mat) <- "double"

  # lookup_mat must be integer matrix for IntegerMatrix in Rcpp
  storage.mode(lookup_mat) <- "integer"

  # --- Step 4: Call C++ kernel ---
  message("Computing neighbor statistics in C++ ...")
  stats_mat <- compute_neighbor_stats_cpp(
    cell_idx_vec, year_idx_vec,
    csr$ptr, csr$nbr,
    lookup_mat,
    var_mat
  )

  # --- Step 5: Attach results to cell_data ---
  message("Attaching new columns...")
  stat_suffixes <- c("_neighbor_max", "_neighbor_min", "_neighbor_mean")
  col <- 1L
  for (var_name in neighbor_source_vars) {
    for (sfx in stat_suffixes) {
      new_col_name <- paste0(var_name, sfx)
      set(cell_data, j = new_col_name, value = stats_mat[, col])
      col <- col + 1L
    }
  }

  message("Done. Added ", length(neighbor_source_vars) * 3L, " neighbor-feature columns.")
  cell_data
}

# ============================================================
# 5. Run it
# ============================================================
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cell_data <- add_all_neighbor_features(
  cell_data,
  id_order,
  rook_neighbors_unique,
  neighbor_source_vars
)

# The trained Random Forest model is unchanged.
# Prediction proceeds exactly as before:
# predictions <- predict(rf_model, newdata = cell_data)
```

---

## Summary of Changes and Guarantees

| Aspect | Original | Optimized |
|---|---|---|
| Neighbor lookup | `paste` + named-vector matching per row | Integer matrix `(cell, year) â†’ row`, O(1) |
| Neighbor traversal | R `lapply` over 6.46M-element list | CSR arrays traversed in compiled C++ |
| Stats computation | 5 separate R `lapply` passes | Single C++ pass over all 5 variables |
| Estimated wall-clock | 86+ hours | **5â€“20 minutes** |
| Peak RAM | >10 GB (list overhead) | **~6.1 GB** (fits in 16 GB) |
| Numerical output | max, min, mean per var per row | **Identical** (same arithmetic, same NA handling) |
| Trained RF model | â€” | **Untouched** (no retraining) |
| Column names | `{var}_neighbor_max/min/mean` | **Same names, same order** |
END_RESPONSE>>>
