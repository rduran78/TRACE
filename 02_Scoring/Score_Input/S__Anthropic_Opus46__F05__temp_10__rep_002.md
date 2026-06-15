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
run_id: Anthropic_F05_T10_R002
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F05
prompt_family_slug: STRING_KEY_REDUNDANCY
family_label: String-key probe
family_group: Line-level probe
replicate: 2
raw_output_file: Experiment_Output/Anthropic/Temp_1.0/output_F05_Anthropic_temp_10_rep_002.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

---

## 1. Diagnosis

The repeated string-key work is **not merely a local inefficiency â€” it is a symptom of a larger, fundamentally misaligned algorithmic pattern**. Here is the full chain of waste:

### The Core Problem: Row-Level Iteration Over 6.46M Rows in Pure R

`build_neighbor_lookup` calls `lapply` over every one of the ~6.46 million rows. Inside each iteration it:

1. **Constructs paste keys** (`paste(neighbor_cell_ids, data$year[i], sep = "_")`) â€” string allocation and concatenation for every neighbor of every row.
2. **Looks up those keys** in a named character vector (`idx_lookup[neighbor_keys]`) â€” this is an O(1)-amortized hash lookup *per key*, but the constant factor of R's named-vector hashing on character strings is large, and it is invoked **~6.46M Ã— avg_neighbors â‰ˆ 25+ billion character operations**.

The `idx_lookup` named vector itself (`paste(data$id, data$year, sep = "_")` with 6.46M entries) is built once â€” that part is fine. But the *per-row consumption* of it is the bottleneck.

### Why It's a Broader Pattern

The neighbor lookup produces a **list of integer row-index vectors** â€” one per row. This is then consumed by `compute_neighbor_stats`, which itself iterates over all 6.46M rows again, subsetting a numeric vector and computing `max/min/mean`. This is repeated for **each of the 5 variables**, meaning 5 Ã— 6.46M additional R-level iterations.

So the full cost is:

| Phase | Iterations | Cost Driver |
|---|---|---|
| `build_neighbor_lookup` | 6.46M | String paste + hash lookup per neighbor |
| `compute_neighbor_stats` | 5 Ã— 6.46M | R-level `lapply`, vector subset, `max/min/mean` |
| **Total R-level loop iterations** | **~38.8M** | Plus billions of string ops in phase 1 |

At ~86+ hours, this is dominated by the string construction in phase 1 and the interpretive overhead of pure-R loops in both phases.

### Key Insight for Reformulation

The string-keyed lookup is solving a simple structural problem: **"given a cell ID and a year, find the row index."** Because the panel is balanced (344,208 cells Ã— 28 years = 9,637,824 potential slots, with 6.46M populated), and years are a small integer domain, this can be solved with **integer arithmetic instead of string hashing**. Moreover, the neighbor structure is **invariant across years** â€” the same cell has the same neighbors in every year. This means:

> **We don't need to build a per-row neighbor list at all. We need a per-cell neighbor list (344K entries), and then we can join across all years simultaneously using vectorized operations.**

---

## 2. Optimization Strategy

### Algorithmic Reformulation

**Replace the row-level string-key approach with a vectorized sparse-matrix multiplication.**

The neighbor relationships form a **sparse adjacency matrix** `W` of dimension 344,208 Ã— 344,208. Computing the mean of a variable across neighbors is simply:

```
neighbor_mean = (W %*% x) / (W %*% 1)  # where 1 is a vector of non-NA indicators
```

And max/min can be computed via a sparse row-sweep using the adjacency structure.

For a balanced panel, we compute these statistics **per year** (each year has the same cell set and the same adjacency), which reduces to 28 sparse matrixâ€“vector multiplications per variable â€” not 6.46M R-level iterations.

### Concrete Steps

1. **Build the sparse adjacency matrix once** from `rook_neighbors_unique` (a `nb` object â†’ `nb2listw` â†’ sparse matrix, or directly).
2. **For each year and each variable**, extract the 344,208-length vector, apply sparse operations to get `neighbor_max`, `neighbor_min`, `neighbor_mean`.
3. **Write results back** into the data frame.
4. **No string keys. No per-row `lapply`. No `paste`.**

### Complexity Comparison

| | Original | Reformulated |
|---|---|---|
| Neighbor lookup | 6.46M string-hash lookups | 1 sparse matrix (built once) |
| Stats computation | 5 vars Ã— 6.46M R iterations | 5 vars Ã— 28 years Ã— 3 sparse ops |
| Dominant cost | ~25B character ops | ~5 Ã— 28 Ã— 2 sparse mat-vec products (~1.37M nonzeros each) |
| Expected time | 86+ hours | **Minutes** |

### RAM Check

- Sparse matrix: 1,373,394 nonzeros Ã— 12 bytes â‰ˆ 16 MB.
- Data frame: 6.46M rows Ã— 110 cols Ã— 8 bytes â‰ˆ 5.7 GB. Already in memory.
- Year-sliced vectors: 344K Ã— 8 bytes â‰ˆ 2.7 MB each. Negligible.
- Well within 16 GB.

---

## 3. Working R Code

```r
# =============================================================================
# OPTIMIZED NEIGHBOR FEATURE CONSTRUCTION
# =============================================================================
# Prerequisites:
#   cell_data           â€” data.frame with columns: id, year, ntl, ec, pop_density, def, usd_est_n2, ...
#   rook_neighbors_unique â€” nb object (from spdep) with 344,208 entries
#   id_order            â€” integer/numeric vector of cell IDs in the order matching rook_neighbors_unique
#
# This code preserves the exact same numerical output as the original:
#   For each row, neighbor_max_<var>, neighbor_min_<var>, neighbor_mean_<var>
# =============================================================================

library(Matrix)  # for sparse matrices
library(data.table)  # for fast grouped operations

# ---- Step 1: Build sparse adjacency matrix (once) --------------------------

build_sparse_adjacency <- function(nb_obj, n) {
  # nb_obj: spdep nb object (list of integer neighbor index vectors)
  # n: number of spatial units
  # Returns: sparse logical/numeric adjacency matrix (n x n), row i has 1s at neighbor columns
  
  # Enumerate all (i, j) pairs
  i_idx <- rep(seq_len(n), times = lengths(nb_obj))
  j_idx <- unlist(nb_obj)
  
  # Remove zero-length / zero entries (spdep uses 0L for no-neighbor indicator in some versions)
  valid <- j_idx > 0L
  i_idx <- i_idx[valid]
  j_idx <- j_idx[valid]
  
  sparseMatrix(i = i_idx, j = j_idx, x = 1, dims = c(n, n))
}

n_cells <- length(id_order)  # 344,208
W <- build_sparse_adjacency(rook_neighbors_unique, n_cells)

# ---- Step 2: Map cell IDs to matrix row/col indices ------------------------

# id_order[k] is the cell ID for the k-th row/col of W
# We need a lookup: cell_id -> matrix index
cell_id_to_mat_idx <- setNames(seq_along(id_order), as.character(id_order))

# ---- Step 3: Convert to data.table for fast year-grouped operations ---------

dt <- as.data.table(cell_data)

# Add matrix index column (maps each row's cell ID to the row/col index in W)
dt[, mat_idx := cell_id_to_mat_idx[as.character(id)]]

# Sort by year and mat_idx to enable fast vectorized access
setkey(dt, year, mat_idx)

# ---- Step 4: Compute neighbor stats per variable ----------------------------

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# Pre-allocate result columns
for (var_name in neighbor_source_vars) {
  dt[, paste0("neighbor_max_", var_name) := NA_real_]
  dt[, paste0("neighbor_min_", var_name) := NA_real_]
  dt[, paste0("neighbor_mean_", var_name) := NA_real_]
}

# Get sorted unique years
years <- sort(unique(dt$year))

# For max and min via sparse matrix, we need a direct approach since
# sparse mat-vec only gives sums. We iterate over years (only 28)
# and use the sparse structure directly.

# Extract the adjacency list from the sparse matrix for row-wise max/min
# This is done once and is fast.
W_dgC <- as(W, "dgCMatrix")  # ensure CSC format for column slicing
W_dgR <- as(W, "dgRMatrix")  # CSR format for fast row access

# Actually, for max/min we need to iterate over rows of W.
# With 344K cells and avg ~4 neighbors each (rook), this is very fast in vectorized R.
# We'll extract the neighbor list from the sparse matrix once.

# Extract neighbor indices from sparse matrix (CSR format)
# W_dgR@j is 0-based column indices, W_dgR@p is row pointer
get_neighbor_list_from_sparse <- function(W_csr) {
  n <- nrow(W_csr)
  p <- W_csr@p
  j <- W_csr@j + 1L  # convert to 1-based
  lapply(seq_len(n), function(i) {
    start <- p[i] + 1L
    end   <- p[i + 1L]
    if (end >= start) j[start:end] else integer(0)
  })
}

# This gives us neighbor mat_idx for each cell (same as rook_neighbors_unique but guaranteed consistent)
nb_list <- get_neighbor_list_from_sparse(W_dgR)

# ---- Step 5: Year-by-year, variable-by-variable computation ----------------

# Strategy:
#   For each year, the subset of dt with that year has cells ordered by mat_idx (due to setkey).
#   We build a full-length vector (length n_cells) with the variable values placed at their mat_idx.
#   Then we compute neighbor stats using the nb_list.
#
#   For mean: use sparse matrix-vector product for sum and count, then divide.
#   For max/min: use vectorized C-level operations via vapply on nb_list.
#
#   Since nb_list has only 344K entries with ~4 neighbors each, vapply over it is ~344K iterations
#   with trivial work â€” takes < 1 second per variable per year.

for (yr in years) {
  cat("Processing year:", yr, "\n")
  
  # Get the rows for this year
  yr_rows <- dt[.(yr)]  # keyed access: all rows with this year
  
  # Map mat_idx to positions within this year-slice
  # yr_rows is sorted by mat_idx due to setkey
  yr_mat_indices <- yr_rows$mat_idx
  
  # For each variable, build a full vector of length n_cells (NA for missing cells)
  for (var_name in neighbor_source_vars) {
    
    # Full vector (NA-initialized)
    full_vec <- rep(NA_real_, n_cells)
    full_vec[yr_mat_indices] <- yr_rows[[var_name]]
    
    # -- Neighbor mean via sparse matrix-vector multiply --
    # Replace NA with 0 for sum, track non-NA for count
    non_na <- !is.na(full_vec)
    vec_zero <- full_vec
    vec_zero[!non_na] <- 0
    
    neighbor_sum   <- as.numeric(W %*% vec_zero)       # sum of neighbor values
    neighbor_count <- as.numeric(W %*% as.numeric(non_na))  # count of non-NA neighbors
    
    neighbor_mean_full <- ifelse(neighbor_count > 0, neighbor_sum / neighbor_count, NA_real_)
    
    # -- Neighbor max and min via nb_list --
    # vapply over 344K cells â€” fast because each has ~4 neighbors
    neighbor_max_full <- vapply(nb_list, function(idx) {
      if (length(idx) == 0L) return(NA_real_)
      v <- full_vec[idx]
      v <- v[!is.na(v)]
      if (length(v) == 0L) return(NA_real_)
      max(v)
    }, numeric(1))
    
    neighbor_min_full <- vapply(nb_list, function(idx) {
      if (length(idx) == 0L) return(NA_real_)
      v <- full_vec[idx]
      v <- v[!is.na(v)]
      if (length(v) == 0L) return(NA_real_)
      min(v)
    }, numeric(1))
    
    # -- Write results back to the year-slice rows --
    # yr_mat_indices tells us which mat_idx each row corresponds to
    max_col  <- paste0("neighbor_max_", var_name)
    min_col  <- paste0("neighbor_min_", var_name)
    mean_col <- paste0("neighbor_mean_", var_name)
    
    # Use set() for in-place modification (no copy)
    row_indices <- dt[year == yr, which = TRUE]
    set(dt, i = row_indices, j = max_col,  value = neighbor_max_full[yr_mat_indices])
    set(dt, i = row_indices, j = min_col,  value = neighbor_min_full[yr_mat_indices])
    set(dt, i = row_indices, j = mean_col, value = neighbor_mean_full[yr_mat_indices])
  }
}

# ---- Step 6: Convert back to data.frame if needed --------------------------
cell_data <- as.data.frame(dt)

# ---- Done. The trained Random Forest model is untouched. --------------------
cat("Neighbor feature construction complete.\n")
```

---

## 4. Complexity and Runtime Estimate

| Operation | Count | Unit Cost | Total |
|---|---|---|---|
| Build sparse matrix | 1 | O(1.37M) | < 1 sec |
| Sparse mat-vec (`W %*% x`) | 5 vars Ã— 28 yrs Ã— 2 = 280 | O(1.37M) | ~30 sec |
| `vapply` for max/min | 5 vars Ã— 28 yrs Ã— 2 Ã— 344K cells | O(4) per cell | ~5â€“10 min |
| `data.table` keyed access/set | 28 years Ã— 5 vars | O(230K) per slice | ~1 min |
| **Total** | | | **~10â€“15 minutes** |

This is a **~350Ã—â€“500Ã— speedup** over the original 86+ hours.

---

## 5. Why This Preserves the Numerical Estimand

- The sparse matrix `W` encodes **exactly the same neighbor relationships** as `rook_neighbors_unique`.
- For each cell-year, the set of neighbor values considered is identical.
- `max`, `min`, and `mean` are computed on the same non-NA neighbor values.
- The mean computation `(W %*% x) / (W %*% 1_{non-NA})` is algebraically equivalent to `mean(neighbor_vals[!is.na(neighbor_vals)])`.
- The resulting columns are identically named and placed into `cell_data`.
- **The trained Random Forest model is never touched** â€” only the feature-construction step is replaced.
END_RESPONSE>>>
