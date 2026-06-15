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
run_id: Anthropic_F07_T00_R014
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F07
prompt_family_slug: RASTER_INVALID_IRREGULAR
family_label: Raster invalid irregular topology
family_group: Raster/kernel bridge
replicate: 14
raw_output_file: Experiment_Output/Anthropic/Temp_0.0/output_F07_Anthropic_temp_00_rep_014.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The bottleneck is **`build_neighbor_lookup`**, which creates a list of ~6.46 million elements, each built by character-key lookups inside an `lapply` over every row. Specifically:

1. **`paste` + named-vector lookup per row**: For each of the 6.46M rows, the code converts neighbor cell IDs to character keys (`paste(id, year, sep="_")`), then does named-vector indexing (`idx_lookup[neighbor_keys]`). Named-vector lookup in R is O(n) per query in the worst case and involves repeated string hashing. Over 6.46M rows Ã— ~4 neighbors each â‰ˆ 26M string operations.

2. **`lapply` over 6.46M rows for `compute_neighbor_stats`**: Each iteration subsets a numeric vector, removes NAs, and computes max/min/mean. The per-element overhead of R-level `lapply` (closure call, allocation of 3-element vectors) across 6.46M iterations is enormous.

3. **Memory**: Building a 6.46M-element list of integer vectors, plus the `do.call(rbind, ...)` on 6.46M 3-element vectors, creates massive intermediate allocations.

**Estimated cost**: The 86+ hour runtime is almost entirely attributable to the row-level R loop in `build_neighbor_lookup` and the repeated `lapply` in `compute_neighbor_stats`.

---

## Optimization Strategy

### Key Insight: Vectorize via Sparse Matrix Multiplication

The neighbor relationships are a **fixed spatial graph** (344,208 cells, ~1.37M directed edges). The panel has 28 years. For each year, the neighbor-max, neighbor-min, and neighbor-mean of a variable can be computed by operating on the **sparse adjacency matrix** directly â€” no per-row R loop needed.

**Plan:**

1. **Build a sparse binary adjacency matrix `W`** (344,208 Ã— 344,208) from `rook_neighbors_unique` once. This is tiny (~1.37M non-zero entries).

2. **For each variable and each year**, extract the value vector `v` (length 344,208), then:
   - **Neighbor mean** = `(W %*% v) / (W %*% ones)` â€” sparse matrix-vector multiply, microseconds.
   - **Neighbor max / min** â€” use a grouped operation on the sparse matrix's structure (CSC column indices), or use `data.table` grouped operations on an edge list.

3. **Join results back** to the panel `data.table` by `(id, year)`.

This replaces 6.46M R-level iterations with ~28 sparse matrix-vector multiplies per variable (one per year), each taking milliseconds. Total runtime drops from 86+ hours to **minutes**.

---

## Working R Code

```r
library(data.table)
library(Matrix)

# â”€â”€ 0. Convert to data.table if not already â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
if (!is.data.table(cell_data)) cell_data <- as.data.table(cell_data)

# â”€â”€ 1. Build sparse adjacency matrix from spdep nb object (once) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
build_sparse_adjacency <- function(nb_obj, id_order) {

  # nb_obj: list of integer vectors (neighbor indices into id_order)
  # Returns: sparse binary matrix W (n x n), where W[i,j]=1 means j is

  #          a rook neighbor of i.
  n <- length(nb_obj)
  stopifnot(n == length(id_order))

  # Build COO triplets
  from <- rep(seq_len(n), lengths(nb_obj))
  to   <- unlist(nb_obj)

  # Remove 0-neighbor entries (spdep uses integer(0) or 0L for no neighbors)
  valid <- to > 0L
  from  <- from[valid]
  to    <- to[valid]

  W <- sparseMatrix(i = from, j = to, x = 1, dims = c(n, n))
  return(W)
}

W <- build_sparse_adjacency(rook_neighbors_unique, id_order)

# Precompute the number of neighbors per cell (constant across years)
ones_vec    <- rep(1, length(id_order))
n_neighbors <- as.numeric(W %*% ones_vec)  # length = n_cells

# â”€â”€ 2. Build edge list for max/min (grouped ops) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Extract COO from W
W_coo <- summary(W)  # data.frame with columns i, j, x
edge_dt <- data.table(focal = W_coo$i, neighbor = W_coo$j)
# focal's neighbor is 'neighbor', so for focal cell i we want values at j.

# Map from cell index (1..344208) to id_order value
idx_to_id <- data.table(cell_idx = seq_along(id_order), id = id_order)

# â”€â”€ 3. Compute neighbor stats for all variables â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

# Ensure id_order mapping in cell_data
# Create a cell_idx column: position of each cell's id in id_order
id_map <- data.table(id = id_order, cell_idx = seq_along(id_order))
cell_data <- merge(cell_data, id_map, by = "id", all.x = TRUE, sort = FALSE)

# Key for fast subsetting
setkey(cell_data, year, cell_idx)

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
years <- sort(unique(cell_data$year))
n_cells <- length(id_order)

for (var_name in neighbor_source_vars) {

  cat("Processing variable:", var_name, "\n")

  max_col  <- paste0("neighbor_max_",  var_name)
  min_col  <- paste0("neighbor_min_",  var_name)
  mean_col <- paste0("neighbor_mean_", var_name)

  # Pre-allocate result columns with NA
  cell_data[, (max_col)  := NA_real_]
  cell_data[, (min_col)  := NA_real_]
  cell_data[, (mean_col) := NA_real_]

  for (yr in years) {

    # Extract the value vector for this year, ordered by cell_idx
    yr_data <- cell_data[.(yr)]  # keyed on year first
    # Ensure we have a full vector aligned to cell_idx 1..n_cells
    v <- rep(NA_real_, n_cells)
    v[yr_data$cell_idx] <- yr_data[[var_name]]

    # â”€â”€ Neighbor mean via sparse mat-vec â”€â”€
    Wv <- as.numeric(W %*% v)
    # Handle NA propagation: count only non-NA neighbors
    not_na   <- as.numeric(!is.na(v))
    Wv_notna <- as.numeric(W %*% not_na)          # count of non-NA neighbors
    # Sum only non-NA values: replace NA with 0 for the multiply
    v_zero        <- v
    v_zero[is.na(v_zero)] <- 0
    Wv_sum        <- as.numeric(W %*% v_zero)
    neighbor_mean <- ifelse(Wv_notna > 0, Wv_sum / Wv_notna, NA_real_)

    # â”€â”€ Neighbor max and min via edge-list grouped ops â”€â”€
    # Look up neighbor values
    edge_vals <- data.table(
      focal = edge_dt$focal,
      val   = v[edge_dt$neighbor]
    )
    # Remove edges where neighbor value is NA
    edge_vals <- edge_vals[!is.na(val)]

    agg <- edge_vals[, .(nmax = max(val), nmin = min(val)), by = focal]

    # Build full-length vectors
    neighbor_max <- rep(NA_real_, n_cells)
    neighbor_min <- rep(NA_real_, n_cells)
    neighbor_max[agg$focal] <- agg$nmax
    neighbor_min[agg$focal] <- agg$nmin

    # â”€â”€ Write back into cell_data â”€â”€
    # yr_data$cell_idx gives the cell indices present this year
    cidx <- yr_data$cell_idx
    # Use data.table set() for speed â€” need the actual row numbers in cell_data
    row_nums <- which(cell_data$year == yr)
    # But these are already aligned because we subset by key.
    # Safer: match back by cell_idx
    set(cell_data, i = row_nums, j = max_col,  value = neighbor_max[cidx])
    set(cell_data, i = row_nums, j = min_col,  value = neighbor_min[cidx])
    set(cell_data, i = row_nums, j = mean_col, value = neighbor_mean[cidx])
  }

  cat("  Done:", var_name, "\n")
}

# â”€â”€ 4. Clean up helper column â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Remove cell_idx if you don't want it in the final data
# cell_data[, cell_idx := NULL]

cat("All neighbor features computed.\n")
```

---

## Why This Preserves Correctness

| Concern | Guarantee |
|---|---|
| **Same numerical estimand** | For each cell-year, we compute max, min, and mean over exactly the same set of rook-neighbor values as the original code. The sparse matrix `W` encodes the identical adjacency from `rook_neighbors_unique`. |
| **NA handling** | Matches original: NAs are excluded before computing stats; if all neighbors are NA (or a cell has no neighbors), the result is `NA`. |
| **Irregular topology / masked cells** | The sparse matrix naturally handles cells with 0â€“4 neighbors, missing cells, and boundary irregularities â€” no rectangular-raster assumption. |
| **Trained Random Forest** | No model retraining. We only recompute the input features identically, then feed them to the existing model. |

## Expected Performance

| Step | Cost |
|---|---|
| Build sparse matrix | ~1 second (1.37M entries) |
| Per variable, per year: 2 sparse mat-vec multiplies | ~5 ms each Ã— 28 years Ã— 5 vars = ~1.4 seconds total |
| Per variable, per year: edge-list grouped max/min | ~0.1 s each Ã— 28 Ã— 5 = ~14 seconds total |
| Write-back via `set()` | Negligible |
| **Total** | **< 1 minute** (down from 86+ hours) |

Memory peak stays well under 4 GB (sparse matrix ~22 MB, edge list ~22 MB, value vectors ~2.6 MB each, `cell_data` itself is the dominant object).
END_RESPONSE>>>
