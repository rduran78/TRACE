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
run_id: Anthropic_F07_T10_R003
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F07
prompt_family_slug: RASTER_INVALID_IRREGULAR
family_label: Raster invalid irregular topology
family_group: Raster/kernel bridge
replicate: 3
raw_output_file: Experiment_Output/Anthropic/Temp_1.0/output_F07_Anthropic_temp_10_rep_003.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

## Diagnosis

The bottleneck is **`build_neighbor_lookup`**. It creates a list of 6.46 million elements, and for each row it:

1. Performs character-based key lookups (`paste` + named-vector indexing) â€” O(n) string operations repeated ~6.46M times.
2. Allocates a new character vector (`neighbor_keys`) per row.
3. Uses `as.character` and `paste` inside a per-row `lapply`, which defeats R's vectorization.

The `compute_neighbor_stats` function then iterates over the 6.46M-element list again per variable (Ã—5 variables). Each iteration subsets `vals[idx]` and computes `max/min/mean` â€” this is actually fine per-call, but the 6.46M-element `lapply` in pure R is slow.

**Root cause summary:**

| Component | Cost | Why |
|---|---|---|
| `build_neighbor_lookup` | ~80+ hours | Per-row string paste + named-vector hash lookup Ã—6.46M |
| `compute_neighbor_stats` | ~6 hours (Ã—5 vars) | Per-row lapply over 6.46M list elements, Ã—5 |
| **Total** | **~86+ hours** | Pure-R row-level iteration, no vectorization |

The `spdep::nb` object (`rook_neighbors_unique`) has ~344K cells with ~1.37M directed edges â€” this is a **sparse graph** and should be represented as a **sparse matrix**, which enables fully vectorized neighbor aggregation via matrix multiplication.

---

## Optimization Strategy

**Replace the entire row-level lookup + loop with a single sparse adjacency matrix and matrix-vector products.**

1. **Build a sparse binary adjacency matrix `W`** (344,208 Ã— 344,208) from `rook_neighbors_unique`. This has ~1.37M nonzeros â€” trivial in memory (~16 MB).

2. **Expand to the panel** using a cell-to-row mapping. For each year, the neighbor of cell `i` in year `t` is the row of that neighbor cell in year `t`. Rather than building a 6.46M Ã— 6.46M matrix (too large), we process **year-by-year**: for each year, extract the column of values, do `W %*% x` (sparse mat-vec), and divide by neighbor counts. This gives `neighbor_sum`, from which `neighbor_mean = neighbor_sum / neighbor_count`.

3. For **neighbor max and min**, sparse matrix multiplication doesn't directly help, but we can use a **long-form edge table** + `data.table` grouped aggregation â€” vectorized C-level groupby.

**Expected speedup:** From ~86 hours â†’ **~2â€“5 minutes**.

---

## Working R Code

```r
library(Matrix)
library(data.table)

# ==============================================================
# STEP 1: Build sparse adjacency matrix from spdep::nb object
# ==============================================================
build_sparse_adjacency <- function(nb_obj) {
  # nb_obj is a list of length n_cells; nb_obj[[i]] gives integer 
  # vector of neighbor indices (or 0L if no neighbors)
  n <- length(nb_obj)
  
  from <- rep(seq_len(n), lengths(nb_obj))
  to   <- unlist(nb_obj)
  
  # spdep encodes "no neighbors" as a single 0; remove those
  valid <- to > 0L
  from  <- from[valid]
  to    <- to[valid]
  
  W <- sparseMatrix(i = from, j = to, x = 1, dims = c(n, n))
  W
}

W <- build_sparse_adjacency(rook_neighbors_unique)
n_cells <- length(rook_neighbors_unique)

# Neighbor count per cell (constant across years)
neighbor_count <- as.numeric(W %*% rep(1, n_cells))

# ==============================================================
# STEP 2: Create cell-index and year columns in data.table
# ==============================================================
# id_order is the vector mapping position index -> cell id
# We need the reverse: cell id -> position index in W

dt <- as.data.table(cell_data)

id_to_pos <- setNames(seq_along(id_order), as.character(id_order))
dt[, cell_pos := id_to_pos[as.character(id)]]

# Sort for efficiency (optional but helps cache locality)
setkey(dt, year, cell_pos)

years <- sort(unique(dt$year))

# ==============================================================
# STEP 3: Build edge table (long form) for max/min
#          This is ~1.37M edges Ã— 28 years â‰ˆ 38.5M rows
#          (~300 MB with 2 int + 1 double column â€” fits in RAM)
# ==============================================================
# We build the edge list once from W
W_coo <- summary(W)  # gives i, j, x columns (data.frame)
edges <- data.table(from = W_coo$i, to = W_coo$j)

# ==============================================================
# STEP 4: Function to compute all three stats for one variable
# ==============================================================
compute_neighbor_features_fast <- function(dt, var_name, W, edges, 
                                            neighbor_count, id_order, years) {
  
  col_max  <- paste0("neighbor_max_",  var_name)
  col_min  <- paste0("neighbor_min_",  var_name)
  col_mean <- paste0("neighbor_mean_", var_name)
  
  # Pre-allocate result columns
  dt[, (col_max)  := NA_real_]
  dt[, (col_min)  := NA_real_]
  dt[, (col_mean) := NA_real_]
  
  n_cells <- nrow(W)
  
  for (yr in years) {
    # Extract rows for this year, ordered by cell_pos
    yr_mask <- dt$year == yr
    yr_dt   <- dt[yr_mask]
    
    # Build a full-length vector indexed by cell_pos
    # (some cells may be missing in a year; they stay NA)
    vals_full <- rep(NA_real_, n_cells)
    vals_full[yr_dt$cell_pos] <- yr_dt[[var_name]]
    
    # --- MEAN via sparse matrix-vector product ---
    # Replace NA with 0 for summation, track valid counts
    vals_zero <- vals_full
    vals_zero[is.na(vals_zero)] <- 0
    valid_indicator <- as.numeric(!is.na(vals_full))
    
    neighbor_sum   <- as.numeric(W %*% vals_zero)
    neighbor_valid <- as.numeric(W %*% valid_indicator)
    
    n_mean <- ifelse(neighbor_valid > 0, 
                     neighbor_sum / neighbor_valid, 
                     NA_real_)
    
    # --- MAX and MIN via edge table ---
    # Look up neighbor values
    neighbor_vals_vec <- vals_full[edges$to]
    
    # Create temporary data.table for grouped aggregation
    agg_dt <- data.table(
      from = edges$from,
      val  = neighbor_vals_vec
    )
    # Remove NA neighbor values before aggregation
    agg_dt <- agg_dt[!is.na(val)]
    
    if (nrow(agg_dt) > 0) {
      stats <- agg_dt[, .(nmax = max(val), nmin = min(val)), by = from]
      
      n_max_full <- rep(NA_real_, n_cells)
      n_min_full <- rep(NA_real_, n_cells)
      n_max_full[stats$from] <- stats$nmax
      n_min_full[stats$from] <- stats$nmin
    } else {
      n_max_full <- rep(NA_real_, n_cells)
      n_min_full <- rep(NA_real_, n_cells)
    }
    
    # --- Write results back to dt for this year's rows ---
    # Map from cell_pos back to the rows
    pos_vals <- yr_dt$cell_pos
    
    set(dt, which = yr_mask, j = col_max,  value = n_max_full[dt$cell_pos[yr_mask]])
    set(dt, which = yr_mask, j = col_min,  value = n_min_full[dt$cell_pos[yr_mask]])
    set(dt, which = yr_mask, j = col_mean, value = n_mean[dt$cell_pos[yr_mask]])
  }
  
  dt
}

# ==============================================================
# STEP 5: Run for all 5 neighbor source variables
# ==============================================================
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {
  message("Computing neighbor features for: ", var_name)
  dt <- compute_neighbor_features_fast(
    dt, var_name, W, edges, neighbor_count, id_order, years
  )
}

# ==============================================================
# STEP 6: Remove helper column and convert back if needed
# ==============================================================
dt[, cell_pos := NULL]

# If downstream code expects a data.frame:
cell_data <- as.data.frame(dt)

# The trained Random Forest model is untouched â€” 
# the output columns are numerically identical to the originals,
# so predict(rf_model, cell_data) works without retraining.
```

---

## Complexity Comparison

| Step | Original | Optimized |
|---|---|---|
| Build lookup | O(6.46M) string ops, ~80 hrs | O(1.37M) sparse triplet construction, **~1 sec** |
| Compute stats (per var, per year) | O(230K) list iterations, R-level | Sparse mat-vec O(1.37M) + `data.table` groupby O(1.37M), C-level |
| Total (5 vars Ã— 28 years) | **~86+ hours** | **~2â€“5 minutes** |
| Memory | ~6.46M-element list of int vectors | Sparse matrix (~16 MB) + edge table (~33 MB) |

## Numerical Equivalence

The sparse matrix approach computes **exactly the same** quantities:

- `neighbor_mean` = sum of non-NA neighbor values / count of non-NA neighbors (identical to the original `mean(neighbor_vals[!is.na(neighbor_vals)])`)
- `neighbor_max` / `neighbor_min` = grouped max/min over the same edge set (identical to original `max`/`min` on the same `neighbor_vals`)

The trained Random Forest model is **never touched** â€” only the input feature columns are recomputed with identical values, so all downstream predictions remain unchanged.
END_RESPONSE>>>
