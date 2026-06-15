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
run_id: Anthropic_F05_T03_R022
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.3
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F05
prompt_family_slug: STRING_KEY_REDUNDANCY
family_label: String-key probe
family_group: Line-level probe
replicate: 22
raw_output_file: Experiment_Output/Anthropic/Temp_0.3/output_F05_Anthropic_temp_03_rep_022.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis and Optimized Solution

## Diagnosis

The repeated string-key work is **a symptom of a larger repeated-lookup pattern**, not merely a local inefficiency. Here's the full picture:

### The Core Problem: O(N) `lapply` with String Operations Inside the Loop

`build_neighbor_lookup` iterates over all ~6.46 million rows. For **each row**, it:

1. Converts `data$id[i]` to character and looks it up in `id_to_ref` â€” 6.46M character coercions + named-vector lookups.
2. Retrieves neighbor cell IDs from the `nb` object.
3. Calls `paste()` to build string keys for every neighbor of that row â€” across all rows this creates roughly `6.46M Ã— avg_neighbors â‰ˆ 25.8M+` string concatenations.
4. Indexes into `idx_lookup` (a ~6.46M-element named character vector) by string matching â€” this is an **O(n)** hash probe per key on a very large vector, repeated ~25.8M times.

The string-key approach is fundamentally the wrong data structure. The neighbor topology is **time-invariant** â€” cell ð‘–'s neighbors are the same in every year. Yet the code re-discovers this for every cell-year row by encoding `(id, year)` as a string. This means the spatial lookup is repeated 28 times (once per year), and all the string machinery is pure overhead.

### The Second Problem: Serial `compute_neighbor_stats`

`compute_neighbor_stats` loops over 6.46M row-index vectors, subsetting and computing `max/min/mean` in R-level `lapply`. This is repeated 5 times (once per variable). That's ~32.3M R-level function calls with repeated subsetting.

### Summary

| Layer | Waste Factor |
|---|---|
| String key construction | ~25.8M `paste()` calls, entirely eliminable |
| String key lookup | ~25.8M named-vector probes on a 6.46M-element vector |
| Year-invariant topology repeated per year | 28Ã— redundant spatial work |
| R-level `lapply` for stats | ~32.3M interpreted iterations across 5 variables |

**Estimated speedup from the reformulation below: ~200â€“500Ã—**, bringing runtime from 86+ hours to roughly 10â€“25 minutes.

---

## Optimization Strategy

1. **Separate space and time.** Build the neighbor lookup once in cell-space (344K cells), then expand to cell-year space via integer arithmetic â€” no strings.
2. **Use integer indexing throughout.** Map cell IDs to integer positions once. Map years to integer offsets once. Compute row indices as `(cell_position - 1) * n_years + year_offset` â€” pure integer math.
3. **Vectorize the statistics computation.** Unroll the neighbor list into a flat edge list, use vectorized subsetting and `rowsum()`/group-by operations to compute `max`, `min`, `mean` without any per-row `lapply`.
4. **Process all 5 variables in one pass** over the edge structure.

---

## Working R Code

```r
# =============================================================================
# OPTIMIZED NEIGHBOR FEATURE CONSTRUCTION
# Drop-in replacement for build_neighbor_lookup + compute_neighbor_stats loop
# Preserves the exact numerical estimand (max, min, mean of rook neighbors).
# =============================================================================

build_neighbor_features_optimized <- function(cell_data,
                                               id_order,
                                               rook_neighbors_unique,
                                               neighbor_source_vars) {
  # ------------------------------------------------------------------
  # 1. Build integer mappings (no strings anywhere)
  # ------------------------------------------------------------------
  
  # Unique cell IDs in the order matching the nb object
  # id_order[k] is the cell ID whose neighbors are rook_neighbors_unique[[k]]
  n_cells <- length(id_order)
  
  # Map cell ID -> position in id_order (integer)
  id_to_pos <- integer(max(id_order))
  id_to_pos[id_order] <- seq_len(n_cells)
  # If IDs are not contiguous integers, use a hash:
  # But for grid cells they typically are. Fallback:
  if (max(id_order) > 10L * n_cells) {
    # Sparse IDs â€” use environment-based hash
    id_to_pos_env <- new.env(hash = TRUE, size = n_cells)
    for (k in seq_len(n_cells)) {
      id_to_pos_env[[as.character(id_order[k])]] <- k
    }
    get_pos <- function(ids) {
      vapply(as.character(ids), function(x) id_to_pos_env[[x]], integer(1),
             USE.NAMES = FALSE)
    }
  } else {
    get_pos <- function(ids) id_to_pos[ids]
  }
  
  # Unique sorted years and year -> offset mapping
  years_unique <- sort(unique(cell_data$year))
  n_years      <- length(years_unique)
  year_to_offset <- integer(max(years_unique))
  year_to_offset[years_unique] <- seq_len(n_years)
  
  # ------------------------------------------------------------------
  # 2. Ensure cell_data is sorted by (id, year) so we can use arithmetic
  #    indexing: row = (cell_pos - 1) * n_years + year_offset
  # ------------------------------------------------------------------
  cell_data <- cell_data[order(cell_data$id, cell_data$year), ]
  
  # Verify the sort produces the expected layout
  cell_positions <- get_pos(cell_data$id)
  year_offsets   <- year_to_offset[cell_data$year]
  expected_row   <- (cell_positions - 1L) * n_years + year_offsets
  
  if (!all(expected_row == seq_len(nrow(cell_data)))) {
    # Some cells may not have all years â€” build explicit row index
    # This handles unbalanced panels
    row_index <- integer(n_cells * n_years)  # NA-filled
    row_index[(cell_positions - 1L) * n_years + year_offsets] <- seq_len(nrow(cell_data))
    balanced <- FALSE
    message("Panel is unbalanced; using explicit row-index mapping.")
  } else {
    row_index <- NULL
    balanced <- TRUE
    message("Panel is balanced; using arithmetic row indexing.")
  }
  
  # Helper: given cell_pos (vector) and year_offset (scalar or vector),
  # return row numbers in cell_data
  get_rows <- if (balanced) {
    function(cpos, yoff) (cpos - 1L) * n_years + yoff
  } else {
    function(cpos, yoff) {
      idx <- (cpos - 1L) * n_years + yoff
      row_index[idx]  # may contain 0 or NA for missing cell-years
    }
  }
  
  # ------------------------------------------------------------------
  # 3. Build flat edge list from nb object (cell-space, time-invariant)
  #    from_pos -> to_pos (directed: each neighbor pair appears once per

  #    direction, matching the original code's behavior)
  # ------------------------------------------------------------------
  message("Building flat edge list from nb object...")
  
  # Pre-calculate total edges for memory allocation
  n_edges <- sum(vapply(rook_neighbors_unique, function(x) {
    if (length(x) == 1L && x[1] == 0L) 0L else length(x)
  }, integer(1)))
  
  from_pos <- integer(n_edges)
  to_pos   <- integer(n_edges)
  ptr <- 0L
  
  for (k in seq_len(n_cells)) {
    nb_k <- rook_neighbors_unique[[k]]
    if (length(nb_k) == 1L && nb_k[1] == 0L) next
    n_nb <- length(nb_k)
    from_pos[ptr + seq_len(n_nb)] <- k
    to_pos[ptr + seq_len(n_nb)]   <- nb_k  # nb objects store positions directly
    ptr <- ptr + n_nb
  }
  
  message(sprintf("Edge list: %d directed edges across %d cells.", n_edges, n_cells))
  
  # ------------------------------------------------------------------
  # 4. Expand edge list across years and compute stats (vectorized)
  # ------------------------------------------------------------------
  message("Computing neighbor statistics for ", length(neighbor_source_vars), " variables...")
  
  n_rows <- nrow(cell_data)
  
  for (var_name in neighbor_source_vars) {
    message("  Processing: ", var_name)
    
    vals <- cell_data[[var_name]]
    
    # Allocate output columns
    col_max  <- rep(NA_real_, n_rows)
    col_min  <- rep(NA_real_, n_rows)
    col_mean <- rep(NA_real_, n_rows)
    
    # Process one year at a time to keep memory bounded
    # For each year: expand the spatial edge list, look up values, aggregate
    for (y in seq_len(n_years)) {
      # Row indices for "from" cells in this year
      from_rows <- get_rows(from_pos, y)
      # Row indices for "to" (neighbor) cells in this year
      to_rows   <- get_rows(to_pos, y)
      
      # Remove edges where either endpoint is missing (unbalanced panel)
      valid <- !is.na(from_rows) & !is.na(to_rows) & (from_rows > 0L) & (to_rows > 0L)
      
      fr <- from_rows[valid]
      tr <- to_rows[valid]
      
      # Get neighbor values
      nb_vals <- vals[tr]
      
      # Remove edges where the neighbor value is NA
      not_na <- !is.na(nb_vals)
      fr     <- fr[not_na]
      nb_vals <- nb_vals[not_na]
      
      if (length(fr) == 0L) next
      
      # Aggregate by "from" row using fast grouped operations
      # Use data.table for speed if available, otherwise tapply
      if (requireNamespace("data.table", quietly = TRUE)) {
        dt <- data.table::data.table(fr = fr, v = nb_vals)
        agg <- dt[, .(vmax = max(v), vmin = min(v), vsum = sum(v), vn = .N),
                  keyby = fr]
        col_max[agg$fr]  <- agg$vmax
        col_min[agg$fr]  <- agg$vmin
        col_mean[agg$fr] <- agg$vsum / agg$vn
      } else {
        # Fallback: rowsum is vectorized C code, very fast
        # For max/min we need split-apply; use tapply
        col_max[sort(unique(fr))]  <- tapply(nb_vals, fr, max)
        col_min[sort(unique(fr))]  <- tapply(nb_vals, fr, min)
        sums   <- rowsum(nb_vals, fr, reorder = TRUE)
        counts <- rowsum(rep(1, length(fr)), fr, reorder = TRUE)
        ufr    <- as.integer(rownames(sums))
        col_mean[ufr] <- sums[, 1] / counts[, 1]
      }
    }
    
    # Assign to cell_data with original column naming convention
    cell_data[[paste0(var_name, "_neighbor_max")]]  <- col_max
    cell_data[[paste0(var_name, "_neighbor_min")]]  <- col_min
    cell_data[[paste0(var_name, "_neighbor_mean")]] <- col_mean
  }
  
  message("Done. Neighbor features added.")
  return(cell_data)
}

# =============================================================================
# USAGE â€” drop-in replacement for the original outer loop
# =============================================================================

# Original code (86+ hours):
# neighbor_lookup <- build_neighbor_lookup(cell_data, id_order, rook_neighbors_unique)
# for (var_name in neighbor_source_vars) {
#   cell_data <- compute_and_add_neighbor_features(cell_data, var_name, neighbor_lookup)
# }

# Optimized replacement (~10-25 minutes):
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cell_data <- build_neighbor_features_optimized(
  cell_data            = cell_data,
  id_order             = id_order,
  rook_neighbors_unique = rook_neighbors_unique,
  neighbor_source_vars = neighbor_source_vars
)

# The trained Random Forest model is untouched â€” only the feature-construction
# pipeline is replaced. The output columns have identical names and identical
# numerical values (max, min, mean of rook-neighbor values per cell-year).
```

---

## Why This Preserves Correctness

| Original | Optimized | Equivalence |
|---|---|---|
| For row `i`, finds neighbors of `cell_data$id[i]` via string keys in the same year | For row `i`, finds neighbors via integer position arithmetic in the same year | Same neighbor set, same year filter |
| Computes `max`, `min`, `mean` of non-NA neighbor values | Computes `max`, `min`, `mean` of non-NA neighbor values via vectorized grouped aggregation | Identical numerical result |
| Outputs columns like `ntl_neighbor_max` | Outputs columns like `ntl_neighbor_max` | Same column names |

The Random Forest model is never touched â€” only the upstream feature matrix construction is accelerated.

## Complexity Comparison

| | Original | Optimized |
|---|---|---|
| String constructions | ~32M `paste()` calls | **0** |
| Named-vector probes | ~25.8M on 6.46M-element vector | **0** |
| R-level loop iterations | ~32.3M (across 5 vars) | **140** (28 years Ã— 5 vars) |
| Core aggregation | Interpreted R per row | Vectorized C (data.table / rowsum) |
| Memory for lookup | ~6.46M-element named character vector | ~1.37M-element integer edge list |
| **Estimated wall time** | **86+ hours** | **10â€“25 minutes** |
END_RESPONSE>>>
