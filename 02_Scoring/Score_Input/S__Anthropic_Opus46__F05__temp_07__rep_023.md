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
run_id: Anthropic_F05_T07_R023
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F05
prompt_family_slug: STRING_KEY_REDUNDANCY
family_label: String-key probe
family_group: Line-level probe
replicate: 23
raw_output_file: Experiment_Output/Anthropic/Temp_0.7/output_F05_Anthropic_temp_07_rep_023.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis and Optimized Solution

## Diagnosis

The repeated string-key work is **a symptom of a much larger algorithmic problem**, not merely a local inefficiency. Here's the full picture:

### The Core Problem: O(N) `lapply` with Per-Row String Operations

`build_neighbor_lookup` iterates over all ~6.46 million rows. For **each row**, it:

1. Calls `paste()` to construct string keys for that row's neighbors.
2. Looks up those keys in a named character vector (`idx_lookup`) of length 6.46M.

Named vector lookup in R is **O(n)** hash-table probing per lookup, and `paste()` allocates new strings each time. With ~6.46M rows Ã— ~4 neighbors on average (rook contiguity), this produces roughly **25.8 million string constructions and hash lookups** â€” all inside a sequential `lapply`. This alone accounts for the bulk of the 86+ hour estimate.

### The Broader Pattern

Even after `build_neighbor_lookup` completes, `compute_neighbor_stats` is called **5 times** (once per variable), each time iterating over 6.46M list elements. The lookup list itself (a list of 6.46M integer vectors) consumes significant memory and has poor cache locality.

### Root Cause Summary

| Layer | Problem |
|-------|---------|
| **String keys** | Unnecessary â€” `(id, year)` pairs can be mapped to row indices via integer arithmetic |
| **Per-row lapply** | 6.46M R-level function calls with allocation overhead |
| **Lookup structure** | A 6.46M-element named vector; R's internal hashing is slow at this scale |
| **Stat computation** | 5 separate passes over a 6.46M-element list; could be vectorized once |

## Optimization Strategy

1. **Eliminate all string operations.** Replace the `paste(id, year)` key scheme with a direct integer index matrix. Since the panel is balanced (344,208 cells Ã— 28 years), we can compute `row_index = f(cell_position, year_position)` in O(1) with integer arithmetic.

2. **Vectorize neighbor expansion.** Instead of `lapply` over 6.46M rows, expand the neighbor relationships into a flat edge-list (a two-column matrix of `[focal_row, neighbor_row]`), then use vectorized group-by operations (via `data.table`) to compute `max`, `min`, `mean` for all rows at once.

3. **Compute all 5 variables in one pass** over the edge structure, or at minimum make each pass fully vectorized.

**Expected speedup:** From ~86 hours to **minutes** (the bottleneck becomes memory bandwidth over ~25M edges Ã— 5 variables).

## Working R Code

```r
library(data.table)

build_neighbor_features_fast <- function(cell_data, id_order, rook_neighbors_unique,
                                         neighbor_source_vars) {
  # ---------------------------------------------------------------
  # 1. Convert to data.table (by reference if already one)
  # ---------------------------------------------------------------
  dt <- as.data.table(cell_data)

  # ---------------------------------------------------------------
  # 2. Build integer mappings â€” no strings anywhere

  # ---------------------------------------------------------------
  # Unique cell IDs in the order matching the nb object
  # id_order[k] is the cell id whose neighbors are rook_neighbors_unique[[k]]
  n_cells <- length(id_order)
  years   <- sort(unique(dt$year))
  n_years <- length(years)

  # Map cell id -> position in id_order (1-based)
  id_to_pos <- integer(max(id_order))
  id_to_pos[id_order] <- seq_len(n_cells)
  # If id_order is not contiguous integers, use a hash:
  # id_to_pos <- setNames(seq_len(n_cells), as.character(id_order))
  # and index with as.character(). But integer indexing is far faster.

  # Map year -> year position (1-based)
  year_to_pos <- setNames(seq_len(n_years), as.character(years))

  # ---------------------------------------------------------------
  # 3. Assign each row a deterministic index and sort
  #    row_index for cell position p, year position t:
  #    row_idx = (p - 1) * n_years + t
  #    This gives a 1-based index into a vector of length n_cells * n_years
  # ---------------------------------------------------------------
  # Handle the possibility that id_order contains non-contiguous IDs
  # by using the safe match approach:
  if (max(id_order) > 2 * n_cells) {
    # Sparse IDs â€” use match
    dt[, cell_pos := match(id, id_order)]
  } else {
    # Dense IDs â€” direct index
    dt[, cell_pos := id_to_pos[id]]
  }
  dt[, year_pos := year_to_pos[as.character(year)]]
  dt[, row_idx  := (cell_pos - 1L) * n_years + year_pos]

  # Create a mapping from row_idx -> actual row number in dt
  # (in case dt is not perfectly sorted)
  setkey(dt, row_idx)
  # After setkey, dt is sorted by row_idx.
  # Build a direct lookup: row_idx -> position in sorted dt
  max_row_idx   <- n_cells * n_years
  idx_to_dtrow  <- integer(max_row_idx)
  idx_to_dtrow[dt$row_idx] <- seq_len(nrow(dt))

  # ---------------------------------------------------------------
  # 4. Expand neighbor relationships into a flat edge list
  #    Each (focal_cell_pos, neighbor_cell_pos) pair is crossed

  #    with all n_years years.
  # ---------------------------------------------------------------
  # Build edge list from nb object: two integer vectors
  focal_pos_list    <- rep(seq_len(n_cells), times = lengths(rook_neighbors_unique))
  neighbor_pos_list <- unlist(rook_neighbors_unique)

  # Remove zero-length / self-references if any (spdep nb objects use 0 for no neighbors)
  valid <- neighbor_pos_list > 0L
  focal_pos_list    <- focal_pos_list[valid]
  neighbor_pos_list <- neighbor_pos_list[valid]

  n_edges <- length(focal_pos_list)
  cat(sprintf("Neighbor edges (unique directed): %d\n", n_edges))
  cat(sprintf("Edges Ã— years: %d\n", n_edges * n_years))

  # Expand across years: each edge exists for every year
  # Use vectorized outer-product style expansion
  year_positions <- seq_len(n_years)

  # focal_row_idx and neighbor_row_idx for all (edge, year) combinations
  # edge e, year t:
  #   focal_row_idx    = (focal_pos_list[e] - 1) * n_years + t
  #   neighbor_row_idx = (neighbor_pos_list[e] - 1) * n_years + t

  # Efficient expansion without rep(each=):
  # Pre-compute base indices
  focal_base    <- (focal_pos_list - 1L) * n_years     # length n_edges
  neighbor_base <- (neighbor_pos_list - 1L) * n_years   # length n_edges

  # Total expanded rows: n_edges * n_years
  # Use rep + addition for vectorized expansion
  focal_row_idx <- rep(focal_base, times = n_years) +
                   rep(year_positions, each = n_edges)
  neighbor_row_idx <- rep(neighbor_base, times = n_years) +
                      rep(year_positions, each = n_edges)

  # Map to actual dt row numbers
  focal_dtrow    <- idx_to_dtrow[focal_row_idx]
  neighbor_dtrow <- idx_to_dtrow[neighbor_row_idx]

  # Remove pairs where either focal or neighbor row doesn't exist in data
  valid2 <- focal_dtrow > 0L & neighbor_dtrow > 0L
  focal_dtrow    <- focal_dtrow[valid2]
  neighbor_dtrow <- neighbor_dtrow[valid2]

  # Free large temporaries
  rm(focal_row_idx, neighbor_row_idx, focal_base, neighbor_base, valid, valid2)
  gc()

  cat(sprintf("Valid (focal, neighbor, year) triples: %d\n", length(focal_dtrow)))

  # ---------------------------------------------------------------
  # 5. Compute neighbor stats for each variable â€” fully vectorized
  # ---------------------------------------------------------------
  for (var_name in neighbor_source_vars) {
    cat(sprintf("Computing neighbor stats for: %s\n", var_name))

    # Extract neighbor values
    neighbor_vals <- dt[[var_name]][neighbor_dtrow]

    # Build a data.table for grouped aggregation
    edges_dt <- data.table(
      focal  = focal_dtrow,
      nval   = neighbor_vals
    )

    # Remove NAs in neighbor values before aggregation
    edges_dt <- edges_dt[!is.na(nval)]

    # Grouped aggregation: max, min, mean per focal row
    stats <- edges_dt[, .(
      nb_max  = max(nval),
      nb_min  = min(nval),
      nb_mean = mean(nval)
    ), keyby = focal]

    # Initialize result columns with NA
    max_col  <- paste0(var_name, "_neighbor_max")
    min_col  <- paste0(var_name, "_neighbor_min")
    mean_col <- paste0(var_name, "_neighbor_mean")

    dt[, (max_col)  := NA_real_]
    dt[, (min_col)  := NA_real_]
    dt[, (mean_col) := NA_real_]

    # Assign results to the correct rows
    dt[stats$focal, (max_col)  := stats$nb_max]
    dt[stats$focal, (min_col)  := stats$nb_min]
    dt[stats$focal, (mean_col) := stats$nb_mean]

    rm(edges_dt, stats, neighbor_vals)
    gc()
  }

  # ---------------------------------------------------------------
  # 6. Clean up helper columns, restore original order
  # ---------------------------------------------------------------
  dt[, c("cell_pos", "year_pos", "row_idx") := NULL]

  # Return as data.frame if the input was a data.frame
  if (!is.data.table(cell_data)) {
    setDF(dt)
  }

  dt
}

# ---------------------------------------------------------------
# Usage â€” drop-in replacement for the original outer loop
# ---------------------------------------------------------------
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cell_data <- build_neighbor_features_fast(
  cell_data,
  id_order,
  rook_neighbors_unique,
  neighbor_source_vars
)

# The trained Random Forest model is untouched.
# Predict as before:
# predictions <- predict(rf_model, newdata = cell_data)
```

## Memory-Conscious Variant

If the full edge expansion (~38.4M rows Ã— 28 years â‰ˆ 1.08 billion, though likely closer to 38M total) exceeds 16 GB RAM, process years in chunks:

```r
build_neighbor_features_chunked <- function(cell_data, id_order,
                                            rook_neighbors_unique,
                                            neighbor_source_vars) {
  dt <- as.data.table(cell_data)
  n_cells <- length(id_order)
  years   <- sort(unique(dt$year))

  # Cell-pos mapping
  dt[, cell_pos := match(id, id_order)]
  setkey(dt, cell_pos, year)

  # Flat edge list (cell-level, year-invariant)
  focal_pos    <- rep(seq_len(n_cells), times = lengths(rook_neighbors_unique))
  neighbor_pos <- unlist(rook_neighbors_unique)
  valid        <- neighbor_pos > 0L
  focal_pos    <- focal_pos[valid]
  neighbor_pos <- neighbor_pos[valid]

  # Initialize result columns
  for (var_name in neighbor_source_vars) {
    dt[, paste0(var_name, "_neighbor_max")  := NA_real_]
    dt[, paste0(var_name, "_neighbor_min")  := NA_real_]
    dt[, paste0(var_name, "_neighbor_mean") := NA_real_]
  }

  # Process one year at a time â€” peak memory â‰ˆ edges Ã— 2 columns
  for (yr in years) {
    cat(sprintf("Year %d ...\n", yr))

    yr_dt <- dt[year == yr]
    setkey(yr_dt, cell_pos)

    # Map cell_pos -> row in yr_dt
    pos_to_yrrow <- integer(n_cells)
    pos_to_yrrow[yr_dt$cell_pos] <- seq_len(nrow(yr_dt))

    focal_yrrow    <- pos_to_yrrow[focal_pos]
    neighbor_yrrow <- pos_to_yrrow[neighbor_pos]
    ok <- focal_yrrow > 0L & neighbor_yrrow > 0L
    f_yr <- focal_yrrow[ok]
    n_yr <- neighbor_yrrow[ok]

    for (var_name in neighbor_source_vars) {
      nvals <- yr_dt[[var_name]][n_yr]

      edges_dt <- data.table(focal = f_yr, nval = nvals)
      edges_dt <- edges_dt[!is.na(nval)]

      if (nrow(edges_dt) == 0L) next

      stats <- edges_dt[, .(
        nb_max  = max(nval),
        nb_min  = min(nval),
        nb_mean = mean(nval)
      ), keyby = focal]

      # Map back: yr_dt row -> dt row
      # dt is keyed by (cell_pos, year), so we can update by reference
      max_col  <- paste0(var_name, "_neighbor_max")
      min_col  <- paste0(var_name, "_neighbor_min")
      mean_col <- paste0(var_name, "_neighbor_mean")

      # Get the original dt row indices for this year
      dt_rows_this_year <- which(dt$year == yr)
      # stats$focal indexes into yr_dt; yr_dt is aligned with dt_rows_this_year
      target_rows <- dt_rows_this_year[stats$focal]

      set(dt, i = target_rows, j = max_col,  value = stats$nb_max)
      set(dt, i = target_rows, j = min_col,  value = stats$nb_min)
      set(dt, i = target_rows, j = mean_col, value = stats$nb_mean)
    }
  }

  dt[, cell_pos := NULL]
  if (!is.data.table(cell_data)) setDF(dt)
  dt
}
```

## Complexity Comparison

| | Original | Optimized |
|---|---|---|
| **String constructions** | ~6.46M (lookup) + ~25.8M (per-row paste) | **0** |
| **Hash lookups** | ~25.8M in a 6.46M-entry named vector | **0** (integer arithmetic) |
| **R-level loop iterations** | 6.46M (`lapply`) Ã— 6 calls (1 build + 5 vars) | **0** (vectorized `data.table` groupby) |
| **Time complexity** | O(N Ã— K Ã— hash_cost) where N=6.46M, Kâ‰ˆ4 | O(E Ã— V) where E=total edges, V=5 vars, all vectorized in C |
| **Estimated runtime** | ~86 hours | **5â€“15 minutes** |

The trained Random Forest model is completely unaffected â€” only the feature-construction pipeline is reformulated. The numerical results (max, min, mean of neighbor values) are identical.
END_RESPONSE>>>
