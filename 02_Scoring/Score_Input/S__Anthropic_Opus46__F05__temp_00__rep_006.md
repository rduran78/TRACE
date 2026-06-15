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
run_id: Anthropic_F05_T00_R006
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F05
prompt_family_slug: STRING_KEY_REDUNDANCY
family_label: String-key probe
family_group: Line-level probe
replicate: 6
raw_output_file: Experiment_Output/Anthropic/Temp_0.0/output_F05_Anthropic_temp_00_rep_006.md

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

The string-key lookup on a named vector of length 6.46M is approximately O(N) per probe (R's named vector lookup uses hashing, but the construction and GC pressure of millions of temporary character vectors is enormous). Total cost: **~6.46M Ã— 4 Ã— hash-lookup = ~25.8M hash probes**, plus the string allocation/GC overhead.

### The Deeper Structural Insight

The neighbor relationship is **spatial only** â€” it doesn't change across years. Every cell has the same neighbors in every year. The current code re-discovers this for every cell-year row by encoding `(cell_id, year)` as a string key. This is unnecessary. The neighbor lookup can be built **once at the cell level** (344K cells), and then applied across all 28 years via vectorized indexing.

Similarly, `compute_neighbor_stats` runs an `lapply` over 6.46M rows, extracting neighbor values one row at a time. This can be replaced with a **fully vectorized matrix operation**.

### Summary of Inefficiencies

| Layer | Problem | Scale |
|-------|---------|-------|
| String key construction | `paste()` inside 6.46M-iteration loop | ~25.8M string allocs |
| Named vector lookup | Hash probe on 6.46M-element named vector, per row | ~25.8M probes |
| Row-level `lapply` in `build_neighbor_lookup` | Inherently serial R loop over 6.46M rows | 6.46M iterations |
| Row-level `lapply` in `compute_neighbor_stats` | Another serial R loop over 6.46M rows, repeated 5Ã— | 32.3M iterations |
| Redundant structure | Neighbor topology is year-invariant but re-resolved per cell-year | 28Ã— redundant work |

---

## Optimization Strategy

1. **Exploit year-invariance**: Build a neighbor index at the **cell level** (344K entries), not the cell-year level (6.46M entries). The `spdep::nb` object already provides this.

2. **Convert the ragged neighbor list to a CSR (Compressed Sparse Row) representation**: Two integer vectors (`adj_ptr` and `adj_ids`) replace millions of list elements. This enables fully vectorized operations.

3. **Arrange data so that all years for a given cell are contiguous** (or use a cell-to-rows mapping). Then neighbor stats for all years can be computed via vectorized matrix/column operations.

4. **Replace all `lapply` loops with vectorized `rowSums`/`rowMeans`-style operations** using sparse-matrix or direct indexed arithmetic.

5. **Compute all 5 variables' neighbor stats in one pass** over the adjacency structure.

Expected speedup: from ~86+ hours to **minutes** (roughly 3â€“10 minutes depending on RAM pressure).

---

## Working R Code

```r
# =============================================================================
# OPTIMIZED NEIGHBOR FEATURE CONSTRUCTION
# =============================================================================
# Preserves the exact numerical estimand: for each cell-year row and each
# neighbor source variable, compute max, min, and mean of that variable
# across the cell's rook neighbors in the SAME year.
#
# Requirements:
#   - cell_data: data.frame/data.table with columns 'id', 'year', and the
#     neighbor_source_vars. Rows are cell-year observations.
#   - id_order: integer/numeric vector of cell IDs in the order matching
#     rook_neighbors_unique (i.e., id_order[k] is the cell ID for the
#     k-th element of the nb object).
#   - rook_neighbors_unique: an spdep::nb object (list of integer vectors
#     of neighbor indices, with 0L for no-neighbor entries).
#   - neighbor_source_vars: character vector of variable names.
# =============================================================================

library(data.table)

optimized_neighbor_features <- function(cell_data,
                                        id_order,
                                        rook_neighbors_unique,
                                        neighbor_source_vars) {

  # --- Step 0: Convert to data.table for speed (non-destructive) -----------
  dt <- as.data.table(cell_data)

  # --- Step 1: Build cell-level adjacency in CSR format --------------------
  # Map cell id -> reference index (position in id_order / nb object)
  n_cells <- length(id_order)
  id_to_ref <- integer(max(id_order))
  id_to_ref[id_order] <- seq_len(n_cells)
  # If id_order values are very large/sparse, use a hash instead:
  # id_to_ref_env <- new.env(hash = TRUE, size = n_cells)
  # for (k in seq_len(n_cells)) id_to_ref_env[[as.character(id_order[k])]] <- k

  # Clean the nb object: replace 0L (spdep's "no neighbor" code) with empty
  nb_clean <- lapply(rook_neighbors_unique, function(x) {
    x <- x[x != 0L]
    x
  })

  # Build CSR: adj_ids contains neighbor *cell IDs* (not ref indices),
  # adj_ptr[k]:(adj_ptr[k+1]-1) indexes into adj_ids for cell k (by ref index)
  adj_lengths <- vapply(nb_clean, length, integer(1))
  adj_ptr     <- c(1L, cumsum(adj_lengths) + 1L)  # length n_cells + 1
  adj_ref_ids <- unlist(nb_clean, use.names = FALSE)  # neighbor ref indices
  adj_cell_ids <- id_order[adj_ref_ids]               # neighbor cell IDs

  cat(sprintf("Adjacency CSR built: %d cells, %d directed edges\n",
              n_cells, length(adj_cell_ids)))

  # --- Step 2: Ensure data is keyed by (id, year) for fast join ------------
  setkey(dt, id, year)

  # Create a row-index column so we can map back
  dt[, .row_idx := .I]

  # --- Step 3: Build cell-ref to data-row mapping -------------------------
  # For each cell (by ref index) and each year, we need the row in dt.
  # We'll work year-by-year to keep memory bounded.

  years <- sort(unique(dt$year))
  n_years <- length(years)

  # Pre-allocate output columns
  for (var_name in neighbor_source_vars) {
    max_col <- paste0("neighbor_max_", var_name)
    min_col <- paste0("neighbor_min_", var_name)
    mean_col <- paste0("neighbor_mean_", var_name)
    dt[, (max_col)  := NA_real_]
    dt[, (min_col)  := NA_real_]
    dt[, (mean_col) := NA_real_]
  }

  cat(sprintf("Processing %d years x %d variables...\n",
              n_years, length(neighbor_source_vars)))

  # --- Step 4: Year-by-year vectorized computation -------------------------
  for (yr in years) {

    # Subset rows for this year
    yr_rows <- dt[year == yr]
    n_yr <- nrow(yr_rows)
    if (n_yr == 0) next

    # Map cell_id -> position within this year's subset
    # (We need to look up neighbor values by cell_id within the same year)
    yr_cell_ids <- yr_rows$id
    yr_row_indices <- yr_rows$.row_idx  # original row positions in dt

    # Create a fast lookup: cell_id -> index in yr_rows
    # Using a pre-allocated vector indexed by cell_id (fast if IDs are dense)
    max_id <- max(yr_cell_ids)
    cellid_to_yrpos <- rep(NA_integer_, max_id)
    cellid_to_yrpos[yr_cell_ids] <- seq_len(n_yr)

    # For each cell present this year, find its ref index
    yr_ref_indices <- id_to_ref[yr_cell_ids]  # ref index for each row this year

    # For each row this year, gather neighbor values using CSR adjacency
    # We vectorize this by expanding the adjacency for all cells present this year

    # adj_start and adj_end for each row's cell
    a_start <- adj_ptr[yr_ref_indices]
    a_end   <- adj_ptr[yr_ref_indices + 1L] - 1L
    a_len   <- a_end - a_start + 1L
    a_len[a_len < 0L] <- 0L  # cells with no neighbors

    # Total number of (row, neighbor) pairs this year
    total_pairs <- sum(a_len)

    # Expand: for each row i (1..n_yr), repeat i a_len[i] times
    row_rep <- rep(seq_len(n_yr), times = a_len)

    # Gather the neighbor cell IDs for all rows
    # We need to index into adj_cell_ids using the CSR ranges
    # Build the flat index into adj_cell_ids
    seq_within <- sequence(a_len, from = a_start)
    neighbor_cids <- adj_cell_ids[seq_within]

    # Map neighbor cell IDs to their position in this year's data
    neighbor_yrpos <- cellid_to_yrpos[neighbor_cids]

    # Some neighbors may not be present this year (NA) â€” mark for removal
    valid <- !is.na(neighbor_yrpos)
    row_rep_v <- row_rep[valid]
    neighbor_yrpos_v <- neighbor_yrpos[valid]

    # Now compute stats for each variable
    for (var_name in neighbor_source_vars) {
      vals_yr <- yr_rows[[var_name]]

      # Gather neighbor values
      nvals <- vals_yr[neighbor_yrpos_v]

      # Remove NAs in the variable itself
      valid_val <- !is.na(nvals)
      rr <- row_rep_v[valid_val]
      nv <- nvals[valid_val]

      if (length(nv) == 0) next

      # Compute grouped max, min, sum, count using data.table
      tmp <- data.table(row = rr, val = nv)
      stats <- tmp[, .(
        nb_max  = max(val),
        nb_min  = min(val),
        nb_mean = mean(val)
      ), by = row]

      # Write results back into dt at the correct original row positions
      target_rows <- yr_row_indices[stats$row]

      max_col  <- paste0("neighbor_max_", var_name)
      min_col  <- paste0("neighbor_min_", var_name)
      mean_col <- paste0("neighbor_mean_", var_name)

      set(dt, i = target_rows, j = max_col,  value = stats$nb_max)
      set(dt, i = target_rows, j = min_col,  value = stats$nb_min)
      set(dt, i = target_rows, j = mean_col, value = stats$nb_mean)
    }

    if (yr %% 5 == 0 || yr == years[1] || yr == years[n_years]) {
      cat(sprintf("  Year %d done (%d rows, %d valid neighbor pairs)\n",
                  yr, n_yr, sum(valid)))
    }
  }

  # --- Step 5: Clean up and return -----------------------------------------
  dt[, .row_idx := NULL]

  cat("Neighbor feature construction complete.\n")
  return(dt)
}

# =============================================================================
# USAGE (drop-in replacement for the original outer loop)
# =============================================================================
# cell_data <- optimized_neighbor_features(
#   cell_data,
#   id_order,
#   rook_neighbors_unique,
#   neighbor_source_vars = c("ntl", "ec", "pop_density", "def", "usd_est_n2")
# )
#
# # The trained Random Forest model is untouched â€” just use cell_data for
# # prediction as before. The output columns have identical names and values.
```

---

## Why This Is Correct and Equivalent

| Property | Original | Optimized |
|----------|----------|-----------|
| Neighbor definition | Rook neighbors from `spdep::nb` | Same `spdep::nb` object, same adjacency |
| Year matching | `paste(neighbor_id, year)` key ensures same-year lookup | Explicit year-by-year loop ensures same-year lookup |
| Missing handling | `!is.na(result)` drops neighbors absent in that year; `!is.na(neighbor_vals)` drops NA values | `!is.na(neighbor_yrpos)` drops absent neighbors; `!is.na(nvals)` drops NA values |
| Aggregation | `max`, `min`, `mean` of non-NA neighbor values | Identical `max`, `min`, `mean` via `data.table` grouped aggregation |
| Output columns | `neighbor_max_<var>`, `neighbor_min_<var>`, `neighbor_mean_<var>` | Identical column names and semantics |
| RF model | Untouched | Untouched â€” only the feature table is rebuilt |

## Performance Estimate

| Component | Original | Optimized |
|-----------|----------|-----------|
| String allocations | ~25.8M per variable | **Zero** |
| Named-vector lookups on 6.46M keys | ~25.8M | **Zero** (integer indexing) |
| R-level loop iterations | 6.46M Ã— 6 (build + 5 vars) | **28** (one per year) |
| Memory pattern | 6.46M-element list of integer vectors | Flat integer vectors, year-chunked |
| **Estimated wall time** | **86+ hours** | **3â€“10 minutes** |

The speedup comes from three reinforcing changes: (1) eliminating all string operations, (2) replacing 6.46M R-level loop iterations with 28 vectorized year-passes, and (3) using `data.table` grouped aggregation instead of per-row `lapply`.
END_RESPONSE>>>
