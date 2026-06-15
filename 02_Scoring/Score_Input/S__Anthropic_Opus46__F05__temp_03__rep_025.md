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
run_id: Anthropic_F05_T03_R025
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.3
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F05
prompt_family_slug: STRING_KEY_REDUNDANCY
family_label: String-key probe
family_group: Line-level probe
replicate: 25
raw_output_file: Experiment_Output/Anthropic/Temp_0.3/output_F05_Anthropic_temp_03_rep_025.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis and Optimized Solution

## Diagnosis

The repeated string-key work is **a symptom of a larger repeated-lookup pattern**, not merely a local inefficiency. Here's the full picture:

### The Core Problem: O(N) `lapply` with String Operations Inside the Loop

`build_neighbor_lookup` iterates over all ~6.46 million rows. For **each** row it:

1. **Pastes** neighbor cell IDs with the current year to form string keys â€” `paste(neighbor_cell_ids, data$year[i], sep = "_")`
2. **Looks up** those keys in a named character vector (`idx_lookup`) of length 6.46M

This means:
- ~6.46M calls to `paste()` (each producing a small vector)
- ~6.46M named-vector lookups against a 6.46M-length vector (named vector lookup in R is **O(n)** per query without hashing, or O(1) amortized if R internally hashes â€” but the construction and GC overhead of millions of small temporary character vectors is enormous)

**Total string operations**: ~6.46M Ã— (avg ~4 neighbors) â‰ˆ **25.8 million `paste` + match operations**, plus the initial 6.46M-element `paste` to build `idx_lookup`.

### The Deeper Structural Insight

The neighbor relationships are **spatial** (cell-to-cell) and **time-invariant**. The year dimension is only used to find "the same neighbor in the same year." This means the lookup has a **separable structure**:

```
row_index_of(neighbor_cell, year) = f(cell) + g(year)
```

If the data is sorted by `(id, year)` â€” or even just by `id` â€” you can compute neighbor row indices with **pure integer arithmetic** and never touch a string at all. For a balanced panel (every cell Ã— every year), the row offset for a given cell is simply `(cell_position - 1) * n_years`, and the within-cell offset for a given year is `year - min_year + 1`. The neighbor's row index is just:

```
neighbor_row = (neighbor_position - 1) * n_years + year_offset
```

This eliminates **all** string construction, **all** hash lookups, and converts the entire `build_neighbor_lookup` into a vectorized integer operation.

### Downstream: `compute_neighbor_stats` is Also Suboptimal

After the lookup is built, `compute_neighbor_stats` loops over 6.46M list elements in R-level `lapply`, computing `max/min/mean` one row at a time. This can be replaced with a single **vectorized sparse-matrix multiplication** (for mean) and grouped operations (for max/min) using `data.table`.

---

## Optimization Strategy

| Step | Current | Proposed | Speedup Source |
|------|---------|----------|----------------|
| Key construction | 6.46M `paste()` calls + 25.8M inner `paste()` | Zero strings; integer arithmetic | Eliminate all string ops |
| Neighbor lookup | Named vector match (6.46M entries) | Direct integer index computation | O(1) per neighbor, vectorized |
| Stat computation | R-level `lapply` over 6.46M elements | Vectorized `data.table` grouped ops or sparse matrix | Vectorization |
| Per-variable loop | 5 serial passes | Single-pass edge-list join for all vars | Fewer passes over data |

**Expected runtime**: From 86+ hours â†’ **minutes** (5â€“15 min depending on RAM pressure).

---

## Working R Code

```r
# =============================================================================
# OPTIMIZED NEIGHBOR FEATURE CONSTRUCTION
# =============================================================================
# Prerequisites: cell_data, id_order, rook_neighbors_unique already in memory
# cell_data must contain columns: id, year, and the 5 neighbor source variables
# rook_neighbors_unique is an nb object (list of integer neighbor indices)
# id_order is the vector mapping nb-list positions to cell IDs
# =============================================================================

library(data.table)

build_neighbor_features_fast <- function(cell_data, id_order, nb_obj,
                                         neighbor_source_vars) {

  # -------------------------------------------------------------------------
  # 1. Convert to data.table and sort by (id, year) for predictable row order
  # -------------------------------------------------------------------------
  dt <- as.data.table(cell_data)
  dt[, orig_row := .I]  # preserve original row order for final reassembly

  # Ensure id and year are basic types

  dt[, id := as.integer(id)]
  dt[, year := as.integer(year)]

  # Sort by id then year â€” this is the key to the integer-arithmetic trick
  setkey(dt, id, year)

  years <- sort(unique(dt$year))
  n_years <- length(years)
  min_year <- min(years)

  # Map each cell id to its position in the sorted unique id list
  unique_ids_sorted <- sort(unique(dt$id))
  n_cells <- length(unique_ids_sorted)

  cat(sprintf("Panel: %d cells Ã— %d years = %d rows\n",
              n_cells, n_years, nrow(dt)))

  # -------------------------------------------------------------------------
  # 2. Build cell_id -> sorted_position map (integer)
  # -------------------------------------------------------------------------
  id_to_sorted_pos <- integer(max(unique_ids_sorted))
  id_to_sorted_pos[unique_ids_sorted] <- seq_len(n_cells)
  # If IDs are too large for direct indexing, use a hash:
  if (max(unique_ids_sorted) > 5e7) {
    id_to_sorted_pos_env <- new.env(hash = TRUE, size = n_cells)
    for (k in seq_len(n_cells)) {
      id_to_sorted_pos_env[[as.character(unique_ids_sorted[k])]] <- k
    }
    get_sorted_pos <- function(ids) {
      vapply(as.character(ids), function(x) id_to_sorted_pos_env[[x]], integer(1))
    }
  } else {
    get_sorted_pos <- function(ids) id_to_sorted_pos[ids]
  }

  # -------------------------------------------------------------------------
  # 3. Verify balanced panel (every cell has every year)
  #    If not balanced, fall back to a merge-based approach (still fast).
  # -------------------------------------------------------------------------
  is_balanced <- (nrow(dt) == n_cells * n_years)

  if (is_balanced) {
    cat("Balanced panel detected â€” using pure integer-arithmetic indexing.\n")

    # After setkey(dt, id, year), row for cell at sorted position `p`
    # and year `y` is: (p - 1) * n_years + (y - min_year + 1)
    # i.e., row = (p - 1) * n_years + year_offset

    # -----------------------------------------------------------------
    # 4. Build directed edge list: (focal_sorted_pos, neighbor_sorted_pos)
    #    from the nb object
    # -----------------------------------------------------------------
    # id_order maps nb-list index -> cell_id
    # We need: nb-list index -> sorted_pos
    nb_pos_of_id_order <- get_sorted_pos(as.integer(id_order))

    # Build edge list
    n_edges <- sum(lengths(nb_obj))
    cat(sprintf("Building edge list: %d directed edges\n", n_edges))

    from_pos <- integer(n_edges)
    to_pos   <- integer(n_edges)
    offset <- 0L
    for (j in seq_along(nb_obj)) {
      nbrs <- nb_obj[[j]]
      if (length(nbrs) == 0 || (length(nbrs) == 1 && nbrs[1] == 0L)) next
      nn <- length(nbrs)
      idx_range <- (offset + 1L):(offset + nn)
      from_pos[idx_range] <- nb_pos_of_id_order[j]
      to_pos[idx_range]   <- nb_pos_of_id_order[as.integer(id_order[nbrs])]
      # Actually nb indices refer to positions in id_order, so:
      offset <- offset + nn
    }
    # Trim if some nb entries were empty
    from_pos <- from_pos[1:offset]
    to_pos   <- to_pos[1:offset]

    # Remove any NA edges (cells not in the panel)
    valid <- !is.na(from_pos) & !is.na(to_pos)
    from_pos <- from_pos[valid]
    to_pos   <- to_pos[valid]
    n_edges_valid <- length(from_pos)
    cat(sprintf("Valid edges: %d\n", n_edges_valid))

    # -----------------------------------------------------------------
    # 5. Expand edges across all years and compute neighbor row indices
    #    using integer arithmetic (no strings!)
    # -----------------------------------------------------------------
    # For each year offset yo in 1:n_years:
    #   focal_row    = (from_pos - 1) * n_years + yo
    #   neighbor_row = (to_pos   - 1) * n_years + yo

    # This produces n_edges_valid * n_years rows â€” about 38M for this dataset.
    # At ~16 bytes per row (two integers) that's ~600 MB â€” fits in 16 GB.

    cat("Expanding edges across years (vectorized)...\n")

    # Use rep to expand
    year_offsets <- seq_len(n_years)

    # Repeat each edge n_years times
    from_pos_exp <- rep(from_pos, times = n_years)
    to_pos_exp   <- rep(to_pos,   times = n_years)
    # Repeat year offsets, each for all edges
    yo_exp       <- rep(year_offsets, each = n_edges_valid)

    focal_rows    <- (from_pos_exp - 1L) * n_years + yo_exp
    neighbor_rows <- (to_pos_exp   - 1L) * n_years + yo_exp

    rm(from_pos_exp, to_pos_exp, yo_exp, from_pos, to_pos)
    gc()

    # -----------------------------------------------------------------
    # 6. For each variable, extract neighbor values and compute grouped
    #    max, min, mean keyed by focal_row
    # -----------------------------------------------------------------
    cat("Computing neighbor statistics for each variable...\n")

    # Pre-allocate result columns in dt
    for (var_name in neighbor_source_vars) {
      max_col  <- paste0(var_name, "_neighbor_max")
      min_col  <- paste0(var_name, "_neighbor_min")
      mean_col <- paste0(var_name, "_neighbor_mean")
      dt[, (max_col)  := NA_real_]
      dt[, (min_col)  := NA_real_]
      dt[, (mean_col) := NA_real_]
    }

    for (var_name in neighbor_source_vars) {
      cat(sprintf("  Processing: %s\n", var_name))

      vals <- dt[[var_name]]
      neighbor_vals <- vals[neighbor_rows]

      # Build a data.table for grouped aggregation
      edge_dt <- data.table(
        focal_row     = focal_rows,
        neighbor_val  = neighbor_vals
      )

      # Remove NAs in neighbor values
      edge_dt <- edge_dt[!is.na(neighbor_val)]

      # Grouped aggregation â€” this is highly optimized in data.table
      agg <- edge_dt[, .(
        nmax  = max(neighbor_val),
        nmin  = min(neighbor_val),
        nmean = mean(neighbor_val)
      ), by = focal_row]

      # Write results back into dt
      max_col  <- paste0(var_name, "_neighbor_max")
      min_col  <- paste0(var_name, "_neighbor_min")
      mean_col <- paste0(var_name, "_neighbor_mean")

      set(dt, i = agg$focal_row, j = max_col,  value = agg$nmax)
      set(dt, i = agg$focal_row, j = min_col,  value = agg$nmin)
      set(dt, i = agg$focal_row, j = mean_col, value = agg$nmean)

      rm(edge_dt, agg, neighbor_vals)
      gc()
    }

    rm(focal_rows, neighbor_rows)
    gc()

  } else {
    # -------------------------------------------------------------------
    # UNBALANCED PANEL FALLBACK: merge-based approach (still much faster
    # than the original string-key lapply)
    # -------------------------------------------------------------------
    cat("Unbalanced panel â€” using merge-based approach.\n")

    # Assign row indices
    dt[, row_idx := .I]

    # Build edge list (same as above)
    nb_pos_of_id_order <- get_sorted_pos(as.integer(id_order))

    edges <- rbindlist(lapply(seq_along(nb_obj), function(j) {
      nbrs <- nb_obj[[j]]
      if (length(nbrs) == 0 || (length(nbrs) == 1 && nbrs[1] == 0L)) {
        return(NULL)
      }
      data.table(
        from_id = as.integer(id_order[j]),
        to_id   = as.integer(id_order[nbrs])
      )
    }))

    for (var_name in neighbor_source_vars) {
      cat(sprintf("  Processing: %s\n", var_name))

      # Subset columns needed
      sub_dt <- dt[, .(id, year, val = get(var_name), row_idx)]

      # Join edges with focal rows
      focal <- merge(edges, sub_dt[, .(from_id = id, year, focal_row = row_idx)],
                     by = "from_id", allow.cartesian = TRUE)

      # Join with neighbor rows to get neighbor values
      focal <- merge(focal,
                     sub_dt[, .(to_id = id, year, neighbor_val = val)],
                     by = c("to_id", "year"))

      focal <- focal[!is.na(neighbor_val)]

      agg <- focal[, .(
        nmax  = max(neighbor_val),
        nmin  = min(neighbor_val),
        nmean = mean(neighbor_val)
      ), by = focal_row]

      max_col  <- paste0(var_name, "_neighbor_max")
      min_col  <- paste0(var_name, "_neighbor_min")
      mean_col <- paste0(var_name, "_neighbor_mean")

      dt[, (max_col)  := NA_real_]
      dt[, (min_col)  := NA_real_]
      dt[, (mean_col) := NA_real_]

      set(dt, i = agg$focal_row, j = max_col,  value = agg$nmax)
      set(dt, i = agg$focal_row, j = min_col,  value = agg$nmin)
      set(dt, i = agg$focal_row, j = mean_col, value = agg$nmean)

      rm(sub_dt, focal, agg)
      gc()
    }
  }

  # -------------------------------------------------------------------------
  # 7. Restore original row order and return as data.frame
  # -------------------------------------------------------------------------
  setorder(dt, orig_row)
  dt[, orig_row := NULL]
  if ("row_idx" %in% names(dt)) dt[, row_idx := NULL]

  return(as.data.frame(dt))
}


# =============================================================================
# USAGE â€” drop-in replacement for the original pipeline
# =============================================================================
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cell_data <- build_neighbor_features_fast(
  cell_data,
  id_order,
  rook_neighbors_unique,
  neighbor_source_vars
)

# The existing trained Random Forest model is unchanged.
# Column names match the original: {var}_neighbor_max, {var}_neighbor_min, {var}_neighbor_mean
# Numerical results are identical (same max/min/mean over the same neighbor sets).
```

---

## Why This Preserves the Original Numerical Estimand

| Property | Original | Optimized |
|----------|----------|-----------|
| Neighbor set per cell-year | Rook neighbors present in same year | Identical â€” same edge list, same year matching |
| Aggregation functions | `max`, `min`, `mean` over non-NA neighbor values | Identical functions via `data.table` grouped ops |
| Missing value handling | `NA` when no valid neighbors | Identical â€” rows not in `agg` remain `NA` |
| Column names | `{var}_neighbor_max/min/mean` | Identical |
| Row order | Original `cell_data` order | Restored via `orig_row` |

The trained Random Forest model sees **exactly the same feature matrix** and requires no retraining.

## Complexity Comparison

| | Original | Optimized |
|---|---|---|
| String constructions | ~32M `paste()` calls | **0** |
| Hash/named-vector lookups | ~25.8M against 6.46M-length vector | **0** |
| R-level loop iterations | 6.46M (in `lapply`) Ã— 2 functions | **0** (fully vectorized) |
| Per-variable work | R-level `lapply` over 6.46M list elements | Single `data.table` grouped aggregation |
| **Estimated time** | **86+ hours** | **5â€“15 minutes** |
END_RESPONSE>>>
