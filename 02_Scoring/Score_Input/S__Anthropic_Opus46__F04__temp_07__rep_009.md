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
run_id: Anthropic_F04_T07_R009
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F04
prompt_family_slug: NEIGHBOR_BOTTLENECK_HINT
family_label: Neighbor bottleneck hint
family_group: Spatial-neighbor cue
replicate: 9
raw_output_file: Experiment_Output/Anthropic/Temp_0.7/output_F04_Anthropic_temp_07_rep_009.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The bottleneck is overwhelmingly in `build_neighbor_lookup`, not `compute_neighbor_stats`. Here's why:

**`build_neighbor_lookup`** iterates over **~6.46 million rows** with an `lapply` call that, for each row:

1. Looks up the cell's reference index via a named character vector (`id_to_ref`).
2. Retrieves neighbor cell IDs from the `nb` object.
3. Constructs **string keys** by pasting neighbor IDs with the current year (`paste(..., sep="_")`).
4. Looks up those string keys in a **named character vector** (`idx_lookup`) of length 6.46 million.

String construction and named-vector lookup (which is O(n) hashing per call in base R) repeated ~6.46 million times, each with ~4 neighbors on average (~4 paste + ~4 lookups per row), yields **~25.8 million string allocations and hash lookups**. This is catastrophically slow in interpreted R. The `idx_lookup` named vector of length 6.46M makes each `[` lookup expensive because R rehashes or scans a large internal hash table.

**`compute_neighbor_stats`** is a secondary bottleneck: 6.46M `lapply` iterations with small numeric subsetting and summary stats. It's slow but not as devastating as the string-key construction.

**The Random Forest inference itself** over 6.46M rows Ã— 110 features is comparatively fast (minutes, not hours) with `ranger` or `randomForest` predict methods.

---

## Optimization Strategy

### Principle: Replace string-key lookups with integer-arithmetic indexing.

**Key Insight:** If the data is sorted by `(id, year)` and every cell has the same contiguous block of 28 years (1992â€“2019), then the row index for any `(cell_index, year)` pair can be computed by pure arithmetic:

```
row = (cell_index - 1) * 28 + (year - 1991)
```

No strings. No hash tables. No `paste`. This reduces `build_neighbor_lookup` from billions of character operations to a vectorized integer computation.

### Steps:

1. **Sort data** by `(id, year)` and verify completeness (balanced panel).
2. **Replace `build_neighbor_lookup`** with a fully vectorized version that computes row indices via integer arithmetic, returning a flat structure (vectors of row indices and group pointers) rather than a list-of-lists.
3. **Replace `compute_neighbor_stats`** with a vectorized grouped computation using the flat structure, avoiding `lapply` over 6.46M elements entirely.
4. **Preserve numerical output exactly** â€” same max, min, mean per cell-year per variable.

Expected speedup: from ~86+ hours to **minutes** (roughly 2â€“10 minutes total for all 5 variables).

---

## Working R Code

```r
# ==============================================================================
# OPTIMIZED SPATIAL NEIGHBOR FEATURE CONSTRUCTION
# Drop-in replacement. Preserves the trained RF model and numerical estimand.
# ==============================================================================

#' Step 0: Ensure data is sorted and panel is balanced.
#' This is a prerequisite. If your data is already balanced and sorted, this is
#' nearly free. If not, it fixes it.
prepare_balanced_panel <- function(data, id_col = "id", year_col = "year") {
  # Sort by id then year
  data <- data[order(data[[id_col]], data[[year_col]]), ]
  rownames(data) <- NULL

  years <- sort(unique(data[[year_col]]))
  n_years <- length(years)
  ids <- unique(data[[id_col]])
  n_ids <- length(ids)

  expected_rows <- n_ids * n_years
  if (nrow(data) != expected_rows) {
    stop(
      sprintf(
        "Panel is not balanced: expected %d rows (%d ids x %d years), got %d.",
        expected_rows, n_ids, n_years, nrow(data)
      )
    )
  }

  # Verify contiguous blocks
  if (!all(data[[year_col]] == rep(years, times = n_ids))) {
    stop("After sorting, year blocks are not contiguous. Check for duplicates or gaps.")
  }

  attr(data, "panel_ids") <- ids
  attr(data, "panel_years") <- years
  attr(data, "panel_n_years") <- n_years
  attr(data, "panel_n_ids") <- n_ids
  data
}


#' Step 1: Build neighbor lookup as flat vectors (fully vectorized, no strings).
#'
#' Returns a list with:
#'   $neighbor_row_idx : integer vector of all neighbor row indices (flat)
#'   $group_start      : integer vector of length nrow(data), start position in neighbor_row_idx
#'   $group_len        : integer vector of length nrow(data), number of neighbors per row
build_neighbor_lookup_fast <- function(data, id_order, neighbors) {
  # id_order: vector of cell IDs in the order matching the nb object
  # neighbors: spdep nb object (list of integer vectors, indices into id_order)

  ids   <- attr(data, "panel_ids")
  years <- attr(data, "panel_years")
  n_years <- attr(data, "panel_n_years")
  n_ids   <- attr(data, "panel_n_ids")

  # Map from id_order position -> panel cell index (1-based position in 'ids')
  # id_order gives cell IDs in the order of the nb object.
  # ids gives cell IDs in the order of the sorted data.
  # We need: for each id_order[j], what is its position in 'ids'?
  id_to_panel_idx <- match(id_order, ids)
  # id_to_panel_idx[j] = panel cell index for the j-th element of id_order

  # For each cell in panel order, find its neighbors' panel indices
  # First, map panel cell index -> id_order index
  panel_to_nb_idx <- match(ids, id_order)
  # panel_to_nb_idx[k] = index into neighbors/id_order for the k-th panel cell

  # Build flat neighbor structure per cell (not per row yet)
  # cell_neighbor_panel_idx[[k]] = integer vector of panel cell indices of neighbors of cell k
  cell_neighbor_panel_idx <- vector("list", n_ids)
  for (k in seq_len(n_ids)) {
    nb_idx <- panel_to_nb_idx[k]
    if (is.na(nb_idx)) {
      cell_neighbor_panel_idx[[k]] <- integer(0)
    } else {
      nb_positions <- neighbors[[nb_idx]]
      # Remove 0s (spdep convention for no neighbors)
      nb_positions <- nb_positions[nb_positions > 0L]
      if (length(nb_positions) == 0L) {
        cell_neighbor_panel_idx[[k]] <- integer(0)
      } else {
        # Convert nb positions (indices into id_order) to panel cell indices
        mapped <- id_to_panel_idx[nb_positions]
        cell_neighbor_panel_idx[[k]] <- mapped[!is.na(mapped)]
      }
    }
  }

  # Now expand to row-level.
  # Row for panel cell k, year index t (1-based) is: (k - 1) * n_years + t
  # Neighbor rows for that row: for each neighbor panel cell index m,
  #   the neighbor row is (m - 1) * n_years + t  (same year)

  # Compute group lengths: each row's number of neighbors = its cell's number of neighbors
  cell_n_neighbors <- vapply(cell_neighbor_panel_idx, length, integer(1))
  # Repeat each cell's count n_years times (one per year)
  group_len <- rep(cell_n_neighbors, each = n_years)

  total_entries <- sum(group_len)
  neighbor_row_idx <- integer(total_entries)

  # Fill neighbor_row_idx using vectorized arithmetic per cell
  pos <- 1L
  for (k in seq_len(n_ids)) {
    n_nb <- cell_n_neighbors[k]
    if (n_nb == 0L) next
    nb_panel_indices <- cell_neighbor_panel_idx[[k]]  # vector of length n_nb
    # For each year t in 1:n_years, neighbor rows are (nb_panel_indices - 1) * n_years + t
    # We need to fill n_years blocks, each of length n_nb
    base_offsets <- (nb_panel_indices - 1L) * n_years  # length n_nb
    for (t in seq_len(n_years)) {
      idx_start <- pos
      idx_end <- pos + n_nb - 1L
      neighbor_row_idx[idx_start:idx_end] <- base_offsets + t
      pos <- idx_end + 1L
    }
  }

  # Compute group_start (1-based start positions into neighbor_row_idx)
  group_start <- integer(length(group_len))
  cumlen <- cumsum(c(0L, group_len[-length(group_len)]))
  group_start <- cumlen + 1L

  list(
    neighbor_row_idx = neighbor_row_idx,
    group_start      = group_start,
    group_len        = group_len
  )
}


#' Step 2: Compute neighbor stats (max, min, mean) fully vectorized.
#'
#' Uses the flat neighbor structure. Avoids lapply over 6.46M rows.
compute_neighbor_stats_fast <- function(data, neighbor_lookup_flat, var_name) {
  vals <- data[[var_name]]
  n <- length(neighbor_lookup_flat$group_len)

  neighbor_row_idx <- neighbor_lookup_flat$neighbor_row_idx
  group_start      <- neighbor_lookup_flat$group_start
  group_len        <- neighbor_lookup_flat$group_len

  # Initialize output
  out_max  <- rep(NA_real_, n)
  out_min  <- rep(NA_real_, n)
  out_mean <- rep(NA_real_, n)

  # Rows that have at least one neighbor
  has_neighbors <- which(group_len > 0L)

  if (length(has_neighbors) == 0L) {
    return(cbind(out_max, out_min, out_mean))
  }

  # Extract all neighbor values at once
  all_neighbor_vals <- vals[neighbor_row_idx]

  # Create a grouping vector: which row does each entry in neighbor_row_idx belong to?
  # We can use rep() with group_len
  group_id <- rep.int(seq_len(n), group_len)

  # Handle NAs in neighbor values: set to NA and we'll use na.rm = TRUE logic
  # For efficiency, use tapply or data.table or direct C-level grouping.
  # Since tapply over 25M entries with 6.46M groups can be slow,
  # we use a data.table approach for grouped aggregation.

  if (requireNamespace("data.table", quietly = TRUE)) {
    dt <- data.table::data.table(
      group = group_id,
      val   = all_neighbor_vals
    )
    # Remove NAs
    dt <- dt[!is.na(val)]
    agg <- dt[, .(
      vmax  = max(val),
      vmin  = min(val),
      vmean = mean(val)
    ), by = group]

    out_max[agg$group]  <- agg$vmax
    out_min[agg$group]  <- agg$vmin
    out_mean[agg$group] <- agg$vmean
  } else {
    # Fallback: use split + vapply (slower but no extra dependency)
    split_vals <- split(all_neighbor_vals, group_id)
    stats <- vapply(split_vals, function(v) {
      v <- v[!is.na(v)]
      if (length(v) == 0L) return(c(NA_real_, NA_real_, NA_real_))
      c(max(v), min(v), mean(v))
    }, numeric(3))
    grp_names <- as.integer(names(split_vals))
    out_max[grp_names]  <- stats[1, ]
    out_min[grp_names]  <- stats[2, ]
    out_mean[grp_names] <- stats[3, ]
  }

  cbind(out_max, out_min, out_mean)
}


#' Step 3: Compute and attach neighbor features for one variable.
compute_and_add_neighbor_features_fast <- function(data, var_name, neighbor_lookup_flat) {
  stats <- compute_neighbor_stats_fast(data, neighbor_lookup_flat, var_name)
  data[[paste0(var_name, "_nb_max")]]  <- stats[, 1]
  data[[paste0(var_name, "_nb_min")]]  <- stats[, 2]
  data[[paste0(var_name, "_nb_mean")]] <- stats[, 3]
  data
}


# ==============================================================================
# MAIN EXECUTION (drop-in replacement for the original outer loop)
# ==============================================================================

# --- Prepare balanced panel (sort + validate) ---
cell_data <- prepare_balanced_panel(cell_data, id_col = "id", year_col = "year")

# --- Build flat neighbor lookup (runs once, ~30-90 seconds) ---
neighbor_lookup_flat <- build_neighbor_lookup_fast(
  cell_data, id_order, rook_neighbors_unique
)

# --- Compute neighbor features for all 5 variables (~1-3 min total) ---
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {
  message("Computing neighbor features for: ", var_name)
  cell_data <- compute_and_add_neighbor_features_fast(
    cell_data, var_name, neighbor_lookup_flat
  )
}

# --- The trained Random Forest model is untouched. Proceed to predict: ---
# predictions <- predict(rf_model, cell_data)$predictions
# (or however your existing prediction code works)
```

---

## Summary

| Aspect | Original | Optimized |
|---|---|---|
| **Lookup construction** | 6.46M `paste` + named-vector hash lookups (string-based) | Integer arithmetic: `(cell-1)*28 + year_offset` â€” no strings |
| **Neighbor stats** | `lapply` over 6.46M rows, each subsetting + summarizing | Flat vector extraction + `data.table` grouped aggregation in one pass |
| **Data structure** | List of 6.46M integer vectors (high memory overhead) | 3 flat integer vectors (neighbor_row_idx, group_start, group_len) |
| **Estimated runtime** | ~86+ hours | **~5â€“15 minutes** (lookup build ~1â€“2 min; 5 variables Ã— ~1 min each) |
| **RAM peak** | High (millions of small list elements + string table) | ~200â€“400 MB for flat vectors |
| **Numerical result** | max/min/mean per cell-year | Identical max/min/mean per cell-year |
| **Trained RF model** | â€” | Untouched; no retraining |

The critical insight is that a balanced panel with a known sort order turns every neighbor-row lookup into a closed-form integer calculation, completely eliminating the string construction and hash-table lookups that made the original code take days.
END_RESPONSE>>>
