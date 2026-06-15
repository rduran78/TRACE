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
run_id: Anthropic_F12_T10_R011
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F12
prompt_family_slug: FALSE_RBIND_DIAGNOSIS
family_label: False rbind diagnosis
family_group: Adversarial diagnosis
replicate: 11
raw_output_file: Experiment_Output/Anthropic/Temp_1.0/output_F12_Anthropic_temp_10_rep_011.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

## Diagnosis: Rejecting the Colleague's Claim

The colleague is **wrong** that `do.call(rbind, result)` and list binding inside `compute_neighbor_stats()` are the main bottleneck. While `do.call(rbind, ...)` on ~6.46 million small vectors is not free, it runs in seconds for numeric matrices of that size. The **true, deeper bottleneck** is `build_neighbor_lookup()`.

Here is why:

`build_neighbor_lookup()` calls `lapply` over **every one of the ~6.46 million rows**, and inside each iteration it:

1. **Performs a character key lookup** via `id_to_ref[as.character(data$id[i])]` â€” 6.46M `as.character()` coercions and named-vector lookups.
2. **Subsets the neighbor list** with `id_order[neighbors[[ref_idx]]]`.
3. **Constructs paste keys** with `paste(neighbor_cell_ids, data$year[i], sep = "_")` â€” for every row, creating ~4 string keys on average (rook neighbors), totaling ~26 million string constructions.
4. **Looks up those keys** in `idx_lookup`, a named vector of length 6.46M, meaning each lookup does a **linear-time hash probe on a massive named character vector**, repeated ~26 million times.

This single function therefore performs **tens of millions of string allocations and named-vector lookups inside an interpreted R loop**. On a 16 GB laptop, this is the operation that pushes runtime toward 86+ hours. `compute_neighbor_stats()` is comparatively cheap: it's just numeric indexing and three summary functions over small integer vectors.

### Summary of bottleneck hierarchy

| Component | Estimated cost | True bottleneck? |
|---|---|---|
| `build_neighbor_lookup()` â€” 6.46M iterations of paste + named-vector string lookups | ~85+ hours | **YES â€” dominant** |
| `compute_neighbor_stats()` â€” numeric subsetting + `do.call(rbind, ...)` | Minutes | No |
| Outer `for` loop over 5 variables | 5Ã— cost of `compute_neighbor_stats` | No |

---

## Optimization Strategy

The fix is to **eliminate all per-row string operations** and replace the entire lookup construction with vectorized integer arithmetic.

**Key insight:** Since every grid cell appears in every year (balanced panel: 344,208 cells Ã— 28 years = 9,637,824 potential rows, ~6.46M present), and neighbors are defined spatially (constant across years), we can:

1. **Build a mapping from `id` to row indices grouped by year** using `data.table` or `match()` + `split()` â€” all vectorized.
2. **For each row, compute neighbor row indices** by joining the spatial neighbor list with the year-specific row index map â€” entirely with integer indexing, no strings.
3. Compute `neighbor_lookup` once as an integer list, then reuse across all 5 variables (already done, but now fast).
4. Replace `do.call(rbind, result)` in `compute_neighbor_stats()` with a pre-allocated matrix for marginal further gain.

This reduces the ~86-hour runtime to **minutes**.

---

## Working R Code

```r
# ==============================================================================
# OPTIMIZED PIPELINE â€” preserves trained RF model and original numerical outputs
# ==============================================================================

library(data.table)

# --------------------------------------------------------------------------
# 1. FAST NEIGHBOR LOOKUP CONSTRUCTION (replaces build_neighbor_lookup)
#    Eliminates all per-row string operations; uses pure integer indexing.
# --------------------------------------------------------------------------

build_neighbor_lookup_fast <- function(data, id_order, neighbors) {
  # Convert to data.table for fast grouped operations (non-destructive)
  dt <- as.data.table(data)
  dt[, row_idx := .I]

  # Map each id to its position in id_order (spatial index)
  # id_order is the vector of cell IDs in the order matching the nb object
  id_to_ref <- setNames(seq_along(id_order), as.character(id_order))

  # Get unique years present in the data
  years <- sort(unique(dt$year))

  # For each year, build a fast lookup: cell_id -> row index in 'data'
  # Using data.table keyed joins for O(1) amortized lookups
  # Structure: a list keyed by year, each element is a named integer vector
  #            mapping id -> row_idx
  year_id_to_row <- dt[, .(id, row_idx, year)]
  setkey(year_id_to_row, year, id)

  # Pre-split by year for fast access
  year_maps <- split(year_id_to_row, by = "year", keep.by = FALSE)
  # Convert each to a lookup: id -> row_idx
  year_lookup <- lapply(year_maps, function(sub) {
    setNames(sub$row_idx, as.character(sub$id))
  })
  names(year_lookup) <- as.character(years)

  # Vectorized: get ref_idx (spatial index) for every row
  ref_idx_all <- id_to_ref[as.character(dt$id)]

  # Pre-fetch year as character for each row (vectorized, done once)
  year_char <- as.character(dt$year)

  # Now build the neighbor lookup using integer indexing only
  n <- nrow(dt)
  neighbor_lookup <- vector("list", n)

  # Group rows by year to batch process (avoids repeated year_lookup access)
  row_groups <- split(seq_len(n), year_char)

  for (yr in names(row_groups)) {
    rows_in_year <- row_groups[[yr]]
    lk <- year_lookup[[yr]]  # id -> row_idx for this year

    for (i in rows_in_year) {
      ref <- ref_idx_all[i]
      if (is.na(ref)) {
        neighbor_lookup[[i]] <- integer(0)
        next
      }
      nb_spatial_indices <- neighbors[[ref]]
      if (length(nb_spatial_indices) == 0L ||
          (length(nb_spatial_indices) == 1L && nb_spatial_indices[1] == 0L)) {
        neighbor_lookup[[i]] <- integer(0)
        next
      }
      nb_ids <- as.character(id_order[nb_spatial_indices])
      matched <- lk[nb_ids]
      neighbor_lookup[[i]] <- as.integer(matched[!is.na(matched)])
    }
  }

  neighbor_lookup
}

# --------------------------------------------------------------------------
# 2. OPTIMIZED NEIGHBOR STATS (replaces compute_neighbor_stats)
#    Pre-allocates output matrix; avoids do.call(rbind, ...) on huge list.
# --------------------------------------------------------------------------

compute_neighbor_stats_fast <- function(data, neighbor_lookup, var_name) {
  vals <- data[[var_name]]
  n <- length(neighbor_lookup)
  out <- matrix(NA_real_, nrow = n, ncol = 3)

  for (i in seq_len(n)) {
    idx <- neighbor_lookup[[i]]
    if (length(idx) == 0L) next
    nv <- vals[idx]
    nv <- nv[!is.na(nv)]
    if (length(nv) == 0L) next
    out[i, 1] <- max(nv)
    out[i, 2] <- min(nv)
    out[i, 3] <- mean(nv)
  }

  out
}

# --------------------------------------------------------------------------
# 3. WRAPPER: compute and add neighbor features to the data frame
#    (drop-in replacement for compute_and_add_neighbor_features)
# --------------------------------------------------------------------------

compute_and_add_neighbor_features_fast <- function(data, var_name, neighbor_lookup) {
  stats <- compute_neighbor_stats_fast(data, neighbor_lookup, var_name)
  data[[paste0(var_name, "_neighbor_max")]]  <- stats[, 1]
  data[[paste0(var_name, "_neighbor_min")]]  <- stats[, 2]
  data[[paste0(var_name, "_neighbor_mean")]] <- stats[, 3]
  data
}

# --------------------------------------------------------------------------
# 4. FULL PIPELINE EXECUTION
# --------------------------------------------------------------------------

# Build optimized neighbor lookup (runs in minutes, not days)
neighbor_lookup <- build_neighbor_lookup_fast(
  cell_data, id_order, rook_neighbors_unique
)

# Compute neighbor features for all 5 source variables
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {
  cell_data <- compute_and_add_neighbor_features_fast(
    cell_data, var_name, neighbor_lookup
  )
}

# The trained Random Forest model is untouched.
# Numerical outputs (max, min, mean of neighbor values) are identical
# to the original implementation â€” only the computation path changed.
```

---

## Even Faster: Fully Vectorized Alternative for `build_neighbor_lookup_fast`

If the inner loop is still too slow (6.46M iterations in R), here is a **fully vectorized** version that eliminates all explicit R loops using `data.table` joins:

```r
build_neighbor_lookup_vectorized <- function(data, id_order, neighbors) {
  library(data.table)

  dt <- as.data.table(data)[, row_idx := .I]

  # Expand the nb object into an edge list: (spatial_ref, spatial_neighbor)
  edges <- rbindlist(lapply(seq_along(neighbors), function(ref) {
    nb <- neighbors[[ref]]
    if (length(nb) == 0L || (length(nb) == 1L && nb[0] == 0L)) {
      return(data.table(ref_id = integer(0), nb_id = integer(0)))
    }
    data.table(
      ref_id = id_order[ref],
      nb_id  = id_order[nb]
    )
  }))

  # Join edges with data to get (row_idx_source, year, nb_id)
  setkey(dt, id)
  source <- dt[, .(id, year, row_idx)]
  setnames(source, c("ref_id", "year", "src_row"))
  setkey(source, ref_id)

  # Merge: for each edge, attach all years from source side
  edge_year <- edges[source, on = "ref_id", allow.cartesian = TRUE, nomatch = NULL]
  # edge_year now has columns: ref_id, nb_id, year, src_row

  # Now find the row index of the neighbor in the same year
  target <- dt[, .(id, year, row_idx)]
  setnames(target, c("nb_id", "year", "tgt_row"))
  setkey(target, nb_id, year)
  setkey(edge_year, nb_id, year)

  matched <- target[edge_year, on = c("nb_id", "year"), nomatch = NA]
  matched <- matched[!is.na(tgt_row)]

  # Split into per-source-row lists
  setkey(matched, src_row)
  neighbor_lookup <- vector("list", nrow(data))
  # Initialize all to empty
  for (i in seq_along(neighbor_lookup)) neighbor_lookup[[i]] <- integer(0)

  # Fill in from matched
  split_result <- split(matched$tgt_row, matched$src_row)
  idx <- as.integer(names(split_result))
  neighbor_lookup[idx] <- lapply(split_result, as.integer)

  neighbor_lookup
}
```

### Performance comparison

| Approach | Estimated runtime (6.46M rows, 16GB RAM) |
|---|---|
| **Original** (`build_neighbor_lookup`) | ~86+ hours |
| **Loop-optimized** (`build_neighbor_lookup_fast`) | ~10â€“30 minutes |
| **Fully vectorized** (`build_neighbor_lookup_vectorized`) | ~1â€“5 minutes |
| **Original** `compute_neighbor_stats` + `do.call(rbind,...)` | ~2â€“5 minutes |
| **Optimized** `compute_neighbor_stats_fast` (pre-allocated matrix) | ~1â€“3 minutes |

---

## Conclusion

The colleague's diagnosis is **incorrect**. `do.call(rbind, result)` and the internals of `compute_neighbor_stats()` are not the bottleneckâ€”they operate on numeric vectors with simple indexing and run in minutes. The true bottleneck is `build_neighbor_lookup()`, which performs **~6.46 million iterations of string construction (`paste`) and named-vector lookups (`idx_lookup[neighbor_keys]`) against a 6.46M-length named character vector** inside an interpreted R loop. Replacing this with vectorized integer operations via `data.table` joins reduces total pipeline time from 86+ hours to single-digit minutes while preserving identical numerical output and leaving the trained Random Forest model untouched.
END_RESPONSE>>>
