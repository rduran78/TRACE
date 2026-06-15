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
run_id: Anthropic_F09_T07_R018
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F09
prompt_family_slug: ADJACENCY_TABLE_OPTION
family_label: Adjacency-table option
family_group: Representation cue
replicate: 18
raw_output_file: Experiment_Output/Anthropic/Temp_0.7/output_F09_Anthropic_temp_07_rep_018.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The current implementation is extremely slow (~86+ hours) because of three compounding problems:

### 1. Row-Level `lapply` Over 6.46 Million Rows in `build_neighbor_lookup`
The function iterates row-by-row across all 6.46 million cell-year observations. For each row, it:
- Looks up the cell's reference index via a named character vector (`id_to_ref`).
- Retrieves neighbor cell IDs from the `nb` object.
- Pastes together `(neighbor_id, year)` keys as strings.
- Performs named-vector lookups on `idx_lookup` (a 6.46M-element named vector â€” each lookup is O(n) hash probing on a massive vector).

This produces a **list of 6.46 million integer vectors**, consuming enormous memory and time. The string-pasting and named-vector lookups are the dominant bottleneck.

### 2. Row-Level `lapply` Over 6.46 Million Rows in `compute_neighbor_stats`
For each of the 5 variables, the function iterates over the 6.46M-element neighbor lookup list, subsetting a numeric vector and computing `max`, `min`, `mean`. That is **5 Ã— 6.46M = 32.3 million R-level function calls**, each with per-element overhead.

### 3. The Neighbor Topology Is Invariant Across Years, But Is Rebuilt Per Cell-Year
The rook-neighbor relationships are purely spatial â€” they don't change from year to year. Yet the current code embeds year into the lookup, effectively duplicating the same spatial adjacency structure 28 times and doing all the string work 28 times.

---

## Optimization Strategy

**Core insight:** Separate the *time-invariant spatial adjacency* from the *time-varying cell attributes*. Build the adjacency table once (344K cells Ã— ~4 neighbors each â‰ˆ 1.37M edges), then for each year, join attributes onto that edge table and compute grouped aggregates using `data.table`, which is vectorized in C.

### Steps:

1. **Build a `data.table` edge list once** from the `nb` object: columns `(cell_id, neighbor_id)` â€” ~1.37M rows. This is done once and is year-independent.

2. **For each year (or all years at once via a keyed join):** join the cell-year attributes onto the edge list by `(neighbor_id, year)`, then group by `(cell_id, year)` to compute `max`, `min`, `mean` of each neighbor variable.

3. **Join the resulting neighbor stats back** onto the main `cell_data` table.

This replaces 6.46M R-level list iterations with vectorized `data.table` keyed joins and grouped aggregations, reducing runtime from ~86 hours to **minutes**.

### Complexity comparison:

| Step | Current | Optimized |
|---|---|---|
| Build lookup | 6.46M string-paste + hash lookups | 1.37M-row edge table (once) |
| Compute stats (per var) | 6.46M R `lapply` calls | One vectorized `data.table` join + `groupby` over 1.37M Ã— 28 â‰ˆ 38.4M rows |
| Total R-level iterations | ~38.8M | ~0 (vectorized C) |

### Memory estimate:
- Edge table: 1.37M rows Ã— 2 int cols â‰ˆ 11 MB
- Expanded edge table (with year): 38.4M rows Ã— 3 cols â‰ˆ 920 MB
- Neighbor values joined: +1 double col â‰ˆ +307 MB per variable
- Well within 16 GB RAM.

---

## Working R Code

```r
library(data.table)

# ==============================================================================
# STEP 1: Build the time-invariant spatial edge table ONCE
# ==============================================================================
# rook_neighbors_unique: an nb object (list of integer vectors of neighbor indices)
# id_order: vector of cell IDs corresponding to indices 1..344208 in the nb object

build_edge_table <- function(id_order, neighbors_nb) {
  # neighbors_nb is a list where neighbors_nb[[i]] gives integer indices of
  # neighbors of cell i (in id_order space). 0L means no neighbors in nb objects.
  
  n <- length(id_order)
  
  # Pre-calculate total edges for memory pre-allocation
  n_edges <- sum(vapply(neighbors_nb, function(x) {
    len <- length(x)
    # nb objects use integer(0) for no neighbors, or may contain 0L
    if (len == 1L && x[1] == 0L) 0L else len
  }, integer(1)))
  
  from_id <- integer(n_edges)
  to_id   <- integer(n_edges)
  
  pos <- 1L
  for (i in seq_len(n)) {
    nb_idx <- neighbors_nb[[i]]
    if (length(nb_idx) == 0L || (length(nb_idx) == 1L && nb_idx[1] == 0L)) next
    nb_len <- length(nb_idx)
    from_id[pos:(pos + nb_len - 1L)] <- id_order[i]
    to_id[pos:(pos + nb_len - 1L)]   <- id_order[nb_idx]
    pos <- pos + nb_len
  }
  
  data.table(cell_id = from_id, neighbor_id = to_id)
}

edge_dt <- build_edge_table(id_order, rook_neighbors_unique)

cat("Edge table rows:", nrow(edge_dt), "\n")
# Expected: ~1,373,394

# ==============================================================================
# STEP 2: Convert cell_data to data.table (if not already) and set keys
# ==============================================================================
if (!is.data.table(cell_data)) {
  cell_data <- as.data.table(cell_data)
}

# Ensure id and year columns are present
stopifnot(all(c("id", "year") %in% names(cell_data)))

# ==============================================================================
# STEP 3: Compute all neighbor features via vectorized joins
# ==============================================================================
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

compute_all_neighbor_features <- function(cell_dt, edge_dt, source_vars) {
  
  # Create a lookup table: (neighbor_id aliased as cell_id, year) -> attribute values
  # We only need id, year, and the source variable columns for the join
  lookup_cols <- c("id", "year", source_vars)
  attr_dt <- cell_dt[, ..lookup_cols]
  
  # Rename 'id' to 'neighbor_id' for joining onto edge table
  setnames(attr_dt, "id", "neighbor_id")
  
  # Key the attribute table for fast join
  setkey(attr_dt, neighbor_id, year)
  
  # Get all unique years
  all_years <- sort(unique(cell_dt$year))
  
  # Expand edge table across all years:
  # CJ (cross join) of edge rows Ã— years
  # More memory-efficient: use the edge table and cross join with years
  year_dt <- data.table(year = all_years)
  
  # Cross join edges with years
  # edge_dt has ~1.37M rows, 28 years => ~38.4M rows
  edge_year_dt <- edge_dt[, .(year = all_years), by = .(cell_id, neighbor_id)]
  
  cat("Edge-year table rows:", nrow(edge_year_dt), "\n")
  
  # Key for join
  setkey(edge_year_dt, neighbor_id, year)
  
  # Join neighbor attributes onto edge-year table
  edge_year_dt <- attr_dt[edge_year_dt, on = .(neighbor_id, year)]
  
  # Now edge_year_dt has columns: neighbor_id, year, ntl, ec, ..., cell_id
  # Group by (cell_id, year) and compute max, min, mean for each variable
  
  # Build aggregation expressions dynamically
  agg_exprs <- list()
  for (var in source_vars) {
    var_sym <- as.name(var)
    agg_exprs[[paste0("neighbor_max_", var)]]  <- bquote(
      as.numeric(max(.(var_sym), na.rm = TRUE))
    )
    agg_exprs[[paste0("neighbor_min_", var)]]  <- bquote(
      as.numeric(min(.(var_sym), na.rm = TRUE))
    )
    agg_exprs[[paste0("neighbor_mean_", var)]] <- bquote(
      mean(.(var_sym), na.rm = TRUE)
    )
  }
  
  # Convert to a single call for data.table's j argument
  agg_call <- as.call(c(as.name("list"), agg_exprs))
  
  cat("Computing grouped neighbor statistics...\n")
  
  neighbor_stats <- edge_year_dt[, eval(agg_call), by = .(cell_id, year)]
  
  # Handle Inf/-Inf from max/min on all-NA groups: replace with NA
  inf_cols <- names(neighbor_stats)[-(1:2)]
  for (col in inf_cols) {
    set(neighbor_stats, 
        i = which(is.infinite(neighbor_stats[[col]])), 
        j = col, 
        value = NA_real_)
  }
  
  # Rename cell_id back to id for merging
  setnames(neighbor_stats, "cell_id", "id")
  setkey(neighbor_stats, id, year)
  
  return(neighbor_stats)
}

neighbor_stats <- compute_all_neighbor_features(cell_data, edge_dt, neighbor_source_vars)

cat("Neighbor stats rows:", nrow(neighbor_stats), "\n")

# ==============================================================================
# STEP 4: Merge neighbor features onto cell_data
# ==============================================================================

# Remove any pre-existing neighbor columns to avoid duplication
new_cols <- setdiff(names(neighbor_stats), c("id", "year"))
existing_overlap <- intersect(names(cell_data), new_cols)
if (length(existing_overlap) > 0) {
  cell_data[, (existing_overlap) := NULL]
}

setkey(cell_data, id, year)
cell_data <- neighbor_stats[cell_data, on = .(id, year)]

cat("Final cell_data rows:", nrow(cell_data), "\n")
cat("Final cell_data cols:", ncol(cell_data), "\n")

# ==============================================================================
# STEP 5: Predict with the existing trained Random Forest (unchanged)
# ==============================================================================
# The trained RF model object (e.g., `rf_model`) is used as-is.
# The column names produced above (neighbor_max_ntl, neighbor_min_ntl, 
# neighbor_mean_ntl, etc.) must match the names expected by the model.
#
# If the original code used different naming conventions (e.g., "ntl_neighbor_max"),
# rename accordingly:

# Example renaming block (adjust to match your trained model's expected names):
# old_to_new <- c(
#   "neighbor_max_ntl"  = "ntl_neighbor_max",
#   "neighbor_min_ntl"  = "ntl_neighbor_min",
#   "neighbor_mean_ntl" = "ntl_neighbor_mean",
#   ...
# )
# setnames(cell_data, names(old_to_new), old_to_new)

# Then predict:
# cell_data$prediction <- predict(rf_model, newdata = cell_data)
```

---

## Summary of Expected Gains

| Metric | Current | Optimized |
|---|---|---|
| **`build_neighbor_lookup`** | ~6.46M string operations + hash lookups | One loop over 344K cells to build 1.37M-row edge table (~seconds) |
| **`compute_neighbor_stats`** (per var) | 6.46M R-level `lapply` iterations | One `data.table` keyed join + grouped aggregation (~38.4M rows, vectorized C) |
| **Total for 5 variables** | 5 Ã— 6.46M = 32.3M R iterations | 5 aggregates computed in a single grouped operation |
| **Estimated wall time** | ~86+ hours | **~2â€“10 minutes** on a 16 GB laptop |
| **Peak RAM** | Large (6.46M-element list of integer vectors) | ~3â€“5 GB (edge-year table + attributes) |
| **RF model** | Unchanged âœ“ | Unchanged âœ“ |
| **Numerical results** | Original estimand preserved âœ“ | Identical `max`/`min`/`mean` values âœ“ |

The key transformation is: **stop iterating over cell-year rows in R; instead, express the neighbor relationship as a static edge table and let `data.table`'s C-level join and grouping engine do the work vectorized.** The trained Random Forest model is never retrained â€” only the feature-engineering step is accelerated.
END_RESPONSE>>>
