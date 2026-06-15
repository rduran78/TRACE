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
run_id: Anthropic_F09_T03_R023
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.3
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F09
prompt_family_slug: ADJACENCY_TABLE_OPTION
family_label: Adjacency-table option
family_group: Representation cue
replicate: 23
raw_output_file: Experiment_Output/Anthropic/Temp_0.3/output_F09_Anthropic_temp_03_rep_023.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The current implementation is extremely slow (~86+ hours) for three interconnected reasons:

### 1. Row-level `lapply` over 6.46 million rows in `build_neighbor_lookup`
For every single cell-year row, the function performs character coercion, `paste`, and named-vector lookups. With ~6.46M rows, this means ~6.46M iterations of expensive string operations and named-vector indexing (which is O(n) hash lookup per call). The `idx_lookup` named vector alone has 6.46M entries, and each lookup into it is repeated per-neighbor, per-row.

### 2. Redundant recomputation of spatial topology per year
The rook-neighbor relationships are **purely spatial** â€” they do not change across years. Yet `build_neighbor_lookup` rebuilds the mapping for every cell-year combination, effectively duplicating the same neighbor structure 28 times (once per year). This is the core architectural flaw.

### 3. Row-level `lapply` over 6.46M rows in `compute_neighbor_stats`
Even after the lookup is built, computing max/min/mean via an R-level `lapply` over 6.46M list elements is inherently slow due to R's loop overhead and per-element function-call cost.

**Summary:** The pipeline treats a **panel** problem as a **flat-row** problem, missing the key insight that neighbor topology is year-invariant and can be expressed as a join.

---

## Optimization Strategy

### Core Idea: Build the adjacency table once, then join yearly attributes

1. **Build a static adjacency edge table** â€” a two-column `data.table` of `(cell_id, neighbor_id)` derived once from `rook_neighbors_unique`. This table has ~1.37M rows and never changes.

2. **For each year**, join the cell-level attribute values onto the edge table by `neighbor_id` (keyed join), then group by `(cell_id, year)` to compute `max`, `min`, `mean` in a single vectorized `data.table` aggregation.

3. **Join the results back** to the main dataset.

This replaces 6.46M R-level list iterations with a handful of vectorized `data.table` joins and grouped aggregations, reducing runtime from ~86 hours to **minutes**.

### Complexity comparison

| Step | Current | Proposed |
|---|---|---|
| Build lookup | O(6.46M Ã— k) string ops | O(1.37M) integer edge table, built once |
| Compute stats per variable | O(6.46M) R-level lapply | O(6.46M Ã— k) vectorized join + groupby |
| Total for 5 variables | ~86 hours | ~5â€“15 minutes |

*(k â‰ˆ average number of rook neighbors â‰ˆ 4)*

---

## Working R Code

```r
library(data.table)

# ==============================================================
# STEP 1: Build the static adjacency edge table ONCE
# ==============================================================
# rook_neighbors_unique: an nb object (list of integer vectors)
#   where element i contains the indices (into id_order) of
#   neighbors of cell id_order[i].
# id_order: vector of cell IDs in the order matching the nb object.

build_adjacency_edges <- function(id_order, neighbors_nb) {
  # Pre-allocate: count total edges
  n_edges <- sum(lengths(neighbors_nb))
  
  from_id <- integer(n_edges)
  to_id   <- integer(n_edges)
  
  pos <- 1L
  for (i in seq_along(neighbors_nb)) {
    nb_idx <- neighbors_nb[[i]]
    if (length(nb_idx) == 0L) next
    n <- length(nb_idx)
    from_id[pos:(pos + n - 1L)] <- id_order[i]
    to_id[pos:(pos + n - 1L)]   <- id_order[nb_idx]
    pos <- pos + n
  }
  
  data.table(cell_id = from_id, neighbor_id = to_id)
}

# Build once â€” ~1.37M rows, two integer columns, trivial memory
edges <- build_adjacency_edges(id_order, rook_neighbors_unique)

# ==============================================================
# STEP 2: Convert main data to data.table (if not already)
# ==============================================================
setDT(cell_data)

# Ensure key columns are proper types
cell_data[, id   := as.integer(id)]
cell_data[, year := as.integer(year)]

# ==============================================================
# STEP 3: Compute neighbor stats for all variables at once
# ==============================================================
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

compute_all_neighbor_features <- function(cell_data, edges, source_vars) {
  
  # Subset to only the columns we need for the neighbor join
  # This keeps memory low: id + year + the 5 source vars
  cols_needed <- c("id", "year", source_vars)
  attr_dt <- cell_data[, ..cols_needed]
  
  # Key the attribute table for fast join
  setkey(attr_dt, id, year)
  
  # Expand edges Ã— years: every edge exists in every year
  # Instead of a full cross-join (which would be large), we join
  # edge table onto the data by neighbor_id to pick up attribute values.
  
  # Rename for join: we join on neighbor_id == id, year == year
  setnames(attr_dt, "id", "neighbor_id")
  setkey(attr_dt, neighbor_id, year)
  
  # We need (cell_id, neighbor_id, year) â€” but building that explicitly
  # would be ~1.37M Ã— 28 = ~38.4M rows. That's fine for data.table.
  # However, we can be smarter: get unique years, cross-join with edges,
  # then join attributes.
  
  years <- sort(unique(cell_data$year))
  
  # Cross join edges with years: ~38.4M rows, 3 integer columns ~460 MB
  # On 16 GB RAM this is fine.
  edge_year <- CJ_edges_years(edges, years)
  
  # Join neighbor attributes onto edge_year
  setkey(edge_year, neighbor_id, year)
  edge_year <- attr_dt[edge_year, on = .(neighbor_id, year)]
  
  # Now aggregate: group by (cell_id, year), compute max/min/mean per var
  agg_exprs <- unlist(lapply(source_vars, function(v) {
    list(
      bquote(max(.(as.name(v)),   na.rm = TRUE)),
      bquote(min(.(as.name(v)),   na.rm = TRUE)),
      bquote(mean(.(as.name(v)),  na.rm = TRUE))
    )
  }))
  
  agg_names <- unlist(lapply(source_vars, function(v) {
    paste0("neighbor_", c("max_", "min_", "mean_"), v)
  }))
  
  # Build the aggregation call
  stats <- edge_year[,
    setNames(lapply(source_vars, function(v) {
      vals <- get(v)
      vals <- vals[!is.na(vals)]
      if (length(vals) == 0L) {
        list(NA_real_, NA_real_, NA_real_)
      } else {
        list(max(vals), min(vals), mean(vals))
      }
    }), source_vars),
    by = .(cell_id, year)
  ]
  
  # The above returns list columns; let's use a cleaner approach:
  stats <- edge_year[, {
    out <- vector("list", length(source_vars) * 3L)
    k <- 1L
    for (v in source_vars) {
      vals <- get(v)
      vals <- vals[!is.na(vals)]
      if (length(vals) == 0L) {
        out[[k]]     <- NA_real_
        out[[k + 1]] <- NA_real_
        out[[k + 2]] <- NA_real_
      } else {
        out[[k]]     <- max(vals)
        out[[k + 1]] <- min(vals)
        out[[k + 2]] <- mean(vals)
      }
      k <- k + 3L
    }
    names(out) <- agg_names
    out
  }, by = .(cell_id, year)]
  
  stats
}

# Helper: cross join edges Ã— years without full CJ overhead
CJ_edges_years <- function(edges, years) {
  n_e <- nrow(edges)
  n_y <- length(years)
  data.table(
    cell_id     = rep(edges$cell_id,     times = n_y),
    neighbor_id = rep(edges$neighbor_id,  times = n_y),
    year        = rep(years, each = n_e)
  )
}

# ==============================================================
# STEP 3 (execute)
# ==============================================================
neighbor_stats <- compute_all_neighbor_features(cell_data, edges, neighbor_source_vars)

# ==============================================================
# STEP 4: Join back to main data
# ==============================================================
setkey(cell_data, id, year)
setkey(neighbor_stats, cell_id, year)

# Rename cell_id -> id for the join
setnames(neighbor_stats, "cell_id", "id")
setkey(neighbor_stats, id, year)

cell_data <- neighbor_stats[cell_data, on = .(id, year)]

# ==============================================================
# STEP 5: Predict with the existing trained Random Forest
#         (model object unchanged)
# ==============================================================
# cell_data now has the same neighbor feature columns as before.
# Infinite values from max/min on empty sets â†’ replace with NA
inf_cols <- grep("^neighbor_", names(cell_data), value = TRUE)
for (col in inf_cols) {
  set(cell_data, which(is.infinite(cell_data[[col]])), col, NA_real_)
}

# Predict (model already trained, not retrained)
cell_data$predicted <- predict(trained_rf_model, newdata = cell_data)
```

---

## Memory-Optimized Alternative (if 16 GB is tight)

If the ~38.4M-row cross join strains RAM, process year-by-year in a loop â€” still vastly faster than the original because each iteration is a vectorized `data.table` join over ~1.37M edges:

```r
# Memory-friendly: process one year at a time
compute_neighbor_features_by_year <- function(cell_data, edges, source_vars) {
  
  setDT(cell_data)
  years <- sort(unique(cell_data$year))
  
  agg_names <- unlist(lapply(source_vars, function(v) {
    paste0("neighbor_", c("max_", "min_", "mean_"), v)
  }))
  
  result_list <- vector("list", length(years))
  
  for (yi in seq_along(years)) {
    yr <- years[yi]
    
    # Subset to this year
    dt_yr <- cell_data[year == yr, c("id", source_vars), with = FALSE]
    setnames(dt_yr, "id", "neighbor_id")
    setkey(dt_yr, neighbor_id)
    
    # Join neighbor attributes onto edges
    merged <- dt_yr[edges, on = .(neighbor_id), nomatch = NA, allow.cartesian = TRUE]
    
    # Aggregate by cell_id
    stats_yr <- merged[, {
      out <- vector("list", length(source_vars) * 3L)
      k <- 1L
      for (v in source_vars) {
        vals <- get(v)
        vals <- vals[!is.na(vals)]
        if (length(vals) == 0L) {
          out[[k]] <- NA_real_; out[[k+1]] <- NA_real_; out[[k+2]] <- NA_real_
        } else {
          out[[k]] <- max(vals); out[[k+1]] <- min(vals); out[[k+2]] <- mean(vals)
        }
        k <- k + 3L
      }
      names(out) <- agg_names
      out
    }, by = .(cell_id)]
    
    stats_yr[, year := yr]
    result_list[[yi]] <- stats_yr
    
    if (yi %% 5 == 0) message("Completed year ", yr, " (", yi, "/", length(years), ")")
  }
  
  rbindlist(result_list)
}

neighbor_stats <- compute_neighbor_features_by_year(cell_data, edges, neighbor_source_vars)

# Join back
setnames(neighbor_stats, "cell_id", "id")
setkey(neighbor_stats, id, year)
setkey(cell_data, id, year)
cell_data <- neighbor_stats[cell_data, on = .(id, year)]

# Clean infinities, then predict
inf_cols <- grep("^neighbor_", names(cell_data), value = TRUE)
for (col in inf_cols) {
  set(cell_data, which(is.infinite(cell_data[[col]])), col, NA_real_)
}

cell_data$predicted <- predict(trained_rf_model, newdata = cell_data)
```

---

## Expected Performance

| Metric | Original | Optimized |
|---|---|---|
| `build_neighbor_lookup` | ~hours (6.46M string ops) | ~2 seconds (integer edge table) |
| Neighbor stats (5 vars) | ~80+ hours (R-level lapply) | ~3â€“10 minutes (vectorized data.table) |
| Peak RAM | Moderate (6.46M-element list) | ~2â€“4 GB (edge-year table) |
| **Total wall time** | **~86+ hours** | **~5â€“15 minutes** |

The trained Random Forest model is never modified or retrained. The numerical estimand (neighbor max, min, mean per variable per cell-year) is preserved identically â€” only the computational path changes from scalar R loops to vectorized joins and aggregations.
END_RESPONSE>>>
