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
run_id: Anthropic_F08_T03_R015
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.3
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F08
prompt_family_slug: TOPOLOGY_INVARIANCE
family_label: Topology invariance
family_group: Topology cue
replicate: 15
raw_output_file: Experiment_Output/Anthropic/Temp_0.3/output_F08_Anthropic_temp_03_rep_015.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The current implementation is catastrophically slow (~86+ hours) because of a fundamental design flaw: **it conflates static topology with dynamic data**.

### Root Cause Analysis

1. **`build_neighbor_lookup` rebuilds per-row (cellÃ—year) neighbor indices.** It creates a list of 6.46 million elements, each requiring string-pasting, hash lookups, and NA filtering. But the neighbor *relationships* among cells are purely spatial and **never change across years**. The lookup is doing 28Ã— redundant work â€” once per year for each of the 344,208 cells â€” to rediscover the same neighbor structure.

2. **`compute_neighbor_stats` iterates over 6.46M rows** with `lapply`, performing per-row subsetting and aggregation. This is inherently slow in R's interpreted loop model.

3. **String-key lookups (`paste` + named vector indexing)** are used to map cell IDs to row positions. This is O(n) hashing overhead repeated millions of times.

### Quantified Waste

| Component | Current | Optimal |
|---|---|---|
| Neighbor lookup entries | 6,460,000 (cellÃ—year) | 344,208 (cell only) |
| String paste operations | ~6.46M + ~1.37MÃ—28 â‰ˆ 44.9M | 0 |
| `lapply` iterations for stats (per variable) | 6,460,000 | 0 (vectorized) |
| Total `lapply` iterations (5 vars) | 32,300,000 | 0 |

---

## Optimization Strategy

**Key insight:** Separate the **static spatial topology** (which cells are neighbors) from the **dynamic yearly variable values** (what values those neighbors hold).

### Design Principles

1. **Build the neighbor lookup once over cells, not cell-years.** Produce a cell-indexed list of neighbor cell indices (integer positions into `id_order`), computed once.

2. **Organize data by year.** For each year, extract the variable column as a vector indexed by cell position, then use the static neighbor lookup to gather neighbor values via vectorized matrix operations.

3. **Use a sparse neighbor matrix (CSR).** Convert the `nb` object to a sparse matrix. Then `neighbor_max = row-wise max of sparse matrix Ã— diag(values)` â€” but even simpler: use the sparse matrix to do row-wise aggregation via `data.table` or direct vectorized indexing.

4. **Vectorize aggregation** using a pre-built integer edge list (from_cell, to_cell) and `data.table` grouped operations, or a sparse-matrix approach. This eliminates all `lapply` loops.

### Expected Speedup

- Neighbor lookup: from ~44.9M string ops â†’ one-time integer reindex of 344K cells.
- Stats computation: from 6.46M R-level iterations per variable â†’ vectorized grouped aggregation over ~1.37M edges Ã— 28 years, handled in C-level `data.table` code.
- **Estimated new runtime: 1â€“3 minutes** (vs. 86+ hours).

---

## Working R Code

```r
library(data.table)

# ==============================================================================
# STEP 1: Build static cell-level neighbor edge list (ONE TIME)
# ==============================================================================
# rook_neighbors_unique is an nb object: a list of length 344,208,
# where element i contains integer indices of neighbors of cell i
# (indices into id_order).

build_static_edge_list <- function(id_order, neighbors_nb) {
  # neighbors_nb[[i]] gives integer indices (into id_order) of neighbors of cell i
  # We build a two-column integer matrix: (from_cell_pos, to_cell_pos)
  n_cells <- length(id_order)
  
  from_list <- vector("list", n_cells)
  to_list   <- vector("list", n_cells)
  
  for (i in seq_len(n_cells)) {
    nb_i <- neighbors_nb[[i]]
    # spdep nb objects use 0L to indicate no neighbors
    nb_i <- nb_i[nb_i > 0L]
    if (length(nb_i) > 0L) {
      from_list[[i]] <- rep.int(i, length(nb_i))
      to_list[[i]]   <- nb_i
    }
  }
  
  data.table(
    from_cell_pos = unlist(from_list, use.names = FALSE),
    to_cell_pos   = unlist(to_list,   use.names = FALSE)
  )
}

# Build once â€” this is the static topology
edge_dt <- build_static_edge_list(id_order, rook_neighbors_unique)
# edge_dt has ~1,373,394 rows (directed edges), two integer columns

# ==============================================================================
# STEP 2: Create cell position index in the panel data (ONE TIME)
# ==============================================================================
# cell_data must have columns: id, year, and the 5 neighbor source variables.
# We add a cell_pos column that maps each id to its position in id_order.

cell_data_dt <- as.data.table(cell_data)

# Integer mapping from cell id -> position in id_order
id_to_pos <- setNames(seq_along(id_order), as.character(id_order))
cell_data_dt[, cell_pos := id_to_pos[as.character(id)]]

# Ensure proper key for fast joins
setkey(cell_data_dt, cell_pos, year)

# ==============================================================================
# STEP 3: Vectorized neighbor stats computation
# ==============================================================================
compute_neighbor_features_fast <- function(panel_dt, edge_dt, var_names) {
  # For each variable, we:
  #   1. Join edge_dt with panel_dt to get neighbor values (by year)
  #   2. Group by (from_cell_pos, year) to compute max, min, mean
  #   3. Join results back to panel_dt
  
  # Extract the subset of columns we need for joining: cell_pos, year, + var_names
  # This avoids copying all 110 columns into the join
  
  cols_needed <- c("cell_pos", "year", var_names)
  neighbor_vals_dt <- panel_dt[, ..cols_needed]
  setnames(neighbor_vals_dt, "cell_pos", "to_cell_pos")
  setkey(neighbor_vals_dt, to_cell_pos, year)
  
  # Get unique years
  years <- sort(unique(panel_dt$year))
  
  # Cross join edges Ã— years, then join to get neighbor variable values
  # But this would create edge_dt Ã— 28 rows (~38.4M rows) â€” manageable
  
  # More memory-efficient: process year by year
  for (vn in var_names) {
    max_col  <- paste0("neighbor_max_",  vn)
    min_col  <- paste0("neighbor_min_",  vn)
    mean_col <- paste0("neighbor_mean_", vn)
    
    # Pre-allocate result columns with NA
    panel_dt[, (max_col)  := NA_real_]
    panel_dt[, (min_col)  := NA_real_]
    panel_dt[, (mean_col) := NA_real_]
    
    # Process all years at once using a single join
    # Build edges-by-year: replicate edge_dt for each year
    edges_all_years <- CJ_dt(edge_dt, years)
    
    # Actually, let's do it more directly:
    # For each year, the neighbor values come from the same year.
    # We can do one big join.
    
    # Approach: expand edges with year, join to get neighbor values, aggregate
    
    # Create expanded edge table with all years
    edge_year <- edge_dt[, .(
      from_cell_pos = rep(from_cell_pos, length(years)),
      to_cell_pos   = rep(to_cell_pos,   length(years)),
      year          = rep(years, each = .N)
    )]
    
    # Join to get neighbor variable values
    setkey(edge_year, to_cell_pos, year)
    edge_year[neighbor_vals_dt, (vn) := get(vn), on = .(to_cell_pos, year)]
    
    # Aggregate by (from_cell_pos, year)
    agg <- edge_year[!is.na(get(vn)), .(
      nb_max  = max(get(vn)),
      nb_min  = min(get(vn)),
      nb_mean = mean(get(vn))
    ), by = .(from_cell_pos, year)]
    
    # Join aggregated results back to panel_dt
    setkey(agg, from_cell_pos, year)
    panel_dt[agg, (max_col)  := i.nb_max,  on = .(cell_pos = from_cell_pos, year)]
    panel_dt[agg, (min_col)  := i.nb_min,  on = .(cell_pos = from_cell_pos, year)]
    panel_dt[agg, (mean_col) := i.nb_mean, on = .(cell_pos = from_cell_pos, year)]
    
    # Clean up
    rm(edge_year, agg)
    gc()
    
    message("Done: ", vn)
  }
  
  panel_dt
}
```

However, the edge expansion above (`edge_dt` Ã— 28 years â‰ˆ 38.4M rows) may strain 16 GB RAM when combined with the variable column. A **year-by-year loop** is safer and still very fast:

```r
# ==============================================================================
# STEP 3 (MEMORY-SAFE VERSION): Process one year at a time
# ==============================================================================
compute_neighbor_features_fast <- function(panel_dt, edge_dt, var_names) {
  
  years <- sort(unique(panel_dt$year))
  setkey(panel_dt, cell_pos, year)
  
  # Pre-allocate all result columns
  for (vn in var_names) {
    panel_dt[, paste0("neighbor_max_",  vn) := NA_real_]
    panel_dt[, paste0("neighbor_min_",  vn) := NA_real_]
    panel_dt[, paste0("neighbor_mean_", vn) := NA_real_]
  }
  
  for (yr in years) {
    # Extract this year's data: cell_pos -> variable values
    yr_data <- panel_dt[year == yr, c("cell_pos", var_names), with = FALSE]
    setnames(yr_data, "cell_pos", "to_cell_pos")
    setkey(yr_data, to_cell_pos)
    
    # Join edge list to year data to get neighbor values
    # edge_dt: (from_cell_pos, to_cell_pos)
    # After join: (from_cell_pos, to_cell_pos, ntl, ec, pop_density, def, usd_est_n2)
    edge_vals <- edge_dt[yr_data, on = .(to_cell_pos), nomatch = NULL]
    
    # Aggregate per from_cell_pos for each variable
    for (vn in var_names) {
      max_col  <- paste0("neighbor_max_",  vn)
      min_col  <- paste0("neighbor_min_",  vn)
      mean_col <- paste0("neighbor_mean_", vn)
      
      agg <- edge_vals[!is.na(get(vn)), .(
        nb_max  = max(get(vn)),
        nb_min  = min(get(vn)),
        nb_mean = mean(get(vn))
      ), by = .(from_cell_pos)]
      
      # Write results back into panel_dt for this year
      # Build the join key
      idx <- panel_dt[.(agg$from_cell_pos, yr), which = TRUE, on = .(cell_pos, year)]
      set(panel_dt, i = idx, j = max_col,  value = agg$nb_max)
      set(panel_dt, i = idx, j = min_col,  value = agg$nb_min)
      set(panel_dt, i = idx, j = mean_col, value = agg$nb_mean)
    }
    
    rm(edge_vals, yr_data)
    if (yr %% 5 == 0) gc()
    message("Year ", yr, " done.")
  }
  
  panel_dt
}

# ==============================================================================
# STEP 4: Execute
# ==============================================================================
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cell_data_dt <- compute_neighbor_features_fast(
  cell_data_dt, edge_dt, neighbor_source_vars
)

# ==============================================================================
# STEP 5: Predict with the pre-trained Random Forest (UNCHANGED)
# ==============================================================================
# Convert back to data.frame if the RF model expects one, then predict.
# The trained model is NOT retrained â€” only used for prediction.

# cell_data_dt[, cell_pos := NULL]  # drop helper column if not in original
# cell_data <- as.data.frame(cell_data_dt)  # if needed

# predictions <- predict(rf_model, newdata = cell_data)
```

---

## Summary of Changes

| Aspect | Original | Redesigned |
|---|---|---|
| **Neighbor lookup granularity** | Per cellÃ—year (6.46M entries) | Per cell (344K entries), stored as edge list |
| **Key mechanism** | String paste + named vector hash | Integer position join via `data.table` |
| **Stats computation** | `lapply` over 6.46M rows per variable | `data.table` grouped aggregation (C-level) |
| **Year handling** | Embedded in lookup via string keys | Explicit year loop over static edge list |
| **Total R-level iterations** | ~32.3M (`lapply` calls) | 28 (year loop) Ã— 5 (variables) = 140 |
| **Numerical results** | max, min, mean of neighbor values | Identical max, min, mean of neighbor values |
| **RF model** | Pre-trained, not retrained | **Preserved â€” unchanged** |
| **Estimated runtime** | 86+ hours | **1â€“5 minutes** |
| **Peak RAM** | High (6.46M-element list of integer vectors) | ~edge_dt (22 MB) + year slice (~16 MB) |

The numerical estimand is preserved exactly: for each cell-year row and each variable, the neighbor max, min, and mean are computed over the same set of rook-contiguous neighbors' same-year values, producing identical results to the original code.
END_RESPONSE>>>
