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
run_id: Anthropic_F09_T03_R001
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.3
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F09
prompt_family_slug: ADJACENCY_TABLE_OPTION
family_label: Adjacency-table option
family_group: Representation cue
replicate: 1
raw_output_file: Experiment_Output/Anthropic/Temp_0.3/output_F09_Anthropic_temp_03_rep_001.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The current implementation is extremely slow (~86+ hours) for two interconnected reasons:

### 1. `build_neighbor_lookup` is O(NÂ²)-ish in practice
The function iterates over **all ~6.46 million cell-year rows** with `lapply`, and for each row it:
- Looks up the cell's spatial neighbors (fine).
- Constructs `paste()`-based string keys for every neighbor Ã— that row's year (expensive string allocation).
- Matches those keys against a named character vector (`idx_lookup`) of length 6.46M (expensive named-vector lookup â€” R's named vector lookup is O(n) per query in the worst case, not hash-based like an environment or `data.table` key).

This means the function performs **~6.46M Ã— k string constructions and lookups** (where k â‰ˆ average neighbor count ~4 for rook contiguity), each against a 6.46M-length named vector. The string-key approach turns what should be a simple integer-index join into a massive string-matching problem.

### 2. The lookup is rebuilt monolithically across all years
The neighbor topology is **purely spatial** â€” it doesn't change across years. Yet the lookup fuses spatial adjacency with temporal indexing in a single 6.46M-element list, recomputing neighbor row indices for every cell-year combination even though the spatial structure is identical every year.

### 3. `compute_neighbor_stats` is fine algorithmically but bottlenecked by the lookup
Once the lookup list exists, the stats computation is a simple O(NÃ—k) pass. The bottleneck is building the lookup.

---

## Optimization Strategy

**Core insight:** Separate the *spatial* adjacency structure (which is static) from the *temporal* attribute join (which varies by year).

### Step-by-step plan:

1. **Build a spatial neighbor edge table once** â€” a two-column `data.table` of `(cell_id, neighbor_cell_id)` derived from the `nb` object. This has ~1.37M rows and never changes.

2. **For each year, join cell attributes onto the edge table** â€” use `data.table` keyed joins. For each year-slice, join the variable values for the neighbor cells onto the edge table, then compute `max`, `min`, `mean` grouped by `cell_id`.

3. **Join the resulting neighbor stats back** onto the main `cell_data` table by `(cell_id, year)`.

This replaces 6.46M string lookups with ~28 vectorized `data.table` joins (one per year), each operating on ~1.37M edges. Expected runtime: **minutes, not hours**.

### Why this preserves correctness:
- The spatial neighbor set per cell is identical.
- The variable values used are identical (same cell-year attribute values).
- `max`, `min`, `mean` are computed over the same neighbor sets.
- The trained Random Forest model is never touched â€” we only recompute the input features identically.

---

## Working R Code

```r
library(data.table)

# ==============================================================================
# STEP 1: Build a static spatial neighbor edge table (run once, reuse forever)
# ==============================================================================
build_neighbor_edge_table <- function(id_order, nb_object) {
  # nb_object: a list of length length(id_order), each element is an integer

#              vector of neighbor indices into id_order (spdep::nb format).
  # Returns a data.table with columns: cell_id, neighbor_id
  
  edges <- rbindlist(lapply(seq_along(nb_object), function(i) {
    nb_idx <- nb_object[[i]]
    # spdep::nb uses 0 to indicate no neighbors
    nb_idx <- nb_idx[nb_idx > 0L]
    if (length(nb_idx) == 0L) return(NULL)
    data.table(cell_id = id_order[i], neighbor_id = id_order[nb_idx])
  }))
  
  return(edges)
}

# Build it once
neighbor_edges <- build_neighbor_edge_table(id_order, rook_neighbors_unique)
# ~1.37M rows, two integer columns â€” tiny in memory

cat(sprintf("Neighbor edge table: %d rows\n", nrow(neighbor_edges)))

# ==============================================================================
# STEP 2: Convert cell_data to data.table (if not already)
# ==============================================================================
if (!is.data.table(cell_data)) {
  cell_data <- as.data.table(cell_data)
}

# Ensure 'id' and 'year' columns exist
stopifnot(all(c("id", "year") %in% names(cell_data)))

# ==============================================================================
# STEP 3: Compute neighbor stats for all variables, all years â€” vectorized
# ==============================================================================
compute_all_neighbor_features <- function(cell_data, neighbor_edges, var_names) {
  # Pre-allocate output columns with NA
  for (var_name in var_names) {
    col_max  <- paste0("neighbor_max_",  var_name)
    col_min  <- paste0("neighbor_min_",  var_name)
    col_mean <- paste0("neighbor_mean_", var_name)
    cell_data[, (col_max)  := NA_real_]
    cell_data[, (col_min)  := NA_real_]
    cell_data[, (col_mean) := NA_real_]
  }
  
  # Key the edge table for fast joins
  setkey(neighbor_edges, neighbor_id)
  
  # Get unique years
  years <- sort(unique(cell_data$year))
  
  cat(sprintf("Processing %d years Ã— %d variables...\n", 
              length(years), length(var_names)))
  
  for (yr in years) {
    t0 <- proc.time()[3]
    
    # Extract this year's cell attributes (only id + the variables we need)
    yr_attrs <- cell_data[year == yr, c("id", var_names), with = FALSE]
    setnames(yr_attrs, "id", "neighbor_id")
    setkey(yr_attrs, neighbor_id)
    
    # Join neighbor attributes onto the edge table
    # After this join, each row has: cell_id, neighbor_id, ntl, ec, ...
    edges_with_vals <- neighbor_edges[yr_attrs, on = "neighbor_id", nomatch = NULL]
    # edges_with_vals now has columns: cell_id, neighbor_id, <var_names...>
    
    # Compute grouped stats for each variable
    # Group by cell_id to get the neighbor summary for each cell
    agg_exprs <- list()
    for (var_name in var_names) {
      col_max  <- paste0("neighbor_max_",  var_name)
      col_min  <- paste0("neighbor_min_",  var_name)
      col_mean <- paste0("neighbor_mean_", var_name)
      agg_exprs[[col_max]]  <- call("max",  as.name(var_name), na.rm = TRUE)
      agg_exprs[[col_min]]  <- call("min",  as.name(var_name), na.rm = TRUE)
      agg_exprs[[col_mean]] <- call("mean", as.name(var_name), na.rm = TRUE)
    }
    
    # Build and evaluate the aggregation
    agg_call <- as.call(c(as.name("list"), agg_exprs))
    neighbor_stats <- edges_with_vals[, eval(agg_call), by = cell_id]
    
    # Fix Inf/-Inf from max/min on all-NA groups (shouldn't happen with 
    # nomatch=NULL but defensive)
    stat_cols <- setdiff(names(neighbor_stats), "cell_id")
    for (sc in stat_cols) {
      neighbor_stats[is.infinite(get(sc)), (sc) := NA_real_]
    }
    
    # Join back into cell_data for this year
    # We need to match on (id == cell_id) AND (year == yr)
    setkey(neighbor_stats, cell_id)
    
    # Get row indices in cell_data for this year
    yr_row_idx <- cell_data[, which(year == yr)]
    yr_cell_ids <- cell_data$id[yr_row_idx]
    
    # Create a mapping from cell_id to the stats
    match_idx <- match(yr_cell_ids, neighbor_stats$cell_id)
    
    for (sc in stat_cols) {
      set(cell_data, i = yr_row_idx, j = sc, value = neighbor_stats[[sc]][match_idx])
    }
    
    elapsed <- proc.time()[3] - t0
    cat(sprintf("  Year %d done in %.1f seconds\n", yr, elapsed))
  }
  
  return(cell_data)
}

# ==============================================================================
# STEP 4: Run it
# ==============================================================================
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cell_data <- compute_all_neighbor_features(
  cell_data, 
  neighbor_edges, 
  neighbor_source_vars
)

# ==============================================================================
# STEP 5: Predict with the existing trained Random Forest (unchanged)
# ==============================================================================
# The trained RF model object (e.g., `rf_model`) is used as-is.
# cell_data now has the same neighbor feature columns as before:
#   neighbor_max_ntl, neighbor_min_ntl, neighbor_mean_ntl, 
#   neighbor_max_ec,  neighbor_min_ec,  neighbor_mean_ec, ... etc.
#
# Predict:
# cell_data$prediction <- predict(rf_model, newdata = cell_data)$predictions
# (or however your predict call is structured)

cat("Done. Neighbor features computed. RF model is unchanged.\n")
```

---

## Expected Performance

| Metric | Original | Optimized |
|---|---|---|
| **Lookup construction** | ~6.46M string-key lookups against 6.46M named vector | One-time `rbindlist` over 344K cells (~1.37M edge rows) |
| **Per-variable stats** | 6.46M `lapply` iterations with subsetting | 28 keyed `data.table` joins + grouped aggregation |
| **Total estimated time** | **86+ hours** | **~5â€“15 minutes** |
| **Peak RAM** | ~6.46M-element list of integer vectors + named vector | ~1.37M-row edge table + year slices (~350K rows) |
| **RF model** | Unchanged âœ“ | Unchanged âœ“ |
| **Numerical output** | Original `max`/`min`/`mean` | Identical `max`/`min`/`mean` âœ“ |

The speedup comes from:
1. **Eliminating string-key construction and lookup** â€” replaced by integer-keyed `data.table` joins.
2. **Vectorized grouped aggregation** â€” `data.table`'s `[, ..., by=]` replaces 6.46M individual `lapply` calls.
3. **Year-sliced processing** â€” keeps working memory small (~50MB per year-slice join) while still being fully vectorized within each year.
END_RESPONSE>>>
