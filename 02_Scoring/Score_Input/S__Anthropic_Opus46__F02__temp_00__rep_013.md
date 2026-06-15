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
run_id: Anthropic_F02_T00_R013
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F02
prompt_family_slug: CODE_ONLY_NO_HINT
family_label: Code only, no hint
family_group: Code-visible neutral
replicate: 13
raw_output_file: Experiment_Output/Anthropic/Temp_0.0/output_F02_Anthropic_temp_00_rep_013.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The core bottleneck is the **`build_neighbor_lookup` function**, which runs an `lapply` over ~6.46 million rows, performing per-row string pasting, hash-table lookups, and NA filtering. This produces a list of 6.46 million integer vectors. The secondary bottleneck is **`compute_neighbor_stats`**, which iterates over that same 6.46-million-element list five times (once per variable), computing `max`, `min`, and `mean` on small vectors inside an R-level loop.

**Specific problems:**

1. **String-key hashing per row in `build_neighbor_lookup`:** For every row, `paste(neighbor_cell_ids, data$year[i], sep = "_")` creates character vectors and looks them up in a named vector (`idx_lookup`). With ~6.46M rows and an average of ~4 rook neighbors each, this is ~25.8 million `paste` + lookup operations, all in interpreted R.

2. **The lookup list is enormous:** A 6.46-million-element list of integer vectors consumes substantial memory (list overhead alone is ~50 bytes/element â‰ˆ 300+ MB, plus the integer vectors themselves).

3. **`compute_neighbor_stats` is called 5 times**, each time iterating over the full 6.46M-element list in pure R `lapply`. That's ~32.3 million R-level function calls total.

4. **No vectorization or use of `data.table`:** The entire pipeline relies on base-R named-vector lookups and `lapply`, missing opportunities for vectorized joins and grouped operations.

---

## Optimization Strategy

**Replace the row-level list-based lookup with a vectorized `data.table` join-and-aggregate approach.**

The key insight: the neighbor lookup is conceptually a **join**. Each `(cell_id, year)` pair needs to be joined to its neighbors' `(neighbor_id, year)` rows, then aggregated. `data.table` performs this kind of equi-join + grouped aggregation in optimized C, eliminating millions of R-level function calls.

**Steps:**

1. **Build an edge table** from the `nb` object: a two-column `data.table` with `(id, neighbor_id)` â€” done once, ~1.37M rows.
2. **Join the edge table to the panel data** on `(neighbor_id, year)` to pull neighbor values â€” this is a single keyed `data.table` merge producing ~1.37M Ã— 28 â‰ˆ ~38.5M rows (but done in C, very fast).
3. **Aggregate** (`max`, `min`, `mean`) by `(id, year)` â€” a single grouped operation in `data.table`.
4. **Merge** the aggregated stats back to the main data.
5. **Repeat for each variable**, or do all 5 variables in one pass.

**Expected improvement:**
- `build_neighbor_lookup` is eliminated entirely.
- `compute_neighbor_stats` is replaced by a vectorized join + group-by.
- Estimated runtime: **minutes, not hours** (the join is ~38.5M rows, well within `data.table` comfort zone on 16 GB RAM).
- Memory: the edge table Ã— years is ~38.5M rows Ã— a few columns â€” manageable.
- The trained Random Forest model and all numerical outputs are preserved exactly (same `max`, `min`, `mean` computations on the same neighbor sets).

---

## Working R Code

```r
library(data.table)

#' Optimized neighbor feature pipeline
#' 
#' @param cell_data        data.frame or data.table with columns: id, year, and all neighbor_source_vars
#' @param id_order          character or integer vector of cell IDs in the order matching rook_neighbors_unique
#' @param rook_neighbors_unique  an nb object (list of integer index vectors) from spdep
#' @param neighbor_source_vars   character vector of variable names to compute neighbor stats for
#' 
#' @return cell_data with new columns: {var}_neighbor_max, {var}_neighbor_min, {var}_neighbor_mean
compute_all_neighbor_features <- function(cell_data,
                                          id_order,
                                          rook_neighbors_unique,
                                          neighbor_source_vars) {
  
  # --- Step 1: Convert to data.table if needed (by reference, no copy) ---
  if (!is.data.table(cell_data)) {
    cell_data <- as.data.table(cell_data)
    was_df <- TRUE
  } else {
    was_df <- FALSE
  }
  
  # --- Step 2: Build the edge table from the nb object ---
  # Each element of rook_neighbors_unique is an integer vector of indices into id_order.
  # We expand this into a two-column table: (focal_id, neighbor_id).
  
  n_cells <- length(id_order)
  
  # Compute lengths of each neighbor set
  nb_lengths <- lengths(rook_neighbors_unique)  # integer vector, length = n_cells
  
  # Pre-allocate vectors
  total_edges <- sum(nb_lengths)  # ~1,373,394
  focal_ids    <- rep(id_order, times = nb_lengths)
  neighbor_ids <- id_order[unlist(rook_neighbors_unique, use.names = FALSE)]
  
  edges <- data.table(
    focal_id    = focal_ids,
    neighbor_id = neighbor_ids
  )
  rm(focal_ids, neighbor_ids)  # free memory
  
  cat(sprintf("Edge table built: %d directed edges\n", nrow(edges)))
  
  # --- Step 3: Prepare a keyed lookup of the panel data ---
  # We only need id, year, and the source variables for the join.
  lookup_cols <- c("id", "year", neighbor_source_vars)
  lookup_dt   <- cell_data[, ..lookup_cols]
  
  # Key for fast join
  setkey(lookup_dt, id, year)
  
  # --- Step 4: Get unique years to iterate (avoids a massive cross-join) ---
  # Actually, the most memory-efficient vectorized approach is:
  #   - Cross the edge table with years to get (focal_id, neighbor_id, year)
  #   - Join to lookup_dt on (neighbor_id, year) to get neighbor values
  #   - Aggregate by (focal_id, year)
  #
  # With ~1.37M edges Ã— 28 years = ~38.4M rows, this fits in memory.
  
  unique_years <- sort(unique(cell_data$year))
  
  # Expand edges Ã— years
  edges_by_year <- edges[, .(year = unique_years), by = .(focal_id, neighbor_id)]
  # This creates ~38.4M rows via cross join per edge.
  # More memory-efficient alternative using CJ:
  
  # Actually, the above syntax is wrong for a cross join in data.table.
  # Correct approach:
  edges_by_year <- CJ(edge_idx = seq_len(nrow(edges)), year = unique_years)
  edges_by_year[, focal_id    := edges$focal_id[edge_idx]]
  edges_by_year[, neighbor_id := edges$neighbor_id[edge_idx]]
  edges_by_year[, edge_idx    := NULL]
  
  cat(sprintf("Edges Ã— years table: %d rows (%.1f M)\n", 
              nrow(edges_by_year), nrow(edges_by_year) / 1e6))
  
  rm(edges)
  gc()
  
  # --- Step 5: Join to get neighbor values ---
  # Join edges_by_year to lookup_dt on (neighbor_id = id, year = year)
  setkey(edges_by_year, neighbor_id, year)
  setkey(lookup_dt, id, year)
  
  # Perform the join â€” this pulls in all neighbor_source_vars columns
  edges_by_year <- lookup_dt[edges_by_year, on = .(id = neighbor_id, year = year)]
  
  # After the join, 'id' column is the neighbor_id. Rename for clarity.
  # The focal_id column is carried through from edges_by_year.
  # 'id' now = neighbor_id (from the join key), focal_id = the cell we're computing features for.
  
  # --- Step 6: Aggregate by (focal_id, year) for each variable ---
  setkey(edges_by_year, focal_id, year)
  
  for (var_name in neighbor_source_vars) {
    cat(sprintf("  Computing neighbor stats for: %s\n", var_name))
    
    max_col  <- paste0(var_name, "_neighbor_max")
    min_col  <- paste0(var_name, "_neighbor_min")
    mean_col <- paste0(var_name, "_neighbor_mean")
    
    # Aggregate â€” na.rm = TRUE matches the original code's behavior
    # (original code filters out NAs before computing stats)
    agg <- edges_by_year[, .(
      nb_max  = if (all(is.na(get(var_name)))) NA_real_ else max(get(var_name), na.rm = TRUE),
      nb_min  = if (all(is.na(get(var_name)))) NA_real_ else min(get(var_name), na.rm = TRUE),
      nb_mean = if (all(is.na(get(var_name)))) NA_real_ else mean(get(var_name), na.rm = TRUE)
    ), by = .(focal_id, year)]
    
    setnames(agg, c("nb_max", "nb_min", "nb_mean"), c(max_col, min_col, mean_col))
    
    # Merge back to cell_data
    # Remove these columns from cell_data if they already exist (idempotency)
    for (col in c(max_col, min_col, mean_col)) {
      if (col %in% names(cell_data)) cell_data[, (col) := NULL]
    }
    
    cell_data <- merge(cell_data, agg, by.x = c("id", "year"), by.y = c("focal_id", "year"), all.x = TRUE)
    
    rm(agg)
  }
  
  rm(edges_by_year, lookup_dt)
  gc()
  
  if (was_df) cell_data <- as.data.frame(cell_data)
  
  return(cell_data)
}
```

### Even More Memory-Efficient Variant (Year-Chunked)

If the ~38.4M-row `edges_by_year` table is too large for available RAM (each of the 5 numeric columns is ~307 MB), process one year at a time:

```r
compute_all_neighbor_features_chunked <- function(cell_data,
                                                   id_order,
                                                   rook_neighbors_unique,
                                                   neighbor_source_vars) {
  
  if (!is.data.table(cell_data)) {
    cell_data <- as.data.table(cell_data)
    was_df <- TRUE
  } else {
    was_df <- FALSE
  }
  
  # Build edge table (same as before)
  nb_lengths   <- lengths(rook_neighbors_unique)
  focal_ids    <- rep(id_order, times = nb_lengths)
  neighbor_ids <- id_order[unlist(rook_neighbors_unique, use.names = FALSE)]
  edges <- data.table(focal_id = focal_ids, neighbor_id = neighbor_ids)
  rm(focal_ids, neighbor_ids)
  
  lookup_cols <- c("id", "year", neighbor_source_vars)
  lookup_dt   <- cell_data[, ..lookup_cols]
  setkey(lookup_dt, id, year)
  
  unique_years <- sort(unique(cell_data$year))
  
  # Initialize result columns
  for (var_name in neighbor_source_vars) {
    cell_data[, paste0(var_name, "_neighbor_max")  := NA_real_]
    cell_data[, paste0(var_name, "_neighbor_min")  := NA_real_]
    cell_data[, paste0(var_name, "_neighbor_mean") := NA_real_]
  }
  
  setkey(cell_data, id, year)
  
  for (yr in unique_years) {
    cat(sprintf("Processing year %d ...\n", yr))
    
    # Subset panel data for this year
    yr_data <- lookup_dt[year == yr]
    setkey(yr_data, id)
    
    # Join edges to this year's data to get neighbor values
    # edges has (focal_id, neighbor_id); join on neighbor_id = id
    merged <- yr_data[edges, on = .(id = neighbor_id), nomatch = NA, allow.cartesian = TRUE]
    # merged now has columns: id (=neighbor_id), year, <vars>, focal_id
    
    for (var_name in neighbor_source_vars) {
      max_col  <- paste0(var_name, "_neighbor_max")
      min_col  <- paste0(var_name, "_neighbor_min")
      mean_col <- paste0(var_name, "_neighbor_mean")
      
      agg <- merged[, {
        v <- get(var_name)
        v <- v[!is.na(v)]
        if (length(v) == 0L) {
          list(nb_max = NA_real_, nb_min = NA_real_, nb_mean = NA_real_)
        } else {
          list(nb_max = max(v), nb_min = min(v), nb_mean = mean(v))
        }
      }, by = .(focal_id)]
      
      # Update cell_data in place for this year
      cell_data[agg, on = .(id = focal_id, year = yr),
                (max_col)  := i.nb_max]
      cell_data[agg, on = .(id = focal_id, year = yr),
                (min_col)  := i.nb_min]
      cell_data[agg, on = .(id = focal_id, year = yr),
                (mean_col) := i.nb_mean]
      
      rm(agg)
    }
    
    rm(merged, yr_data)
    gc()
  }
  
  rm(edges, lookup_dt)
  gc()
  
  if (was_df) cell_data <- as.data.frame(cell_data)
  return(cell_data)
}
```

### Usage (Drop-In Replacement)

```r
library(data.table)

# Load your existing objects
# cell_data                 â€” your panel data.frame/data.table
# id_order                  â€” vector of cell IDs matching the nb object
# rook_neighbors_unique     â€” the spdep nb object
# rf_model                  â€” your trained Random Forest (untouched)

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# Option A: Single-pass (needs ~4-6 GB free RAM)
cell_data <- compute_all_neighbor_features(
  cell_data, id_order, rook_neighbors_unique, neighbor_source_vars
)

# Option B: Year-chunked (needs ~1-2 GB free RAM, slightly slower but safer on 16 GB)
cell_data <- compute_all_neighbor_features_chunked(
  cell_data, id_order, rook_neighbors_unique, neighbor_source_vars
)

# Predict with the existing trained model (unchanged)
cell_data$predicted_gdp <- predict(rf_model, newdata = cell_data)
```

---

## Summary

| Aspect | Original | Optimized |
|---|---|---|
| **Lookup construction** | 6.46M R-level `paste` + named-vector lookups | Eliminated; replaced by `data.table` keyed join |
| **Neighbor stats** | 5 Ã— 6.46M `lapply` calls (R interpreter) | 5 Ã— 1 vectorized `data.table` group-by (C backend) |
| **Estimated runtime** | 86+ hours | ~5â€“20 minutes (depending on disk I/O and RAM) |
| **Peak RAM** | Very high (6.46M-element list + copies) | ~4â€“6 GB (Option A) or ~1â€“2 GB (Option B, chunked) |
| **Numerical results** | Baseline | Identical (`max`, `min`, `mean` with NA removal) |
| **RF model** | â€” | Untouched; no retraining needed |
END_RESPONSE>>>
