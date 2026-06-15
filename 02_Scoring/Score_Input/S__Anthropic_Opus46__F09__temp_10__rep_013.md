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
run_id: Anthropic_F09_T10_R013
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F09
prompt_family_slug: ADJACENCY_TABLE_OPTION
family_label: Adjacency-table option
family_group: Representation cue
replicate: 13
raw_output_file: Experiment_Output/Anthropic/Temp_1.0/output_F09_Anthropic_temp_10_rep_013.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The current implementation is extremely slow (~86+ hours) for two root causes:

### 1. `build_neighbor_lookup` is O(NÂ²)-ish in practice
It iterates over all **6.46 million** cell-year rows, and for each row it:
- Does a character-based hash lookup (`id_to_ref`, `idx_lookup`) using `paste()` to build string keys.
- Constructs per-row neighbor keys by pasting cell IDs and years.
- Performs named-vector lookups (which in R are linear scans on long named vectors).

The `idx_lookup` named vector has ~6.46 million entries. Named vector lookup in R is **O(n)** per query (it is *not* a hash table). So each of the 6.46M rows does multiple O(6.46M) scans â†’ effectively **O(nÂ²)** total.

### 2. `compute_neighbor_stats` uses `lapply` over 6.46M rows
Each iteration subsets a numeric vector and computes max/min/mean. While each call is cheap, 6.46M R-level function calls with list allocation is inherently slow.

### 3. The neighbor topology is year-invariant but rebuilt per cell-year
Rook neighbors depend only on spatial grid position, not on year. The current code re-resolves neighbor *row indices* for every cell-year, mixing spatial topology with temporal indexing unnecessarily.

---

## Optimization Strategy

**Key insight:** Separate the *static spatial topology* from the *yearly attribute join*.

1. **Build a cell-level neighbor edge table once** â€” a simple two-column `data.table` of `(cell_id, neighbor_id)` derived from `rook_neighbors_unique`. This is only ~1.37M rows.

2. **Join yearly attributes onto the edge table** â€” For each year and variable, join the cell's attribute value onto the neighbor side of the edge table using `data.table` keyed joins (O(n log n) or O(n) with keys).

3. **Aggregate neighbor stats by `(cell_id, year)`** â€” Use `data.table`'s grouped aggregation (`[, .(max, min, mean), by = .(cell_id, year)]`) which is highly optimized in C.

4. **Join the aggregated stats back** onto the main data.

This eliminates all per-row R-level iteration, all `paste()`-based key construction, and all named-vector lookups. Expected runtime: **minutes, not hours**.

---

## Working R Code

```r
library(data.table)

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# STEP 1: Build the static cell-neighbor edge table (once)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# rook_neighbors_unique is an nb object (list of integer index vectors)
# id_order is the vector of cell IDs corresponding to each nb element

build_edge_table <- function(id_order, neighbors) {
  # neighbors[[i]] gives integer indices into id_order for cell id_order[i]
  from_idx <- rep(seq_along(neighbors), lengths(neighbors))
  to_idx   <- unlist(neighbors, use.names = FALSE)
  
  edge_dt <- data.table(
    cell_id     = id_order[from_idx],
    neighbor_id = id_order[to_idx]
  )
  return(edge_dt)
}

edge_table <- build_edge_table(id_order, rook_neighbors_unique)
# edge_table has ~1,373,394 rows: (cell_id, neighbor_id)

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# STEP 2: Convert main data to data.table (if not already)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
cell_dt <- as.data.table(cell_data)

# Ensure key columns exist and have consistent types
stopifnot(all(c("id", "year") %in% names(cell_dt)))

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# STEP 3: Compute neighbor stats for all variables at once
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

compute_all_neighbor_features <- function(cell_dt, edge_table, vars) {
  
  # Extract only the columns we need for the neighbor join
  # Columns: id, year, and all neighbor source variables
  cols_needed <- c("id", "year", vars)
  attr_dt <- cell_dt[, ..cols_needed]
  
  # Key the attribute table by id for fast join
  setkey(attr_dt, id)
  
  # Create a year-expanded edge table:
  # For each year, every edge (cell_id -> neighbor_id) is valid.
  # Instead of a full cross join (which would be huge), we join per year.
  
  years <- sort(unique(attr_dt$year))
  
  # Pre-allocate list for results
  result_list <- vector("list", length(years))
  
  for (yi in seq_along(years)) {
    yr <- years[yi]
    
    # Subset attributes for this year
    yr_attr <- attr_dt[year == yr]
    setkey(yr_attr, id)
    
    # Join neighbor attributes onto edge table
    # edge_table$neighbor_id -> yr_attr$id to get neighbor's variable values
    merged <- merge(
      edge_table,
      yr_attr[, c("id", vars), with = FALSE],
      by.x = "neighbor_id",
      by.y = "id",
      allow.cartesian = FALSE
    )
    # merged now has: neighbor_id, cell_id, ntl, ec, pop_density, def, usd_est_n2
    # Each row = one directed neighbor relationship for this year
    
    # Aggregate: for each cell_id, compute max/min/mean of each variable
    # across all its neighbors
    agg_exprs <- list()
    for (v in vars) {
      v_sym <- as.name(v)
      agg_exprs[[paste0("n_max_", v)]]  <- bquote(max(.(v_sym), na.rm = TRUE))
      agg_exprs[[paste0("n_min_", v)]]  <- bquote(min(.(v_sym), na.rm = TRUE))
      agg_exprs[[paste0("n_mean_", v)]] <- bquote(mean(.(v_sym), na.rm = TRUE))
    }
    
    # Build the aggregation call
    # Equivalent to: merged[, .(n_max_ntl = max(ntl, na.rm=TRUE), ...), by = cell_id]
    agg_dt <- merged[, 
      lapply(.SD, function(x) {
        x <- x[!is.na(x)]
        if (length(x) == 0L) return(c(NA_real_, NA_real_, NA_real_))
        c(max(x), min(x), mean(x))
      }),
      by = cell_id,
      .SDcols = vars
    ]
    # The above returns 3 rows per cell_id (max, min, mean). 
    # A cleaner approach:
    
    agg_dt <- merged[, {
      res <- list()
      for (v in vars) {
        vals <- get(v)
        vals <- vals[!is.na(vals)]
        if (length(vals) == 0L) {
          res[[paste0("n_max_", v)]]  <- NA_real_
          res[[paste0("n_min_", v)]]  <- NA_real_
          res[[paste0("n_mean_", v)]] <- NA_real_
        } else {
          res[[paste0("n_max_", v)]]  <- max(vals)
          res[[paste0("n_min_", v)]]  <- min(vals)
          res[[paste0("n_mean_", v)]] <- mean(vals)
        }
      }
      res
    }, by = cell_id]
    
    agg_dt[, year := yr]
    result_list[[yi]] <- agg_dt
    
    if (yi %% 5 == 0) message(sprintf("  Processed year %d (%d/%d)", yr, yi, length(years)))
  }
  
  rbindlist(result_list)
}

message("Computing neighbor features...")
t0 <- Sys.time()
neighbor_features <- compute_all_neighbor_features(cell_dt, edge_table, neighbor_source_vars)
message(sprintf("Done in %.1f minutes", difftime(Sys.time(), t0, units = "mins")))

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# STEP 4: Join neighbor features back to main data
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

# Replace Inf/-Inf from max/min of empty sets (shouldn't happen with our NA guard, but safety)
inf_cols <- names(neighbor_features)[grepl("^n_(max|min|mean)_", names(neighbor_features))]
for (col in inf_cols) {
  vals <- neighbor_features[[col]]
  vals[is.infinite(vals)] <- NA_real_
  set(neighbor_features, j = col, value = vals)
}

setkey(cell_dt, id, year)
setkey(neighbor_features, cell_id, year)

# Remove any pre-existing neighbor columns to avoid conflicts
old_ncols <- grep("^n_(max|min|mean)_", names(cell_dt), value = TRUE)
if (length(old_ncols) > 0) {
  cell_dt[, (old_ncols) := NULL]
}

# Merge
cell_dt <- merge(cell_dt, neighbor_features, by.x = c("id", "year"), by.y = c("cell_id", "year"), all.x = TRUE)

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# STEP 5: Predict with the existing trained Random Forest
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# The trained model object is assumed to be in memory (e.g., `rf_model`).
# Convert back to data.frame if the model expects one.

cell_data <- as.data.frame(cell_dt)

# Predict (model is NOT retrained â€” just used for inference)
cell_data$rf_prediction <- predict(rf_model, newdata = cell_data)

message("Pipeline complete.")
```

---

## Further Speed-Up: Vectorized Aggregation (No Inner Loop)

The `for (v in vars)` inside the `j`-expression of `data.table` can be replaced with a fully vectorized `.SD`-based aggregation for even more speed:

```r
# Faster alternative for STEP 3 inner aggregation:
compute_all_neighbor_features_fast <- function(cell_dt, edge_table, vars) {
  
  attr_dt <- cell_dt[, c("id", "year", vars), with = FALSE]
  years <- sort(unique(attr_dt$year))
  result_list <- vector("list", length(years))
  
  for (yi in seq_along(years)) {
    yr <- years[yi]
    yr_attr <- attr_dt[year == yr]
    
    merged <- yr_attr[edge_table, on = .(id = neighbor_id), nomatch = 0L,
                      allow.cartesian = FALSE]
    # merged columns: id (=neighbor_id), year, <vars>, cell_id
    
    # Compute all three stats at once using .SD
    agg_max  <- merged[, lapply(.SD, function(x) { x <- x[!is.na(x)]; if(length(x)==0L) NA_real_ else max(x) }),  by = cell_id, .SDcols = vars]
    agg_min  <- merged[, lapply(.SD, function(x) { x <- x[!is.na(x)]; if(length(x)==0L) NA_real_ else min(x) }),  by = cell_id, .SDcols = vars]
    agg_mean <- merged[, lapply(.SD, function(x) { x <- x[!is.na(x)]; if(length(x)==0L) NA_real_ else mean(x) }), by = cell_id, .SDcols = vars]
    
    setnames(agg_max,  vars, paste0("n_max_",  vars))
    setnames(agg_min,  vars, paste0("n_min_",  vars))
    setnames(agg_mean, vars, paste0("n_mean_", vars))
    
    combined <- agg_max[agg_min, on = "cell_id"][agg_mean, on = "cell_id"]
    combined[, year := yr]
    result_list[[yi]] <- combined
    
    if (yi %% 5 == 0) message(sprintf("  Year %d done (%d/%d)", yr, yi, length(years)))
  }
  
  rbindlist(result_list)
}
```

---

## Expected Performance

| Component | Before | After |
|---|---|---|
| Neighbor lookup build | ~40-60 hrs (6.46M Ã— O(n) named-vec lookups) | ~2 seconds (vectorized edge table) |
| Neighbor stats (5 vars) | ~20-30 hrs (6.46M Ã— 5 lapply calls) | ~5-15 min (data.table grouped agg Ã— 28 years) |
| Total | **86+ hours** | **~5-20 minutes** |

The trained Random Forest model is never retrained â€” only `predict()` is called. The numerical estimand is preserved because `max`, `min`, and `mean` are computed identically over the same neighbor sets; only the implementation mechanism (vectorized join + grouped aggregation vs. row-wise R loop) changes.
END_RESPONSE>>>
