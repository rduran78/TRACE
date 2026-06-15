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
run_id: Anthropic_F07_T07_R008
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F07
prompt_family_slug: RASTER_INVALID_IRREGULAR
family_label: Raster invalid irregular topology
family_group: Raster/kernel bridge
replicate: 8
raw_output_file: Experiment_Output/Anthropic/Temp_0.7/output_F07_Anthropic_temp_07_rep_008.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The bottleneck is **`build_neighbor_lookup`**, which creates a list of ~6.46 million elements, each produced by an R-level `lapply` iteration that performs character coercion, `paste`, and named-vector lookups. Specifically:

1. **`build_neighbor_lookup` is O(N Ã— k) with enormous constant factors.** For each of the ~6.46M rows, it converts IDs to character, pastes strings, and does named-vector indexing (`idx_lookup[neighbor_keys]`). Named-vector lookup in R is hash-based but has high per-call overhead when done 6.46M times inside `lapply`. The `paste` and `as.character` calls generate millions of temporary string objects, causing GC pressure.

2. **Redundant recomputation across years.** The neighbor *topology* is identical for every year â€” cell 42's neighbors are always the same cells. Yet the lookup rebuilds the mapping for every cell-year row, effectively repeating the same neighbor resolution 28 times per cell.

3. **`compute_neighbor_stats` is moderately expensive but not the primary bottleneck.** It iterates over the 6.46M-element list, subsetting a numeric vector each time. This is O(N Ã— k) but with lighter operations. Still, it's called 5 times (once per variable).

4. **Memory:** Storing a 6.46M-element list of integer vectors is ~1â€“2 GB, which is feasible on 16 GB but tight when combined with the dataset and RF model.

**Estimated cost of current approach:**
- `build_neighbor_lookup`: ~6.46M iterations Ã— (string ops + hash lookups) â‰ˆ 60â€“80+ hours on a laptop.
- `compute_neighbor_stats`: ~5 variables Ã— 6.46M iterations â‰ˆ 6â€“10 hours.
- Total: ~86+ hours, consistent with the reported estimate.

---

## Optimization Strategy

### Key Insight: Separate Topology from Time

The neighbor graph is **time-invariant**. Instead of building a 6.46M-element row-level lookup, we:

1. **Work at the cell level (344K cells), not the cell-year level (6.46M rows).**
2. **Convert the `nb` object to a sparse adjacency matrix** (or a flat edge list) once.
3. **Use vectorized sparse matrixâ€“dense matrix multiplication** to compute neighbor sums and counts, then derive max/min/mean.

For **mean**: If `A` is the binary row-normalized adjacency matrix and `X` is an NÃ—T matrix of values, then `A %*% X` gives neighbor means. But we also need **max** and **min**, which aren't linear, so matrix multiplication alone won't suffice for those.

### Approach: data.table Join on Edge List

1. Convert `rook_neighbors_unique` (the `nb` object) to a two-column **edge list** data.table: `(focal_id, neighbor_id)` â€” ~1.37M rows.
2. Reshape the panel data so that for each variable, we can join the edge list against the data keyed by `(id, year)`.
3. A single `data.table` merge of the edge list Ã— 28 years (~38.5M rows) against the values table, then `group by (focal_id, year)` to compute `max`, `min`, `mean` â€” all fully vectorized in C via `data.table`.

This replaces 6.46M R-level iterations with a single vectorized join + grouped aggregation, reducing runtime from ~86 hours to **minutes**.

### Why This Preserves the Estimand

- The edge list is an exact representation of the `nb` rook-neighbor topology.
- The `max`, `min`, `mean` computations are identical â€” just vectorized.
- No approximation, sampling, or rounding is introduced.
- The RF model is not retouched; only the feature-engineering pipeline is optimized.

---

## Working R Code

```r
library(data.table)
library(spdep)  # only needed if nb object needs conversion

# ============================================================
# Step 1: Convert nb object to edge-list data.table (one-time)
# ============================================================
nb_to_edge_list <- function(nb_obj, id_order) {
  # nb_obj:   spdep nb object (list of integer index vectors)
  # id_order: vector of cell IDs in the order matching nb_obj
  # Returns:  data.table with columns (focal_id, neighbor_id)
  
  n <- length(nb_obj)
  focal <- vector("list", n)
  neighbor <- vector("list", n)
  
  for (i in seq_len(n)) {
    nbs <- nb_obj[[i]]
    # spdep nb uses 0L to denote "no neighbors"
    nbs <- nbs[nbs > 0L]
    if (length(nbs) > 0L) {
      focal[[i]]    <- rep(id_order[i], length(nbs))
      neighbor[[i]] <- id_order[nbs]
    }
  }
  
  data.table(
    focal_id    = unlist(focal,    use.names = FALSE),
    neighbor_id = unlist(neighbor, use.names = FALSE)
  )
}

edges <- nb_to_edge_list(rook_neighbors_unique, id_order)
# ~1.37M rows, two integer columns â€” trivial memory

# ============================================================
# Step 2: Convert cell_data to data.table (if not already)
# ============================================================
if (!is.data.table(cell_data)) {
  cell_data <- as.data.table(cell_data)
}

# Ensure id and year are keyed for fast joins
setkey(cell_data, id, year)

# ============================================================
# Step 3: Vectorized neighbor stats computation
# ============================================================
compute_neighbor_stats_fast <- function(cell_dt, edges_dt, var_name) {
  # Build a slim lookup: (id, year, value)
  val_dt <- cell_dt[, .(id, year, val = get(var_name))]
  setkey(val_dt, id, year)
  
  # Get the unique years present in the data
  years <- sort(unique(val_dt$year))
  
  # Cross join edges with years to get all (focal, neighbor, year) triples
  # ~1.37M edges Ã— 28 years â‰ˆ 38.5M rows â€” fits in RAM easily
  edge_year <- CJ_dt(edges_dt, years)
  
  # Join neighbor values onto the edge-year table
  setkey(edge_year, neighbor_id, year)
  edge_year[val_dt, neighbor_val := i.val, on = .(neighbor_id = id, year)]
  
  # Aggregate: for each (focal_id, year), compute max, min, mean
  # Dropping NAs to match original behavior
  stats <- edge_year[
    !is.na(neighbor_val),
    .(
      nmax  = max(neighbor_val),
      nmin  = min(neighbor_val),
      nmean = mean(neighbor_val)
    ),
    keyby = .(focal_id, year)
  ]
  
  # Rename to match expected column names
  max_col  <- paste0("neighbor_max_",  var_name)
  min_col  <- paste0("neighbor_min_",  var_name)
  mean_col <- paste0("neighbor_mean_", var_name)
  setnames(stats, c("nmax", "nmin", "nmean"), c(max_col, min_col, mean_col))
  
  stats
}

# Helper: cross join edges with a year vector
CJ_dt <- function(edges_dt, years) {
  # Repeat each edge for every year â€” vectorized
  n_edges <- nrow(edges_dt)
  n_years <- length(years)
  
  data.table(
    focal_id    = rep(edges_dt$focal_id,    times = n_years),
    neighbor_id = rep(edges_dt$neighbor_id,  times = n_years),
    year        = rep(years, each = n_edges)
  )
}

# ============================================================
# Step 4: Outer loop â€” compute and merge all neighbor features
# ============================================================
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {
  cat("Computing neighbor stats for:", var_name, "\n")
  
  stats <- compute_neighbor_stats_fast(cell_data, edges, var_name)
  
  # Merge back into cell_data
  # Left join so cells with no neighbors get NA (matching original behavior)
  cell_data <- merge(
    cell_data, stats,
    by.x = c("id", "year"),
    by.y = c("focal_id", "year"),
    all.x = TRUE
  )
  
  rm(stats)
  gc()
}

# cell_data now has 15 new columns (3 per variable Ã— 5 variables),
# numerically identical to the original implementation.
# The trained Random Forest model is untouched.
```

---

## Memory-Optimized Variant (if 38.5M-row cross join is tight)

If laptop RAM is constrained (RF model + 6.46M-row dataset already consuming significant memory), process years in batches:

```r
compute_neighbor_stats_chunked <- function(cell_dt, edges_dt, var_name, 
                                            chunk_size = 7) {
  val_dt <- cell_dt[, .(id, year, val = get(var_name))]
  setkey(val_dt, id, year)
  years <- sort(unique(val_dt$year))
  
  # Process in chunks of years
  year_chunks <- split(years, ceiling(seq_along(years) / chunk_size))
  
  results <- lapply(year_chunks, function(yr_chunk) {
    # Subset values to this chunk of years
    val_sub <- val_dt[year %in% yr_chunk]
    setkey(val_sub, id, year)
    
    n_edges <- nrow(edges_dt)
    n_yr    <- length(yr_chunk)
    
    edge_year <- data.table(
      focal_id    = rep(edges_dt$focal_id,    times = n_yr),
      neighbor_id = rep(edges_dt$neighbor_id,  times = n_yr),
      year        = rep(yr_chunk, each = n_edges)
    )
    
    setkey(edge_year, neighbor_id, year)
    edge_year[val_sub, neighbor_val := i.val, on = .(neighbor_id = id, year)]
    
    edge_year[
      !is.na(neighbor_val),
      .(nmax = max(neighbor_val),
        nmin = min(neighbor_val),
        nmean = mean(neighbor_val)),
      keyby = .(focal_id, year)
    ]
  })
  
  stats <- rbindlist(results)
  
  max_col  <- paste0("neighbor_max_",  var_name)
  min_col  <- paste0("neighbor_min_",  var_name)
  mean_col <- paste0("neighbor_mean_", var_name)
  setnames(stats, c("nmax", "nmin", "nmean"), c(max_col, min_col, mean_col))
  
  stats
}
```

---

## Expected Performance

| Component | Original | Optimized |
|---|---|---|
| `build_neighbor_lookup` | ~70 hours (6.46M R iterations with string ops) | **~1 second** (nb â†’ edge list, 344K cells) |
| `compute_neighbor_stats` (Ã—5 vars) | ~16 hours (5 Ã— 6.46M R iterations) | **~3â€“8 minutes** (5 Ã— vectorized data.table join + groupby on 38.5M rows) |
| **Total** | **~86 hours** | **~5â€“10 minutes** |

**Speedup: ~500â€“1000Ã—**, achieved entirely through vectorization and separating the time-invariant topology from the panel time dimension. The numerical results are identical, and the trained Random Forest model is preserved unchanged.
END_RESPONSE>>>
