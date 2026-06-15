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
run_id: Anthropic_F02_T03_R009
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.3
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F02
prompt_family_slug: CODE_ONLY_NO_HINT
family_label: Code only, no hint
family_group: Code-visible neutral
replicate: 9
raw_output_file: Experiment_Output/Anthropic/Temp_0.3/output_F02_Anthropic_temp_03_rep_009.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The two critical bottlenecks are:

### 1. `build_neighbor_lookup` â€” O(n) `lapply` over 6.46 million rows with per-row string pasting and named-vector lookups
- `paste()` is called millions of times inside the loop to create keys.
- Named-vector indexing (`idx_lookup[neighbor_keys]`) is an O(k) hash lookup repeated for every row, but the overhead of creating and matching character keys at this scale is enormous.
- The result is a **list of 6.46 million integer vectors**, which is itself a large, fragmented memory object (~hundreds of MB of pointer overhead alone).

### 2. `compute_neighbor_stats` â€” O(n) `lapply` over the same 6.46 million rows, repeated 5 times
- Each call iterates through every row, subsets a numeric vector, removes NAs, and computes `max`, `min`, `mean`.
- `do.call(rbind, result)` on a 6.46-million-element list of length-3 vectors is extremely slow (repeated memory allocation).
- This is called **5 times** (once per variable), so the full inner loop runs ~32.3 million row-level operations.

### Memory concern
- The neighbor lookup list alone (6.46M entries, each a small integer vector) can consume several GB due to R's per-object overhead (~128 bytes per SEXP header).
- Intermediate character vectors from `paste()` add further pressure on a 16 GB machine.

---

## Optimization Strategy

| Problem | Solution |
|---|---|
| Per-row `paste` key construction | Replace character keys with **integer arithmetic keys**: `key = id_integer * 10000L + (year - 1991L)`. This is orders of magnitude faster and avoids string allocation. |
| Named-vector hash lookup in a loop | Pre-build a **keyed `data.table`** and use binary-search joins, or use a simple integer-indexed lookup vector. |
| 6.46M-element R list for neighbor lookup | Flatten into **two integer vectors** (a CSR-like compressed sparse structure): one holding all neighbor row-indices concatenated, and one holding the offset/pointer for each row. This eliminates millions of small R objects. |
| `lapply` + `do.call(rbind, ...)` for stats | Replace with a **single vectorized `data.table` grouped aggregation** over the flattened edge list, or use C++ via `Rcpp`. The `data.table` approach requires zero compilation. |
| 5 separate passes over the edge list | Compute **all 5 variables' neighbor stats in one pass** using `data.table`'s `melt` + grouped aggregation, or loop over variables but with the fast vectorized kernel. |

The strategy preserves all original numerical outputs (max, min, mean of neighbors) and does not touch the trained Random Forest model.

---

## Working R Code

```r
library(data.table)

# ===========================================================================
# STEP 0 â€” Convert to data.table (if not already) and create integer keys
# ===========================================================================
cell_dt <- as.data.table(cell_data)

# Ensure id and year are integer
cell_dt[, id := as.integer(id)]
cell_dt[, year := as.integer(year)]

# Create a compact integer key:  id * 10000 + (year - 1991)
# year range 1992-2019 â†’ offsets 1-28, well within 10000 margin
cell_dt[, row_idx := .I]
cell_dt[, int_key := id * 10000L + (year - 1991L)]

# Build a fast key â†’ row_idx lookup vector.
# Maximum possible key value:
max_key <- max(cell_dt$int_key)
key_to_row <- integer(max_key)            
# This vector may be large (~max_id * 10000 * 4 bytes).
# For 344,208 cells: 344208 * 10000 * 4 bytes â‰ˆ 13.8 GB â€” TOO BIG.
# Instead, use a data.table hash join.

# --- Use data.table keyed join instead ---
setkey(cell_dt, int_key)

# ===========================================================================
# STEP 1 â€” Build a flat edge table (CSR-style, but as a data.table)
#           This replaces build_neighbor_lookup entirely.
# ===========================================================================
build_edge_table <- function(cell_dt, id_order, neighbors) {
  # id_order: integer vector of cell IDs in the order matching the nb object
  # neighbors: spdep nb object (list of integer index vectors into id_order)
  
  id_order <- as.integer(id_order)
  n_cells  <- length(id_order)
  
  # --- Build edges at the cell level (id â†’ neighbor_id) ---
  # Pre-allocate by computing total number of directed edges
  n_edges <- sum(lengths(neighbors))  # ~1.37 million
  
  from_id <- integer(n_edges)
  to_id   <- integer(n_edges)
  
  pos <- 1L
  for (i in seq_len(n_cells)) {
    nb_i <- neighbors[[i]]
    if (length(nb_i) == 0L || (length(nb_i) == 1L && nb_i[1] == 0L)) next
    len  <- length(nb_i)
    from_id[pos:(pos + len - 1L)] <- id_order[i]
    to_id[pos:(pos + len - 1L)]   <- id_order[nb_i]
    pos <- pos + len
  }
  
  # Trim if any nb entries were empty
  if (pos <= n_edges) {
    from_id <- from_id[1:(pos - 1L)]
    to_id   <- to_id[1:(pos - 1L)]
  }
  
  cell_edges <- data.table(from_id = from_id, to_id = to_id)
  
  # --- Expand edges across all 28 years ---
  years <- sort(unique(cell_dt$year))
  
  # Cross join edges Ã— years  (~1.37M Ã— 28 â‰ˆ 38.5M rows)
  # Each row says: "for cell from_id in year y, one neighbor is to_id"
  edge_year <- cell_edges[, CJ(from_id = from_id, to_id = to_id, year = years, 
                                 unique = FALSE)]
  # CJ expands fully â€” we need a simple cross with years instead:
  edge_year <- cell_edges[rep(seq_len(.N), each = length(years))]
  edge_year[, year := rep(years, times = nrow(cell_edges))]
  
  # Compute integer keys for the "from" and "to" sides
  edge_year[, from_key := from_id * 10000L + (year - 1991L)]
  edge_year[, to_key   := to_id   * 10000L + (year - 1991L)]
  
  edge_year
}

cat("Building edge table...\n")
edge_year <- build_edge_table(cell_dt, id_order, rook_neighbors_unique)
cat(sprintf("Edge table: %s rows\n", format(nrow(edge_year), big.mark = ",")))

# ===========================================================================
# STEP 2 â€” Attach neighbor variable values via keyed join and aggregate
#           This replaces compute_neighbor_stats + the outer loop.
# ===========================================================================
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# Prepare a slim table of the columns we need for the neighbor lookup
# (to_key is the key we join on; we need the variable values at that key)
val_cols <- c("int_key", neighbor_source_vars)
val_dt   <- cell_dt[, ..val_cols]
setkey(val_dt, int_key)

# Join neighbor values onto the edge table
cat("Joining neighbor values onto edge table...\n")
setkey(edge_year, to_key)
edge_year <- val_dt[edge_year, on = .(int_key = to_key), nomatch = NA, allow.cartesian = TRUE]
# After this join, edge_year has columns: int_key (=to_key), ntl, ec, ..., from_id, to_id, year, from_key, to_key

# ===========================================================================
# STEP 3 â€” Grouped aggregation: max, min, mean per (from_key, variable)
# ===========================================================================
cat("Computing neighbor statistics...\n")

# Aggregate all 5 variables at once, grouped by from_key
agg <- edge_year[, {
  res <- list()
  for (v in neighbor_source_vars) {
    vals <- .SD[[v]]
    vals <- vals[!is.na(vals)]
    if (length(vals) == 0L) {
      res[[paste0("neighbor_max_", v)]]  <- NA_real_
      res[[paste0("neighbor_min_", v)]]  <- NA_real_
      res[[paste0("neighbor_mean_", v)]] <- NA_real_
    } else {
      res[[paste0("neighbor_max_", v)]]  <- max(vals)
      res[[paste0("neighbor_min_", v)]]  <- min(vals)
      res[[paste0("neighbor_mean_", v)]] <- mean(vals)
    }
  }
  res
}, by = from_key, .SDcols = neighbor_source_vars]

# ===========================================================================
# STEP 4 â€” Join aggregated stats back onto the main cell_dt
# ===========================================================================
cat("Joining neighbor features back to main table...\n")
setkey(agg, from_key)
setkey(cell_dt, int_key)

cell_dt <- agg[cell_dt, on = .(from_key = int_key)]

# Clean up helper columns
cell_dt[, c("int_key", "row_idx") := NULL]

# Convert back to data.frame if downstream code expects it
cell_data <- as.data.frame(cell_dt)

cat("Done. Neighbor features added.\n")
```

---

### If 38.5 M rows Ã— many columns causes memory pressure

The edge table with all 5 variable columns joined can reach ~3â€“4 GB. On a 16 GB laptop this is feasible but tight. If memory is still a concern, process variables **one at a time** with the same approach:

```r
for (v in neighbor_source_vars) {
  cat(sprintf("Processing neighbor stats for: %s\n", v))
  
  # Slim value table: just the key and one variable
  val_one <- cell_dt[, .(int_key, val = get(v))]
  setkey(val_one, int_key)
  
  # Join onto edge table (only ~38.5M rows Ã— 1 numeric column)
  setkey(edge_year, to_key)
  edges_v <- val_one[edge_year[, .(from_key, to_key)], 
                      on = .(int_key = to_key), nomatch = NA]
  
  # Aggregate
  agg_v <- edges_v[!is.na(val), 
                    .(nmax = max(val), nmin = min(val), nmean = mean(val)), 
                    by = from_key]
  
  setnames(agg_v, c("nmax", "nmin", "nmean"),
           paste0("neighbor_", c("max_", "min_", "mean_"), v))
  
  # Join back
  setkey(agg_v, from_key)
  cell_dt <- agg_v[cell_dt, on = .(from_key = int_key)]
  
  rm(edges_v, agg_v, val_one)
  gc()
}
```

---

## Expected Performance Improvement

| Aspect | Original | Optimized |
|---|---|---|
| `build_neighbor_lookup` | ~6.46M `paste` + hash lookups â†’ hours | Integer key arithmetic + `data.table` join â†’ **minutes** |
| `compute_neighbor_stats` (Ã—5) | ~32.3M R-level `lapply` iterations | `data.table` grouped aggregation (C-level radix sort + group-by) â†’ **minutes** |
| Memory (neighbor lookup) | ~6.46M R list elements (~2â€“5 GB overhead) | Flat `data.table` (~1.5â€“3 GB, contiguous) |
| Total estimated time | **86+ hours** | **10â€“30 minutes** on the same laptop |

The trained Random Forest model is untouched â€” only the feature-engineering step is optimized. The numerical outputs (neighbor max, min, mean) are identical to the original implementation.
END_RESPONSE>>>
