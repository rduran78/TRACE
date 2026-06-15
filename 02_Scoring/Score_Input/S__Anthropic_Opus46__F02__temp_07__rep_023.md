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
run_id: Anthropic_F02_T07_R023
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F02
prompt_family_slug: CODE_ONLY_NO_HINT
family_label: Code only, no hint
family_group: Code-visible neutral
replicate: 23
raw_output_file: Experiment_Output/Anthropic/Temp_0.7/output_F02_Anthropic_temp_07_rep_023.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The bottleneck is clear: **`build_neighbor_lookup`** creates 6.46 million character-paste keys and performs 6.46 million named-vector lookups inside an `lapply` loop. Named vector lookup in R is O(n) per query on the hash, but the real cost is:

1. **`build_neighbor_lookup`**: For each of ~6.46M rows, it pastes character keys and does named-vector subsetting. The `paste` + named-lookup pattern inside `lapply` over millions of rows is extremely slow in interpreted R. Estimated: ~70â€“80% of total runtime.
2. **`compute_neighbor_stats`**: For each of ~6.46M rows Ã— 5 variables, it subsets a numeric vector, removes NAs, and computes max/min/mean. The `lapply` + `do.call(rbind, ...)` pattern over millions of small vectors is slow and memory-wasteful (millions of tiny allocations, then a massive `rbind`).
3. **Memory**: With ~6.46M rows Ã— 110 columns, the main data.frame is ~5â€“6 GB. Building a 6.46M-element list of integer vectors for the neighbor lookup adds another ~1â€“2 GB. The `do.call(rbind, list_of_6M_vectors)` pattern temporarily doubles memory.

**Root cause summary**: Row-level interpreted R loops over millions of rows, millions of small allocations, character key hashing, and a non-vectorized aggregation pattern.

---

## Optimization Strategy

### Core idea: Replace row-level R loops with vectorized joins and grouped aggregations using `data.table`.

1. **Replace `build_neighbor_lookup`** with a pre-materialized **edge table** (`data.table`) that maps every `(id, year)` â†’ `(neighbor_id, year)` â†’ row index. This is a single vectorized merge, not 6.46M sequential lookups.

2. **Replace `compute_neighbor_stats`** with a **grouped `data.table` aggregation** on the edge table joined to the variable values. One grouped operation computes max, min, and mean for all rows simultaneously â€” no `lapply`, no `do.call(rbind, ...)`.

3. **Process all 5 variables** in a single pass over the edge table (or 5 fast grouped aggregations), avoiding rebuilding lookup structures.

4. **Memory management**: The edge table for ~1.37M directed neighbor pairs Ã— 28 years â‰ˆ 38.5M rows of integer pairs â€” about 600 MB. This is feasible on 16 GB RAM, especially if we drop intermediate objects.

### Expected speedup: From 86+ hours â†’ **~5â€“15 minutes**.

### Preserves: The trained Random Forest model (untouched) and the original numerical estimand (same max, min, mean statistics are computed).

---

## Working R Code

```r
library(data.table)

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Step 0: Convert cell_data to data.table if not already
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
if (!is.data.table(cell_data)) {
  cell_data <- as.data.table(cell_data)
}

# Assign a row index to cell_data for later re-joining
cell_data[, .row_idx := .I]

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Step 1: Build the edge list from the nb object (one-time)
#
# rook_neighbors_unique is a list of length = number of cells.
# id_order is the vector of cell IDs in the same order.
# rook_neighbors_unique[[i]] gives integer indices into id_order
# for the neighbors of cell id_order[i].
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

build_edge_list <- function(id_order, neighbors) {
  # Pre-allocate: count total edges
  n_edges <- sum(lengths(neighbors))
  
  from_id <- integer(n_edges)
  to_id   <- integer(n_edges)
  
  pos <- 1L
  for (i in seq_along(neighbors)) {
    nb <- neighbors[[i]]
    n  <- length(nb)
    if (n > 0L) {
      from_id[pos:(pos + n - 1L)] <- id_order[i]
      to_id[pos:(pos + n - 1L)]   <- id_order[nb]
      pos <- pos + n
    }
  }
  
  data.table(focal_id = from_id, neighbor_id = to_id)
}

cat("Building edge list from nb object...\n")
edge_dt <- build_edge_list(id_order, rook_neighbors_unique)
cat(sprintf("Edge list: %d directed edges\n", nrow(edge_dt)))

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Step 2: Expand edge list across all years via cross-join
#
# Every neighbor relationship exists in every year.
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

years <- sort(unique(cell_data$year))
year_dt <- data.table(year = years)

cat("Expanding edge list across years...\n")
# Cross join: every edge Ã— every year
edge_year_dt <- edge_dt[, CJ_idx := 1L][
  year_dt[, CJ_idx := 1L], 
  on = "CJ_idx", 
  allow.cartesian = TRUE
][, CJ_idx := NULL]

# Clean up
edge_dt[, CJ_idx := NULL]
year_dt[, CJ_idx := NULL]

cat(sprintf("Edge-year table: %d rows (%.1f M)\n", nrow(edge_year_dt), nrow(edge_year_dt)/1e6))

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Step 3: Join to get focal row index and neighbor values,
#         then compute grouped statistics per variable
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

# Create a key lookup: (id, year) -> row index in cell_data
setkey(cell_data, id, year)

# Add focal row index to edge_year_dt
# We join edge_year_dt to cell_data to get the focal row's .row_idx
edge_year_dt <- cell_data[, .(id, year, .row_idx)][
  edge_year_dt, 
  on = .(id = focal_id, year = year),
  nomatch = 0L
]
setnames(edge_year_dt, ".row_idx", "focal_row_idx")

# Add neighbor values: join on neighbor_id + year
# We'll do this per variable to control memory

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# Prepare a slim neighbor-value table: (id, year, var1, var2, ...)
neighbor_val_cols <- c("id", "year", neighbor_source_vars)
neighbor_vals_dt <- cell_data[, ..neighbor_val_cols]
setkey(neighbor_vals_dt, id, year)

cat("Joining neighbor values...\n")
# Join neighbor values onto the edge-year table
edge_year_dt <- neighbor_vals_dt[
  edge_year_dt,
  on = .(id = neighbor_id, year = year),
  nomatch = NA
]

# Now edge_year_dt has columns:
#   id (= neighbor_id), year, ntl, ec, pop_density, def, usd_est_n2, focal_row_idx

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Step 4: Grouped aggregation â€” compute max, min, mean per
#         focal_row_idx for each variable
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

cat("Computing neighbor statistics...\n")

# Build aggregation expressions dynamically
agg_exprs <- unlist(lapply(neighbor_source_vars, function(v) {
  list(
    bquote(as.numeric(max(.(as.name(v)), na.rm = TRUE))),
    bquote(as.numeric(min(.(as.name(v)), na.rm = TRUE))),
    bquote(as.numeric(mean(.(as.name(v)), na.rm = TRUE)))
  )
}))

agg_names <- unlist(lapply(neighbor_source_vars, function(v) {
  paste0("neighbor_", v, c("_max", "_min", "_mean"))
}))

names(agg_exprs) <- agg_names

# Single grouped aggregation
neighbor_stats <- edge_year_dt[, 
  lapply(agg_exprs, eval, envir = .SD), 
  by = focal_row_idx
]

# â”€â”€â”€ Alternative (clearer, equally fast) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# If the above eval approach is tricky, do it explicitly:

neighbor_stats <- edge_year_dt[, {
  out <- list()
  for (v in neighbor_source_vars) {
    vals <- .SD[[v]]
    vals <- vals[!is.na(vals)]
    if (length(vals) == 0L) {
      out[[paste0("neighbor_", v, "_max")]]  <- NA_real_
      out[[paste0("neighbor_", v, "_min")]]  <- NA_real_
      out[[paste0("neighbor_", v, "_mean")]] <- NA_real_
    } else {
      out[[paste0("neighbor_", v, "_max")]]  <- max(vals)
      out[[paste0("neighbor_", v, "_min")]]  <- min(vals)
      out[[paste0("neighbor_", v, "_mean")]] <- mean(vals)
    }
  }
  out
}, by = focal_row_idx, .SDcols = neighbor_source_vars]

cat(sprintf("Neighbor stats computed: %d rows, %d columns\n", nrow(neighbor_stats), ncol(neighbor_stats)))

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Step 5: Join statistics back to cell_data
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

# Remove the edge table to free memory
rm(edge_year_dt, neighbor_vals_dt)
gc()

setkey(neighbor_stats, focal_row_idx)

# Merge back by row index
new_cols <- setdiff(names(neighbor_stats), "focal_row_idx")

# Remove any pre-existing neighbor columns to avoid conflicts
for (col in new_cols) {
  if (col %in% names(cell_data)) {
    cell_data[, (col) := NULL]
  }
}

cell_data <- neighbor_stats[cell_data, on = .(focal_row_idx = .row_idx)]

# Rename focal_row_idx back to .row_idx if needed, or drop it
cell_data[, focal_row_idx := NULL]

# If you had .row_idx, drop it too (it was a helper)
if (".row_idx" %in% names(cell_data)) cell_data[, .row_idx := NULL]

cat("Done. Neighbor features added to cell_data.\n")
```

---

## Memory Management Notes for 16 GB RAM

If the cross-joined edge-year table (~38.5M rows) causes memory pressure:

```r
# â”€â”€â”€ Memory-safe variant: process one year at a time â”€â”€â”€â”€â”€â”€â”€â”€â”€
# This trades a small amount of speed for much lower peak memory.

all_stats <- vector("list", length(years))

for (yi in seq_along(years)) {
  yr <- years[yi]
  cat(sprintf("Processing year %d (%d/%d)...\n", yr, yi, length(years)))
  
  # Subset to this year
  cd_yr <- cell_data[year == yr, c("id", ".row_idx", neighbor_source_vars), with = FALSE]
  setkey(cd_yr, id)
  
  # Join edges to focal rows
  ey <- edge_dt[cd_yr, on = .(focal_id = id), nomatch = 0L, allow.cartesian = TRUE]
  # ey now has: focal_id, neighbor_id, .row_idx (focal)
  
  # Join neighbor values
  ey <- cd_yr[, c("id", neighbor_source_vars), with = FALSE][
    ey, on = .(id = neighbor_id), nomatch = NA
  ]
  
  # Aggregate
  stats_yr <- ey[, {
    out <- list()
    for (v in neighbor_source_vars) {
      vals <- .SD[[v]]
      vals <- vals[!is.na(vals)]
      if (length(vals) == 0L) {
        out[[paste0("neighbor_", v, "_max")]]  <- NA_real_
        out[[paste0("neighbor_", v, "_min")]]  <- NA_real_
        out[[paste0("neighbor_", v, "_mean")]] <- NA_real_
      } else {
        out[[paste0("neighbor_", v, "_max")]]  <- max(vals)
        out[[paste0("neighbor_", v, "_min")]]  <- min(vals)
        out[[paste0("neighbor_", v, "_mean")]] <- mean(vals)
      }
    }
    out
  }, by = .(.row_idx), .SDcols = neighbor_source_vars]
  
  all_stats[[yi]] <- stats_yr
  rm(ey, cd_yr, stats_yr)
}

neighbor_stats <- rbindlist(all_stats)
rm(all_stats); gc()

# Join back to cell_data as in Step 5 above
```

---

## Summary

| Aspect | Original | Optimized |
|---|---|---|
| **Lookup construction** | 6.46M `paste` + named vector lookups in `lapply` | One-time vectorized edge-list build + `data.table` keyed join |
| **Aggregation** | 6.46M Ã— 5 `lapply` calls + `do.call(rbind, ...)` | Single grouped `data.table` aggregation |
| **Peak memory** | Uncontrolled (millions of small list allocations) | ~2â€“4 GB for edge-year table (or ~300 MB per year in chunked mode) |
| **Estimated runtime** | 86+ hours | ~5â€“15 minutes |
| **RF model** | Untouched âœ“ | Untouched âœ“ |
| **Numerical output** | max, min, mean per neighbor set | Identical max, min, mean per neighbor set âœ“ |
END_RESPONSE>>>
