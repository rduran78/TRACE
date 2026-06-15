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
run_id: Anthropic_F10_T03_R024
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.3
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F10
prompt_family_slug: SPARSE_GRAPH_FRAME
family_label: Sparse graph frame
family_group: Graph cue
replicate: 24
raw_output_file: Experiment_Output/Anthropic/Temp_0.3/output_F10_Anthropic_temp_03_rep_024.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Optimized R Code

---

## Diagnosis

The current implementation has three critical bottlenecks:

1. **`build_neighbor_lookup` creates ~6.46 million R list entries**, each requiring string-pasting (`paste(id, year, sep="_")`), named-vector lookups (`idx_lookup[neighbor_keys]`), and NA filtering. This is an O(NÃ—K) operation done entirely in interpreted R with per-element string allocation â€” catastrophically slow at this scale.

2. **`compute_neighbor_stats` iterates over 6.46M list elements** in an `lapply`, extracting subsets of a numeric vector and computing `max/min/mean` per element. While each operation is trivial, the R-level loop overhead across 6.46M iterations Ã— 5 variables = ~32.3M iterations is enormous.

3. **The topology is year-invariant but the lookup is rebuilt as if it's year-specific.** The rook neighbor structure is purely spatial â€” cell *i*'s neighbors are the same in every year. Yet `build_neighbor_lookup` embeds year into every key, inflating the lookup from ~344K spatial entries to ~6.46M space-time entries. This is a 28Ã— unnecessary expansion.

**Root cause summary:** The implementation treats a *separable* problem (spatial topology Ã— temporal panel) as a monolithic space-time graph, then solves it with interpreted R loops and string-keyed lookups.

---

## Optimization Strategy

### Key Insight: Separate Spatial Topology from Temporal Panel

The rook adjacency is **time-invariant**. We should:

1. **Build a sparse adjacency matrix once** from the `nb` object (~344K Ã— 344K, ~1.37M nonzeros). This is a standard CSC/CSR sparse matrix â€” trivial memory.

2. **Reshape each variable into a cell Ã— year matrix** (344,208 Ã— 28). This enables vectorized column-wise operations.

3. **Compute neighbor aggregates via sparse matrixâ€“dense matrix multiplication** and analogous operations:
   - **Mean:** `(A %*% X) / (A %*% 1_valid)` where `1_valid` masks non-NA entries.
   - **Max/Min:** Use row-wise sparse iteration, but do it in a compiled/vectorized manner.

4. For **max and min**, there is no direct sparse-matrix shortcut (they aren't linear), but we can use the `Matrix` package's sparse structure to iterate efficiently in C-level code, or use `data.table` group-by operations on the edge list.

### Chosen Approach: Edge-List + `data.table` Aggregation

The most robust and efficient pure-R approach:

- Convert the `nb` object to an edge list (source, target) â€” ~1.37M rows, built once.
- Join with the panel data (which is indexed by cell and year) using `data.table` keyed joins.
- Group-by `(source_cell, year)` and compute `max`, `min`, `mean` â€” fully vectorized in `data.table`'s C backend.
- This replaces 6.46M R-level list iterations with a single grouped aggregation over ~1.37M Ã— 28 â‰ˆ 38.4M rows per variable, executed in compiled C code.

**Expected speedup:** From 86+ hours to **minutes** (roughly 1000â€“5000Ã—).

**Numerical equivalence:** Guaranteed â€” same `max`, `min`, `mean` over the identical neighbor sets.

---

## Optimized R Code

```r
# =============================================================================
# OPTIMIZED SPATIAL NEIGHBOR FEATURE PIPELINE
# =============================================================================
# Prerequisites:
#   - cell_data: data.frame/data.table with columns: id, year, ntl, ec, 
#                pop_density, def, usd_est_n2, (and other predictors)
#   - id_order: vector of cell IDs in the order matching rook_neighbors_unique
#   - rook_neighbors_unique: spdep nb object (list of integer neighbor indices)
#   - rf_model: pre-trained Random Forest model (not retrained)
# =============================================================================

library(data.table)

# ---- Step 1: Build the spatial edge list ONCE --------------------------------
# Convert the nb object to a two-column edge list: (source_cell_id, target_cell_id)
# This encodes "target is a rook neighbor of source"

build_edge_list <- function(id_order, nb_obj) {
  # nb_obj[[i]] contains integer indices (into id_order) of neighbors of cell i
  # We need: for each cell i, edges (id_order[i], id_order[j]) for j in nb_obj[[i]]
  
  n <- length(nb_obj)
  
  # Pre-count total edges for pre-allocation
  edge_counts <- vapply(nb_obj, function(x) {
    if (length(x) == 1L && x[1] == 0L) 0L else length(x)
  }, integer(1))
  total_edges <- sum(edge_counts)
  
  source_idx <- integer(total_edges)
  target_idx <- integer(total_edges)
  
  pos <- 1L
  for (i in seq_len(n)) {
    nbrs <- nb_obj[[i]]
    if (length(nbrs) == 1L && nbrs[1] == 0L) next
    k <- length(nbrs)
    source_idx[pos:(pos + k - 1L)] <- i
    target_idx[pos:(pos + k - 1L)] <- nbrs
    pos <- pos + k
  }
  
  data.table(
    source_id = id_order[source_idx],
    target_id = id_order[target_idx]
  )
}

cat("Building spatial edge list...\n")
edge_list <- build_edge_list(id_order, rook_neighbors_unique)
cat(sprintf("  Edge list: %d directed edges\n", nrow(edge_list)))

# ---- Step 2: Convert cell_data to data.table and set keys -------------------

if (!is.data.table(cell_data)) {
  cell_data <- as.data.table(cell_data)
}

# Ensure id and year are keyed for fast joins
setkey(cell_data, id, year)

# ---- Step 3: Compute neighbor features for all variables ---------------------

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

compute_all_neighbor_features <- function(cell_data, edge_list, var_names) {
  
  # Create a slim lookup table: (id, year, var1, var2, ..., var5)
  # This is the "target node attribute" table
  lookup_cols <- c("id", "year", var_names)
  target_attrs <- cell_data[, ..lookup_cols]
  setnames(target_attrs, "id", "target_id")
  setkey(target_attrs, target_id, year)
  
  # Get all unique years
  all_years <- sort(unique(cell_data$year))
  
  # Cross join edge_list Ã— years to get all (source, target, year) triples
  # ~1.37M edges Ã— 28 years â‰ˆ 38.4M rows â€” fits easily in memory
  # (38.4M rows Ã— ~7 columns Ã— 8 bytes â‰ˆ ~2.1 GB, manageable on 16GB)
  
  cat("Expanding edge list across years...\n")
  years_dt <- data.table(year = all_years)
  edge_year <- edge_list[, CJ_dt := TRUE]  # placeholder
  edge_year <- CJ(edge_idx = seq_len(nrow(edge_list)), year = all_years)
  edge_year[, source_id := edge_list$source_id[edge_idx]]
  edge_year[, target_id := edge_list$target_id[edge_idx]]
  edge_year[, edge_idx := NULL]
  
  cat(sprintf("  Expanded edge-year table: %d rows\n", nrow(edge_year)))
  
  # Join target attributes onto edge-year table
  cat("Joining target node attributes...\n")
  setkey(edge_year, target_id, year)
  edge_year <- target_attrs[edge_year, on = .(target_id, year)]
  
  # Now group by (source_id, year) and compute max, min, mean for each variable
  cat("Computing neighbor aggregations...\n")
  
  # Build aggregation expressions dynamically
  agg_exprs <- unlist(lapply(var_names, function(v) {
    list(
      bquote(as.numeric(max(.(as.name(v)), na.rm = TRUE))),
      bquote(as.numeric(min(.(as.name(v)), na.rm = TRUE))),
      bquote(as.numeric(mean(.(as.name(v)), na.rm = TRUE)))
    )
  }))
  
  agg_names <- unlist(lapply(var_names, function(v) {
    paste0("n_", v, c("_max", "_min", "_mean"))
  }))
  
  # Replace Inf/-Inf with NA (from max/min of all-NA groups)
  # We'll handle this after aggregation
  
  setkey(edge_year, source_id, year)
  
  agg_result <- edge_year[, 
    setNames(lapply(var_names, function(v) {
      vals <- get(v)
      vals <- vals[!is.na(vals)]
      if (length(vals) == 0L) {
        list(NA_real_, NA_real_, NA_real_)
      } else {
        list(max(vals), min(vals), mean(vals))
      }
    }), var_names),
    by = .(source_id, year)
  ]
  
  # The above nested-list approach is tricky in data.table. 
  # Let's use a cleaner approach:
  
  # Actually, let's compute each variable's stats in a straightforward way.
  # data.table is extremely fast at grouped aggregation.
  
  # Re-do with explicit expressions:
  agg_list <- vector("list", length(var_names))
  
  for (vi in seq_along(var_names)) {
    v <- var_names[vi]
    cat(sprintf("  Aggregating: %s\n", v))
    
    # Subset to non-NA values of this variable for efficiency
    sub_dt <- edge_year[!is.na(get(v)), .(source_id, year, val = get(v))]
    
    if (nrow(sub_dt) == 0) {
      # All NA â€” create empty result
      agg_v <- unique(edge_year[, .(source_id, year)])
      agg_v[, c(paste0("n_", v, "_max"), paste0("n_", v, "_min"), 
                 paste0("n_", v, "_mean")) := .(NA_real_, NA_real_, NA_real_)]
    } else {
      agg_v <- sub_dt[, .(
        v_max  = max(val),
        v_min  = min(val),
        v_mean = mean(val)
      ), by = .(source_id, year)]
      
      setnames(agg_v, c("v_max", "v_min", "v_mean"),
               paste0("n_", v, c("_max", "_min", "_mean")))
    }
    
    agg_list[[vi]] <- agg_v
  }
  
  # Merge all aggregation results together
  cat("Merging aggregation results...\n")
  merged <- agg_list[[1]]
  setkey(merged, source_id, year)
  for (vi in 2:length(agg_list)) {
    setkey(agg_list[[vi]], source_id, year)
    merged <- merge(merged, agg_list[[vi]], by = c("source_id", "year"), all = TRUE)
  }
  
  return(merged)
}

neighbor_features <- compute_all_neighbor_features(cell_data, edge_list, neighbor_source_vars)

# ---- Step 4: Join neighbor features back to cell_data ------------------------

cat("Joining neighbor features to cell_data...\n")
setnames(neighbor_features, "source_id", "id")
setkey(neighbor_features, id, year)
setkey(cell_data, id, year)

# Remove any pre-existing neighbor columns to avoid duplication
n_cols <- grep("^n_.*_(max|min|mean)$", names(cell_data), value = TRUE)
if (length(n_cols) > 0) {
  cell_data[, (n_cols) := NULL]
}

cell_data <- merge(cell_data, neighbor_features, by = c("id", "year"), all.x = TRUE)

cat("Done. Neighbor features added.\n")
cat(sprintf("  cell_data dimensions: %d rows Ã— %d columns\n", nrow(cell_data), ncol(cell_data)))

# ---- Step 5: Apply the pre-trained Random Forest model -----------------------
# The RF model is NOT retrained. We only call predict().

cat("Generating predictions with pre-trained Random Forest...\n")
cell_data[, rf_prediction := predict(rf_model, newdata = cell_data)]
cat("Pipeline complete.\n")
```

---

## Memory-Optimized Variant (If 38.4M-Row Edge Table Strains RAM)

If the full cross of edges Ã— years (~38.4M rows with joined attributes) approaches memory limits, process year-by-year:

```r
# ---- Memory-conservative variant: process one year at a time -----------------

compute_neighbor_features_by_year <- function(cell_data, edge_list, var_names) {
  
  all_years <- sort(unique(cell_data$year))
  results <- vector("list", length(all_years))
  
  for (yi in seq_along(all_years)) {
    yr <- all_years[yi]
    cat(sprintf("  Year %d (%d/%d)\n", yr, yi, length(all_years)))
    
    # Subset to this year
    yr_data <- cell_data[year == yr, c("id", var_names), with = FALSE]
    setnames(yr_data, "id", "target_id")
    setkey(yr_data, target_id)
    
    # Join target attributes onto edge list
    edge_yr <- merge(edge_list, yr_data, by = "target_id", all.x = FALSE)
    
    # Aggregate per source
    agg_parts <- vector("list", length(var_names))
    for (vi in seq_along(var_names)) {
      v <- var_names[vi]
      sub <- edge_yr[!is.na(get(v)), .(source_id, val = get(v))]
      if (nrow(sub) > 0) {
        agg_v <- sub[, .(
          v_max  = max(val),
          v_min  = min(val),
          v_mean = mean(val)
        ), by = .(source_id)]
      } else {
        agg_v <- data.table(source_id = integer(0), 
                            v_max = numeric(0), v_min = numeric(0), v_mean = numeric(0))
      }
      setnames(agg_v, c("v_max", "v_min", "v_mean"),
               paste0("n_", v, c("_max", "_min", "_mean")))
      agg_parts[[vi]] <- agg_v
    }
    
    merged <- agg_parts[[1]]
    setkey(merged, source_id)
    for (vi in 2:length(agg_parts)) {
      setkey(agg_parts[[vi]], source_id)
      merged <- merge(merged, agg_parts[[vi]], by = "source_id", all = TRUE)
    }
    merged[, year := yr]
    results[[yi]] <- merged
  }
  
  rbindlist(results, use.names = TRUE, fill = TRUE)
}

cat("Computing neighbor features (year-by-year, memory-safe)...\n")
neighbor_features <- compute_neighbor_features_by_year(
  cell_data, edge_list, neighbor_source_vars
)

# Then join and predict as in Steps 4-5 above.
```

---

## Summary

| Aspect | Original | Optimized |
|---|---|---|
| **Topology representation** | String-keyed named vector (6.46M entries) | Sparse edge list (1.37M rows, built once) |
| **Aggregation engine** | R-level `lapply` over 6.46M elements Ã— 5 vars | `data.table` grouped C-level aggregation |
| **Year handling** | Embedded in keys, 28Ã— redundant topology | Factored out; topology reused per year |
| **Estimated runtime** | 86+ hours | ~5â€“15 minutes |
| **Peak memory** | High (large list-of-lists + string keys) | ~2â€“3 GB (edge-year table) or ~500 MB (year-by-year) |
| **Numerical equivalence** | Baseline | Identical `max`, `min`, `mean` over same neighbor sets |
| **RF model** | Preserved (not retrained) | Preserved (not retrained) |
END_RESPONSE>>>
