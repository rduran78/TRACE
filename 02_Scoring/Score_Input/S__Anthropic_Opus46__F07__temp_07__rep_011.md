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
run_id: Anthropic_F07_T07_R011
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F07
prompt_family_slug: RASTER_INVALID_IRREGULAR
family_label: Raster invalid irregular topology
family_group: Raster/kernel bridge
replicate: 11
raw_output_file: Experiment_Output/Anthropic/Temp_0.7/output_F07_Anthropic_temp_07_rep_011.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

## Diagnosis

The bottleneck is **`build_neighbor_lookup`**. It creates a list of ~6.46 million elements, where each element is built by:

1. **Per-row string pasting and named-vector lookup** (`paste(...)`, `idx_lookup[neighbor_keys]`) â€” this is called 6.46 million times inside `lapply`. Each call does character concatenation and hash-table lookups.
2. The `idx_lookup` named vector has ~6.46 million entries. Named-vector lookup in R is O(n) per query in the worst case (hashed, but still slow at scale with millions of keys and millions of lookups).
3. The `neighbor_lookup` list itself will hold ~6.46M elements Ã— ~4 neighbors on average â‰ˆ ~26M integer references, plus R list overhead. That's manageable in memory, but *constructing* it row-by-row in an interpreted loop is the killer.

**`compute_neighbor_stats`** is also slow: it loops over 6.46M list elements in R, extracting and summarizing small vectors. This is repeated 5 times (once per variable).

**Summary:** ~86+ hours is almost entirely spent in interpreted R loops doing millions of string operations and hash lookups.

---

## Optimization Strategy

### Key Insight: Vectorize via a merge/join on an edge table

Instead of building a per-row neighbor list, we construct a **long edge table** of `(row_index_i, row_index_j)` pairs â€” one row per directed neighbor-year pair â€” and then compute grouped statistics using `data.table` aggregation. This replaces all interpreted loops with vectorized C-level operations.

**Steps:**

1. **Build a long edge data.frame** of `(cell_id_i, cell_id_j)` from `rook_neighbors_unique` (the `nb` object). This is ~1.37M directed pairs (spatial only, time-invariant).

2. **Cross with years** â€” each spatial edge exists for each of 28 years â†’ ~1.37M Ã— 28 â‰ˆ ~38.5M edge-year rows. At ~24 bytes per row (3 integer/numeric columns), this is <1 GB. Fits in 16 GB RAM.

3. **Join** the edge table to `cell_data` to attach the neighbor's variable values, then **group-by** the focal row and compute `max`, `min`, `mean`. This is a single `data.table` grouped aggregation â€” extremely fast.

4. **Repeat** for each of the 5 source variables (or do all at once).

This reduces runtime from ~86 hours to **minutes**.

---

## Working R Code

```r
library(data.table)

# ---------------------------------------------------------------
# 0.  Convert panel to data.table and create a row-index key
# ---------------------------------------------------------------
cell_dt <- as.data.table(cell_data)          
cell_dt[, row_idx := .I]                     # preserve original row order

# Fast lookup: (id, year) -> row_idx
setkey(cell_dt, id, year)

# ---------------------------------------------------------------
# 1.  Build the spatial directed-edge table from the nb object
#     rook_neighbors_unique is a list of length 344,208;
#     id_order[i] is the cell id for the i-th element.
# ---------------------------------------------------------------
edges <- rbindlist(lapply(seq_along(rook_neighbors_unique), function(i) {
  nb_i <- rook_neighbors_unique[[i]]
  if (length(nb_i) == 0L || (length(nb_i) == 1L && nb_i[1] == 0L)) {
    return(NULL)
  }
  data.table(focal_id = id_order[i],
             neighbor_id = id_order[nb_i])
}))
# edges has ~1.37 M rows (directed rook pairs, time-invariant)

# ---------------------------------------------------------------
# 2.  Cross edges with all 28 years to get edge-year table
# ---------------------------------------------------------------
years <- sort(unique(cell_dt$year))
edge_years <- CJ(edge_row = seq_len(nrow(edges)), year = years)
edge_years[, `:=`(focal_id    = edges$focal_id[edge_row],
                   neighbor_id = edges$neighbor_id[edge_row])]
edge_years[, edge_row := NULL]
# ~38.5 M rows

# ---------------------------------------------------------------
# 3.  Attach focal row_idx  (for later join-back)
# ---------------------------------------------------------------
# Keyed lookup on cell_dt
focal_key <- cell_dt[, .(id, year, row_idx)]
setkey(focal_key, id, year)

setkey(edge_years, focal_id, year)
edge_years[focal_key, focal_row := i.row_idx,
           on = .(focal_id = id, year = year)]

# ---------------------------------------------------------------
# 4.  Attach neighbor values for ALL source vars at once
# ---------------------------------------------------------------
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

neighbor_vals <- cell_dt[, c("id", "year", neighbor_source_vars), with = FALSE]
setkey(neighbor_vals, id, year)

setkey(edge_years, neighbor_id, year)
edge_years <- neighbor_vals[edge_years,
                            on = .(id = neighbor_id, year = year)]
# edge_years now has columns: id (=neighbor_id), year,
#   ntl, ec, pop_density, def, usd_est_n2, focal_id, focal_row

# ---------------------------------------------------------------
# 5.  Grouped aggregation: max, min, mean per focal_row per var
# ---------------------------------------------------------------
agg_exprs <- unlist(lapply(neighbor_source_vars, function(v) {
  list(
    bquote(as.numeric(max(.(as.name(v)),   na.rm = TRUE))),
    bquote(as.numeric(min(.(as.name(v)),   na.rm = TRUE))),
    bquote(mean(.(as.name(v)), na.rm = TRUE))
  )
}))
agg_names <- unlist(lapply(neighbor_source_vars, function(v) {
  paste0("neighbor_", c("max_", "min_", "mean_"), v)
}))

# Simpler and equally fast approach â€” compute per variable in a loop:
for (v in neighbor_source_vars) {
  
  # Subset to non-NA neighbor values for this variable
  sub <- edge_years[!is.na(get(v)), .(focal_row, val = get(v))]
  
  stats <- sub[, .(
    nmax  = max(val),
    nmin  = min(val),
    nmean = mean(val)
  ), keyby = focal_row]
  
  # Initialise new columns to NA
  max_col  <- paste0("neighbor_max_",  v)
  min_col  <- paste0("neighbor_min_",  v)
  mean_col <- paste0("neighbor_mean_", v)
  
  cell_dt[, (max_col)  := NA_real_]
  cell_dt[, (min_col)  := NA_real_]
  cell_dt[, (mean_col) := NA_real_]
  
  cell_dt[stats$focal_row, (max_col)  := stats$nmax]
  cell_dt[stats$focal_row, (min_col)  := stats$nmin]
  cell_dt[stats$focal_row, (mean_col) := stats$nmean]
}

# ---------------------------------------------------------------
# 6.  Convert back to data.frame, preserving original row order
# ---------------------------------------------------------------
cell_dt[, row_idx := NULL]
cell_data <- as.data.frame(cell_dt)

# ---------------------------------------------------------------
# 7.  Predict with the existing (already-trained) Random Forest
# ---------------------------------------------------------------
# The model object (e.g. `rf_model`) is unchanged.
# predictions <- predict(rf_model, newdata = cell_data)
```

---

## Why This Preserves Correctness

| Requirement | Guarantee |
|---|---|
| **Same numerical estimand** | The `max`, `min`, `mean` with `na.rm = TRUE` over exactly the same rook-neighbor sets produce bit-identical results to the original code. The edge table is constructed from the same `rook_neighbors_unique` nb object and `id_order`. |
| **Irregular/masked topology** | The nb object already encodes the true adjacency; cells with no neighbors get `NA` (same as original). No rectangular-raster assumption is introduced. |
| **Trained RF model preserved** | No retraining; the model object is never touched. Only the feature-engineering (neighbor stats) is recomputed. |

---

## Expected Performance

| Stage | Estimated Time | Memory |
|---|---|---|
| Build spatial edge table (1.37M rows) | ~2 seconds | ~30 MB |
| Cross with 28 years (38.5M rows) | ~5 seconds | ~900 MB |
| Join neighbor values | ~20 seconds | ~1.5 GB peak |
| 5 Ã— grouped aggregation | ~30 seconds each | in-place |
| **Total** | **~3â€“5 minutes** | **<4 GB peak** |

This replaces the estimated **86+ hours** with roughly **3â€“5 minutes** on the same 16 GB laptop, a speedup of approximately **1,000Ã—**, by eliminating all interpreted R loops in favor of vectorized `data.table` joins and grouped aggregations.
END_RESPONSE>>>
