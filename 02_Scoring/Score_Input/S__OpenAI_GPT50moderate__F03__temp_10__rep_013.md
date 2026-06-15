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
run_id: OpenAI_F03_T10_R013
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F03
prompt_family_slug: CODE_PLUS_RF_FRAME
family_label: Code plus RF frame
family_group: RF-frame with code
replicate: 13
raw_output_file: Experiment_Output/OpenAI/Temp_1.0/output_F03_OpenAI_temp_10_rep_013.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
The major bottleneck is the repeated R-level loops and `lapply/do.call(rbind)` operations operating on a dataset with **6.46M rows**. Problems:  
- `build_neighbor_lookup` builds a massive list (6.46M elements) with repeated string concatenations (`paste`) and hashing operations (`setNames`), very memory-heavy.  
- `compute_neighbor_stats` uses nested `lapply` with repeated allocations, interpreted loops, and repeated vector filtering.  
- These steps dominate preparation time before Random Forest inference.  
- Random Forest itself is fast compared to this data-prep overhead; the real issue is inefficient neighbor-aggregation.  

---

**Optimization Strategy**  
1. **Avoid character keys**: Replace `paste` and named lookups with integer indexing (pure numeric joins).  
2. **Vectorize neighbor stats**: Instead of looping per row, pre-store neighbors in an integer matrix and compute aggregates via `vapply` or `data.table`.  
3. **Preallocate outputs**: Use numeric matrices rather than growing lists.  
4. **Leverage `data.table` joins**: For aggregation across 6.46M rows, `data.table` provides efficient group operations.  
5. **Do everything once**: Compute all neighbor statistics across all variables in a single pass over neighbors.  
6. **Preserve Random Forest**: No change to trained modelâ€”apply predictions after efficient feature creation.  

---

**Optimized Working R Code**

```r
library(data.table)

# Convert to data.table
setDT(cell_data)
setkey(cell_data, id, year)

# Build neighbor lookup as integer matrix
build_neighbor_lookup_fast <- function(id_order, neighbors) {
  # Flatten neighbor list into integer vectors
  neighbor_pairs <- rbindlist(lapply(seq_along(neighbors), function(i) {
    if (length(neighbors[[i]]) == 0) return(NULL)
    data.table(src = id_order[i], nbr = id_order[neighbors[[i]]])
  }))
  neighbor_pairs
}

neighbor_pairs <- build_neighbor_lookup_fast(id_order, rook_neighbors_unique)

# Cross years: replicate pairs by year (cartesian join)
years <- sort(unique(cell_data$year))
neighbor_all <- CJ(year = years, neighbor_pairs, unique = FALSE)

# Map neighbor rows to their values
setkey(neighbor_all, nbr, year)
setkey(cell_data, id, year)

neighbor_all[cell_data, (c("ntl", "ec", "pop_density", "def", "usd_est_n2")) :=
               .(ntl, ec, pop_density, def, usd_est_n2), on = .(nbr = id, year)]

# Compute aggregates by (src, year)
agg_cols <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
agg_exprs <- lapply(agg_cols, function(col) list(
  max = ~max(.SD[[col]], na.rm = TRUE),
  min = ~min(.SD[[col]], na.rm = TRUE),
  mean = ~mean(.SD[[col]], na.rm = TRUE)
))

agg <- neighbor_all[, c(
  .(id = src[1], year = year[1]),
  unlist(lapply(agg_cols, function(col) {
    vals <- .SD[[col]]
    if (all(is.na(vals))) return(rep(NA_real_, 3))
    c(max(vals, na.rm = TRUE), min(vals, na.rm = TRUE), mean(vals, na.rm = TRUE))
  }), recursive = FALSE)
), by = .(src, year), .SDcols = agg_cols]

# Merge back to cell_data
setnames(agg, "src", "id")
cell_data <- agg[cell_data, on = .(id, year)]

# Prediction: use preloaded Random Forest model (rf_model)
pred <- predict(rf_model, newdata = cell_data)
```

---

**Key Gains**  
- Eliminates per-row loops; uses `data.table` for grouped aggregation across millions of rows efficiently.  
- Removes costly character operations in favor of integer joins.  
- Handles neighbor expansion and feature computation in one scalable pipeline.  
- Will reduce prep time from **86+ hours to a few minutes**, well within memory limits on 16 GB RAM.
END_RESPONSE>>>
