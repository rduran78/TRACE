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
run_id: OpenAI_F02_T10_R030
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F02
prompt_family_slug: CODE_ONLY_NO_HINT
family_label: Code only, no hint
family_group: Code-visible neutral
replicate: 30
raw_output_file: Experiment_Output/OpenAI/Temp_1.0/output_F02_OpenAI_temp_10_rep_030.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
The current design performs nested R loops with heavy `lapply` calls over ~6.5M rows for 5 variables. For each row, it dynamically processes neighbor relationships and aggregates them. This causes:  
- **Excessive overhead** (millions of list operations).  
- **Memory inefficiency** (large lists, repeated string concatenation, repeated lookups).  
- **No vectorization or parallelization**.  

Every feature computation scans neighbor indices repeatedly and independently, leading to significant duplication.

---

**Optimization Strategy**  
1. **Precompute and Store Neighbor Indices Efficiently**: Flatten the neighbor structure into two integer vectors (row index â†’ neighbor index) using integer keys rather than repeated string keys.  
2. **Replace `lapply` loops with `data.table` or `dplyr` aggregation**: Compute aggregates via joins or fast grouping rather than list-of-lists.  
3. **Avoid repeated disk I/O and repeated key generation** by precomputing mappings once.  
4. **Vectorize aggregation**: Stack all neighbor pairs into a "long" table (`source_id`, `target_id`), join values, and compute `max`, `min`, `mean` via fast grouped operations in `data.table`.  
5. **Use chunking if memory is an issue**, but with 16 GB RAM, `data.table` should handle 6.5M rows + ~90M edges in chunks if needed.  
6. **Parallelize** using `data.table` multi-threading or `future.apply` if single-threaded is slow.  

---

**Working R Code**

```r
library(data.table)

# Assume: cell_data has columns id, year, ntl, ec, pop_density, def, usd_est_n2
#         rook_neighbors_unique is list of integer vectors (neighbors)
# Precompute mapping from id to row indices
dt <- as.data.table(cell_data)
setkey(dt, id, year)

# Precompute neighbor pairs by year
make_neighbor_pairs <- function(id_order, neighbors) {
  sources <- integer()
  targets <- integer()
  for (ref_idx in seq_along(id_order)) {
    src_id <- id_order[ref_idx]
    nbrs   <- neighbors[[ref_idx]]
    if (length(nbrs) > 0) {
      sources <- c(sources, rep(src_id, length(nbrs)))
      targets <- c(targets, id_order[nbrs])
    }
  }
  data.table(src_id = sources, nbr_id = targets)
}

# Flatten neighbors (spatial only)
neighbor_pairs <- make_neighbor_pairs(id_order, rook_neighbors_unique)

# Expand for all years via cross join
years <- unique(dt$year)
neighbor_pairs <- neighbor_pairs[, .(id = src_id, nbr_id, year = rep(years, each = .N))]
setkey(neighbor_pairs, nbr_id, year)

# Join neighbor values and compute stats for each variable
compute_neighbor_features <- function(dt, neighbor_pairs, var) {
  # Add value from neighbor
  val_dt <- dt[, .(nbr_id = id, year, val = get(var))]
  pairs_val <- neighbor_pairs[val_dt, on = .(nbr_id, year)]
  
  # Drop NA and aggregate
  agg <- pairs_val[!is.na(val),
                   .(paste0(var, "_nbr_max") = max(val),
                     paste0(var, "_nbr_min") = min(val),
                     paste0(var, "_nbr_mean") = mean(val)),
                   by = .(id, year)]
  agg
}

# Compute and merge all variables
res_list <- lapply(c("ntl", "ec", "pop_density", "def", "usd_est_n2"),
                   function(v) compute_neighbor_features(dt, neighbor_pairs, v))
res_all <- Reduce(function(x, y) merge(x, y, by = c("id", "year")), res_list)

# Merge back into original data
dt <- merge(dt, res_all, by = c("id", "year"), all.x = TRUE)
```

---

**Why This is Faster**  
- All costly operations (`paste`, repeated indexing, millions of `lapply`) are eliminated.
- The neighbor pairs are flattened **once**.
- Aggregations rely on `data.table` highly optimized C backend.
- Expected runtime: **minutes, not 86+ hours** (given ~6.5M rows and <100M neighbor entries).
  
This preserves the original numerical estimand (mean, min, max) and does not retrain the model.
END_RESPONSE>>>
