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
run_id: OpenAI_F07_T10_R015
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F07
prompt_family_slug: RASTER_INVALID_IRREGULAR
family_label: Raster invalid irregular topology
family_group: Raster/kernel bridge
replicate: 15
raw_output_file: Experiment_Output/OpenAI/Temp_1.0/output_F07_OpenAI_temp_10_rep_015.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
The bottleneck arises because the current implementation uses nested `lapply` over 6.46 million rows for each of 5 variables, resulting in extreme overhead from repeated R function calls and repeated lookups. The algorithm scales as O(N Ã— K) with large constants, where N = 6.46M rows and K = average number of neighbors (~4-8). Additional inefficiencies include:

- Dynamic key lookups inside loops.
- No vectorization or matrix-based aggregation.
- Repeated work per variable.
- Memory fragmentation from millions of small objects (lists).

Estimated runtime (86+ hours) confirms R's list-iteration overhead is the culprit, not the pure arithmetic cost.

---

**Optimization Strategy**  
- **Pre-flatten neighbor relationships** into an edge list (row index â†’ neighbor row index) across all years in one pass.
- Use this edge list to join data and compute aggregations with **data.table** or **collapse**â€”highly optimized for grouping operations in R.
- Avoid repeated computation per variable: pivot the dataset or perform grouped summaries for all variables at once.
- Keep computations fully in R (no retraining, no algorithm change).
- Reuse the trained Random Forest model without modification.

---

**Working R Implementation**

```r
library(data.table)

# Assume cell_data is a data.table with columns: id, year, ntl, ec, pop_density, def, usd_est_n2

# Step 1: Build long edge list once (row -> neighbor_row)
build_edge_list <- function(data, id_order, neighbors) {
  id_to_idx <- setNames(seq_along(id_order), as.character(id_order))
  idx_lookup <- setNames(seq_len(nrow(data)), paste(data$id, data$year, sep = "_"))
  
  years <- unique(data$year)
  edges <- vector("list", length(years))
  
  for (i in seq_along(years)) {
    yr <- years[i]
    year_rows <- which(data$year == yr)
    row_keys <- paste(data$id[year_rows], yr, sep = "_")
    
    # For each year, create edges (source_idx -> neighbor_idx)
    edge_list <- lapply(year_rows, function(r) {
      cell_idx <- id_to_idx[as.character(data$id[r])]
      neigh_ids <- id_order[neighbors[[cell_idx]]]
      neigh_keys <- paste(neigh_ids, yr, sep = "_")
      neigh_rows <- idx_lookup[neigh_keys]
      neigh_rows[!is.na(neigh_rows)]
    })
    
    from <- rep(year_rows, lengths(edge_list))
    to <- unlist(edge_list, use.names = FALSE)
    edges[[i]] <- data.table(from = from, to = to)
  }
  rbindlist(edges)
}

# Build the edge list
edge_list <- build_edge_list(cell_data, id_order, rook_neighbors_unique)

# Step 2: Explode neighbor values and aggregate with data.table fast grouping
# Variables to compute neighbor stats for
vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# Melt only needed columns for efficiency
dt <- as.data.table(cell_data)

# Lookup neighbor values
edge_dt <- edge_list[, (vars) := dt[to, ..vars]]

# For each var, compute stats grouped by 'from' (source row)
for (v in vars) {
  stats <- edge_dt[!is.na(get(v)), .(
    max_v = max(get(v)),
    min_v = min(get(v)),
    mean_v = mean(get(v))
  ), by = from]
  
  # Join back to main table
  setnames(stats, c("max_v", "min_v", "mean_v"),
           paste0(v, c("_nb_max", "_nb_min", "_nb_mean")))
  
  dt[stats$from, (paste0(v, c("_nb_max", "_nb_min", "_nb_mean"))) := 
        .(stats[[2]], stats[[3]], stats[[4]])]
}

# cell_data now enriched
cell_data <- dt
```

---

**Expected Benefits**
- Converts ~6.5M Ã— nested loops into vectorized joins and grouped aggregations.
- **Data.table** uses radix sorting and memory-efficient joins.
- The heavy-lifting shifts to compiled C, reducing runtime to **minutes or a few hours** (depending on disk I/O), well within 16 GB RAM constraints.
- Preserves all estimands and the trained Random Forest model.
- No algorithmic change, only performance optimization.

**Key Insight:**  
Avoid per-row R function calls. Transform to an edge-based table and leverage optimized grouped operations.
END_RESPONSE>>>
