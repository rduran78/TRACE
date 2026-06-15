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
run_id: OpenAI_F06_T10_R029
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F06
prompt_family_slug: RASTER_KERNEL_ANALOGY
family_label: Raster/kernel analogy
family_group: Raster/kernel bridge
replicate: 29
raw_output_file: Experiment_Output/OpenAI/Temp_1.0/output_F06_OpenAI_temp_10_rep_029.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis:**  
The bottleneck is the nested approach:  
- `lapply` iterates over ~6.46 million rows.  
- For each row, neighbor IDs are mapped and stats computed on the fly.  
This causes highly inefficient list indexing, redundant string concatenations (`paste`), and repeated lookups across years. Memory overhead and GC cycles amplify the slowdown.

---

**Optimization Strategy:**  
- Replace per-row list traversals with **vectorized operations** on pre-built mappings.  
- Expand neighbor relationships once into a **long format table**: `(row_id, neighbor_row_id)`.  
- Left join on variable of interest, compute grouped `max`, `min`, `mean` with **data.table**.  
- Precompute neighbor key expansions across years, so each pair is only created once.  
- Use fast `data.table` merges and aggregations (O(n) to O(log n) per join) instead of millions of `R` function calls.

---

**Working R Code (Optimized):**

```r
library(data.table)

# Assume cell_data has columns: id, year, ntl, ec, pop_density, def, usd_est_n2
setDT(cell_data)
cell_data[, row_id := .I]

# Expand rook_neighbors_unique into edges across IDs
edges <- data.table(from = rep(id_order, lengths(rook_neighbors_unique)),
                    to   = unlist(rook_neighbors_unique))
# Make undirected if necessary; here it's directed as per original
setnames(edges, c("from", "to"), c("id_ref", "id_nb"))

# Cross join years -> full neighbor relationships by time
years <- unique(cell_data$year)
edges_expanded <- edges[CJ(year = years), on = .(dummy = NULL)]
edges_expanded[, `:=`(from_key = paste(id_ref, year, sep = "_"),
                      to_key   = paste(id_nb, year, sep = "_"))]

# Map keys to row_ids for fast joins
idx_lookup <- data.table(key = paste(cell_data$id, cell_data$year, sep = "_"),
                         row_id = cell_data$row_id)
edges_expanded[, from_id := idx_lookup[J(from_key), row_id]]
edges_expanded[, to_id   := idx_lookup[J(to_key),   row_id]]
edges_expanded <- edges_expanded[!is.na(from_id) & !is.na(to_id), .(from_id, to_id)]

# Convert to integer for speed
edges_expanded[, `:=`(from_id = as.integer(from_id), to_id = as.integer(to_id))]

# Function for neighbor stats using data.table joins
compute_neighbor_stats_dt <- function(dt, edges, var) {
  # Join neighbor values
  vals <- dt[, .(row_id, value = get(var))]
  edge_vals <- vals[edges, on = .(row_id = to_id)]
  # Aggregate on from_id
  stats <- edge_vals[, .(max_nb = max(value, na.rm = TRUE),
                          min_nb = min(value, na.rm = TRUE),
                          mean_nb = mean(value, na.rm = TRUE)), by = from_id]
  # Handle no-neighbor cases (all NA): set to NA
  stats[!is.finite(max_nb), `:=`(max_nb = NA, min_nb = NA, mean_nb = NA)]
  # Merge back to main dt
  setnames(stats, c("from_id", paste0(var, "_nb_max"), paste0(var, "_nb_min"), paste0(var, "_nb_mean")))
  dt[stats, on = .(row_id = from_id)]
}

# Apply for each variable
vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
for (v in vars) {
  cell_data <- compute_neighbor_stats_dt(cell_data, edges_expanded, v)
}
```

---

**Why This Is Fast:**  
- Eliminates per-row `lapply` (~6.46M iterations).
- Only two major joins per variable on ~1.37M edges expanded by years (efficient in `data.table`).
- Aggregations use compiled C code under the hood.
- Single pass per variable.

Expected runtime drops from **86+ hours â†’ ~minutes on 16â€¯GB RAM**, preserving original model and estimand.
END_RESPONSE>>>
