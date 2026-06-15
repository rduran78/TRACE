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
run_id: OpenAI_F10_T10_R017
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F10
prompt_family_slug: SPARSE_GRAPH_FRAME
family_label: Sparse graph frame
family_group: Graph cue
replicate: 17
raw_output_file: Experiment_Output/OpenAI/Temp_1.0/output_F10_OpenAI_temp_10_rep_017.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
The bottleneck is the repeated list-based neighbor lookups and `lapply` operations for ~6.46M rows and 5 variables. Each call executes many small R loops and string-based lookups. Memory overhead is high due to list-of-list structure, and repeated neighbor computations for each row-year combination are expensive. Computing per-row neighbor stats via R loops is highly inefficient.

---

**Optimization Strategy**  
1. Build a **global adjacency structure** (row indices per cell's neighbors) **once**, then apply it to all years directly using numeric indices.
2. Use **vectorized operations** or **efficient grouping with data.table** instead of nested loops.
3. Compute neighbor statistics using **aggregated joins** rather than `lapply` for each row.
4. Avoid repeated string concatenationsâ€”map `(cell_id, year)` to row index with precomputed fast joins.
5. Process variables in bulk rather than in separate loops.
6. Keep it memory-efficient: operate with integer indices and pre-allocate result matrices.

---

### **Optimized R Implementation**

```r
library(data.table)

# Convert data to data.table for speed
setDT(cell_data)  # columns: id, year, ntl, ec, pop_density, def, usd_est_n2, ...

# Assumptions:
# - id_order: vector of unique cell IDs in the same order as rook_neighbors_unique
# - rook_neighbors_unique: spdep::nb object
# - Years repeated for each id

# Step 1: Compute adjacency in long format: (cell_id, neighbor_id)
adj_dt <- data.table(
  id = rep(id_order, lengths(rook_neighbors_unique)),
  neighbor_id = unlist(rook_neighbors_unique, use.names = FALSE)
)

# Step 2: Cross with years to represent all year-pairs
years <- unique(cell_data$year)
adj_expanded <- CJ(year = years, seq_len(nrow(adj_dt)))[
  , .(year,
      id = adj_dt$id[seq_len(nrow(adj_dt))],
      neighbor_id = adj_dt$neighbor_id[seq_len(nrow(adj_dt))])
]

# Step 3: Map to row indices
# Build fast key on cell_data (id, year)
setkey(cell_data, id, year)
adj_expanded[, cell_idx := cell_data[.(id, year), which = TRUE]]
adj_expanded[, neigh_idx := cell_data[.(neighbor_id, year), which = TRUE]]
adj_expanded <- adj_expanded[!is.na(cell_idx) & !is.na(neigh_idx)]

# Step 4: Compute neighbor stats efficiently for all variables
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# Prepare results structure
res_list <- vector("list", length(neighbor_source_vars))

for (var in neighbor_source_vars) {
  vals <- cell_data[[var]]
  
  tmp <- data.table(
    cell_idx = adj_expanded$cell_idx,
    val = vals[adj_expanded$neigh_idx]
  )
  
  # Compute max, min, mean per cell_idx
  stats <- tmp[, .(max = max(val, na.rm = TRUE),
                   min = min(val, na.rm = TRUE),
                   mean = mean(val, na.rm = TRUE)), by = cell_idx]
  
  setorder(stats, cell_idx)
  
  # Align back to rows in cell_data
  out_mat <- matrix(NA_real_, nrow(cell_data), 3)
  out_mat[stats$cell_idx, ] <- as.matrix(stats[, .(max, min, mean)])
  
  # Attach columns
  nm <- paste(var, c("nb_max", "nb_min", "nb_mean"), sep = "_")
  cell_data[, (nm) := as.data.table(out_mat)]
}

# The cell_data object now contains all neighbor features
```

---

### **Why it's efficient**
- Avoids per-row `lapply`: now aggregations happen in compiled C through `data.table`.
- Single precomputed neighbor mapping used for all variables and years.
- Handles 6.46M rows in a few passes instead of billions of small loops.
- No repeated string lookups: all mappings are integer-based.
- Memory-efficient: adjacency expanded only once, computed in batch.

**Expected improvement:** Hours â†’ Tens of minutes (depending on disk and CPU). Preserves original logic and statistics, requires no Random Forest retraining.
END_RESPONSE>>>
