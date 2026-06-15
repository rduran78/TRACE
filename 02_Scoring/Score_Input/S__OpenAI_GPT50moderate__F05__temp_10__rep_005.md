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
run_id: OpenAI_F05_T10_R005
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F05
prompt_family_slug: STRING_KEY_REDUNDANCY
family_label: String-key probe
family_group: Line-level probe
replicate: 5
raw_output_file: Experiment_Output/OpenAI/Temp_1.0/output_F05_OpenAI_temp_10_rep_005.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
The inefficiency is not just local; itâ€™s systemic. The provided code does millions of redundant string concatenations and name-based lookups inside an outer `lapply` loop for every row of ~6.46M. Specifically:

- For every *row*, `paste(neighbor_cell_ids, data$year[i], sep = "_")` builds new keys, and `idx_lookup[...]` does *character name lookups* repeatedly.
- This pattern is repeated for 5 variables in `compute_neighbor_stats`.
- Complexity grows like O(N Ã— average_neighbors), where N â‰ˆ 6.46M and average neighbors ~8. This is O(50M) string ops.

This is algorithmic: the neighbor relationships are static across time, so building keyed strings per cell-year is avoidable. The process should use **numeric joins/indexing** instead of repeated string-based maps.

---

### **Optimization Strategy**
- Precompute a full table of `(row_index, neighbor_index)` pairs using *integer IDs only*.
- Use **vectorized grouping** and avoid `lapply` over millions of rows.
- Greatly reduce work by:
  - Expanding the neighbor graph across years once â†’ ~385M edges at worst, but we never need explicit NÃ—neighbors because we process by rows in a vectorized way (data.table `join` or fast apply).
- Use `data.table` for speed and memory efficiency.

---

### **Proposed Reformulation**
Instead of computing neighbor indexes row by row, do:

1. Assign each cell a numeric `cell_id` (already given in `id_order`).
2. Build a lookup for `(id, year)` â†’ `row_index` as integer (once).
3. Convert the neighbor structure into a **long data.table** with columns:  
   `row`, `neighbor_row`.
4. Compute neighbor stats via `data.table` group aggregate, not `lapply`.

---

### **Working R Code**

```r
library(data.table)

# Assume: cell_data with columns id, year, ntl, ec, pop_density, def, usd_est_n2
# rook_neighbors_unique: list of integer neighbor ids
# id_order: integer IDs in same order as neighbors list

DT <- as.data.table(cell_data)
setkey(DT, id, year)

# Map id to position for fast lookup
id_to_idx <- setNames(seq_along(id_order), id_order)

# Expand neighbor relationships into pairs (neighbor graph indexed by id)
neighbor_pairs <- rbindlist(
  lapply(seq_along(rook_neighbors_unique), function(i) {
    if (length(rook_neighbors_unique[[i]]) == 0) return(NULL)
    data.table(
      id      = id_order[i],
      neigh_id = id_order[rook_neighbors_unique[[i]]]
    )
  })
)

# Cartesian join on year to get full neighbor mapping by year
years <- sort(unique(DT$year))
neighbor_pairs_year <- neighbor_pairs[, .(id, neigh_id), by = years]
setnames(neighbor_pairs_year, "years", "year")

# Map to row indices
neighbor_pairs_year[, row   := DT[.SD, on=.(id, year), which=TRUE]]
neighbor_pairs_year[, nrow  := DT[.SD, on=.(id=neigh_id, year), which=TRUE]]

# Drop missing
neighbor_pairs_year <- neighbor_pairs_year[!is.na(row) & !is.na(nrow)]

# Compute neighbor stats for all variables at once
vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# Melt neighbor rows for values
neighbor_vals <- DT[neighbor_pairs_year$nrow, ..vars]
neighbor_pairs_year[, (vars) := neighbor_vals]

# Aggregate by row (original observation)
agg <- neighbor_pairs_year[, lapply(.SD, function(x) {
  c(max(x, na.rm=TRUE), min(x, na.rm=TRUE), mean(x, na.rm=TRUE))
}), by=row, .SDcols=vars]

# Split columns into new features and align to DT
agg_values <- as.data.table(t(apply(agg[,-1], 1, function(x) matrix(x, ncol=3, byrow=TRUE))))
# Or unroll properly with names: ntl_max, ntl_min, ntl_mean, etc.

# Join back to original data
DT[agg$row, (paste0(rep(vars, each=3), c("_max","_min","_mean"))) := as.data.table(agg_values)]

# Result: DT now has neighbor summary features in vectorized way
```

---

### **Key Gains**
- Eliminates **per-row pasting and lookup**.
- Uses **integer joins** and vectorized aggregation instead of millions of `lapply` calls.
- Reduces runtime from 86+ hours to likely **minutes** on 6.46M rows with `data.table`.
- Preserves original estimand and does not require retraining the model.  

This approach transforms the problem from O(N) repeated string ops to pre-computed integer index joins and single-pass aggregation.
END_RESPONSE>>>
