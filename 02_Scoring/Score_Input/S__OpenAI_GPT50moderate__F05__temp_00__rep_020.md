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
run_id: OpenAI_F05_T00_R020
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 0.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F05
prompt_family_slug: STRING_KEY_REDUNDANCY
family_label: String-key probe
family_group: Line-level probe
replicate: 20
raw_output_file: Experiment_Output/OpenAI/Temp_0.0/output_F05_OpenAI_temp_00_rep_020.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**

The repeated string concatenation (`paste(data$id, data$year, sep = "_")`) and hash lookups inside the `lapply` loop are only a *symptom* of a larger inefficiency: the algorithm repeatedly performs expensive string-based key generation and hash lookups for every row (â‰ˆ6.46M times). This is not just a local inefficiency; it reflects an algorithmic design that scales poorly with panel size.

The root cause:
- For each of 6.46M rows, the code constructs neighbor keys and looks them up in a hash map.
- This involves repeated string concatenation and repeated indexing into a large named vector.
- The neighbor structure is static across years, but the code recomputes neighbor indices for every row-year combination.

**Optimization Strategy**

Reformulate the algorithm to:
1. **Precompute numeric indices** for neighbors across all years, avoiding string keys entirely.
2. Use **vectorized operations** or `data.table` joins instead of per-row `lapply`.
3. Exploit the fact that the neighbor graph is static across years: replicate neighbor relationships across years in a single precomputed structure.

**Proposed Approach**
- Assign each `(id, year)` a numeric row index once.
- Expand the neighbor relationships across years using integer arithmetic.
- Compute neighbor stats in a fully vectorized way using `data.table` grouping.

---

### **Working R Code**

```r
library(data.table)

# Assume: cell_data has columns id, year, ntl, ec, pop_density, def, usd_est_n2
# rook_neighbors_unique: list of integer vectors (spdep::nb)
# id_order: vector of all unique ids in the same order as rook_neighbors_unique

# Convert to data.table
setDT(cell_data)
setkey(cell_data, id, year)

# Precompute mapping from id to position
id_to_pos <- setNames(seq_along(id_order), id_order)

# Basic dimensions
n_ids   <- length(id_order)
n_years <- length(unique(cell_data$year))

# Precompute neighbor pairs (id -> neighbor_id)
neighbor_pairs <- data.table(
  id = rep(id_order, lengths(rook_neighbors_unique)),
  neighbor_id = unlist(rook_neighbors_unique)
)

# Expand across years
years <- sort(unique(cell_data$year))
neighbor_pairs_expanded <- neighbor_pairs[
  , .(id = rep(id, length(years)),
      neighbor_id = rep(neighbor_id, length(years)),
      year = rep(years, each = .N))
]

# Join to get row indices
setkey(neighbor_pairs_expanded, neighbor_id, year)
setkey(cell_data, id, year)
neighbor_pairs_expanded[
  , neighbor_idx := cell_data[.SD, which = TRUE]
]

# Drop missing (if any)
neighbor_pairs_expanded <- neighbor_pairs_expanded[!is.na(neighbor_idx)]

# For each source variable, compute neighbor stats
compute_neighbor_stats_fast <- function(var_name) {
  vals <- cell_data[[var_name]]
  neighbor_vals <- vals[neighbor_pairs_expanded$neighbor_idx]
  neighbor_pairs_expanded[, value := neighbor_vals]
  
  # Aggregate by (id, year)
  stats <- neighbor_pairs_expanded[
    , .(max = max(value, na.rm = TRUE),
        min = min(value, na.rm = TRUE),
        mean = mean(value, na.rm = TRUE)),
    by = .(id, year)
  ]
  
  # Merge back
  setnames(stats, c("id", "year", paste0(var_name, "_nbr_max"),
                               paste0(var_name, "_nbr_min"),
                               paste0(var_name, "_nbr_mean")))
  cell_data[stats, on = .(id, year)]
}

# Apply for all neighbor source vars
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
for (var_name in neighbor_source_vars) {
  cell_data <- compute_neighbor_stats_fast(var_name)
}
```

---

### **Why This Is Faster**
- Eliminates per-row string concatenation and hash lookups.
- Uses integer joins and `data.table` aggregation (highly optimized in C).
- Precomputes neighbor relationships once and reuses them for all variables.
- Complexity drops from O(N Ã— neighbors Ã— vars) with heavy overhead to a single join + grouped aggregation per variable.

**Expected Performance**
- Memory: manageable on 16 GB (â‰ˆ6.5M rows + neighbor pairs).
- Time: reduces from 86+ hours to minutes (depending on disk I/O and CPU).

**Preserves**
- Original estimand (same neighbor stats).
- Trained Random Forest model (no retraining needed).
END_RESPONSE>>>
