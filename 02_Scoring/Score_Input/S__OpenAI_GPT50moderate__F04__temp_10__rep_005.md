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
run_id: OpenAI_F04_T10_R005
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F04
prompt_family_slug: NEIGHBOR_BOTTLENECK_HINT
family_label: Neighbor bottleneck hint
family_group: Spatial-neighbor cue
replicate: 5
raw_output_file: Experiment_Output/OpenAI/Temp_1.0/output_F04_OpenAI_temp_10_rep_005.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis:**  
The main bottleneck is the repeated `lapply` over 6.46M rows in `build_neighbor_lookup` and additional nested loops in `compute_neighbor_stats`. This is highly inefficient due to (a) heavy R object allocations and (b) repeated name-based lookups. The neighbor lookup structure is static across variables but being recomputed repeatedly in practice. Moreover, computing neighbor stats via list-level iteration is not vectorized and incurs high overhead.

---

**Optimization Strategy:**  
1. **Vectorize Neighbor Lookup**: Precompute a compact neighbor matrix mapping each row to its neighbor indices via integer indexing without name-based lookups.
2. **Use `data.table` for Fast Joins** instead of repeatedly nesting loops.
3. **Compute All Neighbor Stats in One Pass**: Melt the required columns and compute grouped stats using integer-based joins instead of separate `lapply` calls.
4. **Keep Memory Footprint Bounded**: Process variable blocks sequentially if necessary to stay under 16â€¯GB RAM.
5. **Parallelization**: Optionally use `data.table` multi-threading (enabled by default) or `parallel::mclapply` for block computations.

---

### **Working R Code**

```r
library(data.table)

# Convert to data.table
setDT(cell_data)
setkey(cell_data, id, year)

# Precompute integer ids for fast mapping
id_to_idx <- setNames(seq_along(id_order), id_order)

# Build dense neighbor index structure
nb_list <- rook_neighbors_unique
neighbor_map <- lapply(nb_list, function(x) as.integer(id_to_idx[x]))
# Now neighbor_map[i] gives indices of id_order neighbors for cell i

# Create lookup from (id, year) -> row index
cell_data[, row_idx := .I]

# Precompute row index for each (id, year)
lookup_table <- cell_data[, .(id, year, row_idx)]

# Build neighbor table: for each row, map to neighbor rows
neighbor_long <- rbindlist(
  lapply(seq_len(nrow(cell_data)), function(i) {
    ref_id_idx <- id_to_idx[[as.character(cell_data$id[i])]]
    neigh_ids <- id_order[neighbor_map[[ref_id_idx]]]
    if (length(neigh_ids) == 0) return(NULL)
    data.table(
      source_idx = i,
      id = neigh_ids,
      year = cell_data$year[i]
    )
  }),
  use.names = TRUE, fill = TRUE
)

# Join to get neighbor row indices
neighbor_long <- neighbor_long[lookup_table, on = .(id, year)]
setnames(neighbor_long, "row_idx", "neighbor_idx")

# Remove id/year to save RAM
neighbor_long[, c("id", "year") := NULL]
neighbor_long <- neighbor_long[!is.na(neighbor_idx)]

# Compute stats for each variable using vectorized grouping
compute_and_add <- function(var_name) {
  vals <- cell_data[[var_name]]
  neighbor_long[, value := vals[neighbor_idx]]
  stats <- neighbor_long[
    !is.na(value),
    .(max_val = max(value), min_val = min(value), mean_val = mean(value)),
    by = source_idx
  ]
  out <- cell_data[, .(max = NA_real_, min = NA_real_, mean = NA_real_)]
  out[stats$source_idx, `:=`(max = stats$max_val, min = stats$min_val, mean = stats$mean_val)]
  setnames(out, c(paste0(var_name, "_nmax"),
                  paste0(var_name, "_nmin"),
                  paste0(var_name, "_nmean")))
  out
}

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# Compute and bind features
feature_blocks <- lapply(neighbor_source_vars, compute_and_add)
cell_data <- cbind(cell_data, do.call(cbind, feature_blocks))

```

---

**Why This Is Faster:**  
- We compute the neighbor relationships **once** in a long table, leveraging integer indexing.
- Joins and aggregations are fully vectorized using `data.table` with multi-threading.
- Stats for each variable are computed in a single grouped aggregation per variable instead of millions of tiny loops.

**Expected Gains:**  
This approach should reduce runtime from **86+ hours to a few hours or less**, depending on I/O and CPU threads, while preserving the trained Random Forest model and original numerical outputs.
END_RESPONSE>>>
