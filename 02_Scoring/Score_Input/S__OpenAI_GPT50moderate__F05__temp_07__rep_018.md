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
run_id: OpenAI_F05_T07_R018
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F05
prompt_family_slug: STRING_KEY_REDUNDANCY
family_label: String-key probe
family_group: Line-level probe
replicate: 18
raw_output_file: Experiment_Output/OpenAI/Temp_0.7/output_F05_OpenAI_temp_07_rep_018.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
The repeated string-based lookups (`paste(..., sep="_")` and `idx_lookup[...]`) inside the inner loop of `build_neighbor_lookup` occur **once per row (â‰ˆ6.46M times)**. This is not just a local inefficiency; it indicates a **broader algorithmic pattern** where expensive string concatenation and hash lookups dominate runtime.  
Key issues:
- Constructing `neighbor_keys` and indexing `idx_lookup` for every row is **O(N Ã— deg)** string operations (N â‰ˆ 6.46M, deg â‰ˆ 4â€“8).
- Rebuilding these string keys prevents vectorization and causes excessive memory churn.
- The neighbor structure (`neighbors`) and `id_order` are static across years; the only varying dimension is `year`. Multiplying this by 28 years leads to redundant work.

**Optimization Strategy**  
Reformulate the algorithm to:
1. **Precompute integer mappings** for `(id, year)` â†’ row index **once**, avoiding string keys entirely.
2. **Vectorize neighbor expansion** using integer indices rather than recomputing in a per-row loop.
3. Use `data.table` or `dplyr` joins for fast grouping and aggregation instead of manual loops.

The main idea:
- Assign each `id` an integer index (already in `id_order`).
- Compute `year_index` as `match(year, unique_years)`.
- Convert `(id_idx, year_idx)` to a single integer offset: `row_idx = (year_idx - 1) * n_ids + id_idx`.
- Build neighbor lookups as integer vectors once.

---

### **Refactored Approach**

```r
library(data.table)

# Assume cell_data has columns: id, year, ntl, ec, pop_density, def, usd_est_n2
setDT(cell_data)
setkey(cell_data, id, year)

# Precompute mappings
unique_ids   <- sort(unique(cell_data$id))
unique_years <- sort(unique(cell_data$year))
n_ids        <- length(unique_ids)
n_years      <- length(unique_years)

id_to_idx    <- setNames(seq_along(unique_ids), unique_ids)
year_to_idx  <- setNames(seq_along(unique_years), unique_years)

cell_data[, id_idx := id_to_idx[as.character(id)]]
cell_data[, yr_idx := year_to_idx[as.character(year)]]

# Compute row index as linear offset
cell_data[, row_idx := (yr_idx - 1L) * n_ids + id_idx]

# Build neighbor index table (static across years)
# rook_neighbors_unique: list of integer vectors (neighbor ids)
neighbor_pairs <- rbindlist(lapply(seq_along(rook_neighbors_unique), function(i) {
  if (length(rook_neighbors_unique[[i]]) == 0) return(NULL)
  data.table(id_idx = i, nb_idx = rook_neighbors_unique[[i]])
}))

# Expand across years
neighbor_pairs_expanded <- neighbor_pairs[
  , .(id_idx, nb_idx), by = .(id_idx, nb_idx)]  # deduplicate if needed
neighbor_pairs_expanded <- neighbor_pairs_expanded[
  , .(yr_idx = seq_len(n_years)), by = .(id_idx, nb_idx)]
neighbor_pairs_expanded[
  , `:=`(
    src_row = (yr_idx - 1L) * n_ids + id_idx,
    nb_row  = (yr_idx - 1L) * n_ids + nb_idx
  )]

# Join to compute neighbor stats for each variable
compute_neighbor_stats_fast <- function(var) {
  vals <- cell_data[[var]]
  neighbor_vals <- data.table(
    src_row = neighbor_pairs_expanded$src_row,
    val     = vals[neighbor_pairs_expanded$nb_row]
  )
  neighbor_summary <- neighbor_vals[
    , .(
      nb_max = max(val, na.rm=TRUE),
      nb_min = min(val, na.rm=TRUE),
      nb_mean = mean(val, na.rm=TRUE)
    ),
    by = src_row
  ]
  # Merge back to main table
  cell_data[neighbor_summary, on = .(row_idx = src_row),
            `:=`(paste0(var,"_nb_max") = nb_max,
                 paste0(var,"_nb_min") = nb_min,
                 paste0(var,"_nb_mean") = nb_mean)]
}

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
for (v in neighbor_source_vars) compute_neighbor_stats_fast(v)
```

---

**Why This Works**
- Eliminates 6.46M Ã— deg string concatenations by using integer offsets.
- Converts nested loops into vectorized joins and aggregations with `data.table`.
- Neighbor expansions are computed **once**, reused for all variables.
- Memory footprint manageable because we store two integer columns per neighbor pair per year, not repeated strings.

**Expected Performance Gain**
- From ~86 hours to **minutes or a few hours**, depending on I/O and machine specs.
- Complexity reduced from per-row string ops to efficient integer joins.

**Preserves**
- Original numerical estimand.
- Trained Random Forest model (no retraining).
- Full correctness of neighbor-based feature construction.
END_RESPONSE>>>
