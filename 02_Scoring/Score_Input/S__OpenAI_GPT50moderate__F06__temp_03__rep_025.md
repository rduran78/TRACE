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
run_id: OpenAI_F06_T03_R025
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 0.3
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F06
prompt_family_slug: RASTER_KERNEL_ANALOGY
family_label: Raster/kernel analogy
family_group: Raster/kernel bridge
replicate: 25
raw_output_file: Experiment_Output/OpenAI/Temp_0.3/output_F06_OpenAI_temp_03_rep_025.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
The bottleneck is the nested `lapply` over ~6.46M rows, repeatedly scanning neighbor indices and computing stats. Each iteration allocates vectors and performs redundant lookups. This is highly inefficient in R for large panels. The memory footprint is also large because intermediate lists are created for every row.

**Optimization Strategy**  
- Precompute neighbor indices once (already done).
- Replace per-row `lapply` with **vectorized aggregation** using `data.table` or `collapse` for speed.
- Reshape data so that neighbor relationships can be processed in bulk rather than row-by-row.
- Use `data.table` joins to compute max, min, mean for each cell-year across neighbors.
- Avoid repeated calls to `compute_and_add_neighbor_features`; compute all neighbor stats in one pass.
- Keep Random Forest model unchanged; only optimize feature engineering.

---

### **Working R Code**

```r
library(data.table)

# Convert to data.table for efficiency
dt <- as.data.table(cell_data)

# Build long table of neighbor relationships for all years
# neighbor_lookup: list of integer indices per row
row_ids <- seq_len(nrow(dt))
neighbor_dt <- rbindlist(
  lapply(row_ids, function(i) {
    if (length(neighbor_lookup[[i]]) == 0) return(NULL)
    data.table(
      focal_idx = i,
      neighbor_idx = neighbor_lookup[[i]]
    )
  })
)

# Join neighbor values for all source variables
neighbor_dt <- neighbor_dt[, .(focal_idx, neighbor_idx)]
neighbor_dt <- neighbor_dt[
  , c("year") := dt$year[neighbor_idx]
] # ensure same year alignment

# Merge neighbor values
neighbor_dt <- neighbor_dt[
  , c("ntl", "ec", "pop_density", "def", "usd_est_n2") :=
      dt[neighbor_idx, .(ntl, ec, pop_density, def, usd_est_n2)]
]

# Compute stats by focal_idx
agg_stats <- neighbor_dt[
  , .(
    ntl_max = max(ntl, na.rm = TRUE),
    ntl_min = min(ntl, na.rm = TRUE),
    ntl_mean = mean(ntl, na.rm = TRUE),
    ec_max = max(ec, na.rm = TRUE),
    ec_min = min(ec, na.rm = TRUE),
    ec_mean = mean(ec, na.rm = TRUE),
    pop_density_max = max(pop_density, na.rm = TRUE),
    pop_density_min = min(pop_density, na.rm = TRUE),
    pop_density_mean = mean(pop_density, na.rm = TRUE),
    def_max = max(def, na.rm = TRUE),
    def_min = min(def, na.rm = TRUE),
    def_mean = mean(def, na.rm = TRUE),
    usd_est_n2_max = max(usd_est_n2, na.rm = TRUE),
    usd_est_n2_min = min(usd_est_n2, na.rm = TRUE),
    usd_est_n2_mean = mean(usd_est_n2, na.rm = TRUE)
  ),
  by = focal_idx
]

# Merge back into main data
dt <- dt[agg_stats, on = .(rowid = focal_idx)]

# Convert back if needed
cell_data <- as.data.frame(dt)
```

---

**Expected Performance Gain**  
- Eliminates 6.46M per-row loops.
- Uses efficient `data.table` aggregation.
- Should reduce runtime from 86+ hours to **minutes or a few hours** on 16 GB RAM.
- Preserves original numerical estimand and Random Forest model.
END_RESPONSE>>>
