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
run_id: OpenAI_F12_T03_R028
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 0.3
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F12
prompt_family_slug: FALSE_RBIND_DIAGNOSIS
family_label: False rbind diagnosis
family_group: Adversarial diagnosis
replicate: 28
raw_output_file: Experiment_Output/OpenAI/Temp_0.3/output_F12_OpenAI_temp_03_rep_028.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis:**  
The colleagueâ€™s claim that `do.call(rbind, result)` and repeated list binding inside `compute_neighbor_stats()` is the main bottleneck is **incorrect**. While `rbind` on millions of rows is non-trivial, the deeper bottleneck is the **nested `lapply` over 6.46 million rows combined with repeated neighbor lookups and vector filtering**. Each iteration recomputes indices, subsets, and statistics in pure R, which is extremely slow at this scale. The problem is algorithmic: performing 6.46M Ã— 5 passes in R loops is prohibitive.

---

### **Correct Optimization Strategy**
- **Vectorize and precompute:** Avoid per-row loops by flattening neighbor relationships into a long format and aggregating with fast grouped operations (`data.table` or `dplyr`).
- **Compute all neighbor stats in one pass:** Instead of looping over 5 variables, melt them and compute grouped `max`, `min`, `mean` using efficient C-backed aggregation.
- **Preserve trained model and estimands:** Only change feature engineering speed, not the logic or values.

---

### **Optimized Approach**
1. Convert `cell_data` to `data.table`.
2. Flatten neighbor relationships into a two-column mapping: `(source_row, neighbor_row)`.
3. Join neighbor values for all variables at once.
4. Compute grouped stats by `source_row` and `variable`.

---

### **Working R Code**

```r
library(data.table)

# Convert to data.table
setDT(cell_data)

# Build neighbor lookup as a long table instead of list-of-lists
build_neighbor_dt <- function(data, id_order, neighbors) {
  id_to_ref <- setNames(seq_along(id_order), as.character(id_order))
  idx_lookup <- setNames(seq_len(nrow(data)), paste(data$id, data$year, sep = "_"))
  
  # Preallocate list for efficiency
  res_list <- vector("list", length = length(id_order))
  
  for (i in seq_along(id_order)) {
    neighbor_ids <- id_order[neighbors[[i]]]
    if (length(neighbor_ids) > 0) {
      # For each year, map source to neighbor rows
      res_list[[i]] <- data.table(
        src_id = id_order[i],
        nbr_id = neighbor_ids
      )
    }
  }
  rbindlist(res_list)
}

# Build static neighbor pairs (cell-level)
neighbor_pairs <- build_neighbor_dt(cell_data, id_order, rook_neighbors_unique)

# Expand to panel by joining on year
neighbor_dt <- merge(
  cell_data[, .(id, year, src_row = .I)],
  neighbor_pairs,
  by.x = "id", by.y = "src_id",
  allow.cartesian = TRUE
)
neighbor_dt <- merge(
  neighbor_dt,
  cell_data[, .(nbr_id = id, year, nbr_row = .I)],
  by = c("nbr_id", "year"),
  allow.cartesian = TRUE
)

# Keep only needed columns
neighbor_dt <- neighbor_dt[, .(src_row, nbr_row)]

# Melt the 5 variables for neighbors
vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
nbr_vals <- melt(
  cell_data[, c("row" = .I, vars), with = FALSE],
  id.vars = "row",
  variable.name = "var_name",
  value.name = "value"
)

# Join neighbor values
neighbor_dt <- merge(
  neighbor_dt[, .(src_row, nbr_row)],
  nbr_vals,
  by.x = "nbr_row", by.y = "row",
  allow.cartesian = TRUE
)

# Compute stats by src_row and var_name
agg_stats <- neighbor_dt[
  !is.na(value),
  .(nbr_max = max(value), nbr_min = min(value), nbr_mean = mean(value)),
  by = .(src_row, var_name)
]

# Reshape wide and join back to cell_data
agg_wide <- dcast(
  agg_stats,
  src_row ~ var_name,
  value.var = c("nbr_max", "nbr_min", "nbr_mean")
)

# Merge into original data
cell_data <- cbind(cell_data, agg_wide[order(src_row)][, -"src_row"])
```

---

### **Why This is Faster**
- Eliminates 6.46M per-row loops and repeated filtering.
- Uses `data.table` joins and aggregations in compiled C code.
- Computes all 5 variables in one grouped aggregation instead of 5 full passes.

**Expected speedup:** From 86+ hours to **minutes** on a 16 GB laptop.

**Numerical equivalence:** Same `max`, `min`, `mean` per cell-year and variable as original implementation.  
**Model preserved:** No retraining, only feature computation accelerated.
END_RESPONSE>>>
