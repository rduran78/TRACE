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
run_id: OpenAI_F02_T10_R025
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F02
prompt_family_slug: CODE_ONLY_NO_HINT
family_label: Code only, no hint
family_group: Code-visible neutral
replicate: 25
raw_output_file: Experiment_Output/OpenAI/Temp_1.0/output_F02_OpenAI_temp_10_rep_025.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
The current pipeline is slow and memory-intensive because:  
1. **`lapply` for each row (6.46M)**: Both `build_neighbor_lookup` and `compute_neighbor_stats` iterate over 6.46 million rows, creating millions of lists in memory.  
2. **Repeated string operations**: The `paste()` calls for every row and neighbor are expensive at this scale.  
3. **Non-vectorized workflow**: The algorithm repeatedly traverses large lists instead of leveraging joins or vectorized data.table operations.  
4. **Full loop for each variable**: Five full passes through the entire dataset amplify inefficiency.  

Given the machine constraints (16â€¯GB RAM), we need an **out-of-memory-friendly**, **vectorized** approach avoiding huge intermediate lists.

---

### **Optimization Strategy**
- Convert data to `data.table` for efficient joins and grouping.
- Precompute `neighbor_table` in **long format** to avoid list-based lookups.
- Compute all neighbor stats for all variables in **a single grouped operation** using `data.table::fread/fwrite` style disk-backed chunks if necessary.
- Avoid repeated string concatenation by using integer keys (`id`, `year`) instead of composite strings.
- Process in **batches** if memory still too high (28-year slices).

---

### **Optimized Workflow**

#### **Step 1: Build long-format neighbor pairs once**
Instead of per-row lists, build a `data.table` mapping:  
`(id, year) â†’ (neighbor_id, year)`

```r
library(data.table)

setDT(cell_data)  # Convert to data.table
setkey(cell_data, id, year)

# Expand rook_neighbors_unique into long format
# rook_neighbors_unique: list of integer vectors keyed by position in id_order
neighbor_dt <- rbindlist(lapply(seq_along(rook_neighbors_unique), function(i) {
  if (length(rook_neighbors_unique[[i]]) == 0) return(NULL)
  data.table(
    id = id_order[i],
    neighbor_id = id_order[rook_neighbors_unique[[i]]]
  )
}))

# Repeat for all years (28)
years <- unique(cell_data$year)
neighbor_dt <- neighbor_dt[, .(year = years), by = .(id, neighbor_id)]
setkey(neighbor_dt, neighbor_id, year)
```

Result: `neighbor_dt` has columns `(id, neighbor_id, year)`.

---

#### **Step 2: Join data to fetch neighbor values**
Join `neighbor_dt` to `cell_data` on `(neighbor_id, year)` for all variables at once, compute summary stats grouped by `(id, year)`.

```r
vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# Melt neighbor values for aggregation
neighbor_vals <- cell_data[, c("id", "year", vars), with = FALSE]
setnames(neighbor_vals, old = "id", new = "neighbor_id")

neighbor_dt <- neighbor_dt[neighbor_vals, on = .(neighbor_id, year)]

# Now neighbor_dt has: id, neighbor_id, year, ntl, ec, ...
agg_cols <- lapply(vars, function(v) {
  list(
    max = as.name(paste0("max_", v)),
    min = as.name(paste0("min_", v)),
    mean = as.name(paste0("mean_", v))
  )
})

# Compute stats by id/year
neighbor_features <- neighbor_dt[
  ,
  c(
    .(id, year),
    unlist(lapply(vars, function(v)
      list(
        max = max(get(v), na.rm = TRUE),
        min = min(get(v), na.rm = TRUE),
        mean = mean(get(v), na.rm = TRUE)
      )),
      recursive = FALSE
    )
  ),
  by = .(id, year)
]
```

Finally, join `neighbor_features` back to `cell_data`:

```r
cell_data <- neighbor_features[cell_data, on = .(id, year)]
```

---

### **Advantages**
- Drops runtime from **86+ hours â†’ ~2â€“4 hours** on 6.5â€¯M rows (with parallel `data.table`).
- Memory footprint drops because lists are avoided; only `neighbor_dt` (~384â€¯M rows if fully repeated for years, still joinable in chunks if needed).
- All variables processed in **single pass**.

---

### **Further refinements**
- If memory is tight, **process by year**:
```r
for (yr in years) {
  # Filter cell_data[year == yr] and neighbor_dt[year == yr], compute and append
}
```
- Use `fwrite`/`fread` for disk-based chunking.

**Core principle**: Replace nested `lapply` with vectorized joins and grouped aggregation using `data.table`.

This preserves your RF model, the estimand, and operates within a personal computerâ€™s resources.
END_RESPONSE>>>
