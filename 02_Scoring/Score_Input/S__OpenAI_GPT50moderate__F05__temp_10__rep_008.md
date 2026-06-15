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
run_id: OpenAI_F05_T10_R008
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F05
prompt_family_slug: STRING_KEY_REDUNDANCY
family_label: String-key probe
family_group: Line-level probe
replicate: 8
raw_output_file: Experiment_Output/OpenAI/Temp_1.0/output_F05_OpenAI_temp_10_rep_008.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**

The inefficiency is **global and algorithmic, not local**.  
The repeated construction of `neighbor_keys` and string-based indexing inside `build_neighbor_lookup` does not happen once; it is executed **6.46 million times** (once per row). For each row, you:

- Build neighbor keys with `paste()`.
- Do hash/map lookups via `idx_lookup[...]`, which reconstruct conversions and uses string keys.

This is a **symptom of a broader pattern**: you are recomputing a string-based lookup map every time instead of working with direct **integer indices**. Since the dataset is fixed, keys such as `id_year` can be encoded as integer offsets, eliminating character concatenation and repeated environment lookups.

Given the 86+ hour runtime estimate, the primary performance bottleneck is:
- N = 6.46 million rows
- Each row checks multiple neighbors (sum of directed neighbor relationships â‰ˆ 1.37 million per year â†’ tens of millions of lookups overall)
- String operations dominate this cost.

---

### **Optimization strategy**

1. **Precompute direct integer indexing using vectorized arithmetic**:
   - Convert `id` into sequential integers (`1..N_cells`).
   - Encode `(id, year)` to **row index in constant time** using:
     ```
     index = (year_index - 1) * N_cells + id_index
     ```
   - This avoids string concatenation and hash lookups entirely.

2. **Precompute neighbor offsets once**:
   - Use rook neighbor relationships (`nb` object) to create a **matrix or list** of neighbor **cell indices** for each cell.
   - For each cell-year row, you can then apply the same neighbor set but in a different year block via simple arithmetic offset.

3. **Compute features in a single vectorized pass per variable using base vectorization or matrix ops**:
   - Use integer indexing into `vals[...]` instead of repeated string maps.

4. Keep Random Forest model unchanged and preserve the feature definitions; only speed up pipeline.

---

### **Proposed Efficient Implementation**

Assumptions:
- `id_order` is a vector of unique cell IDs of length `n_cells`.
- `cell_data` has columns: `id` (matching `id_order`), `year`, and all variables.
- `rook_neighbors_unique` is a `spdep::nb` neighbor list of length `n_cells`.
- `years` is sorted and unique.

#### Step 1: Encode IDs and years

```r
n_cells <- length(id_order)
years   <- sort(unique(cell_data$year))
n_years <- length(years)

# Maps for integer encoding
id_to_idx    <- setNames(seq_len(n_cells), id_order)
year_to_idx  <- setNames(seq_len(n_years), years)

cell_data$id_idx   <- id_to_idx[as.character(cell_data$id)]
cell_data$year_idx <- year_to_idx[as.character(cell_data$year)]
```

Row index formula:
```
row_index = (year_idx - 1) * n_cells + id_idx
```

We avoid any string concatenation after this.

---

#### Step 2: Build neighbor index structure ONCE

```r
# rook_neighbors_unique: list of neighbor indices (integers into id_order)
neighbor_list <- rook_neighbors_unique  # already integer indices if original nb was built from cell index

# When expanded to all years: neighbors for (cell, year) differ only by offset per year
# Offsets per year block:
year_offsets <- (seq_len(n_years) - 1) * n_cells
```

---

#### Step 3: Build a lookup list of row indices for ALL rows

Instead of lapply over 6.46M rows, we leverage vector arithmetic:

```r
row_ids <- seq_len(nrow(cell_data))

# Preallocate neighbor lookup as a list of length nrow(cell_data)
neighbor_lookup <- vector("list", length = nrow(cell_data))

id_idx_vec   <- cell_data$id_idx
year_idx_vec <- cell_data$year_idx

for (i in seq_along(year_offsets)) {
  y_offset <- year_offsets[i]
  year_rows <- which(year_idx_vec == i)
  # For each of these rows, compute neighbors
  for (r in year_rows) {
    neighbors_for_cell <- neighbor_list[[id_idx_vec[r]]]  # neighbor cell indices
    if (length(neighbors_for_cell) > 0) {
      neighbor_lookup[[r]] <- neighbors_for_cell + y_offset
    }
  }
}
```

But this double loop is still largeâ€”next step simplifies further:

---

#### Step 4: Purely formula-based index computation

Instead of storing 6.46M lists, compute on the fly in `compute_neighbor_stats_using_offsets()` using offsets:

---

#### Final Vectorized Functions

```r
compute_neighbor_stats_fast <- function(data, neighbor_list, n_cells, n_years, var_name) {
  vals <- data[[var_name]]
  id_idx <- data$id_idx
  year_idx <- data$year_idx
  result <- matrix(NA_real_, nrow(data), 3)  # columns: max, min, mean
  
  for (year_i in seq_len(n_years)) {
    y_offset <- (year_i - 1) * n_cells
    rows_this_year <- which(year_idx == year_i)
    
    for (r in rows_this_year) {
      neighbors <- neighbor_list[[id_idx[r]]]
      if (!length(neighbors)) next
      # Compute neighbor absolute row indices
      idx <- neighbors + y_offset
      neighbor_vals <- vals[idx]
      neighbor_vals <- neighbor_vals[!is.na(neighbor_vals)]
      if (!length(neighbor_vals)) next
      result[r, ] <- c(max(neighbor_vals), min(neighbor_vals), mean(neighbor_vals))
    }
  }
  
  result
}
```

---

### **Putting it Together**

```r
n_cells <- length(id_order)
years   <- sort(unique(cell_data$year))
n_years <- length(years)

id_to_idx    <- setNames(seq_len(n_cells), id_order)
year_to_idx  <- setNames(seq_len(n_years), years)

cell_data$id_idx   <- id_to_idx[as.character(cell_data$id)]
cell_data$year_idx <- year_to_idx[as.character(cell_data$year)]

neighbor_list <- rook_neighbors_unique  # integer-based nb list

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {
  stats_mat <- compute_neighbor_stats_fast(cell_data, neighbor_list, n_cells, n_years, var_name)
  colnames(stats_mat) <- paste0(var_name, c("_nbr_max", "_nbr_min", "_nbr_mean"))
  cell_data <- cbind(cell_data, stats_mat)
}
```

---

### **Performance Impact**

- Eliminates 6.46M `paste()` calls.
- Eliminates string-based hash lookups (`idx_lookup[...]`).
- Memory footprint smaller: no giant character vectors, only integer mapping.
- Time complexity drops to O(N + total_edges) rather than O(N Ã— string_ops).
- On 6.46M rows, expect **runtime in hours â†’ minutes**, even in R.

---

âœ… **Preserves original estimand** (same neighbors and year alignment).  
âœ… **Works with existing model** (adds same neighbor-based features, no retraining).  
âœ… **Memory-efficient** for 16 GB machine.

**Bottom line**: The repeated `paste()` and character lookup indicate a fundamental algorithmic inefficiency. Moving to integer-based indexing with arithmetic offsets converts the process from string-hash driven to numeric O(1) lookups, reducing runtime from ~86 hours to a tractable few minutes.
END_RESPONSE>>>
