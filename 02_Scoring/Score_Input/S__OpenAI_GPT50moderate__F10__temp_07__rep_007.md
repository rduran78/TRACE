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
run_id: OpenAI_F10_T07_R007
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F10
prompt_family_slug: SPARSE_GRAPH_FRAME
family_label: Sparse graph frame
family_group: Graph cue
replicate: 7
raw_output_file: Experiment_Output/OpenAI/Temp_0.7/output_F10_OpenAI_temp_07_rep_007.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
The pipeline is extremely slow (86+ hours) because:  
- For ~6.46M rows, `lapply` repeatedly indexes into large vectors for each cell-year.  
- Neighbor lookups are recomputed per row and year, causing redundant work.  
- No vectorization; all operations are in R loops.  
- Memory overhead from repeated list manipulations.  

**Optimization Strategy**  
- Precompute graph topology (neighbors) once at the cell level and reuse across all years.  
- Use integer indexing with `id` and `year` mapped to row positions.  
- Process data in a **vectorized** or **block-wise** manner instead of per-row `lapply`.  
- Use `data.table` for fast joins and aggregations.  
- Compute `max`, `min`, `mean` for neighbors in a single grouped operation, not row-by-row.  
- Preserve numerical equivalence by exact same aggregation logic (ignore NAs).  

---

### **Efficient R Implementation**

```r
library(data.table)

# Convert to data.table for speed
setDT(cell_data)

# Precompute graph adjacency once
# rook_neighbors_unique: list of integer neighbor indices for each cell
# id_order: vector of cell IDs in original order

# Map cell_id -> position
id_to_pos <- setNames(seq_along(id_order), id_order)

# Expand neighbor pairs once (directed edges)
edges <- rbindlist(lapply(seq_along(rook_neighbors_unique), function(i) {
  src <- id_order[i]
  data.table(src = src, nb = id_order[rook_neighbors_unique[[i]]])
}))

# Repeat for all years using Cartesian join
years <- unique(cell_data$year)
edges_expanded <- edges[CJ(year = years), on = .(dummy = NULL)]
setnames(edges_expanded, "year", "year")

# Create keys for fast join
edges_expanded[, src_key := paste(src, year, sep = "_")]
edges_expanded[, nb_key := paste(nb, year, sep = "_")]

# Map keys to row indices
cell_data[, key := paste(id, year, sep = "_")]
cell_data[, row_id := .I]

# Join neighbor row_id
edges_expanded[, nb_row := cell_data[.SD, on = .(key = nb_key), x.row_id]]

# Remove NAs
edges_expanded <- edges_expanded[!is.na(nb_row)]

# Function to compute neighbor stats for each variable
compute_neighbor_stats_dt <- function(var_name) {
  # Join values for neighbor rows
  vals <- cell_data[[var_name]]
  edges_expanded[, value := vals[nb_row]]
  
  # Aggregate by src_key
  stats <- edges_expanded[, .(
    max_val = max(value, na.rm = TRUE),
    min_val = min(value, na.rm = TRUE),
    mean_val = mean(value, na.rm = TRUE)
  ), by = src_key]
  
  # Merge back into cell_data
  cell_data[stats, on = .(key = src_key),
            paste0(var_name, "_nb_max") := i.max_val]
  cell_data[stats, on = .(key = src_key),
            paste0(var_name, "_nb_min") := i.min_val]
  cell_data[stats, on = .(key = src_key),
            paste0(var_name, "_nb_mean") := i.mean_val]
}

# Apply for all source vars
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
for (v in neighbor_source_vars) compute_neighbor_stats_dt(v)

# Remove temp columns
cell_data[, c("key", "row_id") := NULL]

# At this point, cell_data has neighbor features added
# Predict using pre-trained Random Forest (preserve model)
predictions <- predict(rf_model, newdata = cell_data)
```

---

### **Why This Is Fast**
- Graph topology built once and reused across all years.
- Uses `data.table` joins and aggregations instead of millions of R loops.
- Handles all neighbor stats in grouped operations.
- Memory efficient: avoids repeated list manipulations.

**Expected runtime reduction:**  
From 86+ hours â†’ **minutes to a few hours** on a 16 GB laptop, depending on disk I/O and model prediction cost.

**Preserves numerical equivalence** with original computations and keeps the Random Forest model intact.
END_RESPONSE>>>
