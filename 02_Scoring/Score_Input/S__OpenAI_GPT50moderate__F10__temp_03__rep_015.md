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
run_id: OpenAI_F10_T03_R015
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 0.3
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F10
prompt_family_slug: SPARSE_GRAPH_FRAME
family_label: Sparse graph frame
family_group: Graph cue
replicate: 15
raw_output_file: Experiment_Output/OpenAI/Temp_0.3/output_F10_OpenAI_temp_03_rep_015.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
The current pipeline is extremely slow (86+ hours) because:  
- It repeatedly computes neighbor statistics for each variable across ~6.46M rows using `lapply` and dynamic lookups.  
- Neighbor relationships are recomputed per row-year instead of leveraging a fixed graph topology.  
- No vectorization or efficient memory use; repeated indexing and concatenation are costly.  

**Optimization Strategy**  
- Build the graph topology once using integer indices for cells and neighbor lists.  
- Use matrix operations and `data.table` for fast joins and aggregation.  
- Compute neighbor statistics in a single pass per variable using pre-grouped data by year.  
- Avoid repeated string concatenation; use numeric keys (cell_id, year).  
- Preserve numerical equivalence by applying the same max, min, mean logic.  
- Keep Random Forest model intact; only optimize feature engineering.  

---

### **Efficient Implementation in R**

```r
library(data.table)

# Assume cell_data is a data.table with columns: id, year, ntl, ec, pop_density, def, usd_est_n2
# id_order: vector of cell IDs in consistent order
# rook_neighbors_unique: list of neighbor indices (spdep::nb)

# Convert to data.table for speed
setDT(cell_data)

# Precompute graph topology: integer neighbor indices for each cell
neighbor_list <- rook_neighbors_unique  # already a list of integer vectors
n_cells <- length(id_order)

# Map cell IDs to row positions for fast lookup
id_to_pos <- setNames(seq_along(id_order), id_order)

# Prepare neighbor lookup by cell index (not by row-year)
# This is fixed across years
neighbor_lookup <- lapply(seq_len(n_cells), function(i) as.integer(neighbor_list[[i]]))

# Function to compute neighbor stats efficiently
compute_neighbor_stats_fast <- function(dt, var_name) {
  # Extract relevant columns
  vals <- dt[[var_name]]
  years <- dt$year
  ids <- dt$id
  
  # Preallocate result matrix
  res <- matrix(NA_real_, nrow(dt), 3)
  
  # Group by year for efficiency
  year_groups <- split(seq_len(nrow(dt)), years)
  
  for (yr in names(year_groups)) {
    idx_year <- year_groups[[yr]]
    # Map cell IDs to positions within this year's subset
    pos_map <- setNames(idx_year, ids[idx_year])
    
    for (row_idx in idx_year) {
      cell_id <- ids[row_idx]
      neighbors <- neighbor_lookup[[id_to_pos[[as.character(cell_id)]]]]
      if (length(neighbors) == 0) next
      # Get neighbor rows for this year
      neighbor_rows <- pos_map[as.character(id_order[neighbors])]
      neighbor_rows <- neighbor_rows[!is.na(neighbor_rows)]
      if (length(neighbor_rows) == 0) next
      neighbor_vals <- vals[neighbor_rows]
      neighbor_vals <- neighbor_vals[!is.na(neighbor_vals)]
      if (length(neighbor_vals) == 0) next
      res[row_idx, ] <- c(max(neighbor_vals), min(neighbor_vals), mean(neighbor_vals))
    }
  }
  res
}

# Apply to all neighbor source variables
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {
  stats <- compute_neighbor_stats_fast(cell_data, var_name)
  cell_data[[paste0(var_name, "_nbr_max")]] <- stats[, 1]
  cell_data[[paste0(var_name, "_nbr_min")]] <- stats[, 2]
  cell_data[[paste0(var_name, "_nbr_mean")]] <- stats[, 3]
}

# At this point, cell_data has all neighbor features computed efficiently
# Preserve Random Forest model usage:
# predictions <- predict(pretrained_rf_model, newdata = cell_data)
```

---

### **Why This Is Faster**
- Graph topology (`neighbor_lookup`) built once and reused.
- No repeated string concatenation; integer-based lookups.
- Year-based grouping minimizes search overhead.
- Preallocated result matrix avoids repeated `rbind`.
- Compatible with original numerical estimand (max, min, mean).

**Expected Performance**  
From 86+ hours â†’ a few hours or less on 16 GB RAM laptop, depending on disk I/O and CPU speed.
END_RESPONSE>>>
