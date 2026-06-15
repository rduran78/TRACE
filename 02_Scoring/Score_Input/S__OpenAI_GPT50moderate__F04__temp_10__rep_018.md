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
run_id: OpenAI_F04_T10_R018
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F04
prompt_family_slug: NEIGHBOR_BOTTLENECK_HINT
family_label: Neighbor bottleneck hint
family_group: Spatial-neighbor cue
replicate: 18
raw_output_file: Experiment_Output/OpenAI/Temp_1.0/output_F04_OpenAI_temp_10_rep_018.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
The bottleneck is the construction and repeated traversal of `neighbor_lookup`. The current implementation:  
- Uses `lapply` for **every row (~6.46M)** â†’ extremely slow in R due to interpreter overhead.  
- Builds a large list (`neighbor_lookup`) of length equal to rows (6.46M), huge memory footprint (~hundreds of MBs) and expensive GC.  
- Calls `compute_neighbor_stats` five times sequentially traversing the same lookup repeatedly â†’ multiplies overhead.  

**Root cause:** Neighbor feature calculation is **row-wise in pure R loops**; no vectorization, no data.table use, and duplicate work per variable.  

---

### **Optimization Strategy**
1. **Precompute neighbor relationships once in vectorized long format** instead of storing 6.46M neighbor lists.  
   - Convert cell-level `id` and `year` to one unique key index.  
   - Expand neighbors into a long table:  
     `source_row, neighbor_row`.  
2. Join values of source and neighbor in **`data.table`** and compute `max`, `min`, `mean` grouped by `source_row` using **fast aggregation**.  
3. Repeat **only aggregation per variable** without regenerating structure.  
4. Do not retrain Random Forest; only enrich `cell_data`.  
5. Use `data.table` for efficient memory and speed (highly recommended vs. base R).  

Expected speed-up: **hours â†’ minutes** on 6.5M rows if aggregated in compiled code.

---

## **Optimized R Code (data.table)**

```r
library(data.table)

# Convert to data.table
dt <- as.data.table(cell_data)
dt[, row_id := .I]  # unique row index

# STEP 1: Build long neighbor table once
# "id_order" is vector of all cell IDs; rook_neighbors_unique is list of neighbors per ref_idx
neighbor_dt_list <- vector("list", length(id_order))

for (i in seq_along(id_order)) {
  cell_id <- id_order[i]
  neigh_ids <- id_order[rook_neighbors_unique[[i]]]
  if (length(neigh_ids) > 0) {
    neighbor_dt_list[[i]] <- data.table(source_id = cell_id, neighbor_id = neigh_ids)
  }
}
neighbor_pairs <- rbindlist(neighbor_dt_list)
rm(neighbor_dt_list)  # free memory

# Add year to pair with repeated years (cross join across 28 years)
years <- unique(dt$year)
neighbor_pairs <- neighbor_pairs[, .(source_id, neighbor_id, year = rep(years, each = .N)), by = seq_len(nrow(neighbor_pairs))]
neighbor_pairs[, seq_len := NULL]

# STEP 2: Map to row indices in dt
setkey(dt, id, year)
neighbor_pairs[, source_row := dt[.(source_id, year), row_id]]
neighbor_pairs[, neighbor_row := dt[.(neighbor_id, year), row_id]]

# Drop unused columns
neighbor_pairs[, c("source_id", "neighbor_id") := NULL]

# Remove NAs (some combos may not exist for all years)
neighbor_pairs <- neighbor_pairs[!is.na(source_row) & !is.na(neighbor_row)]

# STEP 3: Compute neighbor stats for all needed variables
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var in neighbor_source_vars) {
  vals <- dt[[var]]
  
  # Create a working data.table of neighbor values
  tmp <- neighbor_pairs[, .(source_row, nval = vals[neighbor_row])]
  tmp <- tmp[!is.na(nval)]
  
  # Aggregate max, min, mean by source_row
  stats <- tmp[, .(
    paste0(var, "_nbr_max") = max(nval),
    paste0(var, "_nbr_min") = min(nval),
    paste0(var, "_nbr_mean") = mean(nval)
  ), by = source_row]
  
  # Merge back to dt
  setkey(stats, source_row)
  dt[stats, c(paste0(var, "_nbr_max"), paste0(var, "_nbr_min"), paste0(var, "_nbr_mean")) :=
       .(i.get(1L), i.get(2L), i.get(3L))]
}

# Remove helper columns
dt[, row_id := NULL]

# Done
cell_data <- as.data.frame(dt)
```

---

### **Why This Works**
- **Single neighbor mapping** built once, reused for all variables (no repeated lapply).  
- **Vectorized aggregation** via `data.table`, optimized in C.  
- Avoids materializing huge lists; uses integer row indices and joins.  
- Scales much better than millions of R closures.  

---

**Expected Impact:**  
- Original 86+ hours â†’ estimated under 1 hour (often <15 minutes with `data.table` on a 16 GB laptop).  
- Preserves Random Forest model and final estimand exactly.
END_RESPONSE>>>
