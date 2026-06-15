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
run_id: OpenAI_F02_T07_R017
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F02
prompt_family_slug: CODE_ONLY_NO_HINT
family_label: Code only, no hint
family_group: Code-visible neutral
replicate: 17
raw_output_file: Experiment_Output/OpenAI/Temp_0.7/output_F02_OpenAI_temp_07_rep_017.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
The current approach is extremely slow and memory-heavy because:  
- `build_neighbor_lookup` creates a massive list (size â‰ˆ 6.46M) of integer vectors, each requiring a lookup for every row. This is highly inefficient both in time and memory.  
- `compute_neighbor_stats` repeatedly traverses large lists and subsets vectors (`vals[idx]`) for 6.46M rows Ã— 5 variables = ~32M operations.  
- The nested `lapply` and repeated key lookups create significant overhead.  
- Everything is happening in R loops instead of vectorized or compiled operations.  

**Optimization Strategy**  
- **Avoid per-row operations:** Restructure to compute neighbor statistics by *joining* instead of looping.  
- **Leverage `data.table`:** Use fast joins and grouping on a long neighbor table.  
- **Precompute neighbor relationships at the cell level and expand across years:** Instead of building row-specific neighbor lists repeatedly, create a long table: `(cell_id, year, neighbor_id)` and join values once.  
- **Process all variables in a single pass:** Melt data to long format for neighbor stats, compute grouped aggregations, then merge back.  
- **Optional:** Use `fwrite`/`fread` for intermediate steps if memory pressure is high.  

---

### **Optimized Implementation**

```r
library(data.table)

# Convert to data.table
setDT(cell_data)

# Precompute neighbor table (cell_id -> neighbors)
# rook_neighbors_unique: list of neighbors per cell index corresponding to id_order
neighbor_dt <- rbindlist(lapply(seq_along(id_order), function(i) {
  if (length(rook_neighbors_unique[[i]]) == 0) return(NULL)
  data.table(
    id = id_order[i],
    neighbor_id = id_order[rook_neighbors_unique[[i]]]
  )
}))

# Expand neighbors across years
years <- sort(unique(cell_data$year))
neighbor_dt <- neighbor_dt[, .(year = years), by = .(id, neighbor_id)]

# Merge neighbor table with values from cell_data
# Create key for fast joins
setkey(cell_data, id, year)
setkey(neighbor_dt, neighbor_id, year)

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# Melt to long for joining
long_vars <- melt(cell_data[, c("id", "year", neighbor_source_vars), with=FALSE],
                  id.vars = c("id", "year"),
                  variable.name = "var", value.name = "val")

setkey(long_vars, id, year)

# Join neighbor values
neighbor_vals <- neighbor_dt[, .(id, year, var = NULL)] # structure for aggregation
neighbor_vals <- neighbor_dt[long_vars, on = .(neighbor_id = id, year), allow.cartesian=TRUE]
# neighbor_vals now has: id (focal), neighbor_id, year, var, val

# Aggregate neighbor stats by focal cell/year/var
agg_stats <- neighbor_vals[, .(
  neigh_max = max(val, na.rm = TRUE),
  neigh_min = min(val, na.rm = TRUE),
  neigh_mean = mean(val, na.rm = TRUE)
), by = .(id, year, var)]

# Reshape back to wide
agg_wide <- dcast(agg_stats, id + year ~ var,
                  value.var = c("neigh_max", "neigh_min", "neigh_mean"))

# Merge back to cell_data
cell_data <- merge(cell_data, agg_wide, by = c("id", "year"), all.x = TRUE)
```

---

### **Why this works faster**
- **No per-row lapply:** Uses `data.table` grouping and joins, which are highly optimized in C.
- **Single pass for all variables:** Avoids 5 separate loops.
- **Memory efficiency:** Stores neighbor relationships once and uses references internally.
- **Parallelizable:** `data.table` operations can leverage multi-threading automatically.

**Expected improvement:**  
- From 86+ hours to a few minutes (depending on disk speed and CPU cores).
- Fits within 16 GB because intermediate joins are streamed and aggregated efficiently.

**Result:**  
- Preserves trained Random Forest model (no retraining).
- Keeps identical numerical estimands (max, min, mean for neighbors).
- Suitable for large panel datasets on a standard laptop.
END_RESPONSE>>>
