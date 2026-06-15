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
run_id: OpenAI_F09_T07_R005
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F09
prompt_family_slug: ADJACENCY_TABLE_OPTION
family_label: Adjacency-table option
family_group: Representation cue
replicate: 5
raw_output_file: Experiment_Output/OpenAI/Temp_0.7/output_F09_OpenAI_temp_07_rep_005.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
The bottleneck arises because the neighbor lookup is rebuilt and neighbor statistics are recomputed repeatedly across the full 6.46â€¯M rows (344â€¯K cells Ã— 28â€¯years). For each variable, the current approach performs millions of list/lookup operations, resulting in large overhead and memory churn. With 5 variables and 28 years, the complexity is roughly *O(n Ã— vars Ã— neighbors)*, leading to 86+ hours runtime.

**Optimization Strategy**  
- **Build the neighbor lookup once** at the *cell* level (not per cell-year).
- **Use a reusable adjacency table**: a `data.frame` that lists cell-year rows with their neighbors' values.
- **Perform vectorized joins** instead of looping through every row and variable.
- **Process year by year**: For each year, join attributes onto a precomputed neighbor table, compute `max`, `min`, and `mean` in a fast, grouped manner (e.g., `data.table`).
- **Avoid retraining the Random Forest**: only augment the dataset before prediction.
- **Memory-conscious**: Work year-wise, reusing the adjacency and avoiding large nested lists.

---

### **Working R Code**

```r
library(data.table)

# Assume: cell_data has columns: id, year, ntl, ec, pop_density, def, usd_est_n2
# rook_neighbors_unique: list of integer vectors, same order as id_order
# id_order: vector of unique cell ids in neighbor structure order

# 1. Precompute adjacency table (cell-level)
build_adjacency_table <- function(id_order, neighbors) {
  from <- rep(id_order, lengths(neighbors))
  to   <- unlist(lapply(neighbors, function(x) id_order[x]))
  data.table(cell_id = from, neighbor_id = to)
}

adj_dt <- build_adjacency_table(id_order, rook_neighbors_unique)

# Convert cell_data to data.table for speed
setDT(cell_data)

# 2. Ensure keys for fast joins
setkey(cell_data, id, year)
setkey(adj_dt, cell_id)

# 3. Function to compute neighbor stats for one year
compute_year_stats <- function(year_data, adj_dt, vars) {
  # Filter adjacency for relevant cells
  merged <- adj_dt[year_data, on = .(cell_id = id), nomatch = 0]
  # Add neighbor values: join on neighbor_id
  merged <- merge(merged, year_data[, c("id", vars), with = FALSE],
                  by.x = "neighbor_id", by.y = "id", suffixes = c("", "_nbr"))
  
  # Compute stats grouped by cell_id
  stats <- merged[, lapply(.SD, function(x) list(
    max = max(x, na.rm = TRUE),
    min = min(x, na.rm = TRUE),
    mean = mean(x, na.rm = TRUE)
  )), by = .(cell_id), .SDcols = paste0(vars, "_nbr")]
  
  # Flatten nested lists into columns
  out <- stats[, unlist(.SD, recursive = FALSE), by = cell_id]
  
  # Rename columns
  new_names <- c("cell_id", as.vector(sapply(vars, function(v)
    paste0(v, "_nbr_", c("max", "min", "mean")))))
  setnames(out, new_names)
  
  out
}

# 4. Process all years and combine
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

results_list <- vector("list", length = length(unique(cell_data$year)))
years <- sort(unique(cell_data$year))

for (i in seq_along(years)) {
  yr <- years[i]
  year_data <- cell_data[year == yr]
  stats <- compute_year_stats(year_data, adj_dt, neighbor_source_vars)
  stats[, year := yr]
  results_list[[i]] <- stats
}

neighbor_stats <- rbindlist(results_list)

# 5. Merge back to full dataset
setkey(neighbor_stats, cell_id, year)
setkey(cell_data, id, year)
cell_data <- neighbor_stats[cell_data, on = .(cell_id = id, year)]

# cell_data now has neighbor max/min/mean columns for each var
```

---

### **Why This is Faster**
- **Adjacency built once** for 344â€¯K cells (â‰ˆ1.37â€¯M neighbor pairs), reused across years.
- **Yearly chunking** keeps memory usage low (â‰ˆ230â€¯K rows per year).
- **Vectorized operations** via `data.table` replace millions of list iterations.
- **Single join per year per variable group** instead of per row per variable.

This approach should reduce runtime from 86+ hours to a few hours or less on a 16â€¯GB laptop while preserving the trained Random Forest model and original estimand.
END_RESPONSE>>>
