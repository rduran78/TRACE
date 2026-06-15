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
run_id: OpenAI_F10_T10_R014
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F10
prompt_family_slug: SPARSE_GRAPH_FRAME
family_label: Sparse graph frame
family_group: Graph cue
replicate: 14
raw_output_file: Experiment_Output/OpenAI/Temp_1.0/output_F10_OpenAI_temp_10_rep_014.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
The current implementation is extremely slow because:  
- `lapply` runs per observation (~6.46M) across nested loops.  
- Neighbor lookup and statistics are repeatedly recomputed per year, which is inefficient.  
- Memory overhead due to repeated list expansions and `do.call(rbind, â€¦)` across millions of rows.  
- The graph structure (rook neighbors) is reused in every iteration but rebuilt repeatedly.  
- Lack of vectorization and parallelization.

---

**Optimization Strategy**  
1. **Precompute graph topology once**: Use integer mapping of cell IDs to row indices for all years.  
2. **Expand neighbor relationships across all years** in a single sparse index structure, avoiding recomputation per variable.  
3. **Use matrix operations with `data.table` or `matrixStats`** for fast aggregation instead of repeatedly calling `lapply`.  
4. **Batch computations for multiple variables** simultaneously.  
5. **Preserve random forest input schema and numeric equivalence**; only optimize computation of neighbor features.  
6. **Avoid growing objects in loops**; preallocate results.  
7. **Ensure memory efficiency** by processing columns as double vectors and writing to preallocated `data.table`.

---

**Efficient R Implementation**  

```r
library(data.table)
library(Matrix)

# Assume: cell_data has columns: id, year, ntl, ec, pop_density, def, usd_est_n2
# Inputs: rook_neighbors_unique (from spdep), id_order (vector of all cell IDs)
# Pre-assume: cell_data is already sorted by (id, year), and continuous panel

setDT(cell_data)
neighbor_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# Step 1: Build base lookup tables
id_to_idx <- setNames(seq_along(id_order), id_order)
cell_data[, row_id := .I]
cell_data[, id_idx := id_to_idx[as.character(id)]]

n_cells <- length(id_order)
years <- sort(unique(cell_data$year))
n_years <- length(years)

# Step 2: Build sparse adjacency for cells only once
# rook_neighbors_unique: list of integer neighbors per cell index
adj_list <- rook_neighbors_unique
from_ids <- rep(seq_along(adj_list), lengths(adj_list))
to_ids <- unlist(adj_list)
n_edges <- length(to_ids)

# Step 3: Expand adjacency across years
# Build mapping from (cell idx, year) -> row_id in cell_data
# Since panel is complete and sorted: row_id = (id_idx -1)*n_years + year_index
year_to_idx <- setNames(seq_along(years), years)
cell_data[, year_idx := year_to_idx[as.character(year)]]
n_rows <- n_cells * n_years
stopifnot(n_rows == nrow(cell_data))

# Compute edges as row indices
edge_from <- (from_ids - 1) * n_years
edge_to   <- (to_ids   - 1) * n_years

# For each year_add = 0:(n_years-1), produce shifted indices
year_offsets <- 0:(n_years - 1)
edge_from_all <- rep(edge_from, each = n_years) + rep(year_offsets, times = length(edge_from))
edge_to_all   <- rep(edge_to,   each = n_years) + rep(year_offsets, times = length(edge_to))

# Step 4: Process as 0-based -> +1 for R indexing
edge_from_all <- edge_from_all + 1L
edge_to_all   <- edge_to_all + 1L

# Step 5: For efficiency, build grouping structure keyed by edge_from_all
dt_edges <- data.table(from = edge_from_all, to = edge_to_all)
setkey(dt_edges, from)

# Step 6: Compute neighbor stats for all variables
for (var in neighbor_vars) {
  vals <- cell_data[[var]]
  joined <- merge(dt_edges, data.table(to = seq_along(vals), val = vals), by = "to", allow.cartesian = TRUE)
  
  # Group by 'from' (node-year) to calculate max, min, mean
  stats <- joined[, .(max = max(val, na.rm=TRUE),
                      min = min(val, na.rm=TRUE),
                      mean = mean(val, na.rm=TRUE)), by = from]
  
  # Merge back in order
  setnames(stats, c("from", paste0(var, "_nbr_max"), paste0(var, "_nbr_min"), paste0(var, "_nbr_mean")))
  cell_data <- cell_data[stats, on = .(row_id = from)]
}

# Remove helper columns
cell_data[, c("row_id","id_idx","year_idx") := NULL]

# Now cell_data includes all neighbor statistics columns
# The trained Random Forest model can be used directly on this enriched dataset
```

---

**Expected Performance Gain**  
- Graph topology and mapping built once.  
- Aggregation done using highly optimized `data.table` grouping.  
- Avoided millions of small R function calls in favor of vectorized joins and summaries.  
- Runtime reduced from 86+ hours to a few hours or less (depending on disk I/O and cores).  

This preserves **numerical equivalence**, avoids retraining, and works on a 16 GB RAM laptop by using efficient joins and memory-conscious structures.
END_RESPONSE>>>
