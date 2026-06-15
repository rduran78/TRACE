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
run_id: OpenAI_F10_T10_R005
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F10
prompt_family_slug: SPARSE_GRAPH_FRAME
family_label: Sparse graph frame
family_group: Graph cue
replicate: 5
raw_output_file: Experiment_Output/OpenAI/Temp_1.0/output_F10_OpenAI_temp_10_rep_005.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
- The bottleneck comes from repeatedly building neighbor lookups and computing statistics for each observation using `lapply` over all 6.46M rows and 1.37M neighbor links in pure R loops.  
- Memory overhead is high due to R list expansions and redundant computations across 28 years.  
- Current design does not leverage vectorization or sparse matrix operations and fails to reuse computations per variable efficiently.  

---

**Optimization Strategy**  
- Construct a fixed sparse adjacency matrix **once** for all cells (344,208 nodes).  
- Neighbor aggregation can be done as matrix multiplications on a year-by-year basis using `Matrix` (sparse operations).  
- Compute `sum`, `count`, `min`, `max` without looping rows individually. `mean` = `sum / count`.  
- Avoid re-reading neighbors or recalculating indexes per iteration.  
- Use `data.table` for fast slicing by year.  
- Preserve numerical results exactly (NA handling included).  

---

**Efficient R Implementation**

```r
library(data.table)
library(Matrix)

# Assumptions:
# cell_data with columns: id, year, ntl, ec, pop_density, def, usd_est_n2
# id_order: vector of unique cell IDs in canonical order (length = N)
# rook_neighbors_unique: spdep::nb list of neighbors for each element in id_order

# -------------------------
# 1. Build sparse adjacency once
# -------------------------
N <- length(id_order)
src <- rep(seq_along(rook_neighbors_unique), lengths(rook_neighbors_unique))
dst <- unlist(rook_neighbors_unique, use.names = FALSE)
adj <- sparseMatrix(i = src, j = dst, x = 1, dims = c(N, N))

# -------------------------
# 2. Prepare data
# -------------------------
setDT(cell_data)
setkey(cell_data, id, year)
years <- sort(unique(cell_data$year))
var_list <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# To map id to [1..N]
id_map <- setNames(seq_along(id_order), id_order)

# -------------------------
# 3. Function to compute neighbor stats per year and variable
# -------------------------
compute_neighbor_stats_year <- function(dt_year, var_name) {
  vals <- rep(NA_real_, N)
  vals[id_map[dt_year$id]] <- dt_year[[var_name]]
  # Replace NA with 0 for sum but track non-missing counts
  non_na <- !is.na(vals)
  vals_na0 <- vals; vals_na0[!non_na] <- 0
  
  # Counts of non-NA neighbors
  count <- adj %*% Matrix(as.numeric(non_na), ncol = 1)
  
  # Sum for mean
  sum_x <- adj %*% Matrix(vals_na0, ncol = 1)
  
  mean_x <- sum_x
  mean_x[count > 0] <- sum_x[count > 0] / count[count > 0]
  mean_x[count == 0] <- NA_real_
  
  # Max/Min: compute only among non-NA neighbors
  # Efficient way: iterate neighbors, but in C++ or collapse:
  # Here use spApply for simplicity with summary:
  # We'll implement in pure R:
  res_max <- rep(NA_real_, N)
  res_min <- rep(NA_real_, N)
  
  for (i in which(rowSums(adj) > 0)) {
    neigh <- which(adj[i, ] != 0)
    neigh_vals <- vals[neigh]; neigh_vals <- neigh_vals[!is.na(neigh_vals)]
    if (length(neigh_vals) > 0) {
      res_max[i] <- max(neigh_vals)
      res_min[i] <- min(neigh_vals)
    }
  }
  
  data.table(
    id = dt_year$id,
    year = dt_year$year,
    paste0(var_name, "_nbr_max") := res_max[id_map[dt_year$id]],
    paste0(var_name, "_nbr_min") := res_min[id_map[dt_year$id]],
    paste0(var_name, "_nbr_mean") := as.numeric(mean_x[id_map[dt_year$id]])
  )
}

# -------------------------
# 4. Loop over years and merge results
# -------------------------
result_list <- vector("list", length(years))
for (yi in seq_along(years)) {
  y <- years[yi]
  dt_year <- cell_data[year == y]
  stats_list <- lapply(var_list, function(v) compute_neighbor_stats_year(dt_year, v))
  dt_merged <- Reduce(function(a, b) merge(a, b, by = c("id", "year")), stats_list)
  result_list[[yi]] <- dt_merged
}

neighbor_features <- rbindlist(result_list)
# Merge back to cell_data
cell_data <- merge(cell_data, neighbor_features, by = c("id", "year"), all.x = TRUE)
```

---

**Key Gains**  
- Adjacency built once and reused -> huge time/memory savings.  
- Vectorized sum/mean via sparse matrix multiplication.  
- `max`/`min` still requires row-wise scan but only per node (344K) not per cell-year (6.46M). This is 28Ã— faster than per-record scanning.  
- Memory footprint controlled by `Matrix` and `data.table`.  
- Fully preserves original estimand and works with pre-trained Random Forest without retraining.  

---

**Expected Runtime**: Reduced from ~86 hours to a few hours on 16 GB RAM laptop, primarily dominated by max/min pass. Further optimizations possible in C++ via `Rcpp` or using `igraph::graph_apply`.
END_RESPONSE>>>
