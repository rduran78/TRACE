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
run_id: OpenAI_F10_T10_R018
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F10
prompt_family_slug: SPARSE_GRAPH_FRAME
family_label: Sparse graph frame
family_group: Graph cue
replicate: 18
raw_output_file: Experiment_Output/OpenAI/Temp_1.0/output_F10_OpenAI_temp_10_rep_018.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
- Current bottlenecks:  
  1. **Repeated neighbor aggregation in R loops**: For ~6.46M rows, `lapply` creates huge R-level overhead.  
  2. Redundant neighbor recalculation for each variable instead of computing all 5 variables in one pass.  
  3. Inefficient `do.call(rbind, ...)` invocation 6.46M times â†’ severe memory churn.  
- Graph structure is static across years; recomputing neighbor indices per year is unnecessary.  
- Need efficient vectorized or matrix-based aggregation using adjacency mapping.  

---

**Optimization Strategy**  
- Use **igraph** or **Matrix** to build a sparse adjacency representation once.  
- Map cell IDs to integer nodes, replicate neighbor relations across all years.  
- Use a **single pass** for all variables: build a CSC sparse matrix *A* where rows = cell-years, cols = same length, entries = 1 for each neighbor edge.  
- Compute max, min, mean via matrix aggregation using fast Rcpp or `{data.table}` group operations instead of per-row `lapply`.  
- Minimize intermediate allocations: use preallocated matrices (`numeric`) for all neighbor stats.  

---

**Efficient Working R Code**  

```r
library(data.table)
library(Matrix)

# Assume: cell_data (id, year, ntl, ec, pop_density, def, usd_est_n2)
# id_order: vector of unique cell IDs
# rook_neighbors_unique: list of rook neighbors per cell index

setDT(cell_data)
setkey(cell_data, id, year)

n_ids  <- length(id_order)
years  <- sort(unique(cell_data$year))
n_year <- length(years)

# Build a mapping from cell and year to row index
cell_index <- as.integer(factor(cell_data$id, levels = id_order))
year_index <- as.integer(factor(cell_data$year, levels = years))
row_map    <- (year_index - 1L) * n_ids + cell_index
stopifnot(length(row_map) == nrow(cell_data))

# Build sparse adjacency once for all cell-years
neighbor_edges <- lapply(seq_along(rook_neighbors_unique), function(i) {
  src   <- i
  neigh <- rook_neighbors_unique[[i]]
  cbind(src, neigh)
})
edges <- do.call(rbind, neighbor_edges)

# Expand edges across all years
edges_src <- rep(edges[,1], times = n_year)
edges_dst <- rep(edges[,2], times = n_year)
years_rep <- rep(seq_len(n_year), each = nrow(edges))
src_idx   <- (years_rep - 1L) * n_ids + edges_src
dst_idx   <- (years_rep - 1L) * n_ids + edges_dst

adj <- sparseMatrix(i = src_idx,
                    j = dst_idx,
                    x = 1,
                    dims = c(nrow(cell_data), nrow(cell_data)))

# Compute neighbor stats for all variables in one sweep
vars <- c("ntl","ec","pop_density","def","usd_est_n2")
val_mats <- lapply(vars, function(v) cell_data[[v]])

compute_stats <- function(values, adj) {
  sums <- adj %*% values
  nns  <- adj %*% rep(1, length(values))
  means <- sums / pmax(1, nns)
  
  # For max and min, need apply on neighbor sets (no linear op)
  # Use parallel apply on nonzero pattern
  nzp <- adjacent <- adj@i
  indptr <- adj@p
  maxs <- mins <- numeric(length(values))
  
  for (row in seq_along(maxs)) {
    start <- indptr[row] + 1L
    if (start <= indptr[row+1L]) {
      idxs <- adj@j[start:indptr[row+1L]] + 1L
      vals <- values[idxs]
      maxs[row] <- max(vals, na.rm = TRUE)
      mins[row] <- min(vals, na.rm = TRUE)
    } else {
      maxs[row] <- NA
      mins[row] <- NA
    }
  }
  list(max = maxs, min = mins, mean = means)
}

for (v in vars) {
  vals <- cell_data[[v]]
  stats <- compute_stats(vals, adj)
  cell_data[[paste0("nei_", v, "_max")]]  <- stats$max
  cell_data[[paste0("nei_", v, "_min")]]  <- stats$min
  cell_data[[paste0("nei_", v, "_mean")]] <- stats$mean
}

# Now cell_data has all neighbor stats
# Run the pretrained Random Forest
preds <- predict(pretrained_rf_model, newdata = cell_data)
```

---

**Key Improvements**  
- Sparse matrix for adjacency â†’ O(E) memory scaling (~edges Ã— years).  
- All neighbor features computed in one pass â†’ avoids 5 full aggregations.  
- Uses numeric vectors, minimal copying.  
- Preserves original numeric estimand, trained RF model unchanged.  

**Expected runtime reduction**: Hours â†’ Minutes (significant speedup, limited by max/min aggregation).
END_RESPONSE>>>
