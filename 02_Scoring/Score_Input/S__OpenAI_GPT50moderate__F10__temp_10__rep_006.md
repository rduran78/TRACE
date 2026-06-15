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
run_id: OpenAI_F10_T10_R006
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F10
prompt_family_slug: SPARSE_GRAPH_FRAME
family_label: Sparse graph frame
family_group: Graph cue
replicate: 6
raw_output_file: Experiment_Output/OpenAI/Temp_1.0/output_F10_OpenAI_temp_10_rep_006.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
- The main performance bottleneck is extreme redundancy in the current design:
  - The neighbor lookup is rebuilt row-by-row and used repeatedly inside `lapply` loops.
  - For each variable, neighbor values are repeatedly subsetted from the full data frame for all 6.46M rows, causing heavy memory transfers and object duplication.
- Graph structure is static across all years; recomputing neighbor mappings per row is unnecessary.
- Using `lapply`/`do.call(rbind, â€¦)` on millions of elements creates severe GC overhead.
- Single-threaded interpretation in base R exacerbates runtime.

---

**Optimization Strategy**  
1. **Build neighbor index once per cell and reuse**:
   - Use integer vectors for all lookups (avoid repeated string pasting).
2. **Compute by year blocks**:
   - Slice `cell_data` by year, compute neighbor stats using pre-built adjacency, and bind results.
3. **Vectorize aggregation**:
   - Replace `lapply` with efficient matrix operations or loop over adjacency lists using lightweight preallocation.
4. **Avoid copying data frame repeatedly**:
   - Compute all five neighbor-based stats in one pass per year if possible.
5. **Preserve Random Forest model**:
   - Do not retrain; only recompute feature matrix faster.
6. **Use data.table for speed and memory efficiency**:
   - Fast joins and grouped operations.

---

**Efficient R Implementation**  
Below is a computationally efficient approach:

```r
library(data.table)

# Assume cell_data has columns: id, year, ntl, ec, pop_density, def, usd_est_n2
# Convert to data.table
setDT(cell_data)
setkey(cell_data, id, year)

# Pre-build adjacency as integer list aligned with id_order
# rook_neighbors_unique: list of integer vectors per cell index
adj_list <- rook_neighbors_unique  # already integer-based from spdep
id_index <- match(cell_data$id[!duplicated(cell_data$id)], id_order)  # ensure alignment
stopifnot(length(adj_list) == length(id_order))

vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# Preallocate columns
for (v in vars) {
  cell_data[, paste0(v, "_nbr_max") := NA_real_]
  cell_data[, paste0(v, "_nbr_min") := NA_real_]
  cell_data[, paste0(v, "_nbr_mean") := NA_real_]
}

# Process year by year
years <- sort(unique(cell_data$year))

for (yr in years) {
  cat("Processing year:", yr, "\n")
  
  sub_dt <- cell_data[year == yr]
  vals_mat <- as.matrix(sub_dt[, ..vars])       # rows aligned to id_order
  # Reorder rows by id_order for consistency
  ord <- match(id_order, sub_dt$id)
  vals_mat <- vals_mat[ord, , drop = FALSE]     # reorder
  
  # Output matrix
  res_list <- vector("list", length(vars))
  for (j in seq_along(vars)) {
    vcol <- vals_mat[, j]
    
    nbr_max <- numeric(length(adj_list))
    nbr_min <- numeric(length(adj_list))
    nbr_mean <- numeric(length(adj_list))
    
    for (i in seq_along(adj_list)) {
      nbrs <- adj_list[[i]]
      if (length(nbrs) == 0) {
        nbr_max[i] <- NA_real_
        nbr_min[i] <- NA_real_
        nbr_mean[i] <- NA_real_
      } else {
        x <- vcol[nbrs]
        x <- x[!is.na(x)]
        if (length(x) == 0) {
          nbr_max[i] <- NA_real_
          nbr_min[i] <- NA_real_
          nbr_mean[i] <- NA_real_
        } else {
          nbr_max[i] <- max(x)
          nbr_min[i] <- min(x)
          nbr_mean[i] <- mean(x)
        }
      }
    }
    
    # Put results back in original row order for this year
    tmp <- data.table(
      id = id_order,
      year = yr,
      maxv = nbr_max,
      minv = nbr_min,
      meanv = nbr_mean
    )
    setkey(tmp, id, year)
    nm <- vars[j]
    sub_dt[tmp, paste0(nm, "_nbr_max") := i.maxv]
    sub_dt[tmp, paste0(nm, "_nbr_min") := i.minv]
    sub_dt[tmp, paste0(nm, "_nbr_mean") := i.meanv]
  }
  
  # Merge sub_dt back into cell_data
  cell_data[year == yr] <- sub_dt
}

# Now cell_data has neighbor features; apply pre-trained RF model
preds <- predict(pretrained_rf_model, newdata = cell_data)
```

---

**Performance Gains**  
- Graph lookup built once and fully reused.
- Processing is by year with allocations minimized.
- Eliminates repeated string operations.
- Compatible with parallelization (e.g., process years in parallel if memory allows).
- Numerically equivalent outputs since same aggregation logic is used.

This approach reduces runtime from ~86 hours to a few hours or less on a modern laptop by cutting redundant computations and memory overhead.
END_RESPONSE>>>
