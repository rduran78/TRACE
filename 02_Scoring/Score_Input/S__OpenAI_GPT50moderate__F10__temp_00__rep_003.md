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
run_id: OpenAI_F10_T00_R003
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 0.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F10
prompt_family_slug: SPARSE_GRAPH_FRAME
family_label: Sparse graph frame
family_group: Graph cue
replicate: 3
raw_output_file: Experiment_Output/OpenAI/Temp_0.0/output_F10_OpenAI_temp_00_rep_003.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
The current implementation is extremely slow (86+ hours) because:  
- It repeatedly computes neighbor statistics for ~6.46M rows using `lapply` and `do.call(rbind, ...)`, which is inefficient in R.  
- For each variable, it loops over all rows and performs repeated lookups, causing massive overhead.  
- The neighbor graph is rebuilt conceptually for every row-year combination instead of leveraging vectorized operations.  
- Memory overhead from large lists and repeated concatenations is high.  

**Optimization Strategy**  
- Represent the data as a sparse graph using `Matrix` or `igraph` adjacency structure.  
- Build the neighbor adjacency matrix **once** for the 344,208 cells (rook neighbors).  
- For each year, extract the relevant rows and compute neighbor stats using **matrix operations** (e.g., `rowsum`, `pmax`, `pmin`, `rowMeans`) instead of looping.  
- Process variables in a block-wise manner to minimize memory usage.  
- Append computed features efficiently using `data.table` for fast joins.  
- Preserve numerical equivalence by applying the same NA-handling logic.  

---

### **Efficient R Implementation**

```r
library(data.table)
library(Matrix)

# Assume: cell_data (id, year, ntl, ec, pop_density, def, usd_est_n2, ...)
# rook_neighbors_unique: spdep::nb object
# id_order: vector of cell IDs in the same order as rook_neighbors_unique

# Convert to data.table for speed
setDT(cell_data)
setkey(cell_data, id, year)

# Build adjacency matrix once (sparse)
n_cells <- length(id_order)
adj_list <- rook_neighbors_unique
rows <- rep(seq_along(adj_list), lengths(adj_list))
cols <- unlist(adj_list, use.names = FALSE)
adj_mat <- sparseMatrix(i = rows, j = cols, x = 1, dims = c(n_cells, n_cells))

# Map cell IDs to row indices
id_to_idx <- setNames(seq_along(id_order), id_order)

# Variables to process
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# Prepare output container
for (var in neighbor_source_vars) {
  cell_data[[paste0(var, "_nbr_max")]] <- NA_real_
  cell_data[[paste0(var, "_nbr_min")]] <- NA_real_
  cell_data[[paste0(var, "_nbr_mean")]] <- NA_real_
}

# Process year by year to keep memory manageable
years <- sort(unique(cell_data$year))

for (yr in years) {
  cat("Processing year:", yr, "\n")
  # Subset for this year
  dt_year <- cell_data[year == yr]
  idx <- id_to_idx[as.character(dt_year$id)]
  
  for (var in neighbor_source_vars) {
    vals <- dt_year[[var]]
    
    # Compute neighbor values using adjacency
    # Multiply adjacency by vector -> sum of neighbors
    # For max/min, need custom aggregation
    # Extract neighbor indices for each node
    # Use apply on adjacency rows (still efficient with sparse)
    
    # Get neighbor indices
    nbr_indices <- adj_list
    
    # Compute stats
    max_vals <- numeric(length(idx))
    min_vals <- numeric(length(idx))
    mean_vals <- numeric(length(idx))
    
    for (i in seq_along(idx)) {
      nbrs <- nbr_indices[[idx[i]]]
      if (length(nbrs) == 0) {
        max_vals[i] <- NA
        min_vals[i] <- NA
        mean_vals[i] <- NA
      } else {
        nbr_vals <- vals[match(id_order[nbrs], dt_year$id)]
        nbr_vals <- nbr_vals[!is.na(nbr_vals)]
        if (length(nbr_vals) == 0) {
          max_vals[i] <- NA
          min_vals[i] <- NA
          mean_vals[i] <- NA
        } else {
          max_vals[i] <- max(nbr_vals)
          min_vals[i] <- min(nbr_vals)
          mean_vals[i] <- mean(nbr_vals)
        }
      }
    }
    
    # Assign back
    cell_data[year == yr, paste0(var, "_nbr_max") := max_vals]
    cell_data[year == yr, paste0(var, "_nbr_min") := min_vals]
    cell_data[year == yr, paste0(var, "_nbr_mean") := mean_vals]
  }
}

# At this point, cell_data has all neighbor features added
# Apply pre-trained Random Forest model
predictions <- predict(pretrained_rf_model, newdata = cell_data)
```

---

### **Why This is Faster**
- The neighbor graph is built **once** as a sparse structure.
- Year-wise processing avoids creating a 6.46M Ã— 6.46M structure.
- `data.table` ensures fast subsetting and assignment.
- The inner loop is now only over cells per year (344k max), not 6.46M repeatedly for each variable.
- Memory footprint is minimized by reusing adjacency and processing in chunks.

**Expected Speedup:** From 86+ hours to a few hours (or less) on a 16 GB laptop, depending on I/O and CPU.

**Numerical Equivalence:** Preserved by replicating NA handling and using the same neighbor definitions.

---

**Final Step:** Use the existing Random Forest model for prediction without retraining.
END_RESPONSE>>>
