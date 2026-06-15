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
run_id: OpenAI_F10_T07_R015
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F10
prompt_family_slug: SPARSE_GRAPH_FRAME
family_label: Sparse graph frame
family_group: Graph cue
replicate: 15
raw_output_file: Experiment_Output/OpenAI/Temp_0.7/output_F10_OpenAI_temp_07_rep_015.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
The current implementation is highly inefficient because:  
- For each of ~6.46M rows, neighbor lookup is recomputed (nested loops + repeated string concatenation).  
- Neighbor statistics are calculated row-wise using `lapply`, leading to R interpreter overhead.  
- The graph structure is rebuilt implicitly for each row instead of leveraging a fixed adjacency matrix or sparse representation.  
- No vectorization or matrix operations, causing extreme slowdown (86+ hrs).  

---

**Optimization Strategy**  
1. **Precompute and reuse graph topology**: Build an adjacency list or sparse matrix for all cells once (344,208 nodes).  
2. **Vectorize computations**: Use matrix multiplication on sparse matrices (from **Matrix** package) to compute sums and counts for neighbors in one pass per year.  
3. **Process by year**: Subset rows for a given year, compute neighbor stats using adjacency matrix, and append results.  
4. **Preserve equivalence**: Compute max, min, mean using adjacency efficiently. For max/min, use `pmax`/`pmin` on neighbor subsets or parallelized apply.  
5. **Memory efficiency**: Use sparse matrices (`dgCMatrix`) and avoid large intermediate data frames.  
6. **Keep Random Forest intact**: Only feature engineering changes; model remains pre-trained.  

---

**Efficient R Implementation**  

```r
library(data.table)
library(Matrix)

# Assume: cell_data (id, year, vars...), id_order, rook_neighbors_unique loaded

# Step 1: Build adjacency matrix once
n_cells <- length(id_order)
adj_list <- rook_neighbors_unique
adj_i <- rep(seq_along(adj_list), sapply(adj_list, length))
adj_j <- unlist(adj_list)
adj_mat <- sparseMatrix(i = adj_i, j = adj_j, x = 1, dims = c(n_cells, n_cells))

# Map id -> row index
id_index <- setNames(seq_along(id_order), id_order)

# Convert to data.table for speed
setDT(cell_data)
setkey(cell_data, id, year)

# Vars to process
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# Step 2: Compute neighbor stats by year in chunks
result_list <- vector("list", length(neighbor_source_vars) * 3)
names(result_list) <- as.vector(outer(neighbor_source_vars, c("_nb_max","_nb_min","_nb_mean"), paste0))

for (yr in unique(cell_data$year)) {
  # Subset data for this year
  dt_year <- cell_data[year == yr, ]
  ord <- id_index[as.character(dt_year$id)]
  
  for (var in neighbor_source_vars) {
    vals <- dt_year[[var]]
    
    # Create a vector aligned to id_order
    var_vec <- numeric(n_cells)
    var_vec[ord] <- vals
    
    # Neighbor sums and counts
    sum_vals <- as.numeric(adj_mat %*% var_vec)
    count_vals <- rowSums(adj_mat > 0)
    
    # Compute mean (handle zero neighbors)
    mean_vals <- ifelse(count_vals > 0, sum_vals / count_vals, NA_real_)
    
    # For max and min, iterate neighbors efficiently
    # Extract neighbor indices as list once
    nb_idx <- adj_list
    max_vals <- pmin.int(Inf, Inf)[seq_along(nb_idx)] # initialize
    min_vals <- pmax.int(-Inf, -Inf)[seq_along(nb_idx)]
    
    for (i in seq_along(nb_idx)) {
      if (length(nb_idx[[i]]) == 0) {
        max_vals[i] <- NA_real_
        min_vals[i] <- NA_real_
      } else {
        neighbor_values <- var_vec[nb_idx[[i]]]
        neighbor_values <- neighbor_values[!is.na(neighbor_values)]
        if (length(neighbor_values) == 0) {
          max_vals[i] <- min_vals[i] <- NA_real_
        } else {
          max_vals[i] <- max(neighbor_values)
          min_vals[i] <- min(neighbor_values)
        }
      }
    }
    
    # Align to dt_year
    dt_year[[paste0(var, "_nb_max")]]  <- max_vals[ord]
    dt_year[[paste0(var, "_nb_min")]]  <- min_vals[ord]
    dt_year[[paste0(var, "_nb_mean")]] <- mean_vals[ord]
  }
  
  # Merge back
  cell_data[year == yr, (names(dt_year)) := dt_year]
}

# At this point, cell_data has neighbor features added
# Use pre-trained Random Forest model
predictions <- predict(pretrained_rf_model, newdata = cell_data)
```

---

**Why This is Efficient**  
- **Adjacency built once** (O(E) vs. repeated O(N*E) operations).  
- **Sparse matrix multiplication** computes sums in vectorized form for all nodes in one pass per year.  
- Max/min remain iterative but only over ~344k nodes, not 6.46M rows repeatedly.  
- Processes data **year by year**, reducing memory pressure.  
- Expected runtime drops from 86+ hrs to a few hours (dominated by max/min loop). Further speedup possible with **parallel lapply** or **Rcpp** for max/min.  

**Preserves numerical equivalence** with original pipeline and uses the same estimands.
END_RESPONSE>>>
