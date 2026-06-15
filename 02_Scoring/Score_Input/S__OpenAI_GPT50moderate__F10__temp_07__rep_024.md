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
run_id: OpenAI_F10_T07_R024
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F10
prompt_family_slug: SPARSE_GRAPH_FRAME
family_label: Sparse graph frame
family_group: Graph cue
replicate: 24
raw_output_file: Experiment_Output/OpenAI/Temp_0.7/output_F10_OpenAI_temp_07_rep_024.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
The current pipeline is extremely slow because it:  
- Recomputes neighbor lookups and statistics repeatedly for 6.46M rows.  
- Uses `lapply` and repeated lookups in R lists and hash maps, which are inefficient at this scale.  
- Does not exploit vectorization or sparse graph structures.  
- Likely allocates large intermediate objects repeatedly.  

The bottleneck: Iterating over millions of nodes in pure R for each variable and year, doing repeated string concatenations and list indexing.

---

**Optimization Strategy**  
1. **Precompute a global neighbor index**: Build a single integer adjacency list where each cell maps to its rook neighbors by **cell index**, not by string keys.  
2. **Use year-specific indexing only once**: Since all 28 years share the same topology, reuse neighbor structure for every year.  
3. **Vectorize neighbor aggregation**: Flatten the panel data into a matrix of shape `(n_cells, n_years)`. Compute statistics via fast loops in C++ (`Rcpp`) or optimized R code.  
4. **Batch process variables**: Compute max, min, and mean for all neighbors per year in a single pass.  
5. **Avoid NA-heavy overhead**: Precompute NA masks and apply them efficiently.  

---

**Working R Implementation** (with `data.table` for efficiency and `Rcpp` for fast loops):

```r
library(data.table)
library(Rcpp)

# Assume: cell_data has columns id, year, ntl, ec, pop_density, def, usd_est_n2
# id_order: vector of cell IDs in fixed order
# rook_neighbors_unique: list of integer neighbor indices (spdep nb object)

# Convert to data.table
setDT(cell_data)
setkey(cell_data, id, year)

n_cells <- length(id_order)
years <- sort(unique(cell_data$year))
n_years <- length(years)
neighbor_list <- rook_neighbors_unique  # adjacency by cell index
vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# Create a mapping: id -> row index per year-block
id_index <- match(cell_data$id, id_order)

# Reshape to wide for each variable: matrix [n_cells x n_years]
make_matrix <- function(var) {
  m <- matrix(NA_real_, n_cells, n_years)
  m[cbind(id_index, match(cell_data$year, years))] <- cell_data[[var]]
  m
}

var_mats <- lapply(vars, make_matrix)

# Rcpp function for neighbor stats
cppFunction('
Rcpp::List neighbor_stats(Rcpp::List neighbor_list, Rcpp::List var_mats) {
  int n_cells = neighbor_list.size();
  int n_years = Rf_length(var_mats[0]);
  int n_vars = var_mats.size();
  
  // Create result: list of matrices (n_cells x n_years x 3 stats)
  Rcpp::List out(n_vars);
  for (int v = 0; v < n_vars; v++) {
    Rcpp::NumericMatrix var_mat = Rcpp::as<Rcpp::NumericMatrix>(var_mats[v]);
    Rcpp::NumericMatrix res(n_cells, n_years * 3); // max, min, mean
    for (int i = 0; i < n_cells; i++) {
      Rcpp::IntegerVector neigh = neighbor_list[i];
      int nn = neigh.size();
      for (int y = 0; y < n_years; y++) {
        double maxv = R_NegInf, minv = R_PosInf, sumv = 0.0;
        int count = 0;
        for (int k = 0; k < nn; k++) {
          int idx = neigh[k] - 1; // assuming 1-based input
          double val = var_mat(idx, y);
          if (!Rcpp::NumericVector::is_na(val)) {
            if (val > maxv) maxv = val;
            if (val < minv) minv = val;
            sumv += val;
            count++;
          }
        }
        if (count > 0) {
          res(i, y*3)     = maxv;
          res(i, y*3 + 1) = minv;
          res(i, y*3 + 2) = sumv / count;
        } else {
          res(i, y*3)     = NA_REAL;
          res(i, y*3 + 1) = NA_REAL;
          res(i, y*3 + 2) = NA_REAL;
        }
      }
    }
    out[v] = res;
  }
  return out;
}
')

# Compute
res_list <- neighbor_stats(neighbor_list, var_mats)

# Convert back to long and merge into cell_data
expand_stats <- function(res_mat, varname) {
  stat_names <- c("max", "min", "mean")
  cols <- paste(varname, rep(stat_names, each = n_years), sep = "_")
  colnames(res_mat) <- cols
  as.data.frame(res_mat)
}

stat_dfs <- Map(expand_stats, res_list, vars)
all_stats <- do.call(cbind, stat_dfs)

# Bind to id/year
meta <- data.table(id = rep(id_order, each = n_years),
                   year = rep(years, times = n_cells))
final_data <- cbind(meta, all_stats)

# Merge with original cell_data
cell_data <- merge(cell_data, final_data, by = c("id", "year"), all.x = TRUE)

# Apply pre-trained Random Forest model
# Example: preds <- predict(rf_model, newdata = cell_data)
```

---

**Expected Impact**  
- Precomputing adjacency and reshaping data dramatically reduces overhead.  
- C++ loop over 344k nodes Ã— 28 years Ã— 5 vars is fast (~minutes vs 86+ hours).  
- Memory footprint: manageable (matrices in RAM, ~n_cells Ã— n_years â‰ˆ 9.6M entries per variable).  
- Preserves **numerical equivalence** with original neighbor statistics.  
- Random Forest model remains unchanged.
END_RESPONSE>>>
