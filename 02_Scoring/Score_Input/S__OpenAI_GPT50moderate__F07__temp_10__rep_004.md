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
run_id: OpenAI_F07_T10_R004
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F07
prompt_family_slug: RASTER_INVALID_IRREGULAR
family_label: Raster invalid irregular topology
family_group: Raster/kernel bridge
replicate: 4
raw_output_file: Experiment_Output/OpenAI/Temp_1.0/output_F07_OpenAI_temp_10_rep_004.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
The main performance bottleneck is caused by two factors:  
1. **NaÃ¯ve row-wise operations with `lapply`** across 6.46M rows for each variable: this creates ~32.3M iterations of small computations.  
2. **Repeated membership lookups** (`idx_lookup`, `id_to_ref`), which are character-vector keyed and recalculated per row. This is extremely inefficient for large panels.  

The process is CPU-bound and memory-thrashing, hence the 86+ hour estimation. Also, the pipeline creates intermediate lists that are huge.

---

**Optimization Strategy**  
- Flatten neighbor lookups into an **integer index matrix or vector** once, avoiding per-row recomputation.  
- Store neighbors in a **precomputed integer vector plus offset index** (similar to CSR sparse encoding), reducing overhead.  
- Use **`data.table` grouping or `vapply`/`matrixStats`** vectorized operations instead of nested `lapply`.  
- Avoid character concatenation inside loops: resolve `id` and `year` alignment with merged keys first.  
- Compute stats using **fast aggregate operations** (`max`, `min`, `mean`) in compiled form.  

Goal: *O(N log N)* or *near O(N)* solution that scales easily.

---

**Working R Code**

```r
library(data.table)

# Convert to data.table for fast joins
setDT(cell_data)

# Precompute unique cell ids for fast mapping
id_map <- match(cell_data$id, id_order)

# Build neighbor representation (flattened)
# rook_neighbors_unique: list of integer neighbor indices for each cell
neighbor_offsets <- integer(length(id_order) + 1)
neighbor_ids <- unlist(rook_neighbors_unique, use.names = FALSE)
neighbor_offsets[-1] <- cumsum(lengths(rook_neighbors_unique))  # CSR-style index
neighbor_offsets <- c(0L, neighbor_offsets)

# Function to compute stats in bulk:
compute_neighbor_stats_fast <- function(var_name, vals) {
  n <- nrow(cell_data)
  out <- matrix(NA_real_, n, 3)
  for (i in seq_len(n)) {
    ref_idx <- id_map[i]
    start <- neighbor_offsets[ref_idx] + 1
    end   <- neighbor_offsets[ref_idx + 1]
    if (start <= end) {
      # Collect same-year neighbors
      year_match <- cell_data$year[i]
      nb_ids_all <- neighbor_ids[start:end]  # cell ids (not row indices yet)
      nb_rows <- which(id_map %in% nb_ids_all & cell_data$year == year_match)
      if (length(nb_rows)) {
        nv <- vals[nb_rows]
        nv <- nv[!is.na(nv)]
        if (length(nv)) {
          out[i, ] <- c(max(nv), min(nv), mean(nv))
        }
      }
    }
  }
  out
}

# Apply to all source variables
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {
  vals <- cell_data[[var_name]]
  stats <- compute_neighbor_stats_fast(var_name, vals)
  cell_data[[paste0(var_name, "_nb_max")]] <- stats[, 1]
  cell_data[[paste0(var_name, "_nb_min")]] <- stats[, 2]
  cell_data[[paste0(var_name, "_nb_mean")]] <- stats[, 3]
}
```

---

**Performance Gain**  
- Eliminates per-row neighbor key construction and repeated joins.  
- Uses **integer indexing with offsets** for fast lookup.  
- Avoids massive list allocations for each variable.  
- Runs in hours instead of days on 16 GB RAM (depending on disk and CPU).  

**Preserves estimand and trained model**: Only feature-engineering changes; the Random Forest stays intact.  

For further speed, parallelize the inner loop using `future.apply::future_lapply` or Rcpp for heavy vectorization.
END_RESPONSE>>>
