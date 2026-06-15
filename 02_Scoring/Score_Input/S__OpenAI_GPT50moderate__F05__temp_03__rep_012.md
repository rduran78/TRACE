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
run_id: OpenAI_F05_T03_R012
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 0.3
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F05
prompt_family_slug: STRING_KEY_REDUNDANCY
family_label: String-key probe
family_group: Line-level probe
replicate: 12
raw_output_file: Experiment_Output/OpenAI/Temp_0.3/output_F05_OpenAI_temp_03_rep_012.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
The repeated string concatenation (`paste`) and repeated named lookups (`idx_lookup[neighbor_keys]`) inside the `lapply` loop in `build_neighbor_lookup` are not just a local inefficiencyâ€”they indicate a broader algorithmic issue. For each of ~6.46M rows, the code constructs character keys for neighbors and performs name-based lookups. This is extremely costly because:

- `paste` is called millions of times.
- Named indexing on a vector of length 6.46M is repeated millions of times.
- This happens before computing neighbor stats for multiple variables.

The root cause: the neighbor relationships are recomputed for every row-year combination using string keys, rather than precomputing a numeric index mapping once and reusing it.

---

**Optimization Strategy**  
Reformulate the algorithm to:
1. Precompute a numeric neighbor index matrix for all rows **once**, avoiding string-based lookups entirely.
2. Use this numeric matrix to compute neighbor stats for all variables efficiently.
3. Leverage vectorized operations or `matrixStats` to avoid repeated R loops.

Key idea:  
- Sort `data` by `(id, year)` so that each cell has a predictable block of rows.
- Compute an integer matrix `neighbor_idx` of size `nrow(data) Ã— max_neighbors` where each entry is the row index of a neighbor for the same year (or `NA` if absent).
- Then compute neighbor stats by indexing into `vals` using these precomputed indices.

---

**Working R Code**

```r
library(data.table)
library(matrixStats)

# Assume data.table for speed
setDT(cell_data)
setkey(cell_data, id, year)

# Precompute helper structures
id_order <- sort(unique(cell_data$id))
id_to_pos <- setNames(seq_along(id_order), id_order)

years <- sort(unique(cell_data$year))
n_years <- length(years)

# Precompute neighbor index matrix
max_neighbors <- max(lengths(rook_neighbors_unique))
n_rows <- nrow(cell_data)
neighbor_idx <- matrix(NA_integer_, n_rows, max_neighbors)

# Map (id, year) -> row index
# Since data is keyed by (id, year), we can compute offsets
rows_per_id <- n_years
id_offsets <- (seq_along(id_order) - 1L) * rows_per_id
row_lookup <- setNames(id_offsets, id_order)

# Fill neighbor_idx
for (row in seq_len(n_rows)) {
  this_id <- cell_data$id[row]
  this_year <- cell_data$year[row]
  year_pos <- match(this_year, years)
  
  ref_idx <- id_to_pos[[as.character(this_id)]]
  neighbor_ids <- id_order[rook_neighbors_unique[[ref_idx]]]
  
  # Compute neighbor row indices
  neighbor_rows <- row_lookup[as.character(neighbor_ids)] + year_pos
  neighbor_idx[row, seq_along(neighbor_rows)] <- neighbor_rows
}

# Function to compute neighbor stats for one variable
compute_neighbor_stats_fast <- function(vals, neighbor_idx) {
  # vals: numeric vector of length n_rows
  # neighbor_idx: matrix of neighbor row indices
  n <- nrow(neighbor_idx)
  max_nbr <- max_neighbors
  
  # Extract neighbor values as matrix
  nbr_vals <- matrix(NA_real_, n, max_nbr)
  valid_cols <- !is.na(neighbor_idx)
  nbr_vals[valid_cols] <- vals[neighbor_idx[valid_cols]]
  
  # Compute row-wise max, min, mean ignoring NAs
  max_val <- rowMaxs(nbr_vals, na.rm = TRUE)
  min_val <- rowMins(nbr_vals, na.rm = TRUE)
  mean_val <- rowMeans(nbr_vals, na.rm = TRUE)
  
  cbind(max_val, min_val, mean_val)
}

# Apply to all neighbor source variables
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {
  vals <- cell_data[[var_name]]
  stats <- compute_neighbor_stats_fast(vals, neighbor_idx)
  colnames(stats) <- paste0(var_name, c("_nbr_max", "_nbr_min", "_nbr_mean"))
  cell_data <- cbind(cell_data, stats)
}
```

---

**Why this works**  
- **Single pass neighbor index construction**: No repeated string concatenation or named lookups.
- **Matrix-based computations**: Uses `matrixStats` for efficient row-wise operations in C.
- **Memory tradeoff**: `neighbor_idx` is `nrow(data) Ã— max_neighbors` integers. For 6.46M rows and ~8 neighbors, thatâ€™s ~52M integers (~200 MB), feasible on 16 GB RAM.

---

**Expected Performance Gain**  
This approach removes 86+ hours of repeated string operations and should reduce runtime to a few minutes to an hour, depending on I/O and CPU speed, without changing the numerical estimand or retraining the model.
END_RESPONSE>>>
