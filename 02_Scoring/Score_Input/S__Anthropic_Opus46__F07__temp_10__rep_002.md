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
run_id: Anthropic_F07_T10_R002
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F07
prompt_family_slug: RASTER_INVALID_IRREGULAR
family_label: Raster invalid irregular topology
family_group: Raster/kernel bridge
replicate: 2
raw_output_file: Experiment_Output/Anthropic/Temp_1.0/output_F07_Anthropic_temp_10_rep_002.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The bottleneck is **`build_neighbor_lookup`**. It constructs a list of 6.46 million elements, where each element is produced by an anonymous function inside `lapply` that:

1. **Looks up character-keyed named vectors** (`id_to_ref`, `idx_lookup`) millions of times â€” named-vector lookup in R is **O(n)** per query (linear scan), making the whole construction **O(nÂ²)** in the worst case.
2. **Pastes and matches character keys** (`paste(â€¦, sep="_")`) 6.46 million times, each time creating intermediate character vectors for every neighbor set.
3. **Creates ~6.46 million small integer vectors**, which is brutal on R's memory allocator and GC.

Then `compute_neighbor_stats` iterates over that 6.46M-element list **5 times** (once per variable), each time subsetting a numeric vector with small index vectors â€” this is I/O-bound and GC-heavy.

**Summary:** The 86+ hour runtime is caused by O(n) character-key lookups inside a loop of 6.46M iterations, repeated character allocation via `paste`, and repeated list-of-small-vectors traversal.

---

## Optimization Strategy

| Problem | Solution |
|---|---|
| Named-vector lookup is O(n) | Use **integer hash maps** via `data.table` or environment-based lookup; or eliminate lookup entirely by **pre-joining** with integer keys. |
| `paste` creates millions of temp strings | Replace with **integer arithmetic**: `key = id * 100000L + (year - 1991L)` gives a unique integer key per cell-year. |
| 6.46M-element R list for neighbor_lookup | Replace with a **flat CSR (Compressed Sparse Row)** representation: two integer vectors (`offsets`, `neighbors_flat`). |
| `compute_neighbor_stats` loops 5Ã— over list | Vectorize using `data.table` grouped operations or a single C-level pass via **Rcpp**, or use CSR + vectorized segment operations. |
| 16 GB RAM constraint | CSR is far more compact than a list. All intermediate character vectors are eliminated. |

**Expected speedup:** From 86+ hours â†’ **minutes** (roughly 2â€“10 minutes depending on disk I/O).

**Numerical equivalence:** The neighbor max/min/mean are computed over exactly the same neighbor index sets with the same NA handling, so the trained Random Forest receives identical features.

---

## Working R Code

```r
# =============================================================================
# OPTIMIZED NEIGHBOR FEATURE PIPELINE
# Drop-in replacement. Preserves numerical results exactly.
# =============================================================================

library(data.table)

# â”€â”€ Step 0: Convert cell_data to data.table (non-destructive) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
if (!is.data.table(cell_data)) {
  cell_data <- as.data.table(cell_data)
}

# â”€â”€ Step 1: Build integer-keyed row index using data.table hashing â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Unique integer key per (id, year) pair â€” no paste, no characters.
# id_order and rook_neighbors_unique come from the serialized spdep::nb object.

build_neighbor_lookup_fast <- function(dt, id_order, neighbors) {

  # Map cell id â†’ position in id_order (1-based, matches nb object)
  id_map <- data.table(id = id_order, ref = seq_along(id_order))

  # Map (id, year) â†’ row number in dt
  dt[, .row_idx := .I]
  row_index <- dt[, .(id, year, .row_idx)]
  setkey(row_index, id, year)

  # Unique years in data
  years <- sort(unique(dt$year))

  # â”€â”€ Build CSR representation â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
  # For every row i in dt, we need the row indices of its rook-neighbors
  # in the same year.
  #
  # Pre-expand the nb list into a data.table of directed edges at the

  # cell level, then join with year to get row-level edges.

  # Edges: (from_id, to_id) â€” directed, from nb list
  edge_list <- rbindlist(lapply(seq_along(neighbors), function(ref) {
    nb <- neighbors[[ref]]
    if (length(nb) == 0L || (length(nb) == 1L && nb[1] == 0L)) {
      return(NULL)
    }
    data.table(from_id = id_order[ref], to_id = id_order[nb])
  }))

  # Cross-join edges with years â†’ (from_id, year, to_id)
  edge_year <- edge_list[, CJ(year = years), by = .(from_id, to_id)]

  # Join to get source row index (the "from" row)
  setnames(edge_year, c("from_id", "to_id", "year"))
  edge_year[row_index, on = .(from_id = id, year), from_row := i..row_idx]

  # Join to get neighbor row index (the "to" row)
  edge_year[row_index, on = .(to_id = id, year), to_row := i..row_idx]

  # Drop edges where either side is missing (masked cells / missing years)
  edge_year <- edge_year[!is.na(from_row) & !is.na(to_row)]

  # Sort by from_row for CSR construction

  setorder(edge_year, from_row)

  # Build CSR
  n <- nrow(dt)
  neighbor_to   <- edge_year$to_row
  neighbor_from <- edge_year$from_row

  # offsets: offsets[i] .. offsets[i+1]-1 are the positions in neighbor_to

  offsets <- integer(n + 1L)
  tabulated <- tabulate(neighbor_from, nbins = n)
  offsets[1L] <- 1L
  for (i in seq_len(n)) {
    offsets[i + 1L] <- offsets[i] + tabulated[i]
  }
  # Faster cumsum version:
  offsets <- c(1L, 1L + cumsum(tabulated))

  # Clean up temp column
  dt[, .row_idx := NULL]

  list(offsets = offsets, neighbors = neighbor_to, n = n)
}

# â”€â”€ Step 2: Compute neighbor stats for one variable using CSR â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
compute_neighbor_stats_fast <- function(vals, csr) {
  n        <- csr$n
  offsets  <- csr$offsets
  nb_idx   <- csr$neighbors

  nb_max  <- rep(NA_real_, n)
  nb_min  <- rep(NA_real_, n)
  nb_mean <- rep(NA_real_, n)

  # Vectorized approach: use the flat neighbor vector
  # Get all neighbor values at once
  all_nb_vals <- vals[nb_idx]  # length = total number of edges

  # We need to split by "from_row" â€” use the offsets.
  # For large data, an Rcpp loop is ideal, but we can stay in R
  # with a data.table grouping trick:

  # Build a "from" vector aligned with nb_idx
  from_vec <- rep(seq_len(n), times = diff(offsets))

  edge_dt <- data.table(from = from_vec, val = all_nb_vals)

  # Remove NA values before aggregation
  edge_dt <- edge_dt[!is.na(val)]

  agg <- edge_dt[, .(nb_max = max(val),
                      nb_min = min(val),
                      nb_mean = mean(val)), by = from]

  nb_max[agg$from]  <- agg$nb_max
  nb_min[agg$from]  <- agg$nb_min
  nb_mean[agg$from] <- agg$nb_mean

  data.table(nb_max = nb_max, nb_min = nb_min, nb_mean = nb_mean)
}

# â”€â”€ Step 3: Main pipeline â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
message("Building CSR neighbor lookupâ€¦")
system.time({
  csr <- build_neighbor_lookup_fast(cell_data, id_order, rook_neighbors_unique)
})

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

message("Computing neighbor featuresâ€¦")
system.time({
  for (var_name in neighbor_source_vars) {
    message("  â†’ ", var_name)
    stats <- compute_neighbor_stats_fast(cell_data[[var_name]], csr)
    set(cell_data, j = paste0(var_name, "_nb_max"),  value = stats$nb_max)
    set(cell_data, j = paste0(var_name, "_nb_min"),  value = stats$nb_min)
    set(cell_data, j = paste0(var_name, "_nb_mean"), value = stats$nb_mean)
  }
})

message("Done. cell_data now has neighbor features. RF model is untouched.")
```

---

### Optional: Rcpp Inner Loop (fastest possible)

If the `data.table` grouped aggregation is still not fast enough (it should be ~2â€“5 min), drop in this Rcpp function for the inner computation:

```r
Rcpp::cppFunction('
#include <Rcpp.h>
using namespace Rcpp;

// [[Rcpp::export]]
NumericMatrix neighbor_stats_csr(NumericVector vals,
                                 IntegerVector offsets,
                                 IntegerVector neighbors) {
  int n = offsets.size() - 1;
  NumericMatrix out(n, 3); // max, min, mean
  std::fill(out.begin(), out.end(), NA_REAL);

  for (int i = 0; i < n; i++) {
    int start = offsets[i] - 1; // R 1-based â†’ C 0-based
    int end   = offsets[i + 1] - 1;
    double vmax = R_NegInf, vmin = R_PosInf, vsum = 0.0;
    int cnt = 0;
    for (int j = start; j < end; j++) {
      double v = vals[neighbors[j] - 1]; // R 1-based index
      if (!NumericVector::is_na(v)) {
        if (v > vmax) vmax = v;
        if (v < vmin) vmin = v;
        vsum += v;
        cnt++;
      }
    }
    if (cnt > 0) {
      out(i, 0) = vmax;
      out(i, 1) = vmin;
      out(i, 2) = vsum / cnt;
    }
  }
  return out;
}
')

# Usage (replaces compute_neighbor_stats_fast):
for (var_name in neighbor_source_vars) {
  message("  â†’ ", var_name)
  mat <- neighbor_stats_csr(cell_data[[var_name]], csr$offsets, csr$neighbors)
  set(cell_data, j = paste0(var_name, "_nb_max"),  value = mat[, 1])
  set(cell_data, j = paste0(var_name, "_nb_min"),  value = mat[, 2])
  set(cell_data, j = paste0(var_name, "_nb_mean"), value = mat[, 3])
}
```

**With Rcpp:** The 5-variable neighbor stat computation over 6.46M rows completes in **~10â€“30 seconds** on a modern laptop.

---

## Summary

| Aspect | Original | Optimized |
|---|---|---|
| Lookup structure | Named character vector (O(n) per query) | `data.table` hash join + integer CSR |
| Key construction | `paste(id, year)` â€” 6.46M strings | Integer join, zero string allocation |
| Neighbor storage | R list of 6.46M small vectors (~2â€“5 GB) | Two flat integer vectors (~22 MB) |
| Stat computation | R `lapply` over 6.46M elements Ã— 5 vars | Vectorized `data.table` group-by or Rcpp |
| Runtime | **86+ hours** | **2â€“10 minutes** (pure R) / **< 1 minute** (Rcpp) |
| RAM peak | Likely >16 GB (swapping) | **~1â€“2 GB** |
| Numerical result | Baseline | **Identical** (same indices, same NA logic) |
| RF model | Untouched | **Untouched** |
END_RESPONSE>>>
