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
run_id: Anthropic_F02_T10_R003
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F02
prompt_family_slug: CODE_ONLY_NO_HINT
family_label: Code only, no hint
family_group: Code-visible neutral
replicate: 3
raw_output_file: Experiment_Output/Anthropic/Temp_1.0/output_F02_Anthropic_temp_10_rep_003.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The core bottleneck is the `build_neighbor_lookup` function. It calls `lapply` over **~6.46 million rows**, and inside each iteration it performs:

1. **Character coercion and named-vector lookup** (`id_to_ref[as.character(...)]`) â€” repeated millions of times, this is extremely slow because R's named vector lookup is O(n) in the worst case and involves hashing overhead on every call.
2. **String pasting** (`paste(neighbor_cell_ids, data$year[i], sep = "_")`) â€” allocating ~4â€“8 short strings per row Ã— 6.46M rows = tens of millions of tiny allocations.
3. **Named index lookup** (`idx_lookup[neighbor_keys]`) on a **6.46-million-element named character vector** â€” each lookup hashes the string and probes a very large hash table.

The result is a **list of 6.46 million integer vectors**, which itself consumes several GB of RAM.

`compute_neighbor_stats` then loops over that 6.46M-element list again, computing `max/min/mean` per element â€” manageable in principle, but the R-level `lapply` over millions of small vectors is still slow.

Multiplied by 5 variables, the total work is enormous.

**Summary of problems:**
| Problem | Impact |
|---|---|
| Per-row string pasting & named-vector lookups in `build_neighbor_lookup` | ~90 % of runtime |
| 6.46M-element R list for `neighbor_lookup` | ~2â€“4 GB RAM |
| R-level `lapply` in `compute_neighbor_stats` over 6.46M entries Ã— 5 vars | Significant |
| No vectorization or use of compiled code | Everything is interpreted |

---

## Optimization Strategy

### Key Insight

The neighbor graph is **time-invariant**: cell A's neighbors are the same in every year. The `nb` object already encodes this. We only need to "expand" it across years using **integer arithmetic on a regular panel**, completely avoiding string operations.

### Plan

1. **Exploit the balanced-panel structure.** If cells are ordered consistently, row `(t-1)*N + i` corresponds to cell `i` in year `t`. Neighbor indices for year `t` are simply the cell-level neighbor indices shifted by `(t-1)*N`. This turns `build_neighbor_lookup` into pure integer arithmetic â€” no strings, no hash lookups.

2. **Flatten the neighbor lookup into two parallel vectors** (a CSR-like / adjacency-list-as-vectors representation): a `target` vector and a `neighbor_row` vector. This replaces the 6.46M-element R list with two integer vectors totaling ~20â€“30 M elements, which is ~200 MB instead of several GB.

3. **Vectorize `compute_neighbor_stats`** using `data.table` grouped operations on the flat adjacency vectors. `data.table` performs grouped `max/min/mean` in compiled C code and is orders of magnitude faster than per-element `lapply`.

4. **Process all 5 variables in one pass** over the flat adjacency to avoid redundant subsetting.

**Expected improvement:** From ~86+ hours to **~5â€“20 minutes** on the same laptop, with peak RAM well under 16 GB.

---

## Working R Code

```r
library(data.table)

# ============================================================
# 0. Assumptions / inputs already in the environment:
#    - cell_data        : data.frame or data.table with columns
#                         id, year, ntl, ec, pop_density, def, usd_est_n2, ...
#    - id_order         : integer/character vector of cell IDs
#                         (the ordering that matches rook_neighbors_unique)
#    - rook_neighbors_unique : an nb object (list of integer vectors)
#                              where element i contains the indices
#                              (into id_order) of neighbors of cell i.
#    - neighbor_source_vars : c("ntl","ec","pop_density","def","usd_est_n2")
# ============================================================

# --------------------------------------------------
# STEP 1 : Convert to data.table & ensure sort order
# --------------------------------------------------
if (!is.data.table(cell_data)) cell_data <- as.data.table(cell_data)

# Map each cell id to its position in id_order (1-based).
id_map <- data.table(id = id_order, cell_idx = seq_along(id_order))
cell_data <- merge(cell_data, id_map, by = "id", all.x = TRUE)

# Sort by year then cell_idx so that row number = (year_offset) * N + cell_idx
year_levels <- sort(unique(cell_data$year))           # 1992 .. 2019
year_map    <- data.table(year = year_levels,
                          year_offset = seq_along(year_levels) - 1L)
cell_data   <- merge(cell_data, year_map, by = "year", all.x = TRUE)
setorder(cell_data, year_offset, cell_idx)

# After sorting, the row number for (cell_idx=i, year_offset=t) is
#   row = t * N + i,   where N = number of cells
N <- length(id_order)
T <- length(year_levels)
stopifnot(nrow(cell_data) == N * T)   # balanced panel check

# Assign explicit row numbers (will be used as indices into columns)
cell_data[, row_id := .I]

# Quick sanity: row_id should equal year_offset * N + cell_idx
stopifnot(all(cell_data$row_id == cell_data$year_offset * N + cell_data$cell_idx))

# --------------------------------------------------
# STEP 2 : Build flat adjacency vectors (CSR-style)
#           using only integer arithmetic
# --------------------------------------------------
# For each cell i (1..N), rook_neighbors_unique[[i]] gives
# neighbor cell indices (also in 1..N).
# For year_offset t, the ROW of cell i  = t*N + i
#                     the ROW of cell j  = t*N + j
# So we just need to enumerate (i, j) pairs from the nb object
# and then replicate across T years.

# 2a. Build cell-level edge list
from_cell <- rep(seq_len(N),
                 times = lengths(rook_neighbors_unique))
to_cell   <- unlist(rook_neighbors_unique, use.names = FALSE)

n_edges <- length(from_cell)   # total directed edges at cell level

# 2b. Expand across years: for each year_offset t in 0..(T-1),
#     target_row = t*N + from_cell,  neighbor_row = t*N + to_cell
# This creates two integer vectors of length T * n_edges.

year_offsets <- 0:(T - 1L)

# Use outer arithmetic, then flatten
# target_row[k]   = year_offsets[yt] * N + from_cell[e]
# neighbor_row[k] = year_offsets[yt] * N + to_cell[e]

# Memory estimate: 2 vectors Ã— T Ã— n_edges int32 values
#   2 Ã— 28 Ã— 1,373,394 Ã— 4 bytes â‰ˆ 308 MB  â€” fits in 16 GB
target_row   <- rep(year_offsets, each = n_edges) * N +
                rep(from_cell, times = T)
neighbor_row <- rep(year_offsets, each = n_edges) * N +
                rep(to_cell,   times = T)

# --------------------------------------------------
# STEP 3 : Compute neighbor stats vectorised
# --------------------------------------------------
# For each variable, pull neighbor values, group by target_row,
# compute max / min / mean.

adj_dt <- data.table(target = target_row, nbr = neighbor_row)

# Free the raw vectors to save RAM
rm(target_row, neighbor_row, from_cell, to_cell)
gc()

for (var_name in neighbor_source_vars) {

  # Pull neighbor values in one vectorised step
  adj_dt[, nbr_val := cell_data[[var_name]][nbr]]

  # Remove NAs before aggregation
  stats <- adj_dt[!is.na(nbr_val),
                  .(nb_max  = max(nbr_val),
                    nb_min  = min(nbr_val),
                    nb_mean = mean(nbr_val)),
                  keyby = target]

  # Initialize new columns to NA, then fill matched rows
  max_col  <- paste0(var_name, "_nb_max")
  min_col  <- paste0(var_name, "_nb_min")
  mean_col <- paste0(var_name, "_nb_mean")

  cell_data[, (max_col)  := NA_real_]
  cell_data[, (min_col)  := NA_real_]
  cell_data[, (mean_col) := NA_real_]

  cell_data[stats$target, (max_col)  := stats$nb_max]
  cell_data[stats$target, (min_col)  := stats$nb_min]
  cell_data[stats$target, (mean_col) := stats$nb_mean]

  # Clean up within-loop temp column
  adj_dt[, nbr_val := NULL]
  rm(stats)
  gc()

  message("Done: ", var_name)
}

# --------------------------------------------------
# STEP 4 : Clean up helper columns
# --------------------------------------------------
cell_data[, c("cell_idx", "year_offset", "row_id") := NULL]

# cell_data now has the 15 new neighbor-stat columns
# (5 vars Ã— 3 stats each) alongside all original columns.

# --------------------------------------------------
# STEP 5 : Use the existing trained Random Forest
#           for prediction (unchanged)
# --------------------------------------------------
# predictions <- predict(trained_rf_model, newdata = cell_data)
```

---

## Why This Preserves Correctness

| Concern | Guarantee |
|---|---|
| **Same neighbor relationships** | We use the exact same `rook_neighbors_unique` nb object; we simply translate cell-level indices to row-level indices with deterministic arithmetic. |
| **Same statistics** | `max`, `min`, `mean` are computed on exactly the same sets of non-NA neighbor values as the original code. |
| **Trained RF model untouched** | We only add the same 15 feature columns (same names, same numeric values) that the original pipeline produced. The model object is never modified. |
| **Original numerical estimand** | Because the feature values are numerically identical (same floating-point operations on the same inputs), downstream predictions and any estimand derived from them are unchanged. |

---

## Performance Estimate

| Step | Original | Optimized |
|---|---|---|
| Build neighbor lookup | ~80 hours (string ops Ã— 6.46M) | ~30 seconds (integer arithmetic, vectorised) |
| Compute stats (5 vars) | ~6 hours (lapply Ã— 6.46M Ã— 5) | ~3â€“10 minutes (`data.table` grouped C code) |
| Peak RAM | >10 GB (6.46M-element list) | ~4â€“6 GB (flat int vectors + data.table) |
| **Total** | **~86 hours** | **~5â€“15 minutes** |
END_RESPONSE>>>
