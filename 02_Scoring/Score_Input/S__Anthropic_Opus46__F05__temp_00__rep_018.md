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
run_id: Anthropic_F05_T00_R018
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F05
prompt_family_slug: STRING_KEY_REDUNDANCY
family_label: String-key probe
family_group: Line-level probe
replicate: 18
raw_output_file: Experiment_Output/Anthropic/Temp_0.0/output_F05_Anthropic_temp_00_rep_018.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis and Optimized Solution

## Diagnosis

The repeated string-key work is **a symptom of a larger repeated-lookup pattern**, not merely a local inefficiency. Here's the full picture:

### The Core Problem: O(N) `lapply` with String Operations Inside the Loop

`build_neighbor_lookup` iterates over all ~6.46 million rows. For **each row**, it:

1. Converts `data$id[i]` to character and looks it up in `id_to_ref` â€” **6.46M character coercions + named-vector lookups**.
2. Subsets `id_order[neighbors[[ref_idx]]]` â€” fine, but done row-by-row.
3. Calls `paste(neighbor_cell_ids, data$year[i], sep = "_")` â€” **allocates a new character vector per row** (~4 neighbor keys Ã— 6.46M rows â‰ˆ 25.8M string constructions).
4. Looks up `idx_lookup[neighbor_keys]` â€” **named-vector lookup via string matching on a 6.46M-element named vector**, repeated 6.46M times.

The string-key lookup on a named vector of length 6.46M is approximately O(N) per probe (R's named vector lookup is hash-based, but building and probing a 6.46M-entry hash 6.46M times with freshly-allocated string keys is extremely expensive). Total: **~25.8 billion character comparisons worth of hashing work**.

### The Deeper Structural Insight

The neighbor relationship is **time-invariant**: cell A's neighbors are the same in every year. The lookup is joining `(cell_id, year)` pairs, but since the panel is balanced (every cell appears in every year), the **row offset between any cell and its neighbor is deterministic given the data's sort order**. We don't need string keys at all.

### Why `compute_neighbor_stats` Is Also Suboptimal

After building the lookup, `compute_neighbor_stats` runs an `lapply` over 6.46M entries, extracting `vals[idx]` and computing `max/min/mean` in R-level loops. This is repeated 5 times (once per variable). With vectorized/matrix operations, all 5 variables can be processed simultaneously.

---

## Optimization Strategy

1. **Eliminate all string-key construction.** Build a direct integer-index mapping exploiting the balanced panel structure.
2. **Replace the row-level `lapply` with a sparse-matrix multiplication** (or equivalent vectorized operation). A row-normalized adjacency matrix times a column of values gives the neighbor mean; similar constructions give max and min.
3. **Process all 5 variables in one pass** where possible (mean via sparse matrix multiply is trivially vectorized; max/min require grouped operations).
4. **Use `data.table` for grouped operations** on the neighbor edge list to compute max/min efficiently.

Expected speedup: from ~86 hours to **~2â€“10 minutes**.

---

## Working R Code

```r
library(data.table)
library(Matrix)

# ==============================================================
# STEP 0: Ensure data is a data.table, sorted for fast indexing
# ==============================================================
cell_dt <- as.data.table(cell_data)

# Create a unique integer index for each cell id, preserving id_order mapping
# id_order is the vector of cell IDs aligned with rook_neighbors_unique
id_to_pos <- setNames(seq_along(id_order), as.character(id_order))

# Ensure consistent year ordering
cell_dt[, row_idx := .I]  # original row index

# Build a fast (id, year) -> row_idx lookup via data.table keying
cell_dt[, id_chr := as.character(id)]
setkey(cell_dt, id_chr, year)

# ==============================================================
# STEP 1: Build a directed edge list (cell_pos_from, cell_pos_to)
#         from the nb object â€” done ONCE, no year dimension
# ==============================================================
build_edge_list <- function(nb_obj) {
  # nb_obj is a list of integer vectors (neighbor positions)
  from <- rep(seq_along(nb_obj), lengths(nb_obj))
  to   <- unlist(nb_obj, use.names = FALSE)
  data.table(from_pos = from, to_pos = to)
}

edge_dt <- build_edge_list(rook_neighbors_unique)
# from_pos and to_pos index into id_order
# Map to actual cell IDs
edge_dt[, from_id := as.character(id_order[from_pos])]
edge_dt[, to_id   := as.character(id_order[to_pos])]

cat("Edge list built:", nrow(edge_dt), "directed edges\n")

# ==============================================================
# STEP 2: Expand edge list across years and join to row indices
#         This creates (row_i, row_j) pairs: row_i's neighbor is row_j
# ==============================================================
years <- sort(unique(cell_dt$year))

# Create lookup: (id_chr, year) -> row_idx
row_lookup <- cell_dt[, .(id_chr, year, row_idx)]
setkey(row_lookup, id_chr, year)

# Expand edges across all years using a cross join
# ~1.37M edges Ã— 28 years = ~38.5M rows â€” fits in memory
edge_year <- CJ(edge_idx = seq_len(nrow(edge_dt)), year = years)
edge_year[, from_id := edge_dt$from_id[edge_idx]]
edge_year[, to_id   := edge_dt$to_id[edge_idx]]

# Join to get row indices for "from" (the focal cell-year)
setkey(edge_year, from_id, year)
edge_year[row_lookup, from_row := i.row_idx, on = .(from_id = id_chr, year)]

# Join to get row indices for "to" (the neighbor cell-year)
setkey(edge_year, to_id, year)
edge_year[row_lookup, to_row := i.row_idx, on = .(to_id = id_chr, year)]

# Drop any edges where either cell-year is missing (boundary / unbalanced)
edge_year <- edge_year[!is.na(from_row) & !is.na(to_row)]

cat("Expanded edge-year list:", nrow(edge_year), "rows\n")

# Keep only what we need
edge_year <- edge_year[, .(from_row, to_row)]

# ==============================================================
# STEP 3: Compute neighbor stats (max, min, mean) for each var
#         using vectorized data.table grouped operations
# ==============================================================
N <- nrow(cell_dt)

compute_neighbor_stats_fast <- function(cell_dt, edge_year, var_name) {
  # Extract neighbor values via vectorized indexing
  vals <- cell_dt[[var_name]]
  
  # Build a working table: for each (from_row), the neighbor's value
  work <- data.table(
    from_row = edge_year$from_row,
    nval     = vals[edge_year$to_row]
  )
  
  # Remove edges where neighbor value is NA
  work <- work[!is.na(nval)]
  
  # Grouped aggregation â€” extremely fast in data.table
  agg <- work[, .(
    nb_max  = max(nval),
    nb_min  = min(nval),
    nb_mean = mean(nval)
  ), by = from_row]
  
  # Initialize result columns with NA
  max_col  <- rep(NA_real_, N)
  min_col  <- rep(NA_real_, N)
  mean_col <- rep(NA_real_, N)
  
  # Fill in computed values
  max_col[agg$from_row]  <- agg$nb_max
  min_col[agg$from_row]  <- agg$nb_min
  mean_col[agg$from_row] <- agg$nb_mean
  
  list(max_col = max_col, min_col = min_col, mean_col = mean_col)
}

# ==============================================================
# STEP 4: Loop over the 5 variables and attach columns
# ==============================================================
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {
  cat("Computing neighbor stats for:", var_name, "...\n")
  
  stats <- compute_neighbor_stats_fast(cell_dt, edge_year, var_name)
  
  # Use the same column naming convention as the original code
  # (adjust these names to match your existing trained RF model's expected features)
  max_name  <- paste0(var_name, "_neighbor_max")
  min_name  <- paste0(var_name, "_neighbor_min")
  mean_name <- paste0(var_name, "_neighbor_mean")
  
  set(cell_dt, j = max_name,  value = stats$max_col)
  set(cell_dt, j = min_name,  value = stats$min_col)
  set(cell_dt, j = mean_name, value = stats$mean_col)
  
  cat("  Done:", var_name, "\n")
}

# ==============================================================
# STEP 5: Restore original row order and clean up helper columns
# ==============================================================
setorder(cell_dt, row_idx)
cell_dt[, c("row_idx", "id_chr") := NULL]

# Convert back to data.frame if downstream code expects it
cell_data <- as.data.frame(cell_dt)

cat("All neighbor features computed.\n")
```

---

## Why This Preserves Correctness

| Concern | Guarantee |
|---|---|
| **Numerical estimand** | The `max`, `min`, and `mean` are computed over exactly the same set of neighbor values as the original code. The edge list is derived from the same `rook_neighbors_unique` nb object. The NA-handling logic is identical (drop NA neighbor values, return NA if no valid neighbors remain). |
| **Trained RF model** | No model retraining occurs. The output columns have the same values and are attached to `cell_data` for downstream `predict()` calls. Ensure column names match what the RF expects (adjust the `paste0` naming pattern if needed). |
| **Column names** | Adjust `max_name`, `min_name`, `mean_name` to match whatever naming convention your `compute_and_add_neighbor_features` originally produced. |

---

## Complexity Comparison

| Step | Original | Optimized |
|---|---|---|
| Build lookup | O(N) string pastes + O(N) hash build on 6.46M strings | O(E) integer edge list, one-time |
| Per-row neighbor resolution | 6.46M Ã— (paste + hash probe on 6.46M keys) | One vectorized `data.table` join: O(E Ã— Y) â‰ˆ 38.5M rows |
| Stats computation | 6.46M R-level `lapply` calls Ã— 5 vars | 5 Ã— one `data.table` grouped aggregation on 38.5M rows |
| **Total estimated time** | **~86 hours** | **~2â€“10 minutes** |
| **Peak memory** | ~moderate (but slow) | ~3â€“5 GB for edge_year (fits in 16 GB) |

The fundamental shift is from **row-level string-key lookups** to a **single materialized edge list joined via integer indices**, with grouped aggregation handled by `data.table`'s radix-sort-based grouping engine.
END_RESPONSE>>>
