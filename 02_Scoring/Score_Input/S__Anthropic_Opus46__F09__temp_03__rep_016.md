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
run_id: Anthropic_F09_T03_R016
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.3
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F09
prompt_family_slug: ADJACENCY_TABLE_OPTION
family_label: Adjacency-table option
family_group: Representation cue
replicate: 16
raw_output_file: Experiment_Output/Anthropic/Temp_0.3/output_F09_Anthropic_temp_03_rep_016.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The current implementation is extremely slow (~86+ hours) because of **two compounding bottlenecks**:

### 1. `build_neighbor_lookup` â€” O(nÂ²)-like construction via per-row `lapply`
- It iterates over all **~6.46 million** cell-year rows.
- For each row, it constructs string keys (`paste(id, year)`), looks them up in a named character vector (`idx_lookup`), and filters `NA`s.
- Named vector lookup in R via `[` on character names is **O(n)** in the worst case (hash collisions aside, the overhead of 6.46M character key lookups inside 6.46M iterations is enormous).
- The output is a **list of 6.46 million integer vectors**, which is memory-heavy and cache-unfriendly.

### 2. `compute_neighbor_stats` â€” repeated per-variable `lapply` over 6.46M rows
- Called 5 times (once per neighbor source variable), each time iterating over 6.46M list elements.
- Each iteration subsets a numeric vector, removes `NA`s, and computes `max`, `min`, `mean` â€” all in interpreted R with per-element function-call overhead.

### Root cause
The neighbor topology is **static across years** (rook contiguity doesn't change), yet the lookup is rebuilt at the cell-year level, entangling topology with temporal indexing. This means the same spatial neighbor structure is redundantly encoded 28 times (once per year), and every computation pays the cost of string-key indirection.

---

## Optimization Strategy

### Core idea: Separate topology from time, then vectorize with `data.table`

1. **Build a cell-level adjacency table once** â€” a two-column `data.table` of `(cell_id, neighbor_id)` with ~1.37M rows. This is year-invariant.

2. **Join yearly attributes onto the adjacency table** â€” for each year, the cell attributes are joined to the neighbor side of the adjacency table via a keyed `data.table` join. This is an O(n log n) merge, not an O(nÂ²) string-key lookup.

3. **Compute grouped aggregates** â€” `data.table`'s `[, .(max, min, mean), by = .(cell_id, year)]` computes all neighbor stats in one vectorized pass per variable, leveraging C-level grouped aggregation.

4. **Join results back** to the main dataset.

This eliminates:
- All 6.46M-element `lapply` loops.
- All string-key construction and lookup.
- All per-element R function calls for `max`/`min`/`mean`.

### Expected speedup
- Adjacency table: ~1.37M rows (built once, <1 second).
- Per-variable join + grouped stats: ~1.37M Ã— 28 â‰ˆ 38.4M rows processed via `data.table` vectorized C code â†’ seconds per variable.
- Total for 5 variables: **under 5 minutes** on a 16 GB laptop (vs. 86+ hours).

### Constraints preserved
- The trained Random Forest model is **not retouched**.
- The output columns are numerically identical (`max`, `min`, `mean` of the same neighbor values), preserving the original estimand.

---

## Working R Code

```r
library(data.table)

# ==============================================================
# STEP 1: Build a static cell-level adjacency table (once)
# ==============================================================
# Inputs:
#   id_order             â€” integer/numeric vector of cell IDs in the order
#                          matching rook_neighbors_unique (the nb object)
#   rook_neighbors_unique â€” an nb object (list of integer index vectors)
#
# Output:
#   adj_dt â€” data.table with columns (cell_id, neighbor_id)
#            representing all directed rook-neighbor pairs

build_adjacency_table <- function(id_order, neighbors_nb) {
  # Pre-allocate: count total neighbor links
  n_cells <- length(id_order)
  n_links <- sum(vapply(neighbors_nb, length, integer(1)))

  cell_id    <- integer(n_links)
  neighbor_id <- integer(n_links)

  pos <- 1L
  for (i in seq_len(n_cells)) {
    nb_idx <- neighbors_nb[[i]]
    # nb objects use 0 to denote "no neighbors" in some spdep versions
    nb_idx <- nb_idx[nb_idx != 0L]
    n_nb   <- length(nb_idx)
    if (n_nb > 0L) {
      rng <- pos:(pos + n_nb - 1L)
      cell_id[rng]     <- id_order[i]
      neighbor_id[rng] <- id_order[nb_idx]
      pos <- pos + n_nb
    }
  }

  # Trim if any 0-neighbor cells caused over-allocation
  if (pos - 1L < n_links) {
    cell_id     <- cell_id[1:(pos - 1L)]
    neighbor_id <- neighbor_id[1:(pos - 1L)]
  }

  data.table(cell_id = cell_id, neighbor_id = neighbor_id)
}

adj_dt <- build_adjacency_table(id_order, rook_neighbors_unique)

cat(sprintf(
  "Adjacency table: %s directed neighbor pairs for %s cells\n",
  format(nrow(adj_dt), big.mark = ","),
  format(length(id_order), big.mark = ",")
))

# ==============================================================
# STEP 2: Convert main data to data.table (if not already)
# ==============================================================
if (!is.data.table(cell_data)) {
  cell_data <- as.data.table(cell_data)
}

# ==============================================================
# STEP 3: Compute neighbor stats for each source variable
# ==============================================================
# For each variable, we:
#   a) Select (id, year, variable) from cell_data.
#   b) Join onto adj_dt so each (cell_id, year) row gets its
#      neighbor's variable value.
#   c) Compute grouped max, min, mean by (cell_id, year).
#   d) Join the results back onto cell_data.
#
# Output column names follow the pattern:
#   <var>_neighbor_max, <var>_neighbor_min, <var>_neighbor_mean

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# Key the adjacency table for fast joins
setkey(adj_dt, neighbor_id)

for (var_name in neighbor_source_vars) {

  cat(sprintf("Computing neighbor stats for: %s ...\n", var_name))

  # --- a) Extract the relevant columns from cell_data ---
  # Columns: id (cell identifier), year, and the variable of interest
  attr_dt <- cell_data[, .(id, year, value = get(var_name))]
  setkey(attr_dt, id)

  # --- b) Join neighbor attributes onto the adjacency table ---
  # For every (cell_id, neighbor_id) pair, attach the neighbor's
  # value for each year.
  #
  # We need to cross-join adj_dt with years implicitly:
  # Instead, join adj_dt with attr_dt on neighbor_id == id
  # This gives us (cell_id, neighbor_id, year, value)
  #   where value is the NEIGHBOR's attribute.

  merged <- attr_dt[adj_dt,
    on = .(id = neighbor_id),
    .(cell_id = i.cell_id, year = x.year, value = x.value),
    allow.cartesian = TRUE,
    nomatch = NA
  ]

  # Remove rows where the neighbor had no data for that year
  merged <- merged[!is.na(year)]

  # --- c) Grouped aggregation ---
  stats <- merged[
    !is.na(value),
    .(
      nb_max  = max(value),
      nb_min  = min(value),
      nb_mean = mean(value)
    ),
    by = .(cell_id, year)
  ]

  # --- d) Name the output columns to match original pipeline ---
  col_max  <- paste0(var_name, "_neighbor_max")
  col_min  <- paste0(var_name, "_neighbor_min")
  col_mean <- paste0(var_name, "_neighbor_mean")
  setnames(stats, c("nb_max", "nb_min", "nb_mean"),
                  c(col_max,  col_min,  col_mean))

  # --- e) Remove old columns if they exist (idempotent re-runs) ---
  for (col in c(col_max, col_min, col_mean)) {
    if (col %in% names(cell_data)) {
      cell_data[, (col) := NULL]
    }
  }

  # --- f) Join back onto cell_data ---
  cell_data <- merge(
    cell_data, stats,
    by.x = c("id", "year"),
    by.y = c("cell_id", "year"),
    all.x = TRUE,
    sort = FALSE
  )

  cat(sprintf("  -> Added columns: %s, %s, %s\n", col_max, col_min, col_mean))
}

# ==============================================================
# STEP 4: Predict with the existing trained Random Forest
# ==============================================================
# The trained model object (e.g., `rf_model`) is unchanged.
# Ensure cell_data is a data.frame if the model expects one.

# cell_data_df <- as.data.frame(cell_data)  # if needed
# predictions  <- predict(rf_model, newdata = cell_data_df)

cat("Neighbor feature engineering complete. Ready for RF prediction.\n")
```

---

## Summary

| Aspect | Old Implementation | New Implementation |
|---|---|---|
| **Neighbor lookup** | 6.46M-element `lapply` with string-key hashing | One-time 1.37M-row `data.table` (< 1 sec) |
| **Per-variable stats** | 6.46M-element `lapply` calling `max`/`min`/`mean` in R | Vectorized `data.table` grouped aggregation in C |
| **Total loops** | ~32.3M R-level iterations (5 vars Ã— 6.46M) | 5 keyed joins + 5 grouped aggregations |
| **Estimated time** | ~86+ hours | **< 5 minutes** |
| **RAM peak** | Large list of 6.46M integer vectors | ~38.4M-row intermediate `data.table` (~600 MB) |
| **RF model** | Preserved âœ“ | Preserved âœ“ |
| **Numerical output** | `max`, `min`, `mean` of neighbor values | Identical `max`, `min`, `mean` âœ“ |
END_RESPONSE>>>
