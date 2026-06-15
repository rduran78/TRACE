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
run_id: Anthropic_F09_T10_R009
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F09
prompt_family_slug: ADJACENCY_TABLE_OPTION
family_label: Adjacency-table option
family_group: Representation cue
replicate: 9
raw_output_file: Experiment_Output/Anthropic/Temp_1.0/output_F09_Anthropic_temp_10_rep_009.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The current implementation is extremely slow (~86+ hours) for three interconnected reasons:

### 1. Row-level `lapply` over 6.46 million rows in `build_neighbor_lookup`
For every single cell-year row, the function performs character coercion (`as.character`), string pasting (`paste(..., sep="_")`), and named-vector lookups (`idx_lookup[neighbor_keys]`). With ~6.46M rows, this produces billions of small string operations and hash-table lookups inside a sequential R `lapply`. This is the **dominant bottleneck**.

### 2. Redundant recomputation of the spatial topology per year
The rook-neighbor relationships are **static across years** â€” cell ð‘–'s neighbors are the same in 1992 as in 2019. Yet `build_neighbor_lookup` re-resolves them for every cell-year combination (6.46M times) instead of for the 344,208 unique cells once and then replicating the structure across years.

### 3. Row-level `lapply` in `compute_neighbor_stats`
Even after the lookup is built, `compute_neighbor_stats` loops over 6.46M entries, extracting and summarizing small vectors in pure R. This is slow because R's `lapply` over millions of tiny operations has enormous per-iteration overhead.

### Summary of waste
| Operation | Current scale | Optimal scale | Ratio |
|---|---|---|---|
| Neighbor resolution | 6.46M row lookups | 344,208 cell lookups (once) | ~19Ã— fewer |
| String key creation | ~6.46M `paste` calls + hash lookups | 0 (use integer join) | âˆž |
| Stat computation | 6.46M R-level loops per variable | 1 vectorised `data.table` grouped join per variable | orders of magnitude |

---

## Optimization Strategy

**Core idea:** Build the neighbor edge-list once at the cell level (344K rows, not 6.46M), then use `data.table` keyed joins to attach yearly attribute values and compute grouped `max`, `min`, `mean` â€” all vectorised in C under the hood.

### Steps

1. **Build a static edge table** from `rook_neighbors_unique` (the `nb` object): one row per directed edge `(cell_id, neighbor_id)`. This is done once and is ~1.37M rows.

2. **Key the panel data** by `(id, year)` in a `data.table`.

3. **For each variable**, join the edge table to the panel data to fetch neighbor values, then compute grouped stats with `data.table`'s `:=` and `by=` â€” no R-level loop over rows.

4. **Left-join** the results back onto the main panel. Cells with no neighbors (e.g., boundary cells missing from the panel for a given year) get `NA`, preserving the original numerical estimand exactly.

5. **Predict** with the already-trained Random Forest model â€” unchanged.

**Expected speedup:** From ~86 hours to **minutes** (typically 2â€“10 minutes depending on disk I/O and RAM pressure on a 16 GB laptop). The 6.46M-row `data.table` plus the edge table plus intermediate join results fit comfortably within 16 GB.

---

## Working R Code

```r
library(data.table)

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# STEP 0: Ensure cell_data is a data.table
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
if (!is.data.table(cell_data)) {
 cell_data <- as.data.table(cell_data)
}

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# STEP 1: Build a STATIC edge table from the nb object (once)
#
#   rook_neighbors_unique : an nb object (list of integer vectors)
#   id_order              : vector mapping position â†’ cell id
#
#   Result: edges_dt with columns (cell_id, neighbor_id)
#           ~1.37 M rows (directed pairs)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
build_edge_table <- function(id_order, neighbors_nb) {
  # neighbors_nb[[i]] contains integer indices into id_order
  # that are neighbors of cell id_order[i].
  n <- length(neighbors_nb)
  from_list <- vector("list", n)
  to_list   <- vector("list", n)
  for (i in seq_len(n)) {
    nb_idx <- neighbors_nb[[i]]
    # spdep nb objects use 0L to signal "no neighbors"
    nb_idx <- nb_idx[nb_idx != 0L]
    if (length(nb_idx) > 0L) {
      from_list[[i]] <- rep(id_order[i], length(nb_idx))
      to_list[[i]]   <- id_order[nb_idx]
    }
  }
  data.table(
    cell_id     = unlist(from_list, use.names = FALSE),
    neighbor_id = unlist(to_list,   use.names = FALSE)
  )
}

cat("Building static edge table â€¦\n")
edges_dt <- build_edge_table(id_order, rook_neighbors_unique)
cat(sprintf("  Edge table: %s directed edges\n", format(nrow(edges_dt), big.mark = ",")))

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# STEP 2: Key the panel data for fast joins
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
setkey(cell_data, id, year)

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# STEP 3: For each variable, compute neighbor max / min / mean
#         via a single data.table join + grouped aggregation,
#         then attach back to cell_data.
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

compute_and_add_neighbor_features_dt <- function(panel_dt, edges, var_name) {
  # --- 3a. Build a slim lookup: (id, year, value) ---------
  lookup <- panel_dt[, .(id, year, value = get(var_name))]
  setnames(lookup, "id", "neighbor_id")
  setkey(lookup, neighbor_id, year)

  # --- 3b. Expand edges Ã— years by joining to the panel --
  #     For every (cell_id, year) pair we pull the neighbor's
  #     value from the lookup in one vectorised join.
  #
  #     We add 'year' to the edge table via a cross-join with
  #     unique years?  No â€” more efficient: join edges onto
  #     the panel's (cell_id, year) to get the years that
  #     actually exist, then join neighbor values.
  # --------------------------------------------------------

  # Get (cell_id, year) pairs that exist in the panel
  cell_years <- panel_dt[, .(cell_id = id, year)]
  # Merge with edges to get (cell_id, year, neighbor_id)
  #   â€” one row per cell-year-neighbor triple
  expanded <- edges[cell_years, on = .(cell_id), allow.cartesian = TRUE, nomatch = 0L]
  # Now expanded has: cell_id, neighbor_id, year

  # Join to get the neighbor's value for that year
  setkey(expanded, neighbor_id, year)
  expanded[lookup, value := i.value, on = .(neighbor_id, year)]

  # --- 3c. Grouped aggregation ----------------------------
  stats <- expanded[
    !is.na(value),
    .(
      nb_max  = max(value),
      nb_min  = min(value),
      nb_mean = mean(value)
    ),
    by = .(cell_id, year)
  ]

  # --- 3d. Attach to the panel ----------------------------
  max_col  <- paste0("neighbor_max_",  var_name)
  min_col  <- paste0("neighbor_min_",  var_name)
  mean_col <- paste0("neighbor_mean_", var_name)

  setnames(stats, c("nb_max", "nb_min", "nb_mean"),
                  c(max_col,  min_col,  mean_col))

  # Remove old columns if they exist (re-run safety)
  for (col in c(max_col, min_col, mean_col)) {
    if (col %in% names(panel_dt)) panel_dt[, (col) := NULL]
  }

  # Left join back (preserves rows with no neighbors â†’ NA)
  setkey(stats, cell_id, year)
  panel_dt[stats, (c(max_col, min_col, mean_col)) :=
    mget(paste0("i.", c(max_col, min_col, mean_col))),
    on = .(id = cell_id, year)]

  invisible(panel_dt)
}

cat("Computing neighbor features â€¦\n")
for (var_name in neighbor_source_vars) {
  cat(sprintf("  %s â€¦ ", var_name))
  t0 <- proc.time()
  cell_data <- compute_and_add_neighbor_features_dt(cell_data, edges_dt, var_name)
  elapsed <- (proc.time() - t0)["elapsed"]
  cat(sprintf("done (%.1f s)\n", elapsed))
}

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# STEP 4: Predict with the EXISTING Random Forest model
#         (model object is assumed to be in `rf_model`)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
cat("Generating predictions with the trained Random Forest â€¦\n")
cell_data[, prediction := predict(rf_model, newdata = cell_data)]
cat("Done.\n")
```

---

## Why This Is Correct and Safe

| Concern | Guarantee |
|---|---|
| **Numerical equivalence** | `max`, `min`, `mean` are computed over exactly the same neighbor sets as before (rook contiguity, same year, `NA`-excluded). The only difference is the computation path (vectorised join vs. row-level R loop). |
| **Trained RF model preserved** | The model object (`rf_model`) is never modified or retrained; only `predict()` is called. |
| **Column names preserved** | Output columns follow the same `neighbor_{max,min,mean}_{var}` naming pattern consumed by the RF model's formula/feature list. Adjust the pattern in `setnames` if the original code used a different convention. |
| **Memory** | Peak memory â‰ˆ panel (6.46M Ã— ~115 cols â‰ˆ 5â€“6 GB) + expanded edge-year table (worst case ~1.37M Ã— 28 â‰ˆ 38.4M rows Ã— 3 int/dbl cols â‰ˆ 0.9 GB) + stats table â‰ˆ well within 16 GB. Each variable is processed and released sequentially. |
| **No external dependencies beyond `data.table`** | Already one of the most common R packages; no compilation or special hardware needed. |

### Expected runtime: **2â€“10 minutes total** (vs. 86+ hours), a speedup of roughly **500â€“2,500Ã—**.
END_RESPONSE>>>
