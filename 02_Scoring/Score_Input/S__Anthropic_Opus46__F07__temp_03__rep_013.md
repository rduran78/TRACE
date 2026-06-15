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
run_id: Anthropic_F07_T03_R013
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.3
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F07
prompt_family_slug: RASTER_INVALID_IRREGULAR
family_label: Raster invalid irregular topology
family_group: Raster/kernel bridge
replicate: 13
raw_output_file: Experiment_Output/Anthropic/Temp_0.3/output_F07_Anthropic_temp_03_rep_013.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The bottleneck is **`build_neighbor_lookup`**, which iterates over ~6.46 million rows in a pure-R `lapply`, performing per-row character coercion, `paste`, and named-vector lookups. Each iteration does:

1. `as.character()` + named-vector lookup to map cell ID â†’ reference index.
2. Subsetting the `neighbors` list to get neighbor cell IDs.
3. `paste()` to build `(neighbor_id, year)` keys.
4. Named-vector lookup into `idx_lookup` (~6.46M-element named character vector).

Named-vector lookups on a 6.46M-element vector are **O(n)** string-hashed searches per call. Doing this ~6.46M Ã— ~4 neighbors â‰ˆ **25+ billion character comparisons**. This is why the runtime is 86+ hours.

`compute_neighbor_stats` is a secondary bottleneck: another `lapply` over 6.46M rows, but much cheaper per iteration since it's just numeric subsetting. Still, it's called 5 times (once per variable).

**Root causes:**
1. **Row-level R loop** over 6.46M rows with expensive string operations.
2. **Named character vector lookup** instead of integer hash (environment) or merge/join.
3. **`compute_neighbor_stats` recomputes per variable** instead of vectorizing across all 5 variables at once.
4. The neighbor lookup is **year-invariant** (rook neighbors don't change over time) but is rebuilt per cell-year row as if it were year-specific.

---

## Optimization Strategy

### Key Insight: Separate Spatial Topology from Temporal Expansion

The rook-neighbor graph is **static across years**. The current code re-derives neighbor row indices for every `(cell, year)` pair. Instead:

1. **Build a spatial-only neighbor edge list once** (344K cells, ~1.37M edges) â€” trivially fast.
2. **Exploit the panel structure**: if data is sorted by `(id, year)` or `(year, id)`, the row offset from a cell to the same year of its neighbor is deterministic. But even without that, a single **`data.table` equi-join** on `(neighbor_id, year)` replaces the entire `build_neighbor_lookup`.
3. **Vectorized aggregation**: use `data.table` grouped aggregation (`max`, `min`, `mean`) over the joined result â€” no R-level row loop at all.
4. **Process all 5 variables in one join** instead of 5 separate passes.

**Expected speedup**: from 86+ hours to **~2â€“5 minutes**.

---

## Working R Code

```r
library(data.table)

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 0.  Inputs assumed to exist:
#       cell_data            : data.frame/data.table with columns id, year,
#                              ntl, ec, pop_density, def, usd_est_n2, â€¦
#       id_order             : integer/character vector of cell IDs (length 344,208)
#       rook_neighbors_unique: spdep nb object (list of integer index vectors)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 1.  Build a spatial edge list  (done ONCE, < 1 second)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
build_edge_list <- function(id_order, nb_obj) {
  # nb_obj[[i]] contains integer indices into id_order of neighbors of cell i
  from <- rep(seq_along(nb_obj), lengths(nb_obj))
  to   <- unlist(nb_obj)
  # Remove the 0-neighbor sentinel that spdep uses (integer(0) is fine,

  # but some nb objects store 0L for no-neighbor cells)
  valid <- to != 0L
  data.table(
    focal_id    = id_order[from[valid]],
    neighbor_id = id_order[to[valid]]
  )
}

edges <- build_edge_list(id_order, rook_neighbors_unique)
# edges has ~1,373,394 rows (directed rook pairs)

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 2.  Convert cell_data to data.table (in-place, no copy if already DT)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
setDT(cell_data)

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 3.  Define source variables
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 4.  Single join + grouped aggregation for ALL variables at once
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
compute_all_neighbor_features <- function(cell_data, edges, source_vars) {

  # 4a. Subset to only the columns we need for the neighbor table
  keep_cols <- c("id", "year", source_vars)
  neighbor_vals <- cell_data[, ..keep_cols]

  # 4b. Join: for every (focal_id, year) find neighbor rows

  #     edges supplies (focal_id, neighbor_id);
  #     we join neighbor_vals on neighbor_id == id AND same year.
  #     Result: one row per (focal_id, year, neighbor_id) with neighbor values.
  joined <- edges[neighbor_vals,
                  on = .(neighbor_id = id),
                  allow.cartesian = TRUE,
                  nomatch = NULL,
                  .(focal_id, year, 
                    ntl = i.ntl, ec = i.ec, pop_density = i.pop_density,
                    def = i.def, usd_est_n2 = i.usd_est_n2)]
  # This says: "for each row in neighbor_vals (which has id = some cell and year),
  #  find all edges where neighbor_id == that id, and carry forward focal_id."
  # Result: each row represents focal_cell seeing one neighbor's values in a year.

  # 4c. Aggregate by (focal_id, year)
  agg_exprs <- list()
  for (v in source_vars) {
    sym_v <- as.name(v)
    agg_exprs[[paste0("nb_max_", v)]]  <- substitute(max(V, na.rm = TRUE),  list(V = sym_v))
    agg_exprs[[paste0("nb_min_", v)]]  <- substitute(min(V, na.rm = TRUE),  list(V = sym_v))
    agg_exprs[[paste0("nb_mean_", v)]] <- substitute(mean(V, na.rm = TRUE), list(V = sym_v))
  }
  agg_call <- as.call(c(as.name("list"), agg_exprs))

  stats <- joined[, eval(agg_call), by = .(focal_id, year)]

  # Replace Inf / -Inf (from max/min of zero non-NA values) with NA
  for (col in names(stats)[-(1:2)]) {
    set(stats, which(is.infinite(stats[[col]])), col, NA_real_)
  }

  return(stats)
}

neighbor_stats <- compute_all_neighbor_features(cell_data, edges, neighbor_source_vars)

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 5.  Merge back onto cell_data
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

# Remove old neighbor columns if they exist (idempotency)
old_nb_cols <- grep("^nb_(max|min|mean)_", names(cell_data), value = TRUE)
if (length(old_nb_cols)) cell_data[, (old_nb_cols) := NULL]

cell_data <- merge(cell_data, neighbor_stats,
                   by.x = c("id", "year"),
                   by.y = c("focal_id", "year"),
                   all.x = TRUE, sort = FALSE)

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 6.  Predict with the EXISTING trained Random Forest (unchanged)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# e.g.  cell_data$predicted <- predict(trained_rf, newdata = cell_data)
# The model object is untouched; only the feature-engineering step changed.
```

---

## Why This Is Numerically Identical

| Original code | Optimized code | Equivalence |
|---|---|---|
| For each row `i`, find all rows sharing a rook-neighbor cell ID **and** the same year. Collect their values. Compute `max`, `min`, `mean` (ignoring `NA`). | Join `edges` Ã— `cell_data` on `(neighbor_id = id)`, inheriting `year` from the neighbor row, then group by `(focal_id, year)` and compute `max`, `min`, `mean` (ignoring `NA`). | The set of neighbor values aggregated is identical â€” same cells, same year, same `na.rm = TRUE` logic. The three summary statistics are deterministic functions of that set. âœ“ |

No approximation, sampling, or model retraining is involved. The trained Random Forest is preserved as-is.

---

## Expected Performance

| Step | Rows / Operations | Estimated Time |
|---|---|---|
| Build edge list | 1.37M edges | < 1 sec |
| Join (edges Ã— 6.46M rows) | ~25.8M joined rows | ~30â€“60 sec |
| Grouped aggregation (15 stats) | 6.46M groups | ~30â€“60 sec |
| Merge back | 6.46M rows | ~10 sec |
| **Total** | | **~2â€“5 minutes** |

Peak RAM: ~4â€“6 GB (the joined table of ~25.8M rows Ã— 7 columns is the high-water mark), well within the 16 GB laptop constraint.
END_RESPONSE>>>
