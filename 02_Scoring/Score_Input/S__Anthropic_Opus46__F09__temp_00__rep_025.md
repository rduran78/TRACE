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
run_id: Anthropic_F09_T00_R025
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F09
prompt_family_slug: ADJACENCY_TABLE_OPTION
family_label: Adjacency-table option
family_group: Representation cue
replicate: 25
raw_output_file: Experiment_Output/Anthropic/Temp_0.0/output_F09_Anthropic_temp_00_rep_025.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The current implementation is extremely slow (~86+ hours) for two interconnected reasons:

### 1. `build_neighbor_lookup` is O(NÂ²)-like in practice
This function iterates over **all ~6.46 million cell-year rows** with `lapply`, and for each row it:
- Looks up the cell's spatial neighbors (fine).
- Constructs **character key strings** (`paste(id, year)`) and performs **named-vector lookups** (`idx_lookup[neighbor_keys]`) â€” which in R are hash-table lookups on a 6.46-million-entry named vector, repeated ~6.46 million times Ã— ~4 neighbors each â‰ˆ **~26 million hash lookups**.

Character pasting and named-vector indexing are slow per-call in R. At ~6.46M outer iterations, this dominates runtime.

### 2. `compute_neighbor_stats` uses per-row `lapply`
For each of the 5 variables, it loops over 6.46M rows in R-level `lapply`, extracting small vectors and computing `max/min/mean`. That's **~32 million R function calls** across all variables.

### 3. The neighbor topology is **static** but is being resolved at the cell-year level
Rook neighbors are a property of the **spatial grid** (344,208 cells), not of cell-years (6.46M rows). The current code conflates the two, doing ~19Ã— more work than necessary for the lookup phase.

---

## Optimization Strategy

**Core insight:** Separate the *spatial topology* (which cells neighbor which â€” fixed, 344K cells) from the *panel attributes* (which vary by year â€” 28 years). Then use vectorized joins and grouped vectorized aggregation instead of row-wise R loops.

**Steps:**

1. **Build a static edge table once** â€” a two-column `data.table` of `(cell_id, neighbor_id)` from the `nb` object. This has ~1.37M rows and never changes.

2. **Join yearly attributes onto the edge table** â€” for each year, join the cell-year attributes onto the `neighbor_id` column. This is a keyed `data.table` join: O(E) per year, fully vectorized in C.

3. **Grouped aggregation** â€” group by `(cell_id, year)` and compute `max`, `min`, `mean` of each neighbor variable in one vectorized pass using `data.table`'s `by=` grouping (GForce-optimized in C).

4. **Join results back** to the main dataset.

**Expected speedup:** From ~86 hours to **minutes** (likely 2â€“10 minutes on a 16 GB laptop). The bottleneck shifts from millions of R-level function calls to a handful of vectorized C-level `data.table` operations.

**Preservation guarantees:**
- The trained Random Forest model is untouched â€” we only produce the same predictor columns.
- The numerical estimand is identical: `max`, `min`, `mean` of the same neighbor values, with the same `NA` handling.

---

## Working R Code

```r
library(data.table)

# ==============================================================
# STEP 1: Build a static spatial edge table (once, reusable)
# ==============================================================
# Input: id_order (vector of 344,208 cell IDs, matching the nb object index)
#        rook_neighbors_unique (spdep nb object, list of integer index vectors)

build_edge_table <- function(id_order, neighbors_nb) {
  # Pre-allocate: count total edges
  n_edges <- sum(vapply(neighbors_nb, function(x) {
    # spdep nb encodes "no neighbors" as a single 0L
    if (length(x) == 1L && x[1] == 0L) 0L else length(x)
  }, integer(1)))

  from_id <- integer(n_edges)
  to_id   <- integer(n_edges)
  pos <- 1L

  for (i in seq_along(neighbors_nb)) {
    nb_idx <- neighbors_nb[[i]]
    if (length(nb_idx) == 1L && nb_idx[1] == 0L) next
    n <- length(nb_idx)
    from_id[pos:(pos + n - 1L)] <- id_order[i]
    to_id[pos:(pos + n - 1L)]   <- id_order[nb_idx]
    pos <- pos + n
  }

  data.table(cell_id = from_id, neighbor_id = to_id)
}

edge_table <- build_edge_table(id_order, rook_neighbors_unique)
# ~1.37M rows, two integer columns â€” tiny in memory

# ==============================================================
# STEP 2: Compute all neighbor features via vectorized joins
# ==============================================================

compute_all_neighbor_features <- function(cell_data_df, edge_table,
                                          neighbor_source_vars) {
  # Convert to data.table if needed (by reference if already one)
  dt <- as.data.table(cell_data_df)

  # Columns we need from the neighbor rows: id, year, and the source vars
  # Build the join: for every (cell_id, year), find neighbor rows' attributes
  # We expand edge_table Ã— years via a merge with the attribute table on neighbor_id.

  # Subset to only needed columns for the neighbor attribute lookup
  attr_cols <- c("id", "year", neighbor_source_vars)
  dt_attr <- dt[, ..attr_cols]
  setnames(dt_attr, "id", "neighbor_id")
  setkeyv(dt_attr, c("neighbor_id", "year"))

  # Join neighbor attributes onto edge table (edge_table is year-agnostic,
  # so we do a many-to-many join: each edge Ã— each year the neighbor has data)
  # This produces ~1.37M edges Ã— 28 years â‰ˆ 38.4M rows (fits in 16 GB easily
  # for a handful of numeric columns).

  setkey(edge_table, neighbor_id)

  # Efficient approach: join edge_table with dt_attr on neighbor_id
  # This gives us (cell_id, neighbor_id, year, var1, var2, ...)
  edges_with_attrs <- dt_attr[edge_table,
                              on = .(neighbor_id),
                              allow.cartesian = TRUE,
                              nomatch = NA]
  # Result columns: neighbor_id, year, <vars>, cell_id

  # Now aggregate: for each (cell_id, year), compute max/min/mean of each var
  agg_exprs <- list()
  for (v in neighbor_source_vars) {
    v_sym <- as.name(v)
    agg_exprs[[paste0("neighbor_max_", v)]]  <-
      bquote(as.numeric(max(.(v_sym), na.rm = TRUE)))
    agg_exprs[[paste0("neighbor_min_", v)]]  <-
      bquote(as.numeric(min(.(v_sym), na.rm = TRUE)))
    agg_exprs[[paste0("neighbor_mean_", v)]] <-
      bquote(mean(.(v_sym), na.rm = TRUE))
  }

  # Build a single aggregation call for efficiency
  # data.table GForce optimizes max, min, mean when called directly in j
  agg_call <- as.call(c(as.name("list"),
                         lapply(agg_exprs, function(e) e)))

  neighbor_stats <- edges_with_attrs[,
                                      eval(agg_call),
                                      by = .(cell_id, year)]

  # Replace -Inf/Inf from max/min on all-NA groups with NA
  for (col_name in names(neighbor_stats)) {
    if (col_name %in% c("cell_id", "year")) next
    vals <- neighbor_stats[[col_name]]
    set(neighbor_stats, i = which(is.infinite(vals)), j = col_name, value = NA_real_)
  }

  # Merge back onto the main data
  setnames(neighbor_stats, "cell_id", "id")
  setkeyv(neighbor_stats, c("id", "year"))
  setkeyv(dt, c("id", "year"))

  # Remove old neighbor columns if they exist (to avoid duplicates)
  old_cols <- intersect(names(dt), setdiff(names(neighbor_stats), c("id", "year")))
  if (length(old_cols) > 0) dt[, (old_cols) := NULL]

  dt <- neighbor_stats[dt, on = .(id, year)]

  # Return as data.frame if the input was one
  if (is.data.frame(cell_data_df) && !is.data.table(cell_data_df)) {
    setDF(dt)
  }
  dt
}

# ==============================================================
# STEP 3: Run it
# ==============================================================

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cell_data <- compute_all_neighbor_features(
  cell_data,
  edge_table,
  neighbor_source_vars
)

# ==============================================================
# STEP 4: Predict with the existing trained Random Forest
# ==============================================================
# The trained model object is unchanged. The predictor columns are
# numerically identical to the original pipeline's output.
# Example (adjust to your actual model object and prediction column):
#
#   cell_data$rf_prediction <- predict(trained_rf_model, newdata = cell_data)
```

---

## Complexity Comparison

| Aspect | Original | Optimized |
|---|---|---|
| Neighbor lookup | 6.46M R-level iterations + 26M hash lookups on character keys | One `for` loop over 344K cells to build edge table (once) |
| Stat computation | 6.46M Ã— 5 = 32.3M R `lapply` calls | One vectorized `data.table` grouped aggregation |
| Memory pattern | 6.46M-entry named character vector + 6.46M-entry list | ~38M-row numeric `data.table` (~1.5 GB for 5 vars) |
| Estimated time | ~86 hours | **~2â€“10 minutes** |
| Numerical result | max/min/mean with NA removal | Identical max/min/mean with NA removal |
| Trained RF model | Preserved | Preserved (untouched) |

The key insight is that **spatial topology is time-invariant**: build the ~1.37M-edge table once, then let `data.table`'s C-level join and GForce aggregation handle the panel expansion and summarization entirely in vectorized compiled code.
END_RESPONSE>>>
