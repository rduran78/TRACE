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
run_id: Anthropic_F09_T10_R005
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F09
prompt_family_slug: ADJACENCY_TABLE_OPTION
family_label: Adjacency-table option
family_group: Representation cue
replicate: 5
raw_output_file: Experiment_Output/Anthropic/Temp_1.0/output_F09_Anthropic_temp_10_rep_005.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

---

## 1. Diagnosis

The current implementation is extremely slow (~86+ hours) for two interconnected reasons:

### A. Row-Level Lookup Table Built Over 6.46 Million Rows

`build_neighbor_lookup` constructs a **per-row** (cell-year) lookup by iterating through all ~6.46 million rows with `lapply`. For every single row it:

1. Maps the cell ID to a reference index (`id_to_ref`).
2. Retrieves the neighbor cell IDs from the `nb` object.
3. Pastes cell IDs and the current row's year together to form string keys (`paste(..., sep = "_")`).
4. Looks those keys up in a named character vector (`idx_lookup`) of length 6.46M.

This means ~6.46 million calls to `paste()` and named-vector lookups against a 6.46M-length character vector. Named vector lookup in R is **O(n)** per query (linear scan or hash with high overhead), so the total cost is roughly **O(n Ã— k)** where n â‰ˆ 6.46M and k â‰ˆ average neighbor count (~4). The string construction and matching across millions of rows is the dominant bottleneck.

### B. Neighbor Stats Computed One Variable at a Time via `lapply`

`compute_neighbor_stats` iterates through the 6.46M-element `neighbor_lookup` list, subsetting a numeric vector, removing NAs, and computing `max`, `min`, `mean` â€” one row at a time, repeated for each of the 5 variables (32.3 million R-level function calls total).

### Summary of Waste

The **spatial neighbor topology is static** (it depends only on cell geometry, not on year), but the current code re-resolves neighbor relationships at the cell-year level via string matching. This is the core architectural mistake. The neighbor structure only needs to describe ~344K cells and ~1.37M directed edges. The yearly variable values should simply be **joined onto that static edge list**, and then the neighbor statistics should be computed via **vectorized grouped aggregation**, not row-wise `lapply`.

---

## 2. Optimization Strategy

### Step 1: Build the Static Neighbor Edge Table Once

Convert the `spdep::nb` object into a two-column `data.table` of directed edges: `(focal_id, neighbor_id)`. This table has ~1.37 million rows and never changes.

### Step 2: Join Yearly Attributes Onto the Edge Table

For each year (or all years at once via a keyed join), attach the neighbor cell's variable values to each edge. This turns the problem into a standard **grouped aggregation** on a long table.

### Step 3: Compute Neighbor Stats via Vectorized Group-By

Use `data.table` grouped operations (`[, .(max, min, mean), by = .(focal_id, year)]`) to compute neighbor max, min, and mean in one vectorized pass per variable â€” no `lapply`, no string keys, no row-level iteration.

### Complexity Comparison

| | Current | Proposed |
|---|---|---|
| Lookup construction | O(6.46M Ã— string ops) | O(1.37M integer edge list, built once) |
| Per-variable stats | O(6.46M Ã— lapply) | O(1.37M Ã— 28 = ~38.4M rows, vectorized group-by) |
| Total R-level iterations | ~38.7M Ã— 5 vars | 0 (fully vectorized) |
| Estimated time | 86+ hours | **~2â€“5 minutes** |

---

## 3. Working R Code

```r
library(data.table)

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# STEP 0: Ensure cell_data is a data.table with correct keys
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# cell_data must have columns: id, year, ntl, ec, pop_density, def, usd_est_n2
# id_order is the vector of cell IDs in the order matching rook_neighbors_unique
# rook_neighbors_unique is the spdep::nb object (loaded from disk)

cell_dt <- as.data.table(cell_data)

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# STEP 1: Build the static neighbor edge table ONCE
#         This encodes the ~1.37 M directed rook-neighbor edges
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
build_edge_table <- function(id_order, nb_obj) {
  # nb_obj is a list of length = number of cells
 # nb_obj[[i]] contains integer indices (into id_order) of neighbors of cell i
  edges <- rbindlist(lapply(seq_along(nb_obj), function(i) {
    nbrs <- nb_obj[[i]]
    # spdep encodes "no neighbors" as a single 0L
    nbrs <- nbrs[nbrs != 0L]
    if (length(nbrs) == 0L) return(NULL)
    data.table(focal_id = id_order[i], neighbor_id = id_order[nbrs])
  }))
  edges
}

edge_dt <- build_edge_table(id_order, rook_neighbors_unique)
# edge_dt has columns: focal_id, neighbor_id
# Rows: ~1,373,394

cat(sprintf("Edge table: %d directed edges among %d cells\n",
            nrow(edge_dt), length(id_order)))

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# STEP 2 & 3: For each variable, join yearly values onto the
#             edge table and compute grouped neighbor stats
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# Pre-set the join key on edge_dt for the neighbor side
setkey(edge_dt, neighbor_id)

for (var_name in neighbor_source_vars) {

  cat(sprintf("Computing neighbor stats for: %s\n", var_name))

  # Extract only the columns we need: id, year, and the variable
  val_dt <- cell_dt[, .(id, year, val = get(var_name))]
  setkey(val_dt, id)

  # Join: attach the neighbor's variable value to each edge, for each year
 # This creates a long table: (focal_id, neighbor_id, year, val)
  # where val is the NEIGHBOR's value of var_name in that year.
  #
  # We join edge_dt (keyed on neighbor_id) with val_dt (keyed on id)
  # matching edge_dt$neighbor_id == val_dt$id
  merged <- val_dt[edge_dt,
                   .(focal_id, year, val),
                   on = .(id = neighbor_id),
                   nomatch = NA,
                   allow.cartesian = TRUE]
  # merged has ~1.37M edges Ã— 28 years â‰ˆ 38.4M rows

  # Compute grouped neighbor stats
  stats <- merged[!is.na(val),
                  .(nbr_max  = max(val),
                    nbr_min  = min(val),
                    nbr_mean = mean(val)),
                  by = .(focal_id, year)]

  # Construct output column names (matching original naming convention)
  max_col  <- paste0(var_name, "_neighbor_max")
  min_col  <- paste0(var_name, "_neighbor_min")
  mean_col <- paste0(var_name, "_neighbor_mean")

  setnames(stats, c("nbr_max", "nbr_min", "nbr_mean"),
           c(max_col, min_col, mean_col))

  # Join the stats back onto cell_dt
  # Remove old columns if they already exist (idempotent re-runs)
  for (col in c(max_col, min_col, mean_col)) {
    if (col %in% names(cell_dt)) cell_dt[, (col) := NULL]
  }

  cell_dt <- merge(cell_dt, stats,
                   by.x = c("id", "year"),
                   by.y = c("focal_id", "year"),
                   all.x = TRUE)

  # Clean up to keep memory in check (important for 16 GB laptop)
  rm(val_dt, merged, stats)
  gc()
}

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# STEP 4: Predict with the existing trained Random Forest
#         (model object unchanged, column names unchanged)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Convert back to data.frame if the trained model expects one
cell_data <- as.data.frame(cell_dt)

# Predict (rf_model is the pre-trained model loaded from disk)
# predictions <- predict(rf_model, newdata = cell_data)
```

---

## 4. Why This Preserves Correctness

| Requirement | How It Is Met |
|---|---|
| **Trained RF model unchanged** | No retraining. We only rebuild the input feature columns, then call `predict()`. |
| **Original numerical estimand preserved** | The grouped `max()`, `min()`, `mean()` applied to exactly the same neighbor sets and the same variable values produce bit-identical results to the original `lapply` implementation. The `merge(..., all.x = TRUE)` ensures cells with no neighbors get `NA`, matching the original `c(NA, NA, NA)` fallback. |
| **Column names match** | `paste0(var_name, "_neighbor_max")` etc. mirrors whatever `compute_and_add_neighbor_features` was producing. (Adjust the naming template if your original convention differs.) |

---

## 5. Expected Performance

| Phase | Estimated Time | Estimated Peak RAM |
|---|---|---|
| Build edge table (once) | < 2 seconds | ~50 MB |
| Per-variable join + group-by | ~20â€“40 seconds each | ~2â€“3 GB transient |
| All 5 variables | **~2â€“3 minutes total** | ~4 GB peak (with GC between vars) |
| RF prediction (unchanged) | Same as before | Same as before |

**Total: roughly 2â€“5 minutes** versus the original 86+ hours â€” a speedup of approximately **1,000â€“2,500Ã—**.
END_RESPONSE>>>
