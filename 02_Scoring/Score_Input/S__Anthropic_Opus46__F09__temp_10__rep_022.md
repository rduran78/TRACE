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
run_id: Anthropic_F09_T10_R022
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F09
prompt_family_slug: ADJACENCY_TABLE_OPTION
family_label: Adjacency-table option
family_group: Representation cue
replicate: 22
raw_output_file: Experiment_Output/Anthropic/Temp_1.0/output_F09_Anthropic_temp_10_rep_022.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The current implementation is extremely slow (~86+ hours) for three compounding reasons:

### 1. `build_neighbor_lookup` is O(NÂ²)-flavored
The function iterates over all **6.46 million cell-year rows** with `lapply`, and for each row it:
- Looks up the cell's spatial neighbors (fine).
- Constructs **character paste-keys** (`"id_year"`) for every neighbor Ã— every year.
- Performs **named-vector lookups** (`idx_lookup[neighbor_keys]`) â€” which are hash lookups on a 6.46-million-entry named vector, repeated ~6.46 million times.

This means ~6.46M Ã— ~4 neighbors Ã— (string paste + hash lookup) â‰ˆ **billions of character operations**. This is the primary bottleneck.

### 2. The lookup is **year-redundant**
Neighbor topology is purely spatial â€” cell A's neighbors are the same in 1992 as in 2019. Yet the code rebuilds the full cell-year-level lookup, repeating the identical spatial logic 28 times (once per year).

### 3. `compute_neighbor_stats` uses row-level `lapply`
Even after the lookup is built, `lapply` over 6.46M elements calling `max/min/mean` one-at-a-time is slow in R. A vectorized or `data.table` group-by approach would be orders of magnitude faster.

---

## Optimization Strategy

**Core insight:** Separate the **spatial topology** (which cells are neighbors) from the **temporal attributes** (yearly variable values). Build the topology once as an edge table, then use `data.table` joins and grouped aggregations for each variable.

| Step | What | Complexity |
|------|------|-----------|
| 1 | Build a **cell-level edge table** from `rook_neighbors_unique`: one row per directed neighbor pair `(cell_id, neighbor_id)`. ~1.37M rows, built once. | O(cells Ã— avg_neighbors) |
| 2 | For each variable, **join** the yearly attributes onto the edge table by `(neighbor_id, year)` â€” this is a keyed `data.table` merge, extremely fast. | O(edges Ã— years) â‰ˆ 38M rows |
| 3 | **Group-by** `(cell_id, year)` to compute `max`, `min`, `mean` in one vectorized pass. | O(edges Ã— years) |
| 4 | **Join** the aggregated stats back onto `cell_data`. | O(N) |

**Expected speedup:** From ~86 hours to **minutes** (typically 5â€“15 minutes total on 16 GB RAM).

**Memory:** The edge table expanded by years is ~38M rows Ã— a few columns â€” well within 16 GB.

**Preserves:** The trained Random Forest model is untouched; the numerical outputs (neighbor max, min, mean) are identical.

---

## Working R Code

```r
library(data.table)

# ==============================================================================
# STEP 0: Convert cell_data to data.table (if not already)
# ==============================================================================
cell_data <- as.data.table(cell_data)

# ==============================================================================
# STEP 1: Build the spatial edge table ONCE from the nb object
#         rook_neighbors_unique is a list of length 344,208;
#         rook_neighbors_unique[[i]] contains integer indices of neighbors of
#         the i-th element of id_order.
# ==============================================================================
build_edge_table <- function(id_order, neighbors_nb) {
  # Pre-allocate: count total edges
  n_edges <- sum(lengths(neighbors_nb))  # ~1,373,394
  
  from_id <- integer(n_edges)
  to_id   <- integer(n_edges)
  
  pos <- 1L
  for (i in seq_along(neighbors_nb)) {
    nb_idx <- neighbors_nb[[i]]
    # spdep::nb uses 0 to indicate no neighbors; filter those out
    nb_idx <- nb_idx[nb_idx > 0L]
    n_nb   <- length(nb_idx)
    if (n_nb > 0L) {
      from_id[pos:(pos + n_nb - 1L)] <- id_order[i]
      to_id[pos:(pos + n_nb - 1L)]   <- id_order[nb_idx]
      pos <- pos + n_nb
    }
  }
  
  # Trim if any 0-neighbor cells caused over-allocation
  if (pos - 1L < n_edges) {
    from_id <- from_id[1:(pos - 1L)]
    to_id   <- to_id[1:(pos - 1L)]
  }
  
  data.table(cell_id = from_id, neighbor_id = to_id)
}

cat("Building spatial edge table...\n")
edge_table <- build_edge_table(id_order, rook_neighbors_unique)
cat(sprintf("  Edge table: %d directed edges\n", nrow(edge_table)))

# ==============================================================================
# STEP 2: Expand edge table by years (cross join with unique years)
#         This creates the full (cell_id, neighbor_id, year) table.
#         ~1.37M edges Ã— 28 years â‰ˆ 38.5M rows.
# ==============================================================================
years_dt <- data.table(year = sort(unique(cell_data$year)))
edge_year <- edge_table[, CJ_idx := 1L][
  years_dt[, CJ_idx := 1L],
  on = "CJ_idx",
  allow.cartesian = TRUE
][, CJ_idx := NULL]

cat(sprintf("  Edge-year table: %d rows\n", nrow(edge_year)))

# ==============================================================================
# STEP 3: For each neighbor source variable, compute neighbor stats
#         and join back onto cell_data.
# ==============================================================================
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# Key cell_data for fast joins
# We need to look up variable values by (id, year) for the NEIGHBOR cell
setkey(cell_data, id, year)

for (var_name in neighbor_source_vars) {
  cat(sprintf("Processing neighbor stats for: %s\n", var_name))
  
  # --- 3a: Extract only the columns we need for the neighbor lookup ---
  # Columns: neighbor's id, year, and the variable value
  vals_dt <- cell_data[, .(id, year, val = get(var_name))]
  setkey(vals_dt, id, year)
  
  # --- 3b: Join neighbor values onto the edge-year table ---
  #         Match on neighbor_id == id AND year == year
  edge_vals <- merge(
    edge_year,
    vals_dt,
    by.x = c("neighbor_id", "year"),
    by.y = c("id", "year"),
    all.x = TRUE,      # keep edges even if neighbor has NA
    sort = FALSE
  )
  
  # --- 3c: Aggregate by (cell_id, year) â†’ max, min, mean ---
  #         Remove NAs from val before aggregation (matches original logic)
  agg <- edge_vals[
    !is.na(val),
    .(
      nb_max  = max(val),
      nb_min  = min(val),
      nb_mean = mean(val)
    ),
    by = .(cell_id, year)
  ]
  
  # Name the columns to match the expected feature names
  # Original code uses: <var>_max, <var>_min, <var>_mean
  max_col  <- paste0(var_name, "_max")
  min_col  <- paste0(var_name, "_min")
  mean_col <- paste0(var_name, "_mean")
  
  setnames(agg, c("nb_max", "nb_min", "nb_mean"),
                c(max_col,  min_col,  mean_col))
  
  # --- 3d: Remove old columns from cell_data if they exist (idempotent) ---
  for (col in c(max_col, min_col, mean_col)) {
    if (col %in% names(cell_data)) {
      cell_data[, (col) := NULL]
    }
  }
  
  # --- 3e: Join aggregated stats back onto cell_data ---
  cell_data <- merge(
    cell_data,
    agg,
    by.x = c("id", "year"),
    by.y = c("cell_id", "year"),
    all.x = TRUE,
    sort = FALSE
  )
  
  cat(sprintf("  Added: %s, %s, %s\n", max_col, min_col, mean_col))
}

# Re-key for downstream use
setkey(cell_data, id, year)

cat("Neighbor feature computation complete.\n")

# ==============================================================================
# STEP 4: Predict with the existing trained Random Forest model
#         (model object is assumed to already be in memory, e.g. `rf_model`)
# ==============================================================================
# Ensure cell_data has exactly the predictor columns the model expects.
# predict() on a ranger/randomForest object works directly:
#
#   cell_data$prediction <- predict(rf_model, data = cell_data)$predictions
#
# The trained model is NOT retrained â€” only used for inference.
# ==============================================================================
```

---

## Summary of Changes and Guarantees

| Concern | Guarantee |
|---|---|
| **Numerical equivalence** | `max`, `min`, `mean` are computed on the identical set of non-NA neighbor values per `(cell_id, year)` as the original code. The edge table faithfully reproduces `rook_neighbors_unique`. |
| **Trained RF model** | Completely untouched. No retraining. The code only prepares the same predictor columns for `predict()`. |
| **Memory** | Peak is the 38.5M-row `edge_vals` table (~1â€“2 GB with one numeric column). Well within 16 GB. Each variable's `edge_vals` can be garbage-collected after its iteration. |
| **Speed** | `data.table` keyed merge + grouped aggregation on 38.5M rows typically completes in **seconds per variable**. Total wall time: **5â€“15 minutes** vs. 86+ hours. |
| **Column naming** | Uses `<var>_max`, `<var>_min`, `<var>_mean` â€” adjust the naming pattern if the original `compute_and_add_neighbor_features` used a different convention (inspect the RF model's feature names to confirm). |
END_RESPONSE>>>
