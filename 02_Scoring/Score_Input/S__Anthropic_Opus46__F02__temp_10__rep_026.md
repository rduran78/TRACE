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
run_id: Anthropic_F02_T10_R026
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F02
prompt_family_slug: CODE_ONLY_NO_HINT
family_label: Code only, no hint
family_group: Code-visible neutral
replicate: 26
raw_output_file: Experiment_Output/Anthropic/Temp_1.0/output_F02_Anthropic_temp_10_rep_026.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The core bottleneck is in **`build_neighbor_lookup`**, which uses `lapply` over ~6.46 million rows, performing per-row string pasting and named-vector lookups. Named vector lookups in R via `[` on character keys are **O(n)** hash probes repeated millions of times, and `paste()` inside a per-row loop creates enormous intermediate string allocation. This produces a **list of 6.46 million integer vectors**, consuming substantial memory and taking tens of hours.

**`compute_neighbor_stats`** then iterates over that 6.46M-element list again, once per variable (Ã—5 variables), performing subsetting and summary statistics in pure R loops.

**Specific problems:**

| Component | Issue |
|---|---|
| `build_neighbor_lookup` â€“ `paste(â€¦, sep="_")` inside `lapply` | Creates ~6.46MÃ—k strings per row; massive allocation |
| `build_neighbor_lookup` â€“ `idx_lookup[neighbor_keys]` | Named character vector lookup is slow at scale |
| `build_neighbor_lookup` â€“ output list | ~6.46M list elements, each an integer vector; high memory overhead (~several GB just for list structure) |
| `compute_neighbor_stats` â€“ `lapply` per variable | Pure R loop over 6.46M elements, repeated 5 times |
| General | No vectorization, no use of data.table or matrix operations |

---

## Optimization Strategy

**Key insight:** Instead of building a per-row list lookup, convert the problem to a **tabular join** using `data.table`. The neighbor relationships can be expressed as an edge table `(id, neighbor_id)`. We join this with the panel data on `(neighbor_id, year)` to get neighbor values, then group-by `(id, year)` to compute `max`, `min`, `mean`. This replaces both functions with fully vectorized, indexed operations.

**Steps:**

1. **Expand `rook_neighbors_unique` (nb object) into an edge data.table** `(id, neighbor_id)` â€” done once, ~1.37M rows.
2. **For each variable**, do an equi-join of `edges` with `cell_data` on `(neighbor_id = id, year = year)`, then aggregate by `(id, year)` to get `max`, `min`, `mean`.
3. **Merge** the aggregated stats back into `cell_data`.

**Why this is fast:**
- `data.table` binary-search joins on integer keys are orders of magnitude faster than character named-vector lookups.
- Group-by aggregation in `data.table` is implemented in C and is memory-efficient.
- No list of 6.46M elements; no per-row string allocation.

**Expected improvement:** From ~86+ hours to **minutes** (typically 5â€“20 min depending on disk I/O and RAM pressure). Peak memory stays well within 16 GB.

---

## Working R Code

```r
library(data.table)

# ---------------------------------------------------------------
# STEP 1: Convert the nb object to an edge data.table (one-time)
# ---------------------------------------------------------------
# rook_neighbors_unique is a list of integer vectors (spdep nb object).
# id_order is the vector of cell IDs corresponding to indices 1..N.

build_edge_table <- function(id_order, neighbors) {
  # Pre-allocate by computing total number of edges
  n_edges <- sum(lengths(neighbors))
  
  from_id    <- integer(n_edges)
  to_id      <- integer(n_edges)
  
  pos <- 1L
  for (i in seq_along(neighbors)) {
    nb_i <- neighbors[[i]]
    len  <- length(nb_i)
    if (len == 0L) next
    idx <- pos:(pos + len - 1L)
    from_id[idx] <- id_order[i]
    to_id[idx]   <- id_order[nb_i]
    pos <- pos + len
  }
  
  data.table(id = from_id, neighbor_id = to_id)
}

edges <- build_edge_table(id_order, rook_neighbors_unique)

# ---------------------------------------------------------------
# STEP 2: Convert cell_data to data.table (if not already)
# ---------------------------------------------------------------
if (!is.data.table(cell_data)) {
  setDT(cell_data)
}

# Ensure key columns are integer for fast joins
cell_data[, id   := as.integer(id)]
cell_data[, year := as.integer(year)]
edges[, id          := as.integer(id)]
edges[, neighbor_id := as.integer(neighbor_id)]

# ---------------------------------------------------------------
# STEP 3: For each neighbor source variable, compute stats via join
# ---------------------------------------------------------------
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# Create a minimal keyed lookup table: (id, year, <all source vars>)
# This avoids carrying all 110 columns through the join.
lookup_cols <- c("id", "year", neighbor_source_vars)
lookup_dt   <- cell_data[, ..lookup_cols]
setkey(lookup_dt, id, year)

# Key edges for the join
setnames(edges, c("id", "neighbor_id"))  # ensure names

for (var_name in neighbor_source_vars) {
  
  cat("Processing neighbor stats for:", var_name, "\n")
  
  # Build a small table: (neighbor_id, year, value)
  val_dt <- lookup_dt[, .(neighbor_id = id, year, value = get(var_name))]
  setkey(val_dt, neighbor_id, year)
  
  # Join edges with cell_data to get each cell's year,

  # then join with val_dt to get the neighbor's value for that year.
  # Strategy: 
  #   1. Attach 'year' to each edge by joining edges with cell_data on 'id'
  #   2. Then join with val_dt on (neighbor_id, year) to get neighbor value
  #   3. Aggregate by (id, year)
  
  # To avoid a huge intermediate table, we do it in one chain:
  # First, create (id, year) from cell_data (unique combinations)
  id_year <- unique(cell_data[, .(id, year)])
  setkey(id_year, id)
  setkey(edges, id)
  
  # Expand: each (id, year) Ã— neighbors of id â†’ (id, year, neighbor_id)
  expanded <- edges[id_year, on = "id", allow.cartesian = TRUE, nomatch = 0L]
  # expanded has columns: id, neighbor_id, year
  
  # Now join to get neighbor values
  setkey(expanded, neighbor_id, year)
  expanded[val_dt, on = .(neighbor_id, year), value := i.value]
  
  # Aggregate: max, min, mean per (id, year), ignoring NAs
  agg <- expanded[!is.na(value), 
                  .(nb_max  = max(value),
                    nb_min  = min(value),
                    nb_mean = mean(value)), 
                  by = .(id, year)]
  
  # Rename columns to match expected output pattern
  new_names <- paste0(var_name, c("_nb_max", "_nb_min", "_nb_mean"))
  setnames(agg, c("nb_max", "nb_min", "nb_mean"), new_names)
  
  # Merge back into cell_data
  setkey(agg, id, year)
  setkey(cell_data, id, year)
  
  # Remove old columns if they exist (idempotent re-runs)
  for (nn in new_names) {
    if (nn %in% names(cell_data)) cell_data[, (nn) := NULL]
  }
  
  cell_data <- agg[cell_data, on = .(id, year)]
  
  # Clean up to free memory

  rm(val_dt, expanded, agg)
  gc()
  
  cat("  Done.\n")
}

# ---------------------------------------------------------------
# STEP 4: Use the existing trained Random Forest for prediction
# ---------------------------------------------------------------
# The trained RF model object is unchanged. 
# cell_data now contains the same neighbor feature columns 
# (e.g., ntl_nb_max, ntl_nb_min, ntl_nb_mean, etc.)
# with numerically identical values, so predictions are preserved.
#
# Example (unchanged from original pipeline):
# predictions <- predict(rf_model, newdata = cell_data)
```

---

## Summary of Guarantees

| Concern | Status |
|---|---|
| **Numerical equivalence** | Identical `max`, `min`, `mean` computed on the same non-NA neighbor values; no approximation. |
| **Trained RF model preserved** | No retraining; same feature names and values. |
| **Memory** | `expanded` table is the peak object (~1.37M edges Ã— 28 years â‰ˆ 38M rows Ã— 3 int/dbl cols â‰ˆ ~1 GB per variable, freed each iteration). Total well within 16 GB. |
| **Speed** | `data.table` keyed joins + grouped aggregation: expected ~5â€“20 minutes total vs. 86+ hours. |
| **Dependencies** | Only `data.table` (no Rcpp, no parallelism needed, though both could further help). |
END_RESPONSE>>>
