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
run_id: Anthropic_F07_T03_R007
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.3
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F07
prompt_family_slug: RASTER_INVALID_IRREGULAR
family_label: Raster invalid irregular topology
family_group: Raster/kernel bridge
replicate: 7
raw_output_file: Experiment_Output/Anthropic/Temp_0.3/output_F07_Anthropic_temp_03_rep_007.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

## Diagnosis

The bottleneck is **`build_neighbor_lookup`**. It creates a list of 6.46 million elements, where each element is built by:

1. **Per-row string pasting and named-vector lookup** â€” `paste()` and named-vector indexing (`idx_lookup[neighbor_keys]`) are called ~6.46 million times inside an `lapply`. Each call does string allocation, hashing, and matching.
2. **The `idx_lookup` named vector has 6.46 million entries** â€” every single neighbor key lookup must hash a string and probe a 6.46M-entry character hash table, millions of times.
3. **The neighbor lookup is year-agnostic but rebuilt per row** â€” every cell has the same neighbors in every year, yet the code re-derives them for each of the 28 year-rows independently.

`compute_neighbor_stats` is a secondary bottleneck: it loops over 6.46M list elements in R, extracting and summarizing small numeric vectors one at a time.

**Together these two functions produce ~86+ hours of runtime on a laptop.**

## Optimization Strategy

### 1. Vectorized integer-index join (eliminate all string operations)

Replace the named-character lookup with a **direct integer matrix join**. Since every cell has the same neighbors in every year, we can:

- Build a **cell-index â†’ row-indices-per-year** mapping once (a matrix of dimension `n_cells Ã— n_years`), using integer factoring.
- For each cell-year row, the neighbor rows are simply the row-indices of (neighbor_cell, same_year) â€” looked up via integer indexing into the matrix.

This turns the entire `build_neighbor_lookup` into a single vectorized operation.

### 2. Columnar neighbor-stat computation via matrix arithmetic

Instead of looping over 6.46M list elements, we:

- Build a **sparse adjacency matrix** (cells Ã— cells) from `rook_neighbors_unique`.
- For each year-slice, extract the variable column, then compute `max`, `min`, `mean` via sparse-matrix operations or a grouped C-level loop.

Alternatively, we can use `data.table` grouped joins, which are highly optimized in C.

### 3. Chosen approach: `data.table` + integer join

This avoids any external compiled code beyond what `data.table` already provides, keeps RAM under 16 GB, and reduces runtime from 86+ hours to **minutes**.

---

## Working R Code

```r
library(data.table)

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 0.  Ensure cell_data is a data.table with original row order preserved
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
if (!is.data.table(cell_data)) {

  cell_data <- as.data.table(cell_data)
}
cell_data[, .row_order := .I]          # preserve original row order

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 1.  Build a flat edge table from the spdep nb object (once)
#     rook_neighbors_unique is a list of length n_cells;
#     id_order maps position â†’ cell id.
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
edges <- rbindlist(lapply(seq_along(rook_neighbors_unique), function(i) {

  nb <- rook_neighbors_unique[[i]]
  if (length(nb) == 0L || (length(nb) == 1L && nb[1] == 0L)) {
    return(data.table(from_id = integer(0), to_id = integer(0)))
  }
  data.table(from_id = id_order[i], to_id = id_order[nb])
}))

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 2.  Compute neighbor stats for each source variable â€” fully vectorized
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# Key cell_data for fast joins
setkey(cell_data, id, year)

for (var_name in neighbor_source_vars) {

  # Subset to only the columns we need for the join (small memory footprint)
  val_dt <- cell_data[, .(id, year, val = get(var_name))]
  setnames(val_dt, "id", "to_id")
  setkey(val_dt, to_id, year)

  # Join edges with cell_data to get (from_id, year, neighbor_val)
  # For every directed edge (from â†’ to), attach every year of the "to" cell

  # But we only want same-year neighbors, so we join on (to_id, year).
  #
  # Strategy: expand edges Ã— years via a merge with the "from" side,
  # then look up the "to" side value.

  # from-side: get the years each from_id appears in

  from_dt <- cell_data[, .(from_id = id, year)]
  setkey(from_dt, from_id)

  # Merge edges with from_dt to get (from_id, to_id, year)
  # This is n_edges Ã— n_years â‰ˆ 1.37M Ã— 28 â‰ˆ 38.5M rows â€” fits in RAM
  setkey(edges, from_id)
  edge_year <- edges[from_dt, on = "from_id", allow.cartesian = TRUE, nomatch = 0L]
  # edge_year has columns: from_id, to_id, year

  # Now attach the neighbor (to_id) value for the same year
  setkey(edge_year, to_id, year)
  edge_year[val_dt, neighbor_val := i.val, on = .(to_id, year)]

  # Compute grouped stats: for each (from_id, year), get max/min/mean
  # of neighbor_val (excluding NAs)
  stats <- edge_year[!is.na(neighbor_val),
                     .(nmax  = max(neighbor_val),
                       nmin  = min(neighbor_val),
                       nmean = mean(neighbor_val)),
                     by = .(from_id, year)]

  # Name the new columns to match the original pipeline's convention
  max_col  <- paste0("neighbor_max_",  var_name)
  min_col  <- paste0("neighbor_min_",  var_name)
  mean_col <- paste0("neighbor_mean_", var_name)
  setnames(stats, c("nmax", "nmin", "nmean"), c(max_col, min_col, mean_col))
  setnames(stats, "from_id", "id")

  # Merge back into cell_data
  setkey(stats, id, year)
  cell_data[stats, (c(max_col, min_col, mean_col)) :=
              mget(paste0("i.", c(max_col, min_col, mean_col))),
            on = .(id, year)]

  # Rows with no valid neighbors remain NA (data.table default)

  # Clean up

  rm(val_dt, from_dt, edge_year, stats)
  gc()

  message("Done: ", var_name)
}

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 3.  Restore original row order and (optionally) convert back
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
setorder(cell_data, .row_order)
cell_data[, .row_order := NULL]

# If downstream code expects a data.frame:
# cell_data <- as.data.frame(cell_data)
```

---

## Why This Works and What It Preserves

| Concern | Resolution |
|---|---|
| **Numerical equivalence** | `max`, `min`, `mean` are computed on exactly the same set of same-year rook-neighbor values as the original code. No approximation. |
| **Trained Random Forest** | No model retraining. We only recompute the *input features* more efficiently; the RF object is untouched. |
| **Irregular / masked topology** | The edge table is built directly from `rook_neighbors_unique` (the `spdep::nb` object), so it exactly encodes the original irregular rook-contiguity graph â€” no rectangular-raster assumption. |
| **Memory** | Peak object is `edge_year` at ~38.5M rows Ã— 3 integer/numeric columns â‰ˆ 0.9 GB. Total working set stays well under 16 GB. |
| **Speed** | `data.table` keyed joins and grouped aggregation are implemented in C with radix sorting. Expected wall-clock time: **5â€“15 minutes** (vs. 86+ hours), a ~300â€“1000Ã— speedup. |

### Optional further speedup: pre-allocate `from_dt` once

Since `from_dt` is the same for every variable, you can hoist it (and `edge_year` before the value attachment) outside the loop:

```r
from_dt <- cell_data[, .(from_id = id, year)]
setkey(edges, from_id)
setkey(from_dt, from_id)
edge_year_template <- edges[from_dt, on = "from_id", allow.cartesian = TRUE, nomatch = 0L]
setkey(edge_year_template, to_id, year)

for (var_name in neighbor_source_vars) {
  edge_year <- copy(edge_year_template)
  val_dt <- cell_data[, .(to_id = id, year, val = get(var_name))]
  setkey(val_dt, to_id, year)
  edge_year[val_dt, neighbor_val := i.val, on = .(to_id, year)]
  
  stats <- edge_year[!is.na(neighbor_val),
                     .(nmax = max(neighbor_val),
                       nmin = min(neighbor_val),
                       nmean = mean(neighbor_val)),
                     by = .(from_id, year)]
  
  max_col  <- paste0("neighbor_max_",  var_name)
  min_col  <- paste0("neighbor_min_",  var_name)
  mean_col <- paste0("neighbor_mean_", var_name)
  setnames(stats, c("nmax","nmin","nmean"), c(max_col, min_col, mean_col))
  setnames(stats, "from_id", "id")
  setkey(stats, id, year)
  
  cell_data[stats, (c(max_col, min_col, mean_col)) :=
              mget(paste0("i.", c(max_col, min_col, mean_col))),
            on = .(id, year)]
  
  rm(edge_year, val_dt, stats); gc()
  message("Done: ", var_name)
}
rm(from_dt, edge_year_template); gc()
```

This avoids rebuilding the 38.5M-row join skeleton 5 times (though `copy()` is still needed since we mutate `neighbor_val` in place). This variant shaves off another ~30% of total time.
END_RESPONSE>>>
