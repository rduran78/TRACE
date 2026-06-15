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
run_id: Anthropic_F02_T07_R018
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F02
prompt_family_slug: CODE_ONLY_NO_HINT
family_label: Code only, no hint
family_group: Code-visible neutral
replicate: 18
raw_output_file: Experiment_Output/Anthropic/Temp_0.7/output_F02_Anthropic_temp_07_rep_018.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The core bottleneck is the **`build_neighbor_lookup` function**, which uses `lapply` over ~6.46 million rows to perform per-row string-pasting and named-vector lookups. Named vector lookups in R (via character keys) are O(n) hash lookups repeated millions of times, and `paste()`-based key construction creates enormous temporary character vectors. The secondary bottleneck is **`compute_neighbor_stats`**, which loops over 6.46 million entries again per variable, calling `max`/`min`/`mean` on small vectors inside `lapply` â€” this is slow due to R-level loop overhead and repeated memory allocation. With 5 variables, this is ~32.3 million small R function calls.

**Summary of problems:**

1. **`build_neighbor_lookup`**: 6.46M iterations of string concatenation + named-vector lookup. Named vector indexing on a 6.46M-length character vector is expensive. This alone likely accounts for the majority of the 86+ hour estimate.
2. **`compute_neighbor_stats`**: R-level `lapply` over 6.46M elements Ã— 5 variables = ~32.3M iterations of small numeric aggregations.
3. **Memory**: Storing `neighbor_lookup` as a list of 6.46M integer vectors is memory-heavy (list overhead per element).
4. **No vectorization or data.table/matrix exploitation**: Everything is scalar R loops.

---

## Optimization Strategy

### Key Insight
The neighbor relationships are defined at the **cell level** (344,208 cells), not at the cell-year level (6.46M rows). The lookup inflates cell-level adjacency to cell-year-level adjacency by replicating the same spatial graph across 28 years. We should **never build a 6.46M-element list**. Instead:

1. **Work at the cell level for adjacency** (344K cells, not 6.46M rows).
2. **Use `data.table` for fast indexed joins** instead of named-vector lookups.
3. **Convert the neighbor list to an edge-list (CSR-like structure)** and use vectorized grouped aggregations instead of per-row `lapply`.
4. **Compute all 5 variables' neighbor stats in one pass** over the edge list per year, or in a single large vectorized join.

### Concrete Plan

- Convert `rook_neighbors_unique` (an `nb` object) into an **edge data.table** with columns `(id_from, id_to)` â€” only ~1.37M rows.
- Join this edge table to the main data by `(id_to, year)` to pull neighbor values.
- Use `data.table` grouped aggregation `[, .(max, min, mean), by = .(id_from, year)]` â€” fully vectorized C-level grouping.
- This replaces both `build_neighbor_lookup` and `compute_neighbor_stats` entirely.

**Expected speedup**: From 86+ hours to **minutes**. The join is ~1.37M edges Ã— 28 years = ~38.4M rows in the expanded edge table, but `data.table` handles this with vectorized C code and efficient memory use well within 16 GB.

---

## Working R Code

```r
library(data.table)

# â”€â”€â”€ Step 1: Convert nb object to edge data.table â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# id_order is the vector mapping position in the nb list to cell id.
# rook_neighbors_unique is an nb object (list of integer vectors of neighbor positions).

build_edge_table <- function(id_order, neighbors) {
  # Pre-allocate by computing total edges
  n_edges <- sum(lengths(neighbors))
  from_id <- integer(n_edges)
  to_id   <- integer(n_edges)
  
  pos <- 1L
  for (i in seq_along(neighbors)) {
    nb_i <- neighbors[[i]]
    if (length(nb_i) > 0 && !(length(nb_i) == 1 && nb_i[1] == 0L)) {
      n <- length(nb_i)
      from_id[pos:(pos + n - 1L)] <- id_order[i]
      to_id[pos:(pos + n - 1L)]   <- id_order[nb_i]
      pos <- pos + n
    }
  }
  # Trim if any nb entries were empty (0-coded)
  data.table(from_id = from_id[1:(pos - 1L)],
             to_id   = to_id[1:(pos - 1L)])
}

edge_dt <- build_edge_table(id_order, rook_neighbors_unique)

# â”€â”€â”€ Step 2: Convert main data to data.table (if not already) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

if (!is.data.table(cell_data)) {
  setDT(cell_data)
}

# â”€â”€â”€ Step 3: Compute neighbor stats for all variables via vectorized join â”€â”€â”€â”€â”€

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# Subset columns needed for the join: id, year, and the source variables
join_cols <- c("id", "year", neighbor_source_vars)
neighbor_vals_dt <- cell_data[, ..join_cols]

# Rename 'id' to 'to_id' for the join
setnames(neighbor_vals_dt, "id", "to_id")

# Key for fast join
setkey(neighbor_vals_dt, to_id, year)

# Expand edge table by year: for each edge (from_id -> to_id), 
# we need every year present for from_id.
# But more efficiently: join edges to neighbor data directly.

# We need from_id's years. Get unique (from_id, year) combos from cell_data.
from_year_dt <- cell_data[, .(from_id = id, year)]
setkey(from_year_dt, from_id)

# Merge edges with from_id's years
# For each (from_id, year), get all to_ids, then look up their values at that year.
setkey(edge_dt, from_id)

# This creates all (from_id, to_id, year) triples
edge_year_dt <- edge_dt[from_year_dt, on = "from_id", allow.cartesian = TRUE, nomatch = 0L]
# Columns: from_id, to_id, year

# Now join to get neighbor values
setkey(edge_year_dt, to_id, year)
edge_year_dt <- neighbor_vals_dt[edge_year_dt, on = .(to_id, year), nomatch = NA]
# Columns: to_id, year, ntl, ec, pop_density, def, usd_est_n2, from_id

# â”€â”€â”€ Step 4: Grouped aggregation â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

# Build the aggregation expression dynamically
agg_exprs <- unlist(lapply(neighbor_source_vars, function(v) {
  list(
    bquote(max(.(as.name(v)),   na.rm = TRUE)),
    bquote(min(.(as.name(v)),   na.rm = TRUE)),
    bquote(mean(.(as.name(v)),  na.rm = TRUE))
  )
}))

agg_names <- unlist(lapply(neighbor_source_vars, function(v) {
  paste0("neighbor_", c("max_", "min_", "mean_"), v)
}))

names(agg_exprs) <- agg_names

# Perform the grouped aggregation in one pass
neighbor_stats <- edge_year_dt[,
  eval(as.call(c(as.name("list"), agg_exprs))),
  by = .(from_id, year)
]

# Replace -Inf/Inf from max/min of all-NA groups with NA
for (col_name in agg_names) {
  vals <- neighbor_stats[[col_name]]
  set(neighbor_stats, i = which(is.infinite(vals)), j = col_name, value = NA_real_)
}

# â”€â”€â”€ Step 5: Join results back to cell_data â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

setkey(neighbor_stats, from_id, year)
setkey(cell_data, id, year)

cell_data <- neighbor_stats[cell_data, on = .(from_id = id, year)]

# Restore the 'id' column name (from_id came from the join)
setnames(cell_data, "from_id", "id")

# â”€â”€â”€ Done. cell_data now has all 15 neighbor feature columns. â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# The trained Random Forest model is untouched.
# The numerical values (max, min, mean of neighbors) are identical to the original.
```

---

## Why This Preserves Correctness

| Concern | Guarantee |
|---|---|
| **Trained RF model** | No model code is touched. Only the feature-engineering pipeline is optimized. The output columns have the same names and identical numerical values. |
| **Numerical estimand** | `max`, `min`, `mean` with `na.rm = TRUE` on the same neighbor sets produce bit-identical results. The edge table encodes exactly the same adjacency as the original `nb` object. |
| **Memory** | The largest intermediate object (`edge_year_dt`) is ~38.4M rows Ã— 7 columns of numeric/integer â‰ˆ 2.1 GB, well within 16 GB. `data.table` operations are memory-efficient and modify in place where possible. |
| **Speed** | All operations are vectorized C-level `data.table` joins and group-by aggregations. Expected wall time: **5â€“20 minutes** on a standard laptop, down from 86+ hours. |
END_RESPONSE>>>
