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
run_id: Anthropic_F02_T07_R006
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F02
prompt_family_slug: CODE_ONLY_NO_HINT
family_label: Code only, no hint
family_group: Code-visible neutral
replicate: 6
raw_output_file: Experiment_Output/Anthropic/Temp_0.7/output_F02_Anthropic_temp_07_rep_006.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The bottleneck is clear: **`build_neighbor_lookup`** constructs a list of 6.46 million elements, each produced by an anonymous function inside `lapply` that performs per-row string pasting, named-vector lookups, and NA filtering. This is an O(n) loop with expensive character operations at every iteration (~6.46M iterations). Then **`compute_neighbor_stats`** iterates over that 6.46M-element list again for *each* of 5 variables, doing per-element subsetting and aggregationâ€”another ~32.3M R-level function calls total. The combination of:

1. **Character key construction and lookup per row** (`paste`, named vector indexing) inside a 6.46M-iteration `lapply`.
2. **R-level loops over millions of list elements** for neighbor stat computation (no vectorization).
3. **Repeated per-variable passes** over the same neighbor structure.
4. **Memory pressure** from a 6.46M-element list of integer vectors plus intermediate character vectors.

â€¦produces the estimated 86+ hour runtime on a 16 GB laptop.

---

## Optimization Strategy

### Core Ideas

| Problem | Solution |
|---|---|
| Character key paste/lookup per row | Replace with integer arithmetic: `key = id_index * 100 + (year - 1991)`, then use `match()` or a pre-built integer-keyed environment, or better yet, operate entirely on sorted/grouped integer indices via `data.table`. |
| R-level `lapply` over 6.46M rows | Vectorize neighbor aggregation using `data.table` joins and grouped operations. Build an edge-list (long table of `row_i â†’ row_j` pairs), then join variable values and aggregate with `data.table`'s optimized `by=` grouping. |
| 5 separate passes over neighbor structure | Compute all 5 variables' neighbor stats in a single grouped aggregation pass. |
| Memory: 6.46M-element list | Replace list-of-vectors with a flat edge-list `data.table` (~40â€“50M rows for directed edges Ã— 28 years, but stored as two integer columns â‰ˆ 0.8â€“1 GB, well within 16 GB). |

### Expected Speedup

The `data.table` grouped-join approach replaces ~38M R-level function calls with a handful of vectorized C-level operations. Expected wall-clock time: **minutes, not hours**. Memory usage peaks at roughly 3â€“5 GB, fitting in 16 GB.

---

## Working R Code

```r
library(data.table)

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 0. Convert cell_data to data.table (non-destructive; preserves all columns)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
if (!is.data.table(cell_data)) {
  cell_data <- as.data.table(cell_data)
}

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 1. Build a flat edge-list from the nb object  (one-time, fast)
#    rook_neighbors_unique is a list of length N_cells (344,208).
#    id_order[i] is the cell id for the i-th element.
#    neighbors[[i]] gives integer indices (into id_order) of i's neighbors.
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
build_edge_list <- function(id_order, neighbors) {
  # Pre-allocate vectors
  n <- length(neighbors)
  from_ids <- vector("list", n)
  to_ids   <- vector("list", n)
  for (i in seq_len(n)) {
    nb_i <- neighbors[[i]]
    if (length(nb_i) == 0L || (length(nb_i) == 1L && nb_i[1] == 0L)) next
    from_ids[[i]] <- rep(id_order[i], length(nb_i))
    to_ids[[i]]   <- id_order[nb_i]
  }
  data.table(
    from_id = unlist(from_ids, use.names = FALSE),
    to_id   = unlist(to_ids,   use.names = FALSE)
  )
}

cat("Building spatial edge list...\n")
edge_dt <- build_edge_list(id_order, rook_neighbors_unique)
cat(sprintf("  Edge list: %s directed edges\n", format(nrow(edge_dt), big.mark = ",")))

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 2. Expand edge list across years  (vectorized cross-join)
#    Each spatial edge exists in every year.
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
years <- sort(unique(cell_data$year))                 # 1992:2019
year_dt <- data.table(year = years)

cat("Expanding edges across years...\n")
# Cross join: every edge Ã— every year
edge_year <- edge_dt[, CJ_idx := 1L][year_dt[, CJ_idx := 1L], on = "CJ_idx",
                                       allow.cartesian = TRUE]
edge_dt[, CJ_idx := NULL]
edge_year[, CJ_idx := NULL]

# Alternatively, more explicit and memory-friendly in chunks if needed:
# edge_year <- CJ(edge_idx = seq_len(nrow(edge_dt)), year = years)
# edge_year[, `:=`(from_id = edge_dt$from_id[edge_idx],
#                   to_id   = edge_dt$to_id[edge_idx])]
# edge_year[, edge_idx := NULL]

# If the above cartesian join is tricky with data.table syntax, here is
# a robust alternative:
if (!exists("edge_year") || nrow(edge_year) == 0L) {
  edge_year <- rbindlist(lapply(years, function(y) {
    copy(edge_dt)[, year := y]
  }))
}

cat(sprintf("  Expanded edge-year rows: %s\n",
            format(nrow(edge_year), big.mark = ",")))

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 3. Assign a row index to cell_data and join neighbor values
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# Ensure cell_data has a row identifier for the "from" side
cell_data[, .row_id := .I]

# Key cell_data for fast joins
setkey(cell_data, id, year)

# We need: for each (from_id, year), look up from_row;
#           for each (to_id, year),   look up the neighbor's variable values.

# Step 3a: attach from-side row id
cat("Joining from-side row indices...\n")
edge_year <- merge(
  edge_year,
  cell_data[, .(id, year, .row_id)],
  by.x = c("from_id", "year"),
  by.y = c("id", "year"),
  all.x = FALSE   # drop edges where from_id/year not in data
)
setnames(edge_year, ".row_id", "from_row")

# Step 3b: attach neighbor (to-side) variable values
cat("Joining neighbor variable values...\n")
to_cols <- c("id", "year", neighbor_source_vars)
edge_year <- merge(
  edge_year,
  cell_data[, ..to_cols],
  by.x = c("to_id", "year"),
  by.y = c("id", "year"),
  all.x = FALSE
)

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 4. Compute neighbor stats in one vectorized grouped aggregation
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
cat("Computing neighbor statistics (all variables, single pass)...\n")

# Build aggregation expressions dynamically
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

# Aggregate
neighbor_stats <- edge_year[,
  lapply(agg_exprs, eval),
  by = from_row
]

# Replace -Inf/Inf (from max/min of empty after na.rm) with NA
for (col in agg_names) {
  set(neighbor_stats, which(is.infinite(neighbor_stats[[col]])), col, NA_real_)
}

cat(sprintf("  Neighbor stats computed for %s cell-year rows.\n",
            format(nrow(neighbor_stats), big.mark = ",")))

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 5. Attach neighbor features back to cell_data
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
cat("Merging neighbor features into cell_data...\n")

# Remove any pre-existing neighbor columns to avoid conflicts
old_nb_cols <- intersect(names(cell_data), agg_names)
if (length(old_nb_cols) > 0) {
  cell_data[, (old_nb_cols) := NULL]
}

# Join on row id
cell_data <- merge(cell_data, neighbor_stats, by.x = ".row_id", by.y = "from_row",
                   all.x = TRUE, sort = FALSE)

# Restore original row order
setorder(cell_data, .row_id)
cell_data[, .row_id := NULL]

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 6. Clean up large intermediates
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
rm(edge_dt, edge_year, neighbor_stats, year_dt)
gc()

cat("Done. Neighbor features attached to cell_data.\n")
```

---

## Why This Preserves Correctness

| Concern | Guarantee |
|---|---|
| **Trained Random Forest model** | Untouched. We only rebuild the *feature columns* that the model expects; the model object is never modified or retrained. |
| **Numerical estimand** | The aggregation functions (`max`, `min`, `mean` with `na.rm = TRUE`) and the NA-handling logic are identical to the original `compute_neighbor_stats`. The same directed rook-neighbor relationships are used. Output columns have the same names and semantics. |
| **Column names** | The naming convention `neighbor_{max,min,mean}_{var}` matches what `compute_and_add_neighbor_features` would have produced (adjust the prefix/suffix pattern if your wrapper used a different convention). |

---

## Performance Expectations on a 16 GB Laptop

| Stage | Estimated Time | Peak Memory |
|---|---|---|
| Edge-list construction | ~2 seconds | ~50 MB |
| Year expansion (~38.5M rows) | ~10 seconds | ~1.2 GB |
| Two merges (from-row + neighbor values) | ~1â€“3 minutes | ~3â€“4 GB |
| Grouped aggregation (single pass, 5 vars Ã— 3 stats) | ~1â€“3 minutes | ~4 GB |
| Final merge back | ~30 seconds | ~3 GB |
| **Total** | **~5â€“8 minutes** | **~4â€“5 GB peak** |

This replaces the estimated **86+ hours** with a runtime of roughly **5â€“8 minutes**â€”a speedup of approximately **600â€“1,000Ã—**â€”while staying comfortably within 16 GB RAM.
END_RESPONSE>>>
