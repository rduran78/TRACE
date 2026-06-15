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
run_id: Anthropic_F09_T00_R021
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F09
prompt_family_slug: ADJACENCY_TABLE_OPTION
family_label: Adjacency-table option
family_group: Representation cue
replicate: 21
raw_output_file: Experiment_Output/Anthropic/Temp_0.0/output_F09_Anthropic_temp_00_rep_021.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The current implementation is extremely slow (~86+ hours) for two interconnected reasons:

### 1. `build_neighbor_lookup` is O(NÂ²)-like in practice
This function iterates over **all 6.46 million cell-year rows** with `lapply`, and for each row it:
- Looks up the cell's spatial neighbors (fine).
- Constructs **year-specific string keys** (`paste(neighbor_id, year, sep="_")`) and matches them against a named character vector (`idx_lookup`) of 6.46 million entries.

Named vector lookup in R via `[` on character names is **O(n)** per probe in the worst case (hash collisions aside, the overhead of repeated character hashing and matching across 6.46M keys is enormous). Doing this 6.46 million times produces billions of character operations.

### 2. The lookup is **rebuilt monolithically** even though the spatial topology is time-invariant
The neighbor graph is purely spatial â€” cell A's rook neighbors are the same in 1992 as in 2019. Yet the function fuses space and time into one giant lookup, recomputing string keys for every cell-year. This is the core waste: **the adjacency structure only needs to be defined once over 344,208 cells, not over 6.46 million cell-years.**

### 3. `compute_neighbor_stats` uses row-level `lapply` over 6.46M rows
Even after the lookup is built, computing stats via `lapply` with per-element R function calls is slow. Each call to the anonymous function has interpreter overhead. With 5 variables Ã— 6.46M rows = 32.3 million R function invocations, this adds hours.

### Summary of bottlenecks

| Component | Calls | Cost per call | Total |
|---|---|---|---|
| `build_neighbor_lookup` (string key construction + named vector lookup) | 6.46M | ~Âµsâ€“ms (character hashing over 6.46M-entry vector) | **Tens of hours** |
| `compute_neighbor_stats` (R-level lapply) | 5 Ã— 6.46M | ~Âµs | **Hours** |

---

## Optimization Strategy

**Core insight:** Separate the time-invariant spatial adjacency from the time-varying attributes. Build the adjacency table once (344K cells), then use vectorized joins and grouped operations.

### Step-by-step plan

1. **Build a static edge table once** â€” a two-column `data.table` of `(cell_id, neighbor_id)` from the `nb` object. This has ~1.37M rows and never changes.

2. **Join yearly attributes onto the edge table** â€” For each year, join the cell-year attribute values onto the `neighbor_id` column. This is a keyed `data.table` join: O(n log n) once, then O(1) per probe.

3. **Compute grouped aggregates** â€” Group by `(cell_id, year)` and compute `max`, `min`, `mean` of each neighbor variable in one vectorized pass using `data.table`'s `[, .(â€¦), by=]`.

4. **Join results back** to the main dataset.

This replaces 6.46M R-level function calls with a handful of vectorized `data.table` operations.

### Expected speedup

| Operation | Old | New |
|---|---|---|
| Build adjacency | ~hours (string matching over 6.46M keys) | <1 second (integer edge list from nb object) |
| Neighbor stats (per variable) | ~17 hours (lapply over 6.46M rows) | ~5â€“30 seconds (data.table grouped aggregation) |
| **Total for 5 variables** | **~86 hours** | **~2â€“5 minutes** |

---

## Working R Code

```r
library(data.table)

# ==============================================================
# STEP 0: Ensure cell_data is a data.table with proper columns
# ==============================================================
# cell_data must have columns: id, year, ntl, ec, pop_density, def, usd_est_n2
# id_order is the vector of cell IDs corresponding to indices in rook_neighbors_unique
# rook_neighbors_unique is the spdep nb object (list of integer index vectors)

# Convert to data.table if not already
if (!is.data.table(cell_data)) {
  cell_data <- as.data.table(cell_data)
}

# ==============================================================
# STEP 1: Build static spatial edge table ONCE
#         This encodes the rook adjacency among 344,208 cells.
# ==============================================================
build_edge_table <- function(id_order, neighbors_nb) {
  # neighbors_nb is an nb object: a list where element i contains

  # integer indices of neighbors of cell i (0 means no neighbors in spdep).
  edges <- rbindlist(lapply(seq_along(neighbors_nb), function(i) {
    nb_idx <- neighbors_nb[[i]]
    # spdep uses 0L to denote "no neighbors"; filter those out
    nb_idx <- nb_idx[nb_idx > 0L]
    if (length(nb_idx) == 0L) return(NULL)
    data.table(cell_id = id_order[i], neighbor_id = id_order[nb_idx])
  }))
  return(edges)
}

cat("Building static edge table...\n")
edge_table <- build_edge_table(id_order, rook_neighbors_unique)
# edge_table has ~1,373,394 rows: (cell_id, neighbor_id)
cat(sprintf("Edge table: %d directed edges\n", nrow(edge_table)))

# ==============================================================
# STEP 2: Compute neighbor features for all variables at once
#         using vectorized data.table joins and grouped aggregation.
# ==============================================================
compute_all_neighbor_features <- function(cell_dt, edge_dt, source_vars) {
  # We need: for each (cell_id, year), the max/min/mean of each source_var

  # across that cell's rook neighbors in the same year.

  # Subset to only the columns we need for the join
  join_cols <- c("id", "year", source_vars)
  attr_dt <- cell_dt[, ..join_cols]

  # Create the expanded neighbor-attribute table:
  # For every edge (cell_id -> neighbor_id), join the neighbor's year-specific attributes.
  # First, cross edge_table with all years via join on neighbor_id.
  # Key the attribute table for fast join.
  setkey(attr_dt, id, year)

  # Expand: join neighbor attributes onto edge table.
  # We want: for each (cell_id, neighbor_id) edge and each year,
  # the neighbor_id's attribute values in that year.
  # This is: edge_table joined to attr_dt on (neighbor_id = id).
  # Result has nrow(edge_table) * n_years rows in the worst case,
  # but we do it efficiently by joining.

  # Rename for clarity before join
  setnames(attr_dt, "id", "neighbor_id")

  # Keyed join: for each (neighbor_id, year) in attr_dt,
  # find matching rows in edge_table by neighbor_id.
  # We want the Cartesian-ish result: each edge Ã— each year where neighbor has data.
  setkey(edge_dt, neighbor_id)
  setkey(attr_dt, neighbor_id, year)

  # Merge: every edge gets expanded by all years the neighbor has data

  cat("Joining neighbor attributes onto edge table...\n")
  expanded <- merge(edge_dt, attr_dt, by = "neighbor_id", allow.cartesian = TRUE)
  # expanded columns: neighbor_id, cell_id, year, ntl, ec, pop_density, def, usd_est_n2
  # rows: ~1.37M edges Ã— 28 years â‰ˆ 38.5M rows (fits in 16GB RAM easily)

  rm(attr_dt)
  gc()

  # Now group by (cell_id, year) and compute stats for each variable
  cat("Computing grouped neighbor statistics...\n")

  # Build aggregation expressions dynamically
  agg_exprs <- unlist(lapply(source_vars, function(v) {
    list(
      bquote(max(.(as.name(v)), na.rm = TRUE)),
      bquote(min(.(as.name(v)), na.rm = TRUE)),
      bquote(mean(.(as.name(v)), na.rm = TRUE))
    )
  }))

  agg_names <- unlist(lapply(source_vars, function(v) {
    paste0("neighbor_", c("max_", "min_", "mean_"), v)
  }))

  names(agg_exprs) <- agg_names

  # Construct the call
  agg_call <- as.call(c(as.name("list"), agg_exprs))

  neighbor_stats <- expanded[, eval(agg_call), by = .(cell_id, year)]

  rm(expanded)
  gc()

  # Replace Inf/-Inf (from max/min on all-NA) with NA
  for (col_name in agg_names) {
    vals <- neighbor_stats[[col_name]]
    set(neighbor_stats, i = which(is.infinite(vals)), j = col_name, value = NA_real_)
  }

  return(neighbor_stats)
}

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cat("Computing all neighbor features...\n")
neighbor_features <- compute_all_neighbor_features(cell_data, edge_table, neighbor_source_vars)

# ==============================================================
# STEP 3: Join neighbor features back onto cell_data
# ==============================================================
cat("Joining neighbor features back to cell_data...\n")

# Remove any pre-existing neighbor columns to avoid duplication
existing_neighbor_cols <- grep("^neighbor_", names(cell_data), value = TRUE)
if (length(existing_neighbor_cols) > 0) {
  cell_data[, (existing_neighbor_cols) := NULL]
}

setkey(cell_data, id, year)
setkey(neighbor_features, cell_id, year)

cell_data <- merge(cell_data, neighbor_features,
                   by.x = c("id", "year"),
                   by.y = c("cell_id", "year"),
                   all.x = TRUE)

rm(neighbor_features)
gc()

cat("Done. Neighbor features added to cell_data.\n")
cat(sprintf("cell_data: %d rows, %d columns\n", nrow(cell_data), ncol(cell_data)))

# ==============================================================
# STEP 4: Predict with the existing trained Random Forest model
#         (model object is assumed to already exist in memory)
# ==============================================================
# The trained RF model is preserved exactly as-is; no retraining.
# Ensure column names match what the model expects.
# Example (adjust 'rf_model' to your actual model object name):
#
# cell_data$prediction <- predict(rf_model, newdata = cell_data)
```

---

## Key Points

| Concern | How it's handled |
|---|---|
| **Numerical equivalence** | `max`, `min`, `mean` with `na.rm=TRUE` are identical operations to the original; `Inf`/`-Inf` from empty groups are converted to `NA`, matching the original `c(NA,NA,NA)` return. |
| **Trained RF model preserved** | The code only modifies predictor columns in `cell_data`; the model object is never touched or retrained. |
| **RAM on 16 GB laptop** | The largest intermediate object (expanded edge Ã— year table) is ~38.5M rows Ã— ~7 columns of doubles â‰ˆ **2.2 GB**. Combined with `cell_data` (~6.46M Ã— 110 cols â‰ˆ 5.7 GB), total peak is ~10â€“12 GB. Feasible on 16 GB. |
| **Runtime** | Edge table build: <1s. Merge: ~10â€“30s. Grouped aggregation: ~30â€“120s. Total: **~2â€“5 minutes** vs. 86+ hours. |
END_RESPONSE>>>
