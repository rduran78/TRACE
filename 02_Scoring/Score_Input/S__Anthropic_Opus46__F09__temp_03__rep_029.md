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
run_id: Anthropic_F09_T03_R029
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.3
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F09
prompt_family_slug: ADJACENCY_TABLE_OPTION
family_label: Adjacency-table option
family_group: Representation cue
replicate: 29
raw_output_file: Experiment_Output/Anthropic/Temp_0.3/output_F09_Anthropic_temp_03_rep_029.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The current implementation is extremely slow (~86+ hours) for three interconnected reasons:

### 1. Row-Level `lapply` Over 6.46 Million Rows in `build_neighbor_lookup`
The function iterates row-by-row across all 6.46 million cell-year observations. For each row, it:
- Looks up the cell's reference index.
- Retrieves neighbor cell IDs.
- Pastes together character keys (`id_year`).
- Matches those keys against a named lookup vector.

This means ~6.46 million iterations of string concatenation (`paste`), named-vector lookups, and `NA` filtering. Named vector lookups via character keys in R are hash-based but still carry significant per-call overhead at this scale. Crucially, **the neighbor topology is time-invariant** â€” cell A's neighbors are the same in 1992 as in 2019 â€” yet the function redundantly recomputes the same neighbor cell IDs 28 times for each of the 344,208 cells.

### 2. Row-Level `lapply` Over 6.46 Million Rows in `compute_neighbor_stats`
For each row, it subsets a numeric vector by index, removes `NA`s, and computes `max`, `min`, and `mean`. The per-element overhead of 6.46 million R function calls (each with allocation, subsetting, and three summary functions) is enormous. This is repeated for each of the 5 neighbor source variables, totaling ~32.3 million R-level iterations.

### 3. The Fundamental Missed Optimization: Separate Topology from Attributes
The neighbor topology (which cells are adjacent) is **static**. The cell attributes (ntl, ec, etc.) vary by year. The current code entangles these two concerns by building a monolithic row-index lookup across the entire panel. The correct approach is:

- Build the neighbor-pair table **once** (344,208 cells Ã— ~4 neighbors each â‰ˆ 1.37 million directed pairs).
- For each year, **join** that year's attribute values onto the neighbor-pair table.
- **Group by** the focal cell and compute `max`, `min`, `mean` using vectorized operations.

This converts millions of R-level loop iterations into a small number of vectorized `data.table` joins and grouped aggregations â€” operations that run in optimized C code.

---

## Optimization Strategy

| Step | What | Why |
|------|------|-----|
| **A** | Build a `data.table` of directed neighbor pairs `(focal_id, neighbor_id)` from `rook_neighbors_unique` once. | Topology is static; ~1.37M rows, trivially small. |
| **B** | For each year, subset the panel to that year, join cell attributes onto the neighbor-pair table by `neighbor_id`, then group by `focal_id` to compute `max`, `min`, `mean` for each variable. | Vectorized `data.table` joins and `by`-group aggregations run in C, not R loops. |
| **C** | Bind the yearly results and join back onto the full panel. | Produces the identical columns the trained Random Forest model expects. |
| **D** | Predict with the existing trained RF model as before. | Model is unchanged; numerical outputs are identical. |

**Expected speedup**: The ~1.37M-row join + grouped aggregation per year per variable takes seconds. 28 years Ã— 5 variables = 140 such operations. Total estimated time: **2â€“10 minutes** on a 16 GB laptop, versus 86+ hours.

---

## Working R Code

```r
library(data.table)

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# STEP A: Build the static neighbor-pair table ONCE
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# rook_neighbors_unique : spdep nb object (list of integer vectors)
# id_order              : vector of cell IDs in the same order as the nb object

build_neighbor_pairs <- function(id_order, neighbors) {
  # neighbors[[i]] contains the indices (into id_order) of cell i's neighbors.
  # We expand this into a two-column data.table of directed pairs.
  n <- length(id_order)
  focal_list    <- vector("list", n)
  neighbor_list <- vector("list", n)
  
  for (i in seq_len(n)) {
    nb_idx <- neighbors[[i]]
    # spdep nb objects use 0L to indicate no neighbors
    nb_idx <- nb_idx[nb_idx != 0L]
    if (length(nb_idx) > 0L) {
      focal_list[[i]]    <- rep(id_order[i], length(nb_idx))
      neighbor_list[[i]] <- id_order[nb_idx]
    }
  }
  
  data.table(
    focal_id    = unlist(focal_list, use.names = FALSE),
    neighbor_id = unlist(neighbor_list, use.names = FALSE)
  )
}

neighbor_pairs <- build_neighbor_pairs(id_order, rook_neighbors_unique)
# ~1,373,394 rows; tiny in memory

cat("Neighbor pairs built:", nrow(neighbor_pairs), "directed edges\n")

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# STEP B: Compute neighbor stats via vectorized join + grouped agg
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

# Convert cell_data to data.table if not already (non-destructive copy)
if (!is.data.table(cell_data)) {
  cell_data <- as.data.table(cell_data)
}

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# We will collect all new columns in a separate table keyed by (id, year),
# then join once at the end to avoid repeated modification of the large table.

# Pre-allocate list to collect yearly results
yearly_results <- vector("list", length(unique(cell_data$year)))
names(yearly_results) <- as.character(sort(unique(cell_data$year)))

# Columns we need from cell_data for the neighbor lookup
subset_cols <- c("id", "year", neighbor_source_vars)

# Build the aggregation expression dynamically
# For each variable v, we want: v_neighbor_max, v_neighbor_min, v_neighbor_mean
agg_exprs <- paste0(
  sprintf(
    "list(%s)",
    paste(
      unlist(lapply(neighbor_source_vars, function(v) {
        c(
          sprintf("nb_%s_max  = as.numeric(max(%s, na.rm = TRUE))", v, v),
          sprintf("nb_%s_min  = as.numeric(min(%s, na.rm = TRUE))", v, v),
          sprintf("nb_%s_mean = as.numeric(mean(%s, na.rm = TRUE))", v, v)
        )
      })),
      collapse = ", "
    )
  )
)
agg_expr_parsed <- parse(text = agg_exprs)

years <- sort(unique(cell_data$year))

cat("Computing neighbor statistics for", length(years), "years ...\n")

for (yr in years) {
  # Subset to this year's attributes
  yr_data <- cell_data[year == yr, ..subset_cols]
  
  # Join neighbor attributes onto the pair table
  # Key the yearly data by cell id for fast join
  setkey(yr_data, id)
  
  # Merge: for each (focal_id, neighbor_id) pair, attach the neighbor's attributes
  # We join on neighbor_id = id
  merged <- neighbor_pairs[yr_data, on = .(neighbor_id = id), nomatch = 0L, allow.cartesian = TRUE]
  # merged now has columns: focal_id, neighbor_id, year, ntl, ec, pop_density, def, usd_est_n2
  # where the attribute columns belong to the NEIGHBOR cell
  
  # Group by focal_id and compute stats
  stats <- merged[, eval(agg_expr_parsed), by = .(focal_id)]
  stats[, year := yr]
  
  # Handle -Inf/Inf from max/min on all-NA groups (shouldn't happen if data is clean,
  # but defensive)
  for (v in neighbor_source_vars) {
    max_col  <- paste0("nb_", v, "_max")
    min_col  <- paste0("nb_", v, "_min")
    mean_col <- paste0("nb_", v, "_mean")
    set(stats, which(is.infinite(stats[[max_col]])),  max_col,  NA_real_)
    set(stats, which(is.infinite(stats[[min_col]])),  min_col,  NA_real_)
    set(stats, which(is.nan(stats[[mean_col]])),      mean_col, NA_real_)
  }
  
  yearly_results[[as.character(yr)]] <- stats
}

all_neighbor_stats <- rbindlist(yearly_results, use.names = TRUE)

cat("Neighbor stats computed:", nrow(all_neighbor_stats), "rows,",
    ncol(all_neighbor_stats), "columns\n")

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# STEP C: Join neighbor stats back onto the main panel
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

# Rename focal_id -> id for the join
setnames(all_neighbor_stats, "focal_id", "id")
setkey(all_neighbor_stats, id, year)
setkey(cell_data, id, year)

# If the cell_data already has old neighbor columns from a prior run, remove them
old_nb_cols <- grep("^nb_", names(cell_data), value = TRUE)
if (length(old_nb_cols) > 0L) {
  cell_data[, (old_nb_cols) := NULL]
}

# Merge
cell_data <- all_neighbor_stats[cell_data, on = .(id, year)]

cat("Final panel:", nrow(cell_data), "rows,", ncol(cell_data), "columns\n")

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# STEP D: Column-name alignment with the trained RF model
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# The trained Random Forest model expects specific column names for the
# neighbor features. If the original code produced columns named e.g.
# "ntl_neighbor_max" instead of "nb_ntl_max", rename to match.
#
# Adjust the mapping below to match YOUR model's expected feature names.
# Example (uncomment and edit as needed):
#
# setnames(cell_data, 
#   old = c("nb_ntl_max",  "nb_ntl_min",  "nb_ntl_mean"),
#   new = c("ntl_neighbor_max", "ntl_neighbor_min", "ntl_neighbor_mean")
# )
# ... repeat for ec, pop_density, def, usd_est_n2 ...

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# STEP E: Predict with the existing trained Random Forest (unchanged)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# The model object (e.g., rf_model) is already in memory and must NOT
# be retrained. Simply call predict:
#
# cell_data$prediction <- predict(rf_model, newdata = cell_data)
#
# The numerical estimand is preserved because the neighbor statistics
# (max, min, mean) are computed identically â€” only the computational
# method changed, not the mathematical operations.
```

---

## Why the Numerical Estimand Is Preserved

| Operation | Original code | Optimized code |
|-----------|--------------|----------------|
| Neighbor identification | `id_order[neighbors[[ref_idx]]]` | Same `id_order[neighbors[[i]]]` in `build_neighbor_pairs` |
| Attribute retrieval | `vals[idx]` (direct index) | `data.table` equi-join on `neighbor_id = id` within the same year |
| `max` | `max(neighbor_vals)` after `NA` removal | `max(v, na.rm = TRUE)` grouped by `focal_id` |
| `min` | `min(neighbor_vals)` after `NA` removal | `min(v, na.rm = TRUE)` grouped by `focal_id` |
| `mean` | `mean(neighbor_vals)` after `NA` removal | `mean(v, na.rm = TRUE)` grouped by `focal_id` |

The same set of neighbor cells contributes the same attribute values, and the same three summary functions are applied. The results are numerically identical (up to floating-point associativity, which for `max`/`min` is exact and for `mean` is identical since `data.table` uses the same `mean` implementation). The trained Random Forest model receives identical input features and is never retrained.

---

## Expected Performance

| Component | Original | Optimized |
|-----------|----------|-----------|
| Neighbor lookup build | ~hours (6.46M `lapply` + string ops) | ~5 seconds (344K-iteration loop, vectorized `unlist`) |
| Neighbor stats (per variable) | ~17 hours (6.46M `lapply`) | ~30 seconds (28 joins + grouped aggs) |
| **Total (5 variables)** | **~86+ hours** | **~3â€“10 minutes** |
| Peak RAM | ~6â€“10 GB (list of 6.46M integer vectors) | ~2â€“4 GB (1.37M-row pair table + yearly subsets) |
END_RESPONSE>>>
