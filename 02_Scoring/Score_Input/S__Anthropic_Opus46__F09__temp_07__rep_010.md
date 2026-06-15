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
run_id: Anthropic_F09_T07_R010
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F09
prompt_family_slug: ADJACENCY_TABLE_OPTION
family_label: Adjacency-table option
family_group: Representation cue
replicate: 10
raw_output_file: Experiment_Output/Anthropic/Temp_0.7/output_F09_Anthropic_temp_07_rep_010.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The current implementation is catastrophically slow (~86+ hours) for one core reason: **`build_neighbor_lookup` constructs a per-row (cellÃ—year) lookup list of length ~6.46 million, using character-key hashing inside a sequential `lapply` over every row.** This is unnecessary because the neighbor *topology* is fixed across all 28 yearsâ€”cell A's neighbors are the same in 1992 as in 2019. The function re-discovers this for every cell-year combination, inflating work by a factor of 28.

Specific bottlenecks:

| Bottleneck | Why it hurts |
|---|---|
| `lapply` over 6.46M rows in `build_neighbor_lookup` | O(nÃ—T) character key creation and lookup; ~6.46M iterations with `paste`/`setNames` |
| Character-keyed `idx_lookup` via `paste(id, year)` | String allocation and hash-table lookup for every neighbor of every row |
| `compute_neighbor_stats` with per-row `lapply` | Another 6.46M R-level iterations; no vectorization |
| Repeated `do.call(rbind, ...)` on 6.46M-element list | Builds a huge matrix from millions of 3-element vectors |

**The topology (which cell neighbors which cell) never changes year to year.** Only the *attribute values* change. The entire design should separate structure from attributes.

---

## Optimization Strategy

1. **Build the adjacency table exactly once** â€” a two-column `data.table` of `(cell_id, neighbor_id)` with ~1.37M rows, derived from `rook_neighbors_unique`. This is year-invariant.

2. **For each year, join cell attributes onto the adjacency table** â€” use `data.table` keyed joins. This turns the neighbor-stat computation into a grouped aggregation (`max`, `min`, `mean`) on a ~1.37M-row table per year, fully vectorized in C.

3. **Stack yearly results and join back** to the master panel. No per-row `lapply`, no character keys, no 6.46M-element lists.

**Expected speedup:** From ~86 hours to **minutes** (the adjacency table has ~1.37M rows Ã— 28 years = ~38.4M join+group operations total, which `data.table` handles in seconds per variable).

**Memory:** The adjacency table is ~1.37M rows Ã— 2 integer columns â‰ˆ 11 MB. Yearly attribute slices are â‰¤ 344K rows. Well within 16 GB.

---

## Working R Code

```r
library(data.table)

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# STEP 0 : Convert master data to data.table (if not already)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
setDT(cell_data)

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# STEP 1 : Build the year-invariant adjacency table ONCE
#
#   rook_neighbors_unique : an nb object (list of integer index vectors)
#   id_order              : vector of cell IDs in the same order as the nb object
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
build_adjacency_table <- function(id_order, nb_obj) {
  # nb_obj[[i]] contains integer indices into id_order for neighbors of cell i
  # We expand this into a long (cell_id, neighbor_id) data.table
  n <- length(nb_obj)
  from_idx <- rep(seq_len(n), lengths(nb_obj))
  to_idx   <- unlist(nb_obj)

  # Remove the spdep "0 = no neighbors" sentinel if present
  valid <- to_idx != 0L
  from_idx <- from_idx[valid]
  to_idx   <- to_idx[valid]

  adj <- data.table(
    cell_id     = id_order[from_idx],
    neighbor_id = id_order[to_idx]
  )
  return(adj)
}

adj_table <- build_adjacency_table(id_order, rook_neighbors_unique)
# ~1,373,394 rows, two integer (or numeric) columns â€” tiny

cat("Adjacency table rows:", nrow(adj_table), "\n")

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# STEP 2 : Compute neighbor stats for every source variable
#
#   For each variable we:
#     (a) join yearly attributes onto the adjacency table
#     (b) group by (cell_id, year) and compute max, min, mean
#     (c) join the results back onto cell_data
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# Key the master data for fast joins
setkey(cell_data, id, year)

for (var_name in neighbor_source_vars) {

  cat("Processing neighbor stats for:", var_name, "...\n")

  # --- (a) Subset only the columns we need: id, year, and the variable ------
  attr_cols <- c("id", "year", var_name)
  attr_dt   <- cell_data[, ..attr_cols]

  # --- (b) Join neighbor attributes onto the adjacency table -----------------
  #     For every directed edge (cell_id -> neighbor_id) and every year,

  #     look up the neighbor's attribute value.
  #     We join attr_dt onto adj_table by neighbor_id == id.
  #     This is a many-to-many join (each edge Ã— each year the neighbor appears).
  setnames(attr_dt, old = "id", new = "neighbor_id")
  setkey(attr_dt, neighbor_id)
  setkey(adj_table, neighbor_id)

  edge_year <- adj_table[attr_dt, on = "neighbor_id", allow.cartesian = TRUE, nomatch = 0L]
  # edge_year now has columns: cell_id, neighbor_id, year, <var_name>

  # --- (c) Aggregate: for each (cell_id, year) compute max, min, mean --------
  stats <- edge_year[
    !is.na(get(var_name)),
    .(
      nb_max  = max(get(var_name)),
      nb_min  = min(get(var_name)),
      nb_mean = mean(get(var_name))
    ),
    by = .(cell_id, year)
  ]

  # Rename columns to match the expected feature names
  new_names <- paste0(var_name, c("_nb_max", "_nb_min", "_nb_mean"))
  setnames(stats, c("nb_max", "nb_min", "nb_mean"), new_names)

  # --- (d) Join back onto cell_data ------------------------------------------
  setnames(stats, "cell_id", "id")
  setkey(stats, id, year)
  cell_data <- stats[cell_data, on = .(id, year)]
  setkey(cell_data, id, year)

  # Restore attr_dt name for next iteration
  setnames(attr_dt, old = "neighbor_id", new = "id")

  cat("  Done. Added:", paste(new_names, collapse = ", "), "\n")
}

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# STEP 3 : Predict with the already-trained Random Forest
#
#   The trained model object (e.g., `rf_model`) is unchanged.
#   cell_data now contains the same neighbor-stat columns as before.
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Example (adjust to your actual model object and target column):
# cell_data[, prediction := predict(rf_model, newdata = .SD)]

cat("All neighbor features computed. Ready for prediction.\n")
```

---

## Why This Preserves Correctness

| Requirement | How it is met |
|---|---|
| **Trained RF model unchanged** | No retraining; we only rebuild the *input features* identically, then call `predict()` |
| **Numerical estimand preserved** | `max`, `min`, `mean` are computed on exactly the same rook-neighbor sets and the same variable values; the adjacency table is derived from the same `rook_neighbors_unique` nb object |
| **Column names match** | Output columns are named `<var>_nb_max`, `<var>_nb_min`, `<var>_nb_mean` â€” adjust the naming pattern to match whatever the trained model expects |

---

## Expected Performance

| Stage | Old time | New time (est.) |
|---|---|---|
| Build lookup | ~hours (6.46M `paste`+hash) | < 1 sec (vectorized `unlist`) |
| Neighbor stats (5 vars) | ~80+ hours (30M+ R-level iterations) | **2â€“5 minutes** (`data.table` keyed join + grouped agg) |
| **Total** | **~86 hours** | **< 10 minutes** |

The key insight: **separate topology from attributes, build the adjacency table once, and let `data.table` do the join and aggregation in compiled C code instead of R-level loops.**
END_RESPONSE>>>
