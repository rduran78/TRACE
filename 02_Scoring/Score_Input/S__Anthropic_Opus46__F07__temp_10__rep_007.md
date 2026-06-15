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
run_id: Anthropic_F07_T10_R007
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F07
prompt_family_slug: RASTER_INVALID_IRREGULAR
family_label: Raster invalid irregular topology
family_group: Raster/kernel bridge
replicate: 7
raw_output_file: Experiment_Output/Anthropic/Temp_1.0/output_F07_Anthropic_temp_10_rep_007.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The bottleneck is **`build_neighbor_lookup`**, which runs an R-level `lapply` over **~6.46 million rows**, each iteration performing character coercion, `paste`, and named-vector lookups. This produces a list of 6.46M integer vectors. Then `compute_neighbor_stats` iterates over that list again 5 times (once per variable), each time subsetting with per-element R loops. The combined cost:

| Source | Why it's slow |
|---|---|
| `build_neighbor_lookup` | 6.46M iterations of `paste()` + named-vector lookup (`idx_lookup[neighbor_keys]`). Named vector lookup is O(n) hash probing per call, and `paste` allocates millions of small strings. Estimated: **~60-70 hours alone**. |
| `compute_neighbor_stats` | 6.46M R-level iterations Ã— 5 variables. Each iteration subsets a numeric vector, removes NAs, computes max/min/mean. Estimated: **~15-20 hours**. |
| Memory | Storing a 6.46M-element list of integer vectors is large (~10+ GB with R list overhead), risking swap on a 16 GB machine. |

The fundamental problem: the operation is **embarrassingly vectorizable** but the code uses **scalar R loops** over millions of rows.

---

## Optimization Strategy

### 1. Replace the row-level list with a sparse adjacency matrix approach using `data.table` joins

Instead of building a per-row neighbor list, we:

1. **Expand** the `nb` object into an edge-list of `(cell_id, neighbor_cell_id)` â€” only ~1.37M edges.
2. **Cross-join with years** to get `(cell_id, year, neighbor_cell_id, year)` â€” ~1.37M Ã— 28 = ~38.5M rows, but this is a simple equi-join, not a Cartesian product, because both sides share the same year.
3. **Join** against the data to pull neighbor values.
4. **Aggregate** (max, min, mean) grouped by `(cell_id, year)`.

This replaces 6.46M R-level iterations with **vectorized `data.table` grouped aggregation** on ~38.5M rows â€” typically finishing in **seconds to a few minutes**.

### 2. Memory management

- The edge list is ~1.37M Ã— 2 integers â‰ˆ 11 MB.
- After year-expansion and join: ~38.5M rows Ã— a few columns â‰ˆ < 2 GB.
- Well within 16 GB.

### 3. Preserve the estimand

The new code computes **exactly** the same `max`, `min`, and `mean` of rook-neighbor values per `(cell_id, year)`, excluding `NA` neighbors (same semantics as the original). The trained Random Forest model is never touched.

**Expected runtime: ~2â€“10 minutes total** (down from 86+ hours).

---

## Working R Code

```r
library(data.table)

# ---------------------------------------------------------------
# Step 1: Convert spdep nb object to an edge list (done once)
# ---------------------------------------------------------------
# rook_neighbors_unique is a list where element i contains the
# indices (into id_order) of rook neighbors of cell id_order[i].
# id_order is the vector of cell IDs corresponding to each nb element.

build_edge_list <- function(id_order, neighbors) {
  # neighbors is an nb object (list of integer vectors of neighbor indices)
  from <- rep(
    seq_along(neighbors),
    times = lengths(neighbors)
  )
  to <- unlist(neighbors, use.names = FALSE)
  data.table(
    id          = id_order[from],
    neighbor_id = id_order[to]
  )
}

edge_dt <- build_edge_list(id_order, rook_neighbors_unique)
# edge_dt has columns: id, neighbor_id  (~1.37M rows)

# ---------------------------------------------------------------
# Step 2: Convert cell_data to data.table (if not already)
# ---------------------------------------------------------------
if (!is.data.table(cell_data)) {
  cell_data <- as.data.table(cell_data)
}

# Ensure keyed for fast joins
setkey(cell_data, id, year)

# ---------------------------------------------------------------
# Step 3: For each neighbor source variable, compute neighbor
#         max, min, mean via a single vectorized join + aggregate
# ---------------------------------------------------------------
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

compute_neighbor_features_fast <- function(cell_data, edge_dt, var_name) {
  # Subset to only the columns we need for the join
  # cell_data must have: id, year, <var_name>
  lookup_cols <- c("id", "year", var_name)
  lookup_dt   <- cell_data[, ..lookup_cols]

  # Rename for neighbor side of join
  setnames(lookup_dt, old = c("id", var_name),
           new = c("neighbor_id", "nval"))

  # Join edges with years:
  #   edge_dt (id, neighbor_id) joined to lookup_dt (neighbor_id, year, nval)
  #   This gives us (id, neighbor_id, year, nval) for every
  #   cell-year-neighbor combination.
  setkey(lookup_dt, neighbor_id, year)
  setkey(edge_dt, neighbor_id)

  # Expand edges Ã— years by joining on neighbor_id

  joined <- lookup_dt[edge_dt, on = "neighbor_id", allow.cartesian = TRUE, nomatch = NA]
  # joined now has columns: neighbor_id, year, nval, id
  # (id comes from edge_dt via the join)

  # Remove rows where year is NA (shouldn't happen) or nval is NA
  joined <- joined[!is.na(nval)]

  # Aggregate: for each (id, year), compute max, min, mean of nval
  agg <- joined[, .(
    nmax  = max(nval),
    nmin  = min(nval),
    nmean = mean(nval)
  ), by = .(id, year)]

  # Rename columns to match original naming convention
  # Original code creates columns like: ntl_neighbor_max, ntl_neighbor_min, ntl_neighbor_mean
  new_names <- paste0(var_name, c("_neighbor_max", "_neighbor_min", "_neighbor_mean"))
  setnames(agg, old = c("nmax", "nmin", "nmean"), new = new_names)

  agg
}

# ---------------------------------------------------------------
# Step 4: Compute and merge all neighbor features into cell_data
# ---------------------------------------------------------------
for (var_name in neighbor_source_vars) {
  cat("Computing neighbor stats for:", var_name, "\n")

  agg <- compute_neighbor_features_fast(cell_data, edge_dt, var_name)

  # Remove old columns if they exist (idempotent re-runs)
  old_cols <- paste0(var_name, c("_neighbor_max", "_neighbor_min", "_neighbor_mean"))
  for (col in old_cols) {
    if (col %in% names(cell_data)) cell_data[, (col) := NULL]
  }

  # Merge back onto cell_data
  cell_data <- merge(cell_data, agg, by = c("id", "year"), all.x = TRUE)

  cat("  Done. Rows:", nrow(cell_data), "\n")
}

# Re-key after all merges
setkey(cell_data, id, year)

cat("All neighbor features computed.\n")
cat("Dimensions:", nrow(cell_data), "x", ncol(cell_data), "\n")
```

---

## Why This Is Correct and Safe

| Concern | How it's handled |
|---|---|
| **Irregular topology / masked cells** | The edge list is built directly from the `nb` object â€” only actual rook-neighbor pairs appear. No rectangular-grid assumption is made. |
| **Missing neighbors (boundary cells)** | If a cell has no neighbors in `edge_dt`, the `merge(..., all.x = TRUE)` leaves `NA` â€” identical to the original `c(NA, NA, NA)` return. |
| **NA variable values** | `joined <- joined[!is.na(nval)]` filters them out before aggregation, matching the original `neighbor_vals <- neighbor_vals[!is.na(neighbor_vals)]`. |
| **Numerical equivalence** | `max`, `min`, `mean` are computed on exactly the same sets of values. The estimand is preserved bit-for-bit (within floating-point associativity). |
| **Trained RF model untouched** | No model code is invoked. Only feature columns are computed. Column names follow the same convention. |
| **Memory on 16 GB laptop** | Peak is ~38.5M rows Ã— ~4 columns of the join table â‰ˆ 1.2 GB, plus `cell_data` itself (~5 GB). Total well under 16 GB. |
| **Runtime** | `data.table` keyed joins and grouped aggregation on ~38.5M rows: typically **2â€“10 minutes** total for all 5 variables. |
END_RESPONSE>>>
