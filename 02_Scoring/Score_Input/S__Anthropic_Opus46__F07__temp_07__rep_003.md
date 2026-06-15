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
run_id: Anthropic_F07_T07_R003
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F07
prompt_family_slug: RASTER_INVALID_IRREGULAR
family_label: Raster invalid irregular topology
family_group: Raster/kernel bridge
replicate: 3
raw_output_file: Experiment_Output/Anthropic/Temp_0.7/output_F07_Anthropic_temp_07_rep_003.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The bottleneck is **`build_neighbor_lookup`**. It constructs a list of 6.46 million elements, where each element is built by:

1. Looking up the focal cell's neighbor IDs via `id_to_ref` and `neighbors[[ref_idx]]` â€” this is fast.
2. Pasting together `neighbor_cell_ids` and `data$year[i]` into character keys, then matching them into `idx_lookup` â€” **this is extremely slow**.

The `paste(â€¦, sep="_")` call inside a per-row `lapply` over ~6.46 million rows, each touching ~4 neighbors on average (~1,373,394 directed edges / 344,208 cells â‰ˆ 4), means roughly **25+ million `paste` and hash-lookup operations executed one-at-a-time in interpreted R**. The named-vector lookup `idx_lookup[neighbor_keys]` is also O(n) per probe in the worst case for large named vectors.

Then `compute_neighbor_stats` iterates over the 6.46M-element list again, once **per variable** (Ã—5 variables). Each call extracts neighbor values, removes NAs, and computes max/min/mean. This is another ~32 million interpreted iterations total.

**Summary of costs:**

| Step | Calls | Estimated wall-clock |
|---|---|---|
| `build_neighbor_lookup` (paste + named-vector match, 6.46M rows) | 1 | ~40â€“60 hours |
| `compute_neighbor_stats` (lapply, 6.46M Ã— 5 vars) | 5 | ~25â€“30 hours |
| **Total** | | **~65â€“90 hours** |

---

## Optimization Strategy

### Key insight: Separate the spatial graph from the temporal dimension

Every cell has the **same** neighbors in every year. So the neighbor lookup can be decomposed into:

1. **A spatial edge list** (344K cells Ã— ~4 neighbors = ~1.37M edges) â€” tiny.
2. **A year index** â€” 28 distinct years.

We never need to build a 6.46M-element list. Instead:

### Step 1: Build a sparse adjacency structure once (spatial only)

Convert `rook_neighbors_unique` (an `nb` object) into a two-column edge list (`from_id`, `to_id`) of cell IDs. This is ~1.37M rows â€” trivially small.

### Step 2: Vectorized join to expand edges across years

Use `data.table` to join the edge list against the panel on `(neighbor_id, year)`, pulling in neighbor values. This is a single equi-join of ~1.37M Ã— 28 â‰ˆ ~38.4M rows â€” fast and memory-safe (~2â€“3 GB peak).

### Step 3: Grouped aggregation

Group by `(focal_row, variable)` and compute `max`, `min`, `mean` in one vectorized `data.table` operation.

### Expected performance

| Step | Method | Estimated time |
|---|---|---|
| Build edge list from nb | Vectorized | < 1 sec |
| Expand edges Ã— years + join | `data.table` equi-join | ~10â€“30 sec |
| Grouped max/min/mean Ã— 5 vars | `data.table` grouped agg | ~20â€“60 sec |
| **Total** | | **< 2 minutes** |

This is a **~3,000Ã— speedup** while preserving the exact same numerical output (max, min, mean of non-NA rook-neighbor values per cell-year).

---

## Working R Code

```r
# =============================================================================
# Fast neighbor-stats computation â€” exact drop-in replacement
# =============================================================================
# Requirements: data.table
# Preserves: trained Random Forest model (untouched), original numerical output
# =============================================================================

library(data.table)

compute_all_neighbor_features <- function(cell_data,
                                          id_order,
                                          rook_neighbors_unique,
                                          neighbor_source_vars) {
  # ------------------------------------------------------------------
  # 0.  Convert cell_data to data.table (by reference if already is one)
  # ------------------------------------------------------------------
  if (!is.data.table(cell_data)) {
    cell_data <- as.data.table(cell_data)
  }

  # Ensure a deterministic row identifier so we can write results back
  cell_data[, .row_idx := .I]

  # ------------------------------------------------------------------
  # 1.  Build spatial edge list from the nb object  (~1.37 M rows)
  #     nb object: list of integer vectors; indices into id_order
  # ------------------------------------------------------------------
  from_ref <- rep(
    seq_along(rook_neighbors_unique),
    lengths(rook_neighbors_unique)
  )
  to_ref <- unlist(rook_neighbors_unique, use.names = FALSE)

  # Remove the spdep convention where 0L means "no neighbors"
  valid <- to_ref != 0L
  from_ref <- from_ref[valid]
  to_ref   <- to_ref[valid]

  edges <- data.table(
    focal_id    = id_order[from_ref],
    neighbor_id = id_order[to_ref]
  )

  # ------------------------------------------------------------------
  # 2.  For each source variable, join, aggregate, and attach columns
  # ------------------------------------------------------------------
  # We need a keyed version of cell_data for fast joins
  # Key columns: id (cell identifier) and year
  id_col  <- "id"
  yr_col  <- "year"

  # Minimal neighbor table: just the neighbor's id, year, and value
  # We join edges to cell_data twice:
  #   - once to get the focal cell's row index (so we know where to write)
  #   - once to get the neighbor cell's value

  # Pre-build a lookup: (id, year) -> .row_idx   [for focal side]
  focal_key <- cell_data[, .(id, year, .row_idx)]
  setkey(focal_key, id, year)

  for (var_name in neighbor_source_vars) {
    message("Processing neighbor stats for: ", var_name)

    # Neighbor-side lookup: (id, year) -> value
    neighbor_vals <- cell_data[, .SD, .SDcols = c(id_col, yr_col, var_name)]
    setnames(neighbor_vals, c("neighbor_id", "year", "nval"))
    setkey(neighbor_vals, neighbor_id, year)

    # Expand edges across all years present for the focal cell
    # focal_key gives us every (focal_id, year) with its row index
    # edges gives us every (focal_id -> neighbor_id)
    # We need: (focal_id, year, neighbor_id) then look up neighbor value

    # Step A: join focal_key with edges on focal_id
    #   Result: (focal_id, year, .row_idx, neighbor_id)
    setkey(edges, focal_id)
    expanded <- edges[focal_key, on = .(focal_id = id),
                      .(focal_id, year, .row_idx, neighbor_id),
                      allow.cartesian = TRUE, nomatch = NULL]
    # This is ~1.37M edges Ã— 28 years â‰ˆ 38.4M rows

    # Step B: join with neighbor_vals to get the actual value
    setkey(expanded, neighbor_id, year)
    expanded <- neighbor_vals[expanded, on = .(neighbor_id, year),
                              .(focal_id, year, .row_idx, nval = x.nval),
                              nomatch = NA]

    # Step C: aggregate â€” drop NAs, compute max/min/mean per focal row
    stats <- expanded[!is.na(nval),
                      .(nmax  = max(nval),
                        nmin  = min(nval),
                        nmean = mean(nval)),
                      by = .row_idx]

    # Step D: write results back into cell_data
    max_col  <- paste0("neighbor_max_",  var_name)
    min_col  <- paste0("neighbor_min_",  var_name)
    mean_col <- paste0("neighbor_mean_", var_name)

    # Initialize with NA
    cell_data[, (max_col)  := NA_real_]
    cell_data[, (min_col)  := NA_real_]
    cell_data[, (mean_col) := NA_real_]

    # Fill matched rows
    cell_data[stats, on = .(.row_idx),
              `:=`(
                (max_col)  = i.nmax,
                (min_col)  = i.nmin,
                (mean_col) = i.nmean
              )]

    # Clean up to free memory within the loop
    rm(neighbor_vals, expanded, stats)
    gc(verbose = FALSE)
  }

  # Clean up helper column
  cell_data[, .row_idx := NULL]

  return(cell_data)
}


# =============================================================================
# Usage â€” drop-in replacement for the original outer loop
# =============================================================================

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cell_data <- compute_all_neighbor_features(
  cell_data            = cell_data,
  id_order             = id_order,
  rook_neighbors_unique = rook_neighbors_unique,
  neighbor_source_vars = neighbor_source_vars
)

# cell_data now contains the same 15 new columns as the original code:
#   neighbor_max_ntl, neighbor_min_ntl, neighbor_mean_ntl,
#   neighbor_max_ec,  neighbor_min_ec,  neighbor_mean_ec,
#   ... etc.
#
# The trained Random Forest model is untouched â€” just use cell_data
# as the input to predict() as before.
```

---

## Why This Preserves Correctness

| Property | Original | Replacement |
|---|---|---|
| Neighbor definition | Rook contiguity from `rook_neighbors_unique` nb object | Same nb object, same edges |
| Temporal matching | Neighbors matched within same year | Joined on `(neighbor_id, year)` â€” identical |
| NA handling | `neighbor_vals[!is.na(neighbor_vals)]`; returns `NA` if all missing | `expanded[!is.na(nval), â€¦]`; unmatched rows stay `NA` â€” identical |
| Aggregation functions | `max`, `min`, `mean` | Same functions |
| Output columns | `neighbor_{max,min,mean}_{var}` | Same names and semantics |
| RF model | Not retrained | Not retrained |

The numerical estimand is identical because the same set of non-NA neighbor values is aggregated with the same functions. The only change is **how** those values are gathered â€” vectorized joins instead of interpreted loops.

---

## Memory Profile (16 GB laptop)

| Object | Approximate size |
|---|---|
| `cell_data` (6.46M Ã— 110 cols) | ~5.5 GB |
| `edges` (1.37M Ã— 2 int cols) | ~11 MB |
| `expanded` (peak: ~38.4M Ã— 4 cols) | ~1.2 GB |
| `stats` (â‰¤ 6.46M Ã— 4 cols) | ~200 MB |
| Overhead / gc headroom | ~2 GB |
| **Peak total** | **~9 GB** âœ“ |

Processing each variable sequentially (with `rm` + `gc` inside the loop) keeps peak memory well within 16 GB.
END_RESPONSE>>>
