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
run_id: Anthropic_F07_T10_R004
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F07
prompt_family_slug: RASTER_INVALID_IRREGULAR
family_label: Raster invalid irregular topology
family_group: Raster/kernel bridge
replicate: 4
raw_output_file: Experiment_Output/Anthropic/Temp_1.0/output_F07_Anthropic_temp_10_rep_004.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The bottleneck is **`build_neighbor_lookup`**, which creates a list of ~6.46 million elements, each produced by an R-level `lapply` iteration that does character pasting, named-vector lookups, and `NA` filtering. This is an **O(N Ã— k)** operation executed entirely in interpreted R with per-row string allocation and hash-table lookups. With N â‰ˆ 6.46M and average k â‰ˆ 4 rook neighbors, you're doing ~26M string constructions and named-vector searches inside a single-threaded R loop. Then `compute_neighbor_stats` iterates over 6.46M list elements again, five times (once per variable). The combined cost is roughly:

| Component | Cost driver |
|---|---|
| `build_neighbor_lookup` | 6.46M iterations Ã— string paste + named-vector lookup per neighbor (~86+ hrs) |
| `compute_neighbor_stats` | 6.46M list iterations Ã— 5 variables (moderate but compounding) |
| Memory | 6.46M-element list of integer vectors + repeated `do.call(rbind, ...)` on 6.46M rows |

The fundamental problem: **the lookup is row-level and string-keyed, but the underlying structure is a simple join on (id, year) â€” a fully vectorizable operation.**

---

## Optimization Strategy

### Key Insight
Every cell `i` in year `t` needs the values of its rook neighbors **in the same year `t`**. The neighbor graph is time-invariant. Therefore:

1. **Replace the per-row string-keyed lookup with a vectorized merge/join.** Expand the neighbor list into an edge table `(from_id, to_id)`, join it to the data twice (once for the focal row index, once for the neighbor row index by matching year), and compute grouped statistics with `data.table`.

2. **Compute all 5 variables' neighbor stats in a single grouped aggregation** instead of looping.

3. **Use `data.table` throughout** for memory-efficient, cache-friendly, multi-threaded grouped operations.

This converts ~86 hours of interpreted R into a few vectorized joins and a single `data.table` grouped aggregation â€” expected runtime: **minutes**.

### Numerical Equivalence
The operations `max`, `min`, `mean` over the same neighbor sets with the same `NA`-removal logic are preserved exactly. The trained Random Forest model is untouched.

---

## Working R Code

```r
library(data.table)

# â”€â”€ 0. Convert to data.table (non-destructive; keeps all columns) â”€â”€â”€â”€â”€â”€â”€â”€â”€
dt <- as.data.table(cell_data)

# â”€â”€ 1. Build edge table from the spdep nb object â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
#
#   rook_neighbors_unique is a list of length 344,208.
#   rook_neighbors_unique[[i]] contains the integer indices (into id_order)
#   of the rook neighbors of the cell whose id is id_order[i].

edges <- rbindlist(lapply(seq_along(rook_neighbors_unique), function(i) {
  nb <- rook_neighbors_unique[[i]]
  if (length(nb) == 0L || (length(nb) == 1L && nb[1] == 0L)) {
    return(NULL)
  }
  data.table(from_id = id_order[i], to_id = id_order[nb])
}))

cat(sprintf("Edge table: %d directed edges\n", nrow(edges)))
# Expected: ~1,373,394

# â”€â”€ 2. Attach row indices to the edge table via keyed join â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
#
#   We need to pair every (from_id, year) focal row with every
#   (to_id, same year) neighbor row, then aggregate.

# Create a compact row-index column
dt[, .row_idx := .I]

# Keyed lookup tables (id, year) -> row_idx and variable values
# We only need the neighbor source vars + id + year
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# Join edges with the panel on the "from" side to get year + focal row index
setkey(dt, id, year)

# Expand edges Ã— years:  for every edge (from_id -> to_id),
# we need every year that the from_id appears in the data.
# Instead of a full cross join, we merge edges onto dt.

# Step A: get (from_id, year, focal_row_idx)
focal <- dt[, .(from_id = id, year, focal_row = .row_idx)]
setkey(focal, from_id)
setkey(edges, from_id)

# Merge: for each edge, replicate across all years the focal cell exists
edge_year <- edges[focal, on = .(from_id), allow.cartesian = TRUE, nomatch = NULL]
# Columns: from_id, to_id, year, focal_row

# Step B: attach neighbor values by joining (to_id, year)
# Build a neighbor-value table
nbr_vals <- dt[, c("id", "year", neighbor_source_vars), with = FALSE]
setnames(nbr_vals, "id", "to_id")
setkey(nbr_vals, to_id, year)
setkey(edge_year, to_id, year)

edge_full <- nbr_vals[edge_year, on = .(to_id, year), nomatch = NA]
# Columns: to_id, year, ntl, ec, pop_density, def, usd_est_n2, from_id, focal_row

# â”€â”€ 3. Grouped aggregation: neighbor max, min, mean per focal row â”€â”€â”€â”€â”€â”€â”€â”€â”€
agg_exprs <- unlist(lapply(neighbor_source_vars, function(v) {
  list(
    bquote(max(.(as.name(v)), na.rm = TRUE)),
    bquote(min(.(as.name(v)), na.rm = TRUE)),
    bquote(mean(.(as.name(v)), na.rm = TRUE))
  )
}), recursive = FALSE)

agg_names <- unlist(lapply(neighbor_source_vars, function(v) {
  paste0(v, c("_nb_max", "_nb_min", "_nb_mean"))
}))

# data.table aggregation (multi-threaded via OpenMP)
stats <- edge_full[,
  setNames(lapply(neighbor_source_vars, function(v) {
    vals <- get(v)
    vals <- vals[!is.na(vals)]
    if (length(vals) == 0L) list(NA_real_, NA_real_, NA_real_)
    else list(max(vals), min(vals), mean(vals))
  }), neighbor_source_vars),
  by = focal_row
]

# The above returns nested lists; a cleaner approach:
stats <- edge_full[, {
  out <- vector("list", length(neighbor_source_vars) * 3L)
  k <- 1L
  for (v in neighbor_source_vars) {
    vals <- get(v)
    vals <- vals[!is.na(vals)]
    if (length(vals) == 0L) {
      out[[k]] <- NA_real_; out[[k+1L]] <- NA_real_; out[[k+2L]] <- NA_real_
    } else {
      out[[k]] <- max(vals); out[[k+1L]] <- min(vals); out[[k+2L]] <- mean(vals)
    }
    k <- k + 3L
  }
  names(out) <- agg_names
  out
}, by = focal_row]

# â”€â”€ 4. Merge back into the original data â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Rows with no neighbors (islands) will get NA automatically via nomatch
setkey(stats, focal_row)

for (nm in agg_names) {
  dt[stats$focal_row, (nm) := stats[[nm]]]
}

# For rows not in stats (no neighbors), ensure NA
for (nm in agg_names) {
  if (!nm %in% names(dt)) dt[, (nm) := NA_real_]
}

# â”€â”€ 5. Replace Inf/-Inf from max/min of empty sets (safety) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
for (nm in agg_names) {
  dt[is.infinite(get(nm)), (nm) := NA_real_]
}

# â”€â”€ 6. Convert back to data.frame if the downstream RF predict expects it â”€
dt[, .row_idx := NULL]
cell_data <- as.data.frame(dt)

cat("Done. Neighbor features added for variables:\n")
cat(paste(" ", neighbor_source_vars, collapse = "\n"), "\n")
cat("New columns:\n")
cat(paste(" ", agg_names, collapse = "\n"), "\n")
```

---

## Expected Performance

| Step | Estimated time | Estimated peak RAM |
|---|---|---|
| Edge table construction | ~2 s | ~50 MB |
| `edge_year` join (1.37M edges Ã— 28 yrs) | ~10 s | ~1.5 GB |
| `nbr_vals` join | ~15 s | ~2 GB |
| Grouped aggregation | ~2â€“5 min | ~2.5 GB |
| Merge back | ~5 s | negligible |
| **Total** | **~3â€“6 minutes** | **< 6 GB peak** |

This is well within 16 GB RAM and replaces the 86+ hour runtime with a few minutes while producing **numerically identical** neighbor max, min, and mean values. The trained Random Forest model is never touched.
END_RESPONSE>>>
