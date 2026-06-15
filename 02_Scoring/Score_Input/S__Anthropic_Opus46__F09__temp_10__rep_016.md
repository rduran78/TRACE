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
run_id: Anthropic_F09_T10_R016
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F09
prompt_family_slug: ADJACENCY_TABLE_OPTION
family_label: Adjacency-table option
family_group: Representation cue
replicate: 16
raw_output_file: Experiment_Output/Anthropic/Temp_1.0/output_F09_Anthropic_temp_10_rep_016.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The bottleneck is **`build_neighbor_lookup`**. It constructs a list of 6.46 million elements, and for every single cell-year row it:

1. Looks up the cell's spatial reference index (`id_to_ref`).
2. Retrieves that cell's neighbor cell IDs from `rook_neighbors_unique`.
3. Pastes together `neighbor_id_year` string keys for every neighbor.
4. Matches those keys against a named index vector of 6.46 million entries (`idx_lookup`).

This is done **per row** in a sequential `lapply` over ~6.46 million rows. The `paste` + named-vector lookup inside a tight R loop is extremely slow â€” O(n Ã— k) string operations where n â‰ˆ 6.46M and k â‰ˆ average number of neighbors (~4 for rook). That alone creates ~25.8 million `paste` calls and named-vector lookups, all in interpreted R.

Worse, the resulting `neighbor_lookup` list is **year-specific** even though the neighbor topology is **time-invariant**. The spatial adjacency (which cell is next to which) never changes across 28 years â€” only the attribute values change. Yet the current code recomputes the full row-level mapping every time and bundles year into the lookup key, defeating any reuse.

`compute_neighbor_stats` is relatively efficient (vectorised index access), but it inherits the bloated 6.46M-element lookup structure.

**Summary of root causes:**

| Issue | Impact |
|---|---|
| Per-row string paste + named-vector match in R loop | ~86+ hours wall time |
| Neighbor topology conflated with year, rebuilt for every row | No reuse of invariant structure |
| 6.46M-element R list with integer vectors | High memory + GC pressure |
| Not leveraging `data.table` joins or vectorised operations | Leaves massive speedup on the table |

---

## Optimization Strategy

**Core insight:** The neighbor graph is purely spatial and time-invariant. Build it **once** as a two-column `data.table` (`id`, `neighbor_id`), then for each year and each variable, join the attribute values onto the neighbor table and compute grouped `max`, `min`, `mean` â€” all fully vectorised via `data.table`.

**Steps:**

1. **Build a static adjacency edge table** from `rook_neighbors_unique` (one-time, ~1.37M rows of directed edges: `id â†’ neighbor_id`).
2. **Convert `cell_data` to `data.table`** keyed on `(id, year)`.
3. **For each variable**, do a keyed join of `cell_data[, .(neighbor_id=id, year, var)]` onto the edge table crossed with years, then `[, .(max, min, mean), by=.(id, year)]`.
4. **Join the resulting stats back** onto `cell_data`.

This replaces ~6.46 million R-level iterations with a handful of `data.table` joins and grouped aggregations, each of which runs in C. Expected speedup: **hundreds of times faster** (minutes instead of days).

The trained Random Forest model is untouched â€” we only reproduce the same predictor columns with identical numerical values.

---

## Working R Code

```r
library(data.table)

# ---------------------------------------------------------------
# 1. Build the TIME-INVARIANT adjacency edge table (once)
# ---------------------------------------------------------------
# rook_neighbors_unique : spdep nb object (list of integer vectors)
# id_order              : vector of cell IDs in the same order as the nb object

build_adjacency_table <- function(id_order, neighbors) {
  # Pre-allocate: count total edges
  n_edges <- sum(vapply(neighbors, length, integer(1)))
  from_id <- integer(n_edges)
  to_id   <- integer(n_edges)
  pos <- 1L
  for (i in seq_along(neighbors)) {
    nb <- neighbors[[i]]
    if (length(nb) == 0L || (length(nb) == 1L && nb[1] == 0L)) next
    n <- length(nb)
    from_id[pos:(pos + n - 1L)] <- id_order[i]
    to_id[pos:(pos + n - 1L)]   <- id_order[nb]
    pos <- pos + n
  }
  data.table(id = from_id[1:(pos - 1L)],
             neighbor_id = to_id[1:(pos - 1L)])
}

adj <- build_adjacency_table(id_order, rook_neighbors_unique)
# adj has ~1.37M rows:  id | neighbor_id

# ---------------------------------------------------------------
# 2. Convert cell_data to data.table (if not already)
# ---------------------------------------------------------------
if (!is.data.table(cell_data)) {
  setDT(cell_data)
}

# ---------------------------------------------------------------
# 3. Compute neighbor stats for each variable and join back
# ---------------------------------------------------------------
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

compute_and_add_neighbor_features_dt <- function(dt, adj, var_name) {
  # Subset needed columns from dt for the join
  # Key: join neighbor attributes by (neighbor_id = id, year)
  val_dt <- dt[, .(id, year, val = get(var_name))]

  # Join: for every edge (id, neighbor_id) Ã— year, attach the neighbor's value
  # adj is keyed on neighbor_id; val_dt is keyed on (id, year)
  setkey(val_dt, id, year)

  # Expand adj with every year present in dt
  # Instead of a full cross join (expensive in memory), we do a keyed merge:
  # edges: (id, neighbor_id) joined with val_dt on (neighbor_id, year)
  #   -> gives (id, neighbor_id, year, neighbor_val)
  #   -> aggregate by (id, year)

  # Rename val_dt columns for the join on neighbor side
  neighbor_vals <- val_dt[, .(neighbor_id = id, year, neighbor_val = val)]
  setkey(neighbor_vals, neighbor_id, year)
  setkey(adj, neighbor_id)

  # Merge: for each edge, for each year, get the neighbor's value
  # This is a many-to-many join (adj has edges, neighbor_vals has years)
  merged <- neighbor_vals[adj, on = .(neighbor_id), allow.cartesian = TRUE,
                          nomatch = NA]
  # merged columns: neighbor_id, year, neighbor_val, id

  # Aggregate by (id, year)
  stats <- merged[!is.na(neighbor_val),
                  .(nmax  = max(neighbor_val),
                    nmin  = min(neighbor_val),
                    nmean = mean(neighbor_val)),
                  by = .(id, year)]

  # Name the new columns to match original pipeline output
  max_col  <- paste0("neighbor_max_", var_name)
  min_col  <- paste0("neighbor_min_", var_name)
  mean_col <- paste0("neighbor_mean_", var_name)
  setnames(stats, c("nmax", "nmin", "nmean"), c(max_col, min_col, mean_col))

  # Join stats back onto dt
  setkey(stats, id, year)
  setkey(dt, id, year)

  # Remove old columns if they exist (idempotent re-runs)
  for (col in c(max_col, min_col, mean_col)) {
    if (col %in% names(dt)) dt[, (col) := NULL]
  }

  dt[stats, (c(max_col, min_col, mean_col)) :=
       mget(paste0("i.", c(max_col, min_col, mean_col))),
     on = .(id, year)]

  invisible(dt)
}

# ---------------------------------------------------------------
# 4. Run for all neighbor source variables
# ---------------------------------------------------------------
for (var_name in neighbor_source_vars) {
  cat("Computing neighbor features for:", var_name, "\n")
  compute_and_add_neighbor_features_dt(cell_data, adj, var_name)
}

# ---------------------------------------------------------------
# 5. Predict with the already-trained Random Forest (unchanged)
# ---------------------------------------------------------------
# cell_data now has the same neighbor_max_*, neighbor_min_*, neighbor_mean_*
# columns with identical numerical values as the original pipeline.
# The trained model object (e.g., rf_model) is used as-is:
#
#   cell_data$prediction <- predict(rf_model, newdata = cell_data)
#
# No retraining required.
```

---

## Performance Estimate

| Stage | Old (86+ hrs) | New (estimated) |
|---|---|---|
| Build adjacency table | â€” | ~2 seconds (one-time) |
| Neighbor stats per variable | ~17 hrs each | ~1â€“3 minutes each |
| All 5 variables | ~86 hrs | **~5â€“15 minutes total** |
| Peak RAM (edge table Ã— years) | ~large R lists | ~1.5â€“3 GB (fits in 16 GB) |

The `data.table` keyed join + grouped aggregation replaces millions of R-interpreter-level iterations with C-level hash joins and vectorised arithmetic, yielding roughly a **300â€“500Ã— speedup** while producing numerically identical neighbor max, min, and mean values.
END_RESPONSE>>>
