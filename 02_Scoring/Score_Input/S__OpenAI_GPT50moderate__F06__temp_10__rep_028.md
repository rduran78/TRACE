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
run_id: OpenAI_F06_T10_R028
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F06
prompt_family_slug: RASTER_KERNEL_ANALOGY
family_label: Raster/kernel analogy
family_group: Raster/kernel bridge
replicate: 28
raw_output_file: Experiment_Output/OpenAI/Temp_1.0/output_F06_OpenAI_temp_10_rep_028.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
The performance bottleneck arises because `lapply` loops over ~6.46 million rows (`cell_data`), and for each row retrieves neighbor indices and computes max, min, and mean. This involves millions of small list operations and repeated vector scans in pure R, which is extremely inefficient for such a large dataset. The design essentially results in O(N Ã— K) overhead in R interpreted loops, where N â‰ˆ 6.46M and K â‰ˆ average neighbor count (~4).  

---

**Optimization Strategy**  
1. **Precompute neighbor index relationships at the row level** once, vectorized.
2. Switch to **vectorized aggregation** (using `data.table` or `dplyr`) or a **matrix-based approach** instead of millions of lists.
3. Use **wide format join-based rolling aggregation** for speed, leveraging `data.table`.
4. Avoid recalculating for each variable: gather neighbor data long-form, compute aggregates once, then merge back.
5. Keep everything in-memory, chunk if needed due to RAM constraints.

Expected runtime drops from **86+ hours to minutes** on a 16 GB machine using efficient joins.

---

### **Working R Code (Optimized)**

```r
library(data.table)

# Convert to data.table
setDT(cell_data)

# Step 1: Create row-level unique key
cell_data[, key := paste(id, year, sep = "_")]

# Step 2: Build a long neighbor-edge table for each year
# rook_neighbors_unique: list of neighbors by cell index in id_order
id_to_idx <- setNames(seq_along(id_order), id_order)

# Flatten neighbor relationships into a data.table
edges <- rbindlist(lapply(seq_along(rook_neighbors_unique), function(i) {
  nbrs <- rook_neighbors_unique[[i]]
  if (length(nbrs) == 0) return(NULL)
  data.table(
    src_id = id_order[i],
    nbr_id = id_order[nbrs]
  )
}))

# Step 3: Expand to panel (year match)
years <- sort(unique(cell_data$year))
edges_panel <- edges[CJ(year = years, src_id = src_id, allow.cartesian = TRUE)][,
  .(src_key = paste(src_id, year, sep="_"),
    nbr_key = paste(nbr_id, year, sep="_"))
]

# Step 4: Merge neighbor values in one pass
lookup_cols <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
nbr_vals <- cell_data[, c("key", lookup_cols), with = FALSE]
setkey(nbr_vals, key)
setkey(edges_panel, nbr_key)

edges_panel <- nbr_vals[edges_panel, on = .(key = nbr_key)]
# Now edges_panel has: src_key, neighbor values

# Step 5: Compute aggregates by src_key for all vars
agg <- edges_panel[, lapply(.SD, function(x) {
  x <- x[!is.na(x)]
  if (length(x) == 0) return(rep(NA_real_, 3))
  c(max(x), min(x), mean(x))
}), by = src_key, .SDcols = lookup_cols]

# Step 6: Reshape to wide
agg_long <- melt(agg, id.vars = "src_key", variable.name = "var", value.name = "stats")
agg_long[, c("max", "min", "mean") := tstrsplit(stats, " ", fixed = TRUE)]
agg_long <- dcast(agg_long, src_key ~ var, value.var = c("max", "min", "mean"))

# Step 7: Join back to cell_data
setkey(agg_long, src_key)
cell_data <- agg_long[cell_data, on = .(src_key = key)]
```

---

**Advantages**  
- Avoids per-row R loops.
- Uses `data.table` joins and grouping for vectorized speed.
- Memory-efficient: edges_panel â‰ˆ (1.37M edges Ã— 28 years) â†’ ~38M rows, manageable in 16 GB if processed in chunks.  
- Preserves estimand: same neighbor-based max, min, mean.  
- Random Forest model remains intact since feature columns replicate previous output.  

**Expected Runtime**: Minutes to an hour vs. 86+ hours.  

âœ” Same numerics, much faster execution, feasible on standard laptop.
END_RESPONSE>>>
