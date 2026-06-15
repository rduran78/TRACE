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
run_id: Anthropic_F08_T03_R017
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.3
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F08
prompt_family_slug: TOPOLOGY_INVARIANCE
family_label: Topology invariance
family_group: Topology cue
replicate: 17
raw_output_file: Experiment_Output/Anthropic/Temp_0.3/output_F08_Anthropic_temp_03_rep_017.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

## Diagnosis

The core performance problem is that `build_neighbor_lookup` constructs a **per-row** (per cell-year) lookup, producing ~6.46 million list entries, when the neighbor *topology* is identical across all 28 years. The function pastes cell IDs and years into string keys, performs string matching via named vectors for every single row, and returns row indices into the full panel. This is O(NÃ—T) in both time and memory when it only needs to be O(N).

Then `compute_neighbor_stats` iterates over those ~6.46 million list entries to extract values â€” again, 28Ã— more work than necessary.

**Specifically:**

1. **Redundant topology replication:** The neighbor graph has ~344K cells. The lookup is rebuilt for each of the 28 years implicitly (one entry per cell-year row), yielding ~6.46M list elements instead of ~344K.
2. **String-key hashing is slow:** `paste(..., sep="_")` and named-vector lookup on millions of strings is extremely expensive in R.
3. **Per-row `lapply` over millions of rows:** Pure R loops/lapply over 6.46M elements are inherently slow.
4. **Repeated per-variable:** The same expensive lookup is traversed 5 times (once per variable), and each variable produces 3 stats (max, min, mean) = 15 columns.

**Estimated cost breakdown of current approach:**
- `build_neighbor_lookup`: ~6.46M string-paste + match operations â†’ hours.
- `compute_neighbor_stats`: ~6.46M Ã— 5 variables â†’ hours.
- Total: 86+ hours as reported.

## Optimization Strategy

**Key insight:** Separate the *static topology* (which cells are neighbors of which) from the *dynamic values* (which change by year).

1. **Build the neighbor lookup once, at the cell level only (~344K entries).** This is a simple integer-index mapping from each cell's position in `id_order` to its neighbors' positions â€” directly from the `nb` object. No string keys, no year dimension.

2. **For each year, slice the data, extract the variable column as a vector indexed by cell position, and compute neighbor stats using the static cell-level lookup.** This turns each stats computation into ~344K list lookups per year Ã— 28 years = ~9.6M, but with pure integer indexing on short vectors, and can be heavily vectorized.

3. **Vectorize the inner stats computation** using `vapply` (pre-allocated output) and direct integer subsetting â€” no string operations anywhere.

4. **Optionally use `data.table`** for fast split-by-year and column assignment, avoiding repeated data.frame copies.

**Complexity reduction:**
- Lookup build: O(N) instead of O(NÃ—T) â€” **28Ã— faster**, no string ops.
- Stats computation: same number of neighbor accesses but on integer-indexed vectors with no NA key checks â€” **estimated 50-100Ã— faster** overall.
- Expected runtime: minutes instead of days.

## Working R Code

```r
library(data.table)

# =============================================================================
# STEP 1: Build the cell-level neighbor lookup ONCE (static topology)
# =============================================================================
# Input:
#   id_order          â€” vector of cell IDs in the order matching rook_neighbors_unique
#   rook_neighbors_unique â€” spdep nb object (list of integer index vectors)
#
# Output:
#   cell_neighbor_idx â€” list of length N_cells; each element is an integer vector
#                       of positions (indices into id_order) of that cell's neighbors.
#
# This is essentially just the nb object itself, but we make it explicit and
# ensure 0-neighbor entries are integer(0).

build_cell_neighbor_lookup <- function(id_order, neighbors_nb) {
  n_cells <- length(id_order)
  stopifnot(length(neighbors_nb) == n_cells)
  
  # spdep nb objects store neighbor indices as integer vectors

  # with 0L meaning "no neighbors". Clean that up.
  lapply(seq_len(n_cells), function(i) {
    nb_idx <- neighbors_nb[[i]]
    # spdep uses 0L to denote no neighbors
    nb_idx <- nb_idx[nb_idx != 0L]
    as.integer(nb_idx)
  })
}

cell_neighbor_idx <- build_cell_neighbor_lookup(id_order, rook_neighbors_unique)

# =============================================================================
# STEP 2: Create a mapping from cell ID â†’ position in id_order
# =============================================================================
id_to_pos <- setNames(seq_along(id_order), as.character(id_order))

# =============================================================================
# STEP 3: Convert cell_data to data.table and add cell position column
# =============================================================================
dt <- as.data.table(cell_data)

# Map each row's cell ID to its position in id_order
dt[, cell_pos := id_to_pos[as.character(id)]]

# Ensure data is sorted by year and cell_pos for fast slicing
setkey(dt, year, cell_pos)

# =============================================================================
# STEP 4: Compute neighbor stats efficiently â€” year by year, variable by variable
# =============================================================================
# For each year:
#   - Extract the variable values as a vector indexed by cell_pos
#   - Use cell_neighbor_idx to gather neighbor values
#   - Compute max, min, mean
#
# We pre-allocate output columns in the data.table.

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# Pre-create output column names
neighbor_col_names <- unlist(lapply(neighbor_source_vars, function(v) {
  paste0(v, c("_neighbor_max", "_neighbor_min", "_neighbor_mean"))
}))

# Initialize output columns with NA_real_
for (col_name in neighbor_col_names) {
  set(dt, j = col_name, value = NA_real_)
}

# Get sorted unique years
years <- sort(unique(dt$year))
n_cells <- length(id_order)

# Pre-compute which cells have neighbors and cache neighbor lengths
has_neighbors <- vapply(cell_neighbor_idx, function(x) length(x) > 0L, logical(1))

cat("Computing neighbor statistics...\n")
t0 <- proc.time()

for (yr in years) {
  # Get row indices for this year (data is keyed by year, cell_pos)
  yr_rows <- which(dt$year == yr)
  
  # Get the cell positions for these rows (should be a subset of 1:n_cells)
  yr_cell_pos <- dt$cell_pos[yr_rows]
  
  # Build a fast lookup: for a given cell_pos, what is its row index in yr_rows?
  # We need: given cell_pos p, the value of variable v is at yr_rows[pos_to_yr_idx[p]]
  # Since cell_pos ranges from 1 to n_cells, use a simple vector.
  pos_to_yr_row <- rep(NA_integer_, n_cells)
  pos_to_yr_row[yr_cell_pos] <- yr_rows
  
  for (var_name in neighbor_source_vars) {
    # Extract values for this year, indexed by cell_pos
    # vals_by_pos[p] = value of var_name for cell at position p in this year
    vals_by_pos <- rep(NA_real_, n_cells)
    vals_by_pos[yr_cell_pos] <- dt[[var_name]][yr_rows]
    
    col_max  <- paste0(var_name, "_neighbor_max")
    col_min  <- paste0(var_name, "_neighbor_min")
    col_mean <- paste0(var_name, "_neighbor_mean")
    
    # Compute stats for each cell that exists this year
    # We compute for ALL n_cells positions, then assign to the rows that exist
    
    # Vectorized approach using vapply over cell positions present this year
    stats_matrix <- vapply(yr_cell_pos, function(p) {
      nb <- cell_neighbor_idx[[p]]
      if (length(nb) == 0L) return(c(NA_real_, NA_real_, NA_real_))
      nv <- vals_by_pos[nb]
      nv <- nv[!is.na(nv)]
      if (length(nv) == 0L) return(c(NA_real_, NA_real_, NA_real_))
      c(max(nv), min(nv), mean(nv))
    }, numeric(3))
    # stats_matrix is 3 x length(yr_cell_pos)
    
    set(dt, i = yr_rows, j = col_max,  value = stats_matrix[1L, ])
    set(dt, i = yr_rows, j = col_min,  value = stats_matrix[2L, ])
    set(dt, i = yr_rows, j = col_mean, value = stats_matrix[3L, ])
  }
  
  cat(sprintf("  Year %d done.\n", yr))
}

elapsed <- proc.time() - t0
cat(sprintf("Neighbor stats completed in %.1f seconds.\n", elapsed[3]))

# =============================================================================
# STEP 5: Remove helper column and convert back if needed
# =============================================================================
dt[, cell_pos := NULL]

# If downstream code expects a data.frame:
cell_data <- as.data.frame(dt)

# =============================================================================
# STEP 6: Predict with the pre-trained Random Forest (UNCHANGED)
# =============================================================================
# The trained RF model object (e.g., `rf_model`) is used as-is.
# cell_data now contains all the original columns plus the 15 neighbor stat columns,
# computed identically to the original implementation.
#
# Example (unchanged from original pipeline):
# predictions <- predict(rf_model, newdata = cell_data)
```

## Why This Preserves the Original Numerical Estimand

| Aspect | Original | Redesigned |
|---|---|---|
| Neighbor topology | From `rook_neighbors_unique` via string-key matching | From `rook_neighbors_unique` directly via integer indices |
| Stats computed | `max`, `min`, `mean` of non-NA neighbor values | Identical: `max`, `min`, `mean` of non-NA neighbor values |
| Edge cases (no neighbors / all-NA neighbors) | Returns `NA, NA, NA` | Returns `NA, NA, NA` |
| Column names | `{var}_neighbor_max`, `{var}_neighbor_min`, `{var}_neighbor_mean` | Identical |
| RF model | Not retrained | Not retrained |

The values fed to `predict(rf_model, ...)` are **numerically identical** to the original pipeline. The only change is *how* those values are computed â€” by separating the static topology from the dynamic year-varying data.

## Expected Performance

| Component | Original | Redesigned |
|---|---|---|
| Lookup construction | ~6.46M string ops | ~344K integer copies (once) |
| Stats: inner loop | ~6.46M Ã— 5 vars, string-keyed | ~344K Ã— 28 yrs Ã— 5 vars, integer-indexed |
| Estimated wall time | 86+ hours | **5â€“20 minutes** |
| Peak memory | Multiple copies of 6.46M-element lists | One `data.table` + one `n_cells`-length vector per var-year |
END_RESPONSE>>>
