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
run_id: Anthropic_F12_T10_R006
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F12
prompt_family_slug: FALSE_RBIND_DIAGNOSIS
family_label: False rbind diagnosis
family_group: Adversarial diagnosis
replicate: 6
raw_output_file: Experiment_Output/Anthropic/Temp_1.0/output_F12_Anthropic_temp_10_rep_006.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

## Diagnosis: Rejecting the Colleague's Claim

The colleague's claim that `do.call(rbind, result)` and repeated list binding inside `compute_neighbor_stats()` are the main bottleneck is **incorrect**. While `do.call(rbind, ...)` on a list of 6.46 million small vectors is not ideal, it is a one-shot operation per variable (only 5 calls total) and each inner function returns a fixed-length vector of 3 â€” there is no repeated list binding or growing inside `compute_neighbor_stats()`. The `lapply` produces a pre-allocated list, and `do.call(rbind, ...)` concatenates it once. This is O(n) and takes seconds to low minutes at most.

**The true, deeper bottleneck is `build_neighbor_lookup()`.**

Here is why:

1. **`paste()` key construction for the entire dataset:** `paste(data$id, data$year, sep = "_")` over 6.46 million rows creates a character vector used as names for `idx_lookup`. This is done once and is acceptable.

2. **Per-row `lapply` over 6.46 million rows with repeated `paste()` and character-key lookups:** Inside the `lapply`, for *every single row*, the code:
   - Does `as.character(data$id[i])` â€” character coercion per row.
   - Looks up `id_to_ref[as.character(...)]` â€” named vector lookup by character key.
   - Extracts `neighbor_cell_ids` â€” integer subset.
   - Calls `paste(neighbor_cell_ids, data$year[i], sep = "_")` â€” **constructs new character keys for every neighbor of every row** (billions of `paste` operations total: ~6.46M rows Ã— avg ~4 rook neighbors = ~25.8 million paste operations, each involving string allocation).
   - Does `idx_lookup[neighbor_keys]` â€” named character vector lookup, which in R is **O(n)** per lookup against a vector of 6.46M names (R named vectors use linear hashing but repeated lookups into a 6.46M-element named vector are extremely slow compared to integer indexing or environment/hash lookups).

3. **Total cost accounting:** With ~6.46 million iterations, each doing string construction and named-vector character lookups into a 6.46M-element vector, this function alone accounts for the vast majority of the 86+ hour runtime. The `compute_neighbor_stats` function, by contrast, does only fast integer indexing into a numeric vector.

## Optimization Strategy

1. **Replace the character-key named vector `idx_lookup` with an integer-indexed lookup.** Encode `(id, year)` pairs as integers and use an environment (hash map) or â€” even better â€” a direct integer matrix for O(1) lookup.

2. **Vectorize `build_neighbor_lookup`** by pre-expanding the neighbor relationships across years using a merge/join rather than row-by-row `lapply`. Since the neighbor graph is time-invariant (same spatial neighbors every year), we can construct the full lookup table with vectorized operations.

3. **Replace `do.call(rbind, result)`** in `compute_neighbor_stats` with a pre-allocated matrix for marginal improvement.

The key insight: the neighbor structure is *identical* across all 28 years. There are only 344,208 cells, and each cell's neighbors are the same in every year. So we only need to build the neighbor index mapping for one year and replicate it across all 28 years via integer arithmetic â€” no `paste`, no character lookups.

## Working R Code

```r
# ==============================================================================
# OPTIMIZED build_neighbor_lookup
# ==============================================================================
# Key insight: neighbor relationships are spatial and time-invariant.
# We exploit this by building a per-cell neighbor map once (344K cells),
# then translating to row indices using integer arithmetic, not character keys.

build_neighbor_lookup_fast <- function(data, id_order, neighbors) {
  
  n_rows <- nrow(data)
  
  # Step 1: Create a fast integer-indexed mapping from (id, year) -> row index.
  # Use an environment as a hash map for the id -> integer code mapping.
  unique_ids   <- unique(data$id)
  unique_years <- sort(unique(data$year))
  n_years      <- length(unique_years)
  
  # Integer-code the ids: map each unique id to a contiguous integer 1..N_cells
  id_code_env <- new.env(hash = TRUE, size = length(unique_ids))
  for (j in seq_along(unique_ids)) {
    id_code_env[[as.character(unique_ids[j])]] <- j
  }
  
  # Integer-code the years
  year_min <- min(unique_years)
  # year_code = year - year_min + 1, so 1..28
  
  # Build a matrix: row_index_matrix[id_code, year_code] = row index in data
  # This gives O(1) lookup by integer indexing.
  n_cells <- length(unique_ids)
  row_index_matrix <- matrix(NA_integer_, nrow = n_cells, ncol = n_years)
  
  data_id_codes   <- integer(n_rows)
  data_year_codes <- as.integer(data$year - year_min + 1L)
  
  # Vectorized id coding using match (much faster than per-row env lookup)
  data_id_codes <- match(data$id, unique_ids)
  
  # Fill the matrix
  for (i in seq_len(n_rows)) {
    row_index_matrix[data_id_codes[i], data_year_codes[i]] <- i
  }
  
  # Step 2: Build per-cell neighbor id_codes (time-invariant, only 344K entries)
  # id_order is the vector of cell ids in the order matching the nb object
  id_order_codes <- match(id_order, unique_ids)
  
  # Map each id_order position to its neighbors' id_codes
  # neighbors is an nb object: neighbors[[k]] gives indices into id_order
  cell_neighbor_codes <- lapply(seq_along(id_order), function(k) {
    nb_idx <- neighbors[[k]]
    if (length(nb_idx) == 0 || (length(nb_idx) == 1 && nb_idx[1] == 0L)) {
      return(integer(0))
    }
    id_order_codes[nb_idx]
  })
  
  # Map data ids to their position in id_order for lookup into cell_neighbor_codes
  data_id_order_pos <- match(data$id, id_order)
  
  # Step 3: Build the full row-level neighbor lookup using integer matrix indexing
  # For each row i, its neighbors' row indices are:
  #   row_index_matrix[ cell_neighbor_codes[[ data_id_order_pos[i] ]], data_year_codes[i] ]
  
  neighbor_lookup <- vector("list", n_rows)
  
  for (i in seq_len(n_rows)) {
    nb_codes <- cell_neighbor_codes[[ data_id_order_pos[i] ]]
    if (length(nb_codes) == 0L) {
      neighbor_lookup[[i]] <- integer(0)
    } else {
      yr_col <- data_year_codes[i]
      idx <- row_index_matrix[nb_codes, yr_col]
      neighbor_lookup[[i]] <- idx[!is.na(idx)]
    }
  }
  
  neighbor_lookup
}

# ==============================================================================
# OPTIMIZED compute_neighbor_stats
# ==============================================================================
# Replace do.call(rbind, ...) with pre-allocated matrix.
# The inner lapply with integer indexing is already fast; this is a minor gain.

compute_neighbor_stats_fast <- function(data, neighbor_lookup, var_name) {
  vals   <- data[[var_name]]
  n_rows <- length(neighbor_lookup)
  out    <- matrix(NA_real_, nrow = n_rows, ncol = 3)
  
  for (i in seq_len(n_rows)) {
    idx <- neighbor_lookup[[i]]
    if (length(idx) == 0L) next
    neighbor_vals <- vals[idx]
    neighbor_vals <- neighbor_vals[!is.na(neighbor_vals)]
    if (length(neighbor_vals) == 0L) next
    out[i, 1] <- max(neighbor_vals)
    out[i, 2] <- min(neighbor_vals)
    out[i, 3] <- mean(neighbor_vals)
  }
  out
}

# ==============================================================================
# OPTIMIZED compute_and_add_neighbor_features (drop-in replacement)
# ==============================================================================
compute_and_add_neighbor_features_fast <- function(data, var_name, neighbor_lookup) {
  stats <- compute_neighbor_stats_fast(data, neighbor_lookup, var_name)
  data[[paste0(var_name, "_neighbor_max")]]  <- stats[, 1]
  data[[paste0(var_name, "_neighbor_min")]]  <- stats[, 2]
  data[[paste0(var_name, "_neighbor_mean")]] <- stats[, 3]
  data
}

# ==============================================================================
# OPTIMIZED OUTER PIPELINE
# ==============================================================================
# Build the lookup once (fast version)
neighbor_lookup <- build_neighbor_lookup_fast(cell_data, id_order, rook_neighbors_unique)

# Compute neighbor features for each source variable
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {
  cell_data <- compute_and_add_neighbor_features_fast(cell_data, var_name, neighbor_lookup)
}

# The trained Random Forest model is untouched â€” only the feature-engineering
# pipeline has been optimized. All numerical outputs (max, min, mean of
# neighbor values) are preserved identically.
```

## Summary of Changes and Expected Speedup

| Component | Original Problem | Fix | Impact |
|---|---|---|---|
| `build_neighbor_lookup` | 6.46M `paste()` calls + character named-vector lookups into a 6.46M-element vector â†’ **the true bottleneck (est. 80+ hrs)** | Replace with `match()` + integer matrix `row_index_matrix[id_code, year_code]` for O(1) lookups; exploit time-invariance of neighbor graph | **~1000Ã— speedup** â†’ minutes instead of days |
| `compute_neighbor_stats` | `do.call(rbind, ...)` on 6.46M 3-element vectors â€” **not the main bottleneck** but suboptimal | Pre-allocated matrix, direct `for` loop | ~2-3Ã— minor speedup |
| Overall pipeline | ~86+ hours | Vectorized integer indexing throughout | **Estimated total: 5â€“20 minutes** on a 16 GB laptop |

The colleague's diagnosis was wrong: `do.call(rbind, result)` is a single O(n) allocation per variable and the `lapply` in `compute_neighbor_stats` does only fast integer-vector subsetting. The real bottleneck was `build_neighbor_lookup()` performing billions of string constructions and O(1)-amortized-but-high-constant-factor character hash lookups per row, compounded across 6.46 million rows. The fix eliminates all string operations from the hot path and replaces them with direct integer matrix indexing.
END_RESPONSE>>>
