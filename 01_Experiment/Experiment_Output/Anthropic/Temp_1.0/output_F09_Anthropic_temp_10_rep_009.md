 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The current implementation is extremely slow (~86+ hours) for three interconnected reasons:

### 1. Row-level `lapply` over 6.46 million rows in `build_neighbor_lookup`
For every single cell-year row, the function performs character coercion (`as.character`), string pasting (`paste(..., sep="_")`), and named-vector lookups (`idx_lookup[neighbor_keys]`). With ~6.46M rows, this produces billions of small string operations and hash-table lookups inside a sequential R `lapply`. This is the **dominant bottleneck**.

### 2. Redundant recomputation of the spatial topology per year
The rook-neighbor relationships are **static across years** — cell 𝑖's neighbors are the same in 1992 as in 2019. Yet `build_neighbor_lookup` re-resolves them for every cell-year combination (6.46M times) instead of for the 344,208 unique cells once and then replicating the structure across years.

### 3. Row-level `lapply` in `compute_neighbor_stats`
Even after the lookup is built, `compute_neighbor_stats` loops over 6.46M entries, extracting and summarizing small vectors in pure R. This is slow because R's `lapply` over millions of tiny operations has enormous per-iteration overhead.

### Summary of waste
| Operation | Current scale | Optimal scale | Ratio |
|---|---|---|---|
| Neighbor resolution | 6.46M row lookups | 344,208 cell lookups (once) | ~19× fewer |
| String key creation | ~6.46M `paste` calls + hash lookups | 0 (use integer join) | ∞ |
| Stat computation | 6.46M R-level loops per variable | 1 vectorised `data.table` grouped join per variable | orders of magnitude |

---

## Optimization Strategy

**Core idea:** Build the neighbor edge-list once at the cell level (344K rows, not 6.46M), then use `data.table` keyed joins to attach yearly attribute values and compute grouped `max`, `min`, `mean` — all vectorised in C under the hood.

### Steps

1. **Build a static edge table** from `rook_neighbors_unique` (the `nb` object): one row per directed edge `(cell_id, neighbor_id)`. This is done once and is ~1.37M rows.

2. **Key the panel data** by `(id, year)` in a `data.table`.

3. **For each variable**, join the edge table to the panel data to fetch neighbor values, then compute grouped stats with `data.table`'s `:=` and `by=` — no R-level loop over rows.

4. **Left-join** the results back onto the main panel. Cells with no neighbors (e.g., boundary cells missing from the panel for a given year) get `NA`, preserving the original numerical estimand exactly.

5. **Predict** with the already-trained Random Forest model — unchanged.

**Expected speedup:** From ~86 hours to **minutes** (typically 2–10 minutes depending on disk I/O and RAM pressure on a 16 GB laptop). The 6.46M-row `data.table` plus the edge table plus intermediate join results fit comfortably within 16 GB.

---

## Working R Code

```r
library(data.table)

# ──────────────────────────────────────────────────────────────
# STEP 0: Ensure cell_data is a data.table
# ──────────────────────────────────────────────────────────────
if (!is.data.table(cell_data)) {
 cell_data <- as.data.table(cell_data)
}

# ──────────────────────────────────────────────────────────────
# STEP 1: Build a STATIC edge table from the nb object (once)
#
#   rook_neighbors_unique : an nb object (list of integer vectors)
#   id_order              : vector mapping position → cell id
#
#   Result: edges_dt with columns (cell_id, neighbor_id)
#           ~1.37 M rows (directed pairs)
# ──────────────────────────────────────────────────────────────
build_edge_table <- function(id_order, neighbors_nb) {
  # neighbors_nb[[i]] contains integer indices into id_order
  # that are neighbors of cell id_order[i].
  n <- length(neighbors_nb)
  from_list <- vector("list", n)
  to_list   <- vector("list", n)
  for (i in seq_len(n)) {
    nb_idx <- neighbors_nb[[i]]
    # spdep nb objects use 0L to signal "no neighbors"
    nb_idx <- nb_idx[nb_idx != 0L]
    if (length(nb_idx) > 0L) {
      from_list[[i]] <- rep(id_order[i], length(nb_idx))
      to_list[[i]]   <- id_order[nb_idx]
    }
  }
  data.table(
    cell_id     = unlist(from_list, use.names = FALSE),
    neighbor_id = unlist(to_list,   use.names = FALSE)
  )
}

cat("Building static edge table …\n")
edges_dt <- build_edge_table(id_order, rook_neighbors_unique)
cat(sprintf("  Edge table: %s directed edges\n", format(nrow(edges_dt), big.mark = ",")))

# ──────────────────────────────────────────────────────────────
# STEP 2: Key the panel data for fast joins
# ──────────────────────────────────────────────────────────────
setkey(cell_data, id, year)

# ──────────────────────────────────────────────────────────────
# STEP 3: For each variable, compute neighbor max / min / mean
#         via a single data.table join + grouped aggregation,
#         then attach back to cell_data.
# ──────────────────────────────────────────────────────────────
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

compute_and_add_neighbor_features_dt <- function(panel_dt, edges, var_name) {
  # --- 3a. Build a slim lookup: (id, year, value) ---------
  lookup <- panel_dt[, .(id, year, value = get(var_name))]
  setnames(lookup, "id", "neighbor_id")
  setkey(lookup, neighbor_id, year)

  # --- 3b. Expand edges × years by joining to the panel --
  #     For every (cell_id, year) pair we pull the neighbor's
  #     value from the lookup in one vectorised join.
  #
  #     We add 'year' to the edge table via a cross-join with
  #     unique years?  No — more efficient: join edges onto
  #     the panel's (cell_id, year) to get the years that
  #     actually exist, then join neighbor values.
  # --------------------------------------------------------

  # Get (cell_id, year) pairs that exist in the panel
  cell_years <- panel_dt[, .(cell_id = id, year)]
  # Merge with edges to get (cell_id, year, neighbor_id)
  #   — one row per cell-year-neighbor triple
  expanded <- edges[cell_years, on = .(cell_id), allow.cartesian = TRUE, nomatch = 0L]
  # Now expanded has: cell_id, neighbor_id, year

  # Join to get the neighbor's value for that year
  setkey(expanded, neighbor_id, year)
  expanded[lookup, value := i.value, on = .(neighbor_id, year)]

  # --- 3c. Grouped aggregation ----------------------------
  stats <- expanded[
    !is.na(value),
    .(
      nb_max  = max(value),
      nb_min  = min(value),
      nb_mean = mean(value)
    ),
    by = .(cell_id, year)
  ]

  # --- 3d. Attach to the panel ----------------------------
  max_col  <- paste0("neighbor_max_",  var_name)
  min_col  <- paste0("neighbor_min_",  var_name)
  mean_col <- paste0("neighbor_mean_", var_name)

  setnames(stats, c("nb_max", "nb_min", "nb_mean"),
                  c(max_col,  min_col,  mean_col))

  # Remove old columns if they exist (re-run safety)
  for (col in c(max_col, min_col, mean_col)) {
    if (col %in% names(panel_dt)) panel_dt[, (col) := NULL]
  }

  # Left join back (preserves rows with no neighbors → NA)
  setkey(stats, cell_id, year)
  panel_dt[stats, (c(max_col, min_col, mean_col)) :=
    mget(paste0("i.", c(max_col, min_col, mean_col))),
    on = .(id = cell_id, year)]

  invisible(panel_dt)
}

cat("Computing neighbor features …\n")
for (var_name in neighbor_source_vars) {
  cat(sprintf("  %s … ", var_name))
  t0 <- proc.time()
  cell_data <- compute_and_add_neighbor_features_dt(cell_data, edges_dt, var_name)
  elapsed <- (proc.time() - t0)["elapsed"]
  cat(sprintf("done (%.1f s)\n", elapsed))
}

# ──────────────────────────────────────────────────────────────
# STEP 4: Predict with the EXISTING Random Forest model
#         (model object is assumed to be in `rf_model`)
# ──────────────────────────────────────────────────────────────
cat("Generating predictions with the trained Random Forest …\n")
cell_data[, prediction := predict(rf_model, newdata = cell_data)]
cat("Done.\n")
```

---

## Why This Is Correct and Safe

| Concern | Guarantee |
|---|---|
| **Numerical equivalence** | `max`, `min`, `mean` are computed over exactly the same neighbor sets as before (rook contiguity, same year, `NA`-excluded). The only difference is the computation path (vectorised join vs. row-level R loop). |
| **Trained RF model preserved** | The model object (`rf_model`) is never modified or retrained; only `predict()` is called. |
| **Column names preserved** | Output columns follow the same `neighbor_{max,min,mean}_{var}` naming pattern consumed by the RF model's formula/feature list. Adjust the pattern in `setnames` if the original code used a different convention. |
| **Memory** | Peak memory ≈ panel (6.46M × ~115 cols ≈ 5–6 GB) + expanded edge-year table (worst case ~1.37M × 28 ≈ 38.4M rows × 3 int/dbl cols ≈ 0.9 GB) + stats table ≈ well within 16 GB. Each variable is processed and released sequentially. |
| **No external dependencies beyond `data.table`** | Already one of the most common R packages; no compilation or special hardware needed. |

### Expected runtime: **2–10 minutes total** (vs. 86+ hours), a speedup of roughly **500–2,500×**.