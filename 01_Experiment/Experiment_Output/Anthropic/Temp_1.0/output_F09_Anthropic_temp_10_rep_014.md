 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The bottleneck is in `build_neighbor_lookup`. It constructs a **per-row** (per cell-year) list of neighbor row indices by pasting character keys and doing named-vector lookups across **~6.46 million rows**. This is catastrophically slow because:

1. **Redundant work across years.** The spatial neighbor topology is *identical* for every year. The current code rebuilds string keys and looks up neighbors for every cell-year row — repeating the same spatial logic 28 times.
2. **Character key hashing at scale.** `paste()` + named-vector lookup (`idx_lookup[neighbor_keys]`) for ~6.46M rows, each with ~4 neighbors on average, means tens of millions of string operations inside an `lapply` — effectively an O(N×K) character-matching loop in R's interpreter.
3. **`lapply` over 6.46M elements** returns a list of 6.46M vectors. This is slow to build and slow to consume in `compute_neighbor_stats`, which again `lapply`s over the same 6.46M entries.

The `compute_neighbor_stats` function is comparatively lighter but still loops over 6.46M list elements in pure R, which is suboptimal.

**Summary:** The design conflates *spatial topology* (fixed) with *temporal panel structure* (repeated), causing a 28× blowup in the most expensive operation, all executed in interpreted R loops over millions of rows.

---

## Optimization Strategy

### Core idea: Separate topology from attributes, then vectorize with `data.table` joins.

1. **Build the neighbor edge table once** — a two-column `data.table` of `(cell_id, neighbor_id)` derived from the `spdep::nb` object. This is ~1.37M rows and is year-independent.

2. **Join yearly attributes onto the edge table.** For each year, every cell's neighbors inherit their attribute values via a keyed `data.table` join. This turns the problem into a standard grouped aggregation.

3. **Compute grouped max/min/mean** using `data.table`'s optimized `by=` grouping — no R-level loops, no string key construction, no per-row `lapply`.

4. **Join the aggregated neighbor stats back** onto the main panel `data.table`.

### Complexity reduction

| Aspect | Old | New |
|---|---|---|
| Neighbor lookup construction | 6.46M string-key lookups | 1.37M-row static edge table (built once) |
| Per-variable aggregation | `lapply` over 6.46M list elements | `data.table` grouped aggregation (~38M edge-year rows) |
| Expected wall-clock time | 86+ hours | **~2–10 minutes** |

### Memory check

The edge table replicated across 28 years is ~1.37M × 28 ≈ 38.4M rows × a few columns of doubles — roughly 1–2 GB, well within 16 GB RAM.

---

## Working R Code

```r
library(data.table)

# ──────────────────────────────────────────────────────────────────────
# STEP 0: Ensure cell_data is a data.table with columns: id, year, and
#         the neighbor source variables.
#         rook_neighbors_unique is the spdep::nb object.
#         id_order is the vector mapping nb list index → cell id.
# ──────────────────────────────────────────────────────────────────────
cell_data <- as.data.table(cell_data)

# ──────────────────────────────────────────────────────────────────────
# STEP 1: Build the static spatial edge table ONCE.
#         This replaces build_neighbor_lookup entirely.
# ──────────────────────────────────────────────────────────────────────
build_edge_table <- function(id_order, nb_obj) {
  edges <- rbindlist(lapply(seq_along(nb_obj), function(i) {
    nbs <- nb_obj[[i]]
    # spdep::nb encodes "no neighbors" as a single 0L
    nbs <- nbs[nbs != 0L]
    if (length(nbs) == 0L) return(NULL)
    data.table(cell_id = id_order[i], neighbor_id = id_order[nbs])
  }))
  edges
}

edge_table <- build_edge_table(id_order, rook_neighbors_unique)
# edge_table: ~1,373,394 rows, columns: cell_id, neighbor_id

cat(sprintf("Edge table: %d directed neighbor pairs\n", nrow(edge_table)))

# ──────────────────────────────────────────────────────────────────────
# STEP 2: For each neighbor source variable, compute neighbor max, min,
#         mean via data.table join + grouped aggregation, then attach
#         the results back to cell_data.
# ──────────────────────────────────────────────────────────────────────
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# Key cell_data for fast joins
setkey(cell_data, id, year)

for (var_name in neighbor_source_vars) {

  cat(sprintf("Computing neighbor stats for: %s\n", var_name))

  # --- 2a: Extract the minimal attribute table (id, year, value) ------
  attr_dt <- cell_data[, .(id, year, value = get(var_name))]
  setnames(attr_dt, "id", "neighbor_id")
  setkey(attr_dt, neighbor_id, year)

  # --- 2b: Replicate edges across all years via join ------------------
  #     edge_table has (cell_id, neighbor_id).
  #     Join onto attr_dt by neighbor_id to pick up (year, value).
  #     This gives us one row per (cell_id, neighbor_id, year) with the
  #     neighbor's attribute value.
  edge_year <- merge(edge_table, attr_dt, by = "neighbor_id",
                     allow.cartesian = TRUE)
  # edge_year columns: neighbor_id, cell_id, year, value

  # --- 2c: Grouped aggregation ----------------------------------------
  stats <- edge_year[!is.na(value),
                     .(nb_max  = max(value),
                       nb_min  = min(value),
                       nb_mean = mean(value)),
                     by = .(cell_id, year)]

  # --- 2d: Name the new columns to match original convention ----------
  new_names <- paste0(var_name, c("_neighbor_max",
                                   "_neighbor_min",
                                   "_neighbor_mean"))
  setnames(stats,
           c("nb_max", "nb_min", "nb_mean"),
           new_names)
  setnames(stats, "cell_id", "id")
  setkey(stats, id, year)

  # --- 2e: Remove old columns if they exist (idempotent re-run) -------
  for (nm in new_names) {
    if (nm %in% names(cell_data)) cell_data[, (nm) := NULL]
  }

  # --- 2f: Join back onto cell_data -----------------------------------
  cell_data <- stats[cell_data, on = .(id, year)]
  setkey(cell_data, id, year)

  # Clean up large intermediates each iteration to stay within 16 GB
  rm(attr_dt, edge_year, stats)
  gc()
}

# ──────────────────────────────────────────────────────────────────────
# STEP 3: Predict with the already-trained Random Forest model.
#         The model object (rf_model) is unchanged.
# ──────────────────────────────────────────────────────────────────────
# Example (adjust to your actual model object and predict call):
# cell_data[, prediction := predict(rf_model, newdata = .SD)]
```

---

## Why This Preserves the Original Numerical Estimand

| Concern | Guarantee |
|---|---|
| **Same neighbor topology** | `build_edge_table` reads the same `rook_neighbors_unique` nb object and the same `id_order` mapping. Every `(cell, neighbor)` pair is identical. |
| **Same per-year attribute values** | The join key is `(neighbor_id, year)`, so each neighbor's value for a given year is exactly the value from `cell_data`. |
| **Same aggregation functions** | `max`, `min`, `mean` over non-NA neighbor values — identical to the original `compute_neighbor_stats`. |
| **NA handling** | `edge_year[!is.na(value), ...]` drops NAs before aggregation, matching the original `neighbor_vals <- neighbor_vals[!is.na(neighbor_vals)]`. Cells with zero valid neighbors get `NA` via the outer join (`stats[cell_data]`), matching the original `return(c(NA, NA, NA))`. |
| **Trained RF model** | The model object is never modified or retrained; only `predict()` is called on the enriched `cell_data`. |

**Expected speedup: from ~86+ hours to a few minutes** — dominated by the `merge` and grouped aggregation, both of which `data.table` handles at near-C speed.