 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The current implementation is catastrophically slow (~86+ hours) for one core reason: **`build_neighbor_lookup` constructs a per-row (cell×year) lookup list of length ~6.46 million, using character-key hashing inside a sequential `lapply` over every row.** This is unnecessary because the neighbor *topology* is fixed across all 28 years—cell A's neighbors are the same in 1992 as in 2019. The function re-discovers this for every cell-year combination, inflating work by a factor of 28.

Specific bottlenecks:

| Bottleneck | Why it hurts |
|---|---|
| `lapply` over 6.46M rows in `build_neighbor_lookup` | O(n×T) character key creation and lookup; ~6.46M iterations with `paste`/`setNames` |
| Character-keyed `idx_lookup` via `paste(id, year)` | String allocation and hash-table lookup for every neighbor of every row |
| `compute_neighbor_stats` with per-row `lapply` | Another 6.46M R-level iterations; no vectorization |
| Repeated `do.call(rbind, ...)` on 6.46M-element list | Builds a huge matrix from millions of 3-element vectors |

**The topology (which cell neighbors which cell) never changes year to year.** Only the *attribute values* change. The entire design should separate structure from attributes.

---

## Optimization Strategy

1. **Build the adjacency table exactly once** — a two-column `data.table` of `(cell_id, neighbor_id)` with ~1.37M rows, derived from `rook_neighbors_unique`. This is year-invariant.

2. **For each year, join cell attributes onto the adjacency table** — use `data.table` keyed joins. This turns the neighbor-stat computation into a grouped aggregation (`max`, `min`, `mean`) on a ~1.37M-row table per year, fully vectorized in C.

3. **Stack yearly results and join back** to the master panel. No per-row `lapply`, no character keys, no 6.46M-element lists.

**Expected speedup:** From ~86 hours to **minutes** (the adjacency table has ~1.37M rows × 28 years = ~38.4M join+group operations total, which `data.table` handles in seconds per variable).

**Memory:** The adjacency table is ~1.37M rows × 2 integer columns ≈ 11 MB. Yearly attribute slices are ≤ 344K rows. Well within 16 GB.

---

## Working R Code

```r
library(data.table)

# ──────────────────────────────────────────────────────────────────────
# STEP 0 : Convert master data to data.table (if not already)
# ──────────────────────────────────────────────────────────────────────
setDT(cell_data)

# ──────────────────────────────────────────────────────────────────────
# STEP 1 : Build the year-invariant adjacency table ONCE
#
#   rook_neighbors_unique : an nb object (list of integer index vectors)
#   id_order              : vector of cell IDs in the same order as the nb object
# ──────────────────────────────────────────────────────────────────────
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
# ~1,373,394 rows, two integer (or numeric) columns — tiny

cat("Adjacency table rows:", nrow(adj_table), "\n")

# ──────────────────────────────────────────────────────────────────────
# STEP 2 : Compute neighbor stats for every source variable
#
#   For each variable we:
#     (a) join yearly attributes onto the adjacency table
#     (b) group by (cell_id, year) and compute max, min, mean
#     (c) join the results back onto cell_data
# ──────────────────────────────────────────────────────────────────────
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
  #     This is a many-to-many join (each edge × each year the neighbor appears).
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

# ──────────────────────────────────────────────────────────────────────
# STEP 3 : Predict with the already-trained Random Forest
#
#   The trained model object (e.g., `rf_model`) is unchanged.
#   cell_data now contains the same neighbor-stat columns as before.
# ──────────────────────────────────────────────────────────────────────
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
| **Column names match** | Output columns are named `<var>_nb_max`, `<var>_nb_min`, `<var>_nb_mean` — adjust the naming pattern to match whatever the trained model expects |

---

## Expected Performance

| Stage | Old time | New time (est.) |
|---|---|---|
| Build lookup | ~hours (6.46M `paste`+hash) | < 1 sec (vectorized `unlist`) |
| Neighbor stats (5 vars) | ~80+ hours (30M+ R-level iterations) | **2–5 minutes** (`data.table` keyed join + grouped agg) |
| **Total** | **~86 hours** | **< 10 minutes** |

The key insight: **separate topology from attributes, build the adjacency table once, and let `data.table` do the join and aggregation in compiled C code instead of R-level loops.**