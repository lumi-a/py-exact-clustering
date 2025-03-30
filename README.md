Low-effort way of calling the rust-crate [`exact-clustering`](https://github.com/lumi-a/exact-clustering) from Python. Consult [docs.rs](https://docs.rs/exact-clustering) for more extensive documentation.

```py
import exact_clustering

weighted_points = [
    (1.0, [0, 0]),
    (1.0, [1, 0]),
    (3.0, [0, 2]),
]

print(exact_clustering.weighted_continuous_kmeans_price_of_hierarchy(weighted_points))
```

View provided methods via `help(exact_clustering)` in the python-repl or take a look at `/src/lib.rs`. At time of writing, they are:

- `unweighted_continuous_kmeans_price_of_greedy(points)`
- `unweighted_continuous_kmeans_price_of_hierarchy(points)`
- `unweighted_discrete_kmeans_price_of_greedy(points)`
- `unweighted_discrete_kmeans_price_of_hierarchy(points)`
- `unweighted_discrete_kmedian_price_of_greedy(points)`
- `unweighted_discrete_kmedian_price_of_hierarchy(points)`
- `weighted_continuous_kmeans_price_of_greedy(weighted_points)`
- `weighted_continuous_kmeans_price_of_hierarchy(weighted_points)`
- `weighted_discrete_kmeans_price_of_greedy(weighted_points)`
- `weighted_discrete_kmeans_price_of_hierarchy(weighted_points)`
- `weighted_discrete_kmedian_price_of_greedy(weighted_points)`
- `weighted_discrete_kmedian_price_of_hierarchy(weighted_points)`
