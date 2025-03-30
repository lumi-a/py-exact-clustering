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