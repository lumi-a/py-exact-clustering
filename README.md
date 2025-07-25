![PyPI - Version](https://img.shields.io/pypi/v/exact-clustering)

Low-effort way of calling the rust-crate [`exact-clustering`](https://github.com/lumi-a/exact-clustering) from Python. 

```sh
pip install exact-clustering
```

Consult [docs.rs](https://docs.rs/exact-clustering) for more extensive documentation.

```py
from exact_clustering import *

weighted_points = [
    (1.0, [0, 0]),
    (1.0, [1, 0]),
    (3.0, [0, 2]),
]

print(
    weighted_continuous_kmeans_price_of_hierarchy(
        weighted_points
    )
)
# 1.0
```

View provided methods via `help(exact_clustering)` in the python-repl or take a look at `/src/lib.rs`. At time of writing, they are:

```py
def unweighted_continuous_kmeans_price_of_hierarchy(points: list[tuple[float, list[float]]],) -> float: ...
def weighted_continuous_kmeans_price_of_hierarchy(weighted_points: list[list[float]],) -> float: ...
def unweighted_discrete_kmeans_price_of_hierarchy(points: list[tuple[float, list[float]]],) -> float: ...
def weighted_discrete_kmeans_price_of_hierarchy(weighted_points: list[list[float]],) -> float: ...
def unweighted_discrete_kmedian_price_of_hierarchy(points: list[tuple[float, list[float]]],) -> float: ...
def weighted_discrete_kmedian_price_of_hierarchy(weighted_points: list[list[float]],) -> float: ...
def unweighted_continuous_kmeans_price_of_greedy(points: list[tuple[float, list[float]]],) -> float: ...
def weighted_continuous_kmeans_price_of_greedy(weighted_points: list[list[float]],) -> float: ...
def unweighted_discrete_kmeans_price_of_greedy(points: list[tuple[float, list[float]]],) -> float: ...
def weighted_discrete_kmeans_price_of_greedy(weighted_points: list[list[float]],) -> float: ...
def unweighted_discrete_kmedian_price_of_greedy(points: list[tuple[float, list[float]]],) -> float: ...
def weighted_discrete_kmedian_price_of_greedy(weighted_points: list[list[float]],) -> float: ...
```