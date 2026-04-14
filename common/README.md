# Common utilities

Shared modules used by all Python experiment scripts.

## `liouville_sieve.py`

Segmented sieve computing λ(n) = (−1)^Ω(n) for n in a given range.

### API

| Function | Description |
|---|---|
| `compute_liouville(N)` | Full sieve for n = 0..N, returns int8 array |
| `compute_liouville_segment(a, b)` | Sieve for n ∈ [a, b] only |
| `sieve_primes(limit)` | Primes up to `limit` via Eratosthenes |
| `compute_random_cm(N, seed)` | Random CM function f(p) = ±1 iid |

### Memory

Approximately N bytes for the full sieve.  For N = 10^10, this is ~10 GB.

### Quick test

```python
import sys; sys.path.insert(0, '..')
from common.liouville_sieve import compute_liouville
lam = compute_liouville(10**6)
assert lam[2] == -1 and lam[4] == 1 and lam[30] == -1
print("OK")
```
