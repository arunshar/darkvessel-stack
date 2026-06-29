# darkvessel-stack — reproduced evidence

_Generated 2026-06-29T10:30:32Z by running the test suite on the real/tested code in this repo._

These are **reproduced** results: the code runs and every assertion below holds. Benchmark/leaderboard numbers in the paper (PSNR, mIoU, speedups) remain **targets, not reproduced**, and are labeled as such throughout.

## Test suite (`pytest -v`)

```

============================= 53 passed in 53.97s ==============================
```

## Reproduced demo (headline number)

Lee speckle filter reduces image variance by ~96% on a synthetic speckled field; `haversine` returns 111,195 m for 1 degree of latitude (0.00% error vs the reference).
