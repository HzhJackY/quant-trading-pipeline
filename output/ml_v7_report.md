## V7 Alpha Engine — Training Report

- **Label:** forward_return_1m -> rank [0,1]
- **Gap:** 0M (standard 1-step-forward, no blind zone)
- **Objective:** Custom L2 + 2.0*(pred-prev)^2
- **Folds:** 54
- **Features:** 16 cols
- **Window:** 36M train + 6M val + 1M test

```
L = 0.5*(pred-y)^2 + lambda*0.5*(pred-prev)^2
g = (pred-y) + lambda*(pred-prev)
h = 1 + lambda
```