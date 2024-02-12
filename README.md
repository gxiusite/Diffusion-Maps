# Diffusion-Maps
Clear code of diffusion mapping with fast correlation functions.

```python
from diffmaps import *
import torch
# Load data
data = torch.randn(10000, 200, dtype=torch.float64)
oos_data = torch.randn(100, 200, dtype=torch.float64)

diffmaps = DiffusionMaps(
    data, batch_size=100, n_largest=10, abs_value=False, demean=True, how="pearson"
)
# corr = diffmaps.correlation(print_log=True).unstack()
diffmaps.fit(data, n_eigs=20)
corrds = diffmaps.predict(oos_data)
```