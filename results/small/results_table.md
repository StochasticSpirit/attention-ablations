| Configuration | Test acc (mean ± std) | Train acc | Gap | Params |
|---|---|---|---|---|
| _majority class_ | 0.5080 | 0.5120 | +0.0040 | 1 |
| _tf-idf + multinomial NB_ | 0.6760 | 0.9450 | +0.2690 | 1 |
| _tf-idf + logistic regression_ | 0.7660 | 0.9780 | +0.2120 | 19,553 |
| _tf-idf bigrams + logistic regression_ | 0.7580 | 0.9930 | +0.2350 | 39,157 |
| | | | | |
| full model | 0.7220 ± 0.0069 | 0.9773 | +0.2553 | 2,771,458 |
| no positional encoding | 0.7087 ± 0.0179 | 0.9463 | +0.2377 | 2,771,458 |
| single head | 0.6807 ± 0.0286 | 0.9380 | +0.2573 | 2,771,458 |
| no feed-forward network | 0.7253 ± 0.0181 | 0.9807 | +0.2553 | 2,639,618 |
| no residual connections | 0.7093 ± 0.0050 | 0.9440 | +0.2347 | 2,771,458 |
| no layer norm | 0.7167 ± 0.0153 | 0.9393 | +0.2227 | 2,770,434 |
| unmasked mean pooling | 0.7133 ± 0.0205 | 0.9650 | +0.2517 | 2,771,458 |
| pre-layer-norm | 0.7347 ± 0.0081 | 0.9627 | +0.2280 | 2,771,714 |
| 1 layer | 0.6987 ± 0.0101 | 0.9517 | +0.2530 | 2,638,978 |
| 4 layers | 0.7113 ± 0.0122 | 0.9663 | +0.2550 | 3,036,418 |