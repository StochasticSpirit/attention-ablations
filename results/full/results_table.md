| Configuration | Test acc (mean ± std) | Train acc | Gap | Params |
|---|---|---|---|---|
| _majority class_ | 0.5000 | 0.5023 | +0.0023 | 1 |
| _tf-idf + multinomial NB_ | 0.8333 | 0.8975 | +0.0642 | 1 |
| _tf-idf + logistic regression_ | 0.8824 | 0.9281 | +0.0457 | 20,001 |
| _tf-idf bigrams + logistic regression_ | 0.8922 | 0.9406 | +0.0484 | 40,001 |
| | | | | |
| full model | 0.8440 ± 0.0006 | 0.9526 | +0.1086 | 2,825,474 |
| no positional encoding | 0.8423 ± 0.0056 | 0.9535 | +0.1111 | 2,825,474 |
| single head | 0.8475 ± 0.0011 | 0.9452 | +0.0977 | 2,825,474 |
| no feed-forward network | 0.8373 ± 0.0025 | 0.9506 | +0.1133 | 2,693,634 |
| no residual connections | 0.8320 ± 0.0032 | 0.9114 | +0.0795 | 2,825,474 |
| no layer norm | 0.8410 ± 0.0039 | 0.9055 | +0.0645 | 2,824,450 |
| unmasked mean pooling | 0.8410 ± 0.0013 | 0.9613 | +0.1203 | 2,825,474 |
| pre-layer-norm | 0.8457 ± 0.0040 | 0.9654 | +0.1197 | 2,825,730 |
| 1 layer | 0.8361 ± 0.0014 | 0.9387 | +0.1026 | 2,692,994 |
| 4 layers | 0.8429 ± 0.0010 | 0.9464 | +0.1036 | 3,090,434 |