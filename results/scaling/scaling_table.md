| Train size | Transformer test acc | Best baseline | Baseline used | Margin |
|---|---|---|---|---|
| 500 | 0.6799 ± 0.0058 | 0.7571 | tf-idf + logistic regression | -0.0773 − |
| 1,000 | 0.7083 ± 0.0081 | 0.8210 | tf-idf + multinomial NB | -0.1127 − |
| 2,000 | 0.7395 ± 0.0043 | 0.8336 | tf-idf bigrams + logistic regression | -0.0941 − |
| 5,000 | 0.7874 ± 0.0044 | 0.8600 | tf-idf bigrams + logistic regression | -0.0726 − |
| 10,000 | 0.8210 ± 0.0028 | 0.8776 | tf-idf bigrams + logistic regression | -0.0565 − |
| 20,000 | 0.8378 ± 0.0035 | 0.8922 | tf-idf bigrams + logistic regression | -0.0544 − |

**No crossover found.** The Transformer did not overtake the best linear baseline at any size up to 20,000. On this task, with this architecture and no pretraining, the linear model is the better choice across the whole range swept.