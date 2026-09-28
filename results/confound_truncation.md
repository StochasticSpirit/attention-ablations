# Do the baselines get an unfair advantage by reading the full text?

The Transformer only read the first 256 tokens of each review, set by `max_len`. The tf-idf
baselines see the whole review text, because `run_baselines` is given the untruncated token
lists. A fair number of IMDB reviews are longer than 256 tokens which means that the two model classes are
not being scored on the same text length.

This is worth testing, because the initial headline result of this study states that the linear baseline beats
the Transformer at every training set size we ran it on. If the gap comes from the baselines
reading more, the result is likley because of the preprocessing rather than the models.

## Test

We test by truncating every document to its first 256 tokens, refit the baselines, and score them on the same 25,000 example test set and 20,000 training examples.

| Baseline | Full text | Truncated to 256 tokens | Cost |
|---|---|---|---|
| tf-idf bigrams + logistic regression | 0.8922 | 0.8770 | −1.52 |
| tf-idf + logistic regression | 0.8824 | 0.8696 | −1.28 |
| tf-idf + multinomial NB | 0.8333 | 0.8252 | −0.81 |
| majority class | 0.5000 | 0.5000 | 0.00 |

## What it means

The Transformer scores 0.8440 on the same test set. Against the best baseline on full text, it loses by 4.8 points. Against the same baseline restricted to the text the Transformer actually reads, it still loses by 3.3 points.

So about a third of the gap could be atributed to triuncation, and the rest is because of the model. The linear model still wins on equal footing (truncated data) at 20,000 training examples.

Instead of the 4.8 point gap, 3.3 is more correct as the like for like number.

