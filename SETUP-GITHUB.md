# Pushing this to GitHub

Written for you to run. Nothing here touches your account on its own.

The repo lives at `~/Desktop/Career/projects/attention-ablations`.

---

## Before you start: two decisions

**Repository name.** `attention-ablations` is what the README and the Colab notebook assume. `transformer-from-scratch` is the more searchable name but it puts you in a very crowded field, which is the thing we were trying to avoid. I would keep `attention-ablations`.

**Public or private.** It has to be public to be worth anything to a recruiter. There is nothing sensitive in it. The one thing to check is that `notebooks/original_coursework_submission.ipynb` carries your name in the filename inside the repo, which is fine, but the assignment PDF from IIT Bombay is watermarked "not for sharing" so **do not commit `epgd_assgmt-2.pdf`**. It is not in the folder and should stay out.

---

## 1. Check what you are about to publish

```bash
cd ~/Desktop/Career/projects/attention-ablations
ls -la
```

You should see `README.md`, `LICENSE`, `Makefile`, `pyproject.toml`, `requirements.txt`, `.gitignore`, and the `src`, `tests`, `scripts`, `notebooks`, `results` folders.

## 2. Run the tests locally once

Do not push code you have not seen pass on your own machine.

```bash
pip install -r requirements.txt
make test
```

Expect 47 passed. Then the offline smoke tests, which need no download:

```bash
make smoke
python scripts/run_scaling.py --synthetic --sizes 100 200 400 --epochs 4 --seeds 2
```

If all three work, the repo is sound.

## 3. Initialise git

```bash
git init
git branch -M main
git add .
git status
```

Read the `git status` output before committing. `__pycache__`, `.pytest_cache` and `*.pt` files should **not** appear in the staged list. If they do, `.gitignore` is not being picked up and you should stop and check.

```bash
git commit -m "Transformer encoder from scratch, with equivalence tests, a data-scaling sweep and an ablation study

Implements scaled dot-product attention, multi-head self-attention, sinusoidal
positional encoding, feed-forward network and encoder block using primitive
tensor operations only. Correctness is verified against nn.MultiheadAttention
and F.scaled_dot_product_attention.

Adds masked mean pooling, sqrt(d_model) embedding scaling, validation-based
model selection, tf-idf baselines, a multi-seed ablation runner, and a
data-scaling sweep that reports where the Transformer overtakes tf-idf."
```

## 4. Create the empty repo on GitHub

On github.com: New repository, name `attention-ablations`, Public.

**Do not tick "Add a README", "Add .gitignore" or "Choose a licence."** You already have all three, and adding them creates an initial commit you then have to merge around.

## 5. Connect and push

```bash
git remote add origin https://github.com/YOUR_USERNAME/attention-ablations.git
git push -u origin main
```

If it asks for a password, GitHub wants a personal access token, not your account password. Settings, Developer settings, Personal access tokens, Tokens (classic), Generate new token, tick `repo` scope. Or use the GitHub CLI:

```bash
gh auth login
gh repo create attention-ablations --public --source=. --push
```

## 6. Fill in the results, which is the point of the whole thing

The README currently says the results table is not filled in. **A repo whose headline table is empty is worse than no repo**, so this step is not optional.

1. Open `notebooks/run_study.ipynb` in Colab. File, Upload notebook, or push first and open from GitHub.
2. Runtime, Change runtime type, T4 GPU.
3. Edit cell 1 and replace `YOUR_USERNAME` with your GitHub username.
4. Run the cells in order. Budget roughly two hours on a T4 for everything. Section 6 has a `scaling-quick` cell that gives you the shape of the headline result in about ten minutes, so run that before committing to the long jobs.
5. **Scaling result:** copy the table printed in section 6 into `README.md` between the `<!-- BEGIN SCALING TABLE -->` and `<!-- END SCALING TABLE -->` markers.
6. **Ablation result:** copy the table printed in section 9 into `README.md` between the `<!-- BEGIN RESULTS TABLE -->` and `<!-- END RESULTS TABLE -->` markers.
7. Replace both placeholder rows, then write two or three sentences under each saying what you actually found.

Then commit the results and the figures:

```bash
git add README.md results/
git commit -m "Add results from full IMDB run"
git push
```

## 7. Write the findings paragraphs yourself

This is the part I cannot do for you and it is the part a reviewer reads first. Once you have the numbers, answer in plain language:

**On the scaling curve.** Where is the crossover, and what does it imply for someone deciding whether to reach for a Transformer on a few thousand labelled rows? If there is no crossover up to 20,000, say that plainly. A repository whose headline finding is "the simpler model won across the whole range I tested" is more credible than one that buries it.

**On the ablations.** Which single ablation cost the most accuracy, and does that match what the architecture papers would predict? What did unmasked mean pooling actually cost? That is your own bug, quantified.

**On positional encoding specifically.** If removing it costs close to nothing, say so and connect it to the scaling curve. It would mean the model is not using word order, which is to say it has learned a bag of words, which is exactly why the bag-of-words baseline stays competitive. That link between the two experiments is the most interesting sentence available to you.

**On the attention ranking.** Did it show sentiment words, or mostly common words like "the" and "movie"? If it is the latter, say so. That contradicts the claim in your original coursework notebook, and contradicting yourself with evidence is the single most credible thing in the repository.

Three or four honest sentences per section beats a page of hedging.

## 8. Repo settings, five minutes

- **Description:** "A Transformer encoder written from scratch, verified against PyTorch, plus a data-scaling study of when it starts beating tf-idf and an ablation study of which components earn their place."
- **Topics:** `pytorch`, `transformer`, `attention`, `deep-learning`, `nlp`, `ablation-study`, `from-scratch`, `scaling-laws`
- **Website:** leave blank.
- Untick Wikis, Issues and Projects unless you want them. A tidy repo page reads better.

## 9. Then use it

- Add the URL to your CS resume under the IIT Bombay diploma line.
- Add it to LinkedIn under Projects, and to the Featured section.
- It goes in the Einride application. Their advert asks about model output validation and about explaining AI to non-technical colleagues, and this repo is evidence for both.
- It does **nothing** for the India investing track. Do not put it on the VC resume.

---

## If something goes wrong

**`make test` fails on import.** You are probably not in the repo root, or `src` is not on the path. `PYTHONPATH=src pytest tests/ -v` runs it directly.

**Colab cannot find the repo.** It is private, or the username in the clone command is wrong.

**The IMDB download fails in Colab.** Hugging Face occasionally rate-limits unauthenticated downloads. Add an `HF_TOKEN` in Colab secrets, or just retry.

**Out of memory on the T4.** Drop `--batch-size` to 32, or `--max-len` to 128.

**You want to re-run one configuration only.**

```bash
python scripts/run_ablations.py --only "unmasked mean pooling" --n-train 20000 --seeds 3
```

**The scaling sweep says "no crossover found".** That is a valid result, not an error. It means the Transformer either won at every size or lost at every size. Read the margin column to see which, and report it.

**You want more resolution around the crossover.** Add sizes near where the sign flips:

```bash
python scripts/run_scaling.py --sizes 2000 3000 4000 5000 7000 --epochs 30 --seeds 3
```
