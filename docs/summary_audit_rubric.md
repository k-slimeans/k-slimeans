# Summary audit rubric (issue #34)

Written before any summary was read for this audit. (The only summaries seen earlier were the three hand-read
examples already in `docs/results/prelim_bank77.md` section 6.) One rater; the audit says so wherever it is
reported. Do not look at the key files (`*_key.csv`) while rating: the sheet hides which method wrote which
summary.

## What you get

`docs/results/summary_audit_sheet_<dataset>.csv` (and a `.md` copy for reading). Each row is one summary with
the cluster it was written for: the cluster size and up to 10 of its member texts, drawn at random with a fixed
seed. Rows come in groups of four (one gold intent, four anonymous summaries, A to D, each with its own cluster's
members). Fill the four rating columns and, if you like, `notes`.

## The four ratings

Rate each row independently of the other three rows of its group. Use only the shown members and the summary.
All four ratings are for the summary text as written, not for what a better summary would say.

### 1. `members_fit` (coverage), an integer from 0 to the number of members shown

Count the shown members the summary is "about". A member fits if you would say "yes, that is the kind of text this
summary describes". The summary does not need to repeat the member's words. It covers the cluster when at least
half of the shown members fit. A summary that covers only one sub-group of a cluster with several sub-groups gets
the count of that sub-group.

Do not penalise a summary for a member that is itself off-topic for the cluster.

### 2. `generic` (0 or 1)

1 if the summary says nothing specific to these members: cover the members and read the summary alone; if it would
describe most clusters of this dataset as well as this one, it is generic. Typical cases are text about the task
instead of the content ("what is the main topic of the cluster"), or a topic so broad it spans the whole domain
(for example just "banking" or "customer questions"). 0 if it points at a topic narrower than the dataset.

A summary can be generic and still contain members' words; it is judged on how many other clusters it would fit.

### 3. `unsupported` (0 or 1)

1 if the summary states a topic, detail, number, name, product, condition or constraint that none of the shown
members contains or clearly implies. Paraphrase and generalisation of what the members say are fine. If a
detail might come from members that are not shown (the cluster is larger than 10), mark 0 and write "?" in
`notes`. Mark 1 only when it is clearly outside what the shown members support.

### 4. `fluent` (0 or 1)

1 if the text reads as well-formed English of the shape asked for. For the Qwen3 summaries that is a single
well-formed question with no echoed instruction, no stray quotation marks or markup, and no cut-off sentence.
For the extractive (LSA) summaries, which are concatenated member sentences and never asked for a single
question, it means each sentence is well-formed and nothing is cut off. 0 otherwise.

Fluency says nothing about whether the summary is right: a fluent summary can fail all of the other three.

## Rules of thumb

- Read the members first and form your own idea of the cluster, then read the summary.
- Judge each of the four ratings on its own; do not let a bad coverage rating change the fluency rating.
- Do not guess which method wrote a summary; the group letters are shuffled.
- Leave a cell empty only if you cannot rate it, and say why in `notes`; empty cells are dropped from the
  agreement computation.

## How the hand ratings are compared with the automatic flags

`scripts/audit_agreement.py` turns the ratings into labels and compares each with the automatic flag for the same
summary (definitions in `src/kslimeans/summary_quality.py`):

| Hand label | Rule | Automatic flag |
| --- | --- | --- |
| low coverage | `members_fit` / members shown < 0.5 | `flag_low_coverage`: fewer than half of the cluster's members have this summary as their nearest summary embedding |
| generic | `generic` = 1 | `flag_generic`: duplicate, near-duplicate, mentions the cluster, or a non-positive specificity margin |
| unsupported | `unsupported` = 1 | `flag_unsupported`: at least half of the content lemmas appear in no member, or a number or entity appears in no member |
| not fluent | `fluent` = 0 | `flag_not_fluent`: any fluency proxy fails |

Reported per property: the 2 x 2 table, raw agreement and Cohen's kappa. With about 40 summaries per dataset these
are rough numbers and are reported with their counts.
