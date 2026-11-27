# Dec 2 talk: deck skeleton and Cooper's slides (#83)

30-minute talk plus about 10 minutes of questions, all three of us presenting (#83, #84). Slides are submitted with the
revised manuscript on Nov 30. Draft slides from each speaker are due Nov 28; the rehearsal on Dec 1 aims for 28 minutes.

Contents:

1. **Deck skeleton**: every part of the talk, its speaker, its time, its slides and where its numbers come from. Samiksha and Nabin
   fill in their own parts; the skeleton only fixes titles, order and sources.
2. **Cooper's slides and the backups:** now in [`deck.yaml`](deck.yaml). `build_deck.py` builds the `.pptx` from it.
3. (Merged into 2.)
4. Likely questions for Cooper's parts.
5. Open items.

**Rules for every slide** (from #83 and `synthesis.md` §10):

- Every number comes from `docs/results/`, cited by file and section in the slide footer.
- Every claim carries its scope: step 1 or final; which model; 5 seeds; CLINC150 unless the slide says otherwise.
- Every piece of evidence is labelled **interventional** (we changed the input and re-ran or replayed), **association** (logged
  properties go together), **measurement** (a logged run, nothing compared causally) or **identity** (true by algebra).
- No step-1 result is used to explain a final score.
- The human evaluation (#34) has no numbers yet: its slots are `TODO(#34)` placeholders. The AI-rater stand-in, if it is shown at all, is labelled
  "not a human rating" on the slide itself.
- No BBC article text on any slide (the repo is public); ids only.

**Footer convention**, so the three of us look the same: `dataset · step 1 / final · models · 5 seeds · evidence kind — file §section`.
Example: `CLINC150 · step 1 · old 4B, 1.7B · 5 seeds · interventional — counterfactual_swaps.md §2, §4`.

"Old" means the October prompt (#31); "v2" means the clearer prompt (#46). "Step 1" is the first summary step and the Lloyd block after it;
"final" is the score after all five summary steps.

---

## Part 1. Deck skeleton

| # | Part | Speaker | Min | Slides | Sources | Open inputs |
| --- | --- | --- | ---: | --- | --- | --- |
| 1 | The question; k-means with summaries as centroids; the setup | Cooper | 4 | 1.1 to 1.4 (`deck.yaml`) | `clinc150_summarizers.md` §1, §2, §2.4; README "How the algorithm maps to the code" | none |
| 2 | Closest work and what we add | Samiksha, Nabin | 3 | 2.1 closest work; 2.2 what we add | #80 | `TODO(#80, Samiksha and Nabin)`: related work |
| 3 | What a replacement does; how far the causal evidence goes (step 1 vs final) | Cooper | 8 | 3.1 to 3.9 (`deck.yaml`) | `replacement_diagnostic.md` §3 to §8; `counterfactual_swaps.md`; `summary_guard.md`; `prompt_test.md` §8; `synthesis.md` §5, §10, §11 | none |
| 4 | Is the summary a good representative? Automatic scores and the human evaluation | Samiksha | 5 | 4.1 the four score families; 4.2 readable but a poor representative; 4.3 to 4.5 human evaluation | `summary_quality.md` §2 (definitions) and its results sections; `synthesis.md` §12 | `TODO(#34)`: inter-rater agreement, human against automatic, the paraphrase check |
| 5 | Clusters a summary helped or hurt | Samiksha | 2 | 5.1, 5.2 worked clusters | `synthesis.md` §12 (worked examples); `synthesis_case_studies.md` | `TODO(#50 case studies, Samiksha)` |
| 6 | Why the larger model isn't better here | Nabin | 4 | 6.1 the size result under both prompts; 6.2 what explains it, by kind of evidence; 6.3 what 5 seeds can detect; 6.4 scope | `size_result.md` §1 to §5 (§2(e): the hand-read failure modes; §5: the Qwen3 training notes); `prompt_test.md` §8.7 | none (#36 merged) |
| 7 | What it costs, and when it's worth it: an SLM or an extractive summarizer | Nabin | 3 | 7.1 cost table; 7.2 cost per ACC point; 7.3 SLM or LSA | `cost_justification.md` §2 to §4 (§4: dollars); `cost_justification_pareto.png`; `docs/cost_assumptions.md` | none (#10 merged) |
| 8 | The answer to the main question; limits | Cooper | 1 | 8.1 (`deck.yaml`) | `synthesis.md` §13; `cost_justification.md` §3, §4 | `TODO(#34)` |
| | **Total** | | **30** | | | |

Coordination notes:

- **Parts 3, 4 and 5 share material.** Slide 3.4 (the sign of the purity change goes with purity before) is the step from "assignments change" to "cluster
  quality". Part 4 should not repeat it; part 5's worked clusters can point back to it ("this is the mixed-cluster case of slide 3.4").
- **Parts 3 and 6 share the prompt result.** Slide 3.8 gives the clearer prompt as an intervention on the 4B's final score. The size order
  under both prompts is Nabin's (part 6), so slide 3.8 does not show it.
- **Parts 7 and 8 share the cost result.** Slide 8.1 states it in one line and points to part 7.
- **Timing.** The parts add up to 30 minutes; the rehearsal target is 28 (#83). Cooper's three parts are written for 3.5, 7.5 and 1 minutes.

---

## Part 2. Cooper's slides and the backups: in `deck.yaml`

The final wording of Cooper's slides is in [`deck.yaml`](deck.yaml): parts 1, 3 and 8, the Questions slide and backups B1 to B5. That covers on-slide text, chart data, speaker notes with times, and source lines. `build_deck.py` turns it into the deck.
- The numbers were checked against `docs/results/` on Nov 27. No number was wrong.
- Two wording fixes came out of that check:
  - Slide 3.4: the sign pattern holds against plain Lloyd. Against random moves of the same size, the summaries do better in every stratum.
  - The feedback asked for "no more models or experiments", not "no GPU runs".
- The dollar figures (#10) are in slide 8.1.

Build the deck (needs `python-pptx` and `PyYAML`):

    python docs/talk/build_deck.py --content docs/talk/deck.yaml --out k-slimeans-talk.pptx                    # S&T colours, no template
    python docs/talk/build_deck.py --content docs/talk/deck.yaml --out k-slimeans-talk.pptx --template <S&T template .potx>

The Missouri S&T 16:9 template is at https://brand.mst.edu/digital-templates/powerpoint/ (university sign-in). It is not committed.

## Part 4. Likely questions for Cooper's parts

1. **"If replacing the centroid raises the objective, how can it ever help?"** The identity is about fixed assignments. After points move,
   the replacement run and plain Lloyd are different trajectories: the 1.7B ends 0.18% above plain Lloyd in J and +0.0106 above it in ACC at step 1.
   A lower objective does not mean a better partition against the gold labels. (`replacement_diagnostic.md` §3)
2. **"Does the swap experiment explain the 4B's final scores?"** No. It is step 1 only. Over five steps the guard gives +0.009 [−0.006, +0.024]
   and the 4B stays 0.031 below k-means; the swap itself was never run over five steps. (`synthesis.md` §10)
3. **"Why not run the swap over all five steps?"** Later steps need new summaries of the new partitions, which cannot be produced offline;
   that is a new run, and the Nov 18 feedback asked for no more models or experiments. (`counterfactual_swaps.md` §5; #84)
4. **"Aren't the flagged clusters just harder clusters?"** They may be; that is a stated limit. The swap changes the text in the same clusters,
   and the matched control (the same number of random unflagged clusters) does nothing, so "1.7B summaries are better everywhere" is ruled out; "harder clusters" is not.
   (`counterfactual_swaps.md` §4, §5)
5. **"Is the prompt's gain from copying members verbatim?"** Verbatim copies rose (v2 4B 12% of summaries over all steps, old 1.4%), but at step 1 copying accounts for
   about 0.02 of the +0.16 coverage gain, and its link to ACC was not tested. (`synthesis.md` §2, §9)
6. **"Five seeds and no multiplicity correction: how much should we trust this?"** Each interval is per comparison. The results we lean on are
   far from the noise: the step-1 swap is about 9 times the mean re-encoding noise floor (3.1 times its maximum) and holds in 5 of 5 seeds. "Not detectable" means not detectable
   with five seeds. Seeds 5 to 9 were dropped after the Nov 18 feedback. (`counterfactual_swaps.md` §3; `size_result.md` §3)
## Part 5. Open items

- `TODO(names)`, `TODO(course)`: slide 1.1.
- `TODO(#80)`: part 2 (Samiksha and Nabin).
- `TODO(#34)`: part 4 and slide 8.1. The human evaluation is scored once both rating sets are in.
- `TODO(#50 case studies)`: part 5 (Samiksha).
- Parts 6 and 7 (Nabin): the inputs are in.
  - #36 is merged: `size_result.md` §2(e) and §5, `size_failure_modes.md`, `docs/qwen3_sizes_notes.md`.
  - The #10 dollars are in `cost_justification.md` §4.
- Each speaker fills their own placeholder slides in the shared deck. Cooper makes the deck consistent and submits it on Nov 30.
