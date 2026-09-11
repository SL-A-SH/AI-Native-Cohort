# review-record.md: NPC Tactical Belief Agent

Per §11. Three AI reviews: practitioner, probability model, and preprint.

**Status: awaiting the reviews.** The prompts are in `notes/review-prompts.md`. They are run
separately, in a clean session or a different tool per review, for a reason worth stating here
rather than burying: a review written by the same assistant that built the agent, with its own
accept/reject reasoning attached, is the work marking itself. The **Reason** column below has
to be mine.

Run the reviews **before** drafting the paper, not after. They exist to change it, and a review
collected after the writing is done only ever gets used to defend what is already there.

---

## How to fill this in

One row per review comment. Nothing is accepted automatically: a review that produced twenty
comments and got twenty acceptances was not read. Expect to reject several, including ones that
sound authoritative, and reject them with a reason.

`Evidence` points at a file and a number: a section of `results/failure-analysis.md`, a row of
`results/sweep-seeds.csv`, a case and tick in `results/failures.csv`.

---

## Review 1: practitioner

**AI tool:** _(name and version)_
**Date run:** _(date)_
**Prompt used:** `notes/review-prompts.md`, Review 1

| # | Review comment | Accept / reject | Reason (my words) | Change made | Evidence |
|---|---|---|---|---|---|
| P1 | | | | | |
| P2 | | | | | |
| P3 | | | | | |

---

## Review 2: probability model

**AI tool:** _(name and version)_
**Date run:** _(date)_
**Prompt used:** `notes/review-prompts.md`, Review 2

| # | Review comment | Accept / reject | Reason (my words) | Change made | Evidence |
|---|---|---|---|---|---|
| B1 | | | | | |
| B2 | | | | | |
| B3 | | | | | |

---

## Review 3: preprint

**AI tool:** _(name and version)_
**Date run:** _(date)_
**Prompt used:** `notes/review-prompts.md`, Review 3
**Verdict given:** _(accept / weak accept / weak reject / reject)_

| # | Review comment | Accept / reject | Reason (my words) | Change made | Evidence |
|---|---|---|---|---|---|
| R1 | | | | | |
| R2 | | | | | |
| R3 | | | | | |

---

## Reviews already carried out during the build

These are not the three §11 reviews and do not substitute for them. They are recorded because
each one changed the work and the paper will reference them, and because leaving them out would
make the §11 reviews look like the only scrutiny the project received.

| Source | Comment | Outcome | Evidence |
|---|---|---|---|
| Self-review while building `agent.py` | Illegibility priced at zero when the evidence ratio was exactly 1.0, which is the unattributable case, so the agent flanked confidently having observed nothing | Accepted. Repriced off the unattributable fraction `1/ratio` | `src/agent.py`, `unattributable_fraction` |
| Self-review while building `agent.py` | Flank gain counted absolute mass beyond a chokepoint, so under a uniform belief half the map counted and flanking on no information looked valuable | Accepted. Now measured against the null belief | `src/agent.py`, `_option_flank` |
| First 40-case run | Agent retreated on 54% of decisions: a flat cost let it dodge the staleness penalty on holding | Accepted. Retreat now carries the same staleness | `results/failure-analysis.md` §7 |
| First 40-case run | Agent held on 87% of decisions, including watching the player from three steps away, because gain was credited for cells it had already cleared | Accepted. Gain now counts only uncleared cells | `src/agent.py`, `_expected_gain` |
| First 40-case run | Baselines had no information for 87% of every episode; the experiment measured "does the agent stay put", not "does it search well" | Accepted. Every episode now opens with an inciting noise | `experiments/run.py`, the opening-incident block |
| Failure analysis | Stated reasons were fabricated from the action name: every hold logged as "too little to go on" while reporting traceability of 151 | Accepted. Reasons derived from the winning option's terms | `results/failure-analysis.md` §7 |
| Building the decision record | Sighting likelihood ignored `obs.cell` entirely, discarding the location of the most precise observation available | Accepted. Sightings localise with a tight Gaussian. Changed a headline finding: capture 59.7% to 68.1% across 8 seeds | `results/failure-analysis.md` §7 |
| Self-review of the redundancy result | Low redundant-search count might be an artefact of committing less often rather than of the negative-information model | Rejected: normalised per committed action the gap holds, on comparable commit counts. **But this was the wrong test** — see the row below | `results/decisions.csv`, `results/episodes.csv` |
| Probability review, finding 7 | The redundancy result may be explained entirely by the hand-coded `last_cleared` penalty rather than by probabilistic negative information. The self-review above does not separate them, because both mechanisms are switched on in every arm of it: normalising per commit answers "is it moving less", not "which mechanism causes this" | **Accepted, and the 2x2 ablation was run.** The confound was real and the claim was not established. The ablation does not confirm the reviewer's hypothesis either: negative information alone, with the memory penalty off, cuts redundant searches per commit from 0.363 to 0.029. Memory alone reaches 0.002 but costs 7.5 points of capture. The two are substitutes, not complements | `experiments/ablation.py`, `results/ablation.csv` |
