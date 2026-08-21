# LinkedIn post

**Publish once, with `paper/preprint.pdf` attached.** LinkedIn's limit is 3000 characters;
the text below is 2,996, so it pastes without truncation. Assignment §14 requires seven
elements, each marked in the checklist at the bottom so nothing is lost while editing.

---

I built a small decision agent this week for a problem I live with as a mobile developer. The most useful thing that happened was strangers telling me I was wrong.

The problem, in one sentence: when an in-app purchase clears the store, the developer still cannot see who was behind it (a real customer, a stolen card, a hijacked account, a kid on a parent's phone, or someone who will use the item then demand a refund), and the entitlement must be granted or refused immediately anyway.

Why an agent rather than a rule: the true state is hidden and the evidence is partial, so the honest object is a belief across all five possibilities, not a label. The agent holds a probability for each, updates it with Bayes rule, and picks the action with the lowest expected cost.

Why an explicit probability model rather than an LLM: every prior, likelihood and threshold is written down, so any decision can be replayed and any single number argued with. I did not choose that for elegance. A practitioner asked for it unprompted, wanting the refund decision to come from "a rule I can audit and replay rather than something generative."

The result I did not expect: I priced human review as information you buy. Across 50 simulated cases its value beat its cost exactly once. Even on the most ambiguous case in the set, where no explanation rose above 38%, the model declined to buy the review. My instinct said escalate. The arithmetic said escalating buys nothing worth paying for.

The design change that came from a public discussion: an iOS developer got to the same place with none of my machinery. On delaying a risky purchase a few minutes: "Those few minutes buy you no new information, since Apple won't confirm a stolen card for days, so you're paying UX cost for nothing."

That is a value-of-information argument in plain English, and it exposed a real defect: my simulation treated a human reviewer as someone who could resolve the hidden state. At purchase time nobody can. Sixteen numbered design changes came out of these threads. That one cost me the most.

The largest limitation: the simulation draws cases from the same likelihood tables the agent scores them with, so the model is correct by construction. The results demonstrate arithmetic, not detection power. Every number is an assumption. I found no public base rates for in-app purchase fraud.

What I would like comments on: three practitioners told me the real leverage is not at purchase time at all. Grant instantly, revoke fast, or answer the platform's refund query well. So if you ship IAP: is there any purchase you would actually hold, and what signal would make you hold it? One candidate survived my cost analysis, a high-value consumable where nothing can be clawed back. Is that corner real, or an artifact of my assumptions?

Preprint attached. Code, data and the full discussion record:
https://github.com/SL-A-SH/AI-Native-Cohort

(Course preprint in IJCAI format. Not submitted to or accepted by IJCAI.)

---

## Element checklist (§14)

| Required | Where |
|---|---|
| The problem in one sentence | Paragraph 2 |
| The reason for the agent design | "Why an agent rather than a rule" |
| The reason for the probability model | "Why an explicit probability model" |
| One important result or failure | The value-of-information result |
| One design change from a public discussion | The DC-14 quote and what it broke |
| The largest known limitation | "The largest limitation, stated plainly" |
| One specific request for comments | The closing question |
| PDF attached | `paper/preprint.pdf` |
