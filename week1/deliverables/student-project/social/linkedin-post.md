# LinkedIn post

**Publish once, with `paper/preprint.pdf` attached.** Assignment §14 requires seven
elements; each is marked in the checklist at the bottom so nothing gets edited out by
accident.

---

I spent this week building a small decision agent for a problem I have been living with as
a mobile developer, and the most useful thing that happened was strangers telling me I was
wrong.

**The problem, in one sentence:** when an in-app purchase clears the store, the developer
still cannot see who was behind it — a paying customer, a stolen card, a hijacked account,
a child on a parent's phone, or someone who will consume the item and then demand a refund
and the entitlement has to be granted or refused immediately anyway.

**Why an agent rather than a rule:** the true state is hidden and the evidence is weak and
partial, so the honest object is a belief over the five possibilities, not a label. The
agent keeps a probability for each, updates it with Bayes rule as evidence arrives, and
picks the action with the lowest expected cost.

**Why an explicit probability model rather than an LLM:** every prior, likelihood and
threshold is written down, so any decision can be replayed and any individual number can be
argued with. I did not choose this for elegance. A practitioner asked for it unprompted.
He wanted the refund decision to come from "a rule I can audit and replay rather than
something generative."

**The result I did not expect:** I priced human review as a purchasable piece of
information, and over 50 simulated cases the value of information exceeded its cost exactly
once. Even on the most genuinely ambiguous case in the set, where no explanation rose above 
38% probability, the model declined to buy the review. My instincts said to escalate. The
arithmetic said escalating buys nothing worth paying for.

**The design change that came from a public discussion:** an iOS developer explained why,
and he got there without any of my machinery. On whether to delay a high-risk purchase for
a few minutes: *"Those few minutes buy you no new information, since Apple won't confirm a
stolen card for days, so you're paying UX cost for nothing."* That is a value-of-information
argument in plain English, and it exposed a real defect. My simulation treated a human
reviewer as someone who could resolve the hidden state, when nobody can resolve it at
purchase time. Two routes, one answer, and mine was the one with the wrong assumption
inside it. Sixteen numbered design changes came out of these conversations; that one cost
me the most and taught me the most.

**The largest limitation, stated plainly:** the simulation generates cases from the same
likelihood tables the agent uses to score them, so the model is correct by construction and
the results demonstrate arithmetic, not detection power. Every number is an explicit
assumption. I found no public base rates for in-app purchase fraud. The paper is a study
of a decision procedure, not evidence about how much fraud is out there.

**What I would like comments on:** the agent decides at purchase time, but three separate
practitioners told me the real leverage is elsewhere — grant instantly, then revoke fast, or
answer the platform's refund query well. If you ship IAP: **is there any purchase you would
actually hold, and what signal would make you hold it?** I have one candidate left standing
after the cost analysis which is a high-value consumable, where nothing can be clawed back and I
would like to know whether that corner is real or an artifact of my assumptions.

Preprint attached. Code, data, the discussion record and the review record are in the repo:
https://github.com/SL-A-SH/AI-Native-Cohort

*(Course preprint, IJCAI format. Not submitted to or accepted by IJCAI.)*

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
