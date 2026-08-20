# X thread

**Attach `paper/preprint.pdf` to post 1.** Assignment §14 requires the thread to carry the
problem, the test, the result, and an open question, and requires that a reader gets the
substance without opening the PDF.

Nine posts. Each is under 280 characters.

---

**1/**
An in-app purchase clears the store. You still can't see who was behind it: a real
customer, a stolen card, a hijacked account, a kid on a parent's phone, or someone who'll
consume it and demand a refund.

You have to grant or refuse the entitlement anyway.

Preprint 👇

---

**2/**
So don't pick a label. Keep a belief.

Five hidden states, one probability each, summing to 1. Bayes update as evidence arrives.
Then pick the action with the lowest expected cost: grant, challenge, send to a human,
refuse.

Every prior and threshold written down and replayable.

---

**3/**
The test: 50 simulated cases, true state hidden from every policy, scored afterwards.

Four policies, including a baseline taken from practice — grant everything, ban on refund.

Metrics beyond accuracy: cost, FP/FN, effective FN, review rate, calibration.

---

**4/**
Result: deriving the thresholds from the cost matrix beat the bands I'd tuned by eye.

88.72 vs 92.88 [assumed] units, fewer false positives, same hostile detection.

Over 300 seeds it beats the reactive baseline 99.3% of the time — but is cheapest overall
only 61%.

---

**5/**
The number that mattered more than the ranking: the loss ratio is ~1.75:1.

Payment fraud has asymmetries that justify blocking at <1% suspicion. Low-value digital
goods don't.

Derived refusal boundary: 71.5% belief. You should almost never block.

---

**6/**
I priced human review as information you buy.

Over 50 cases, its value beat its cost once.

On the most ambiguous case in the set — nothing above 38% — the model refused to buy the
review. My instinct said escalate. The arithmetic said escalating buys nothing.

---

**7/**
Then an iOS dev on Reddit said the same thing without any of the machinery:

"Those few minutes buy you no new information, since Apple won't confirm a stolen card for
days, so you're paying UX cost for nothing."

Value of information, in plain English.

---

**8/**
And he was more right than my model.

I'd priced human review as something that *resolves* the hidden state. At purchase time
nothing does — the answer arrives days later in a chargeback.

16 numbered design changes came from these threads. That one hurt.

---

**9/**
Limitation: cases come from the same tables the agent scores with, so the model is right by
construction. It shows arithmetic, not detection.

Open question for anyone shipping IAP: **is there any purchase you'd actually hold?**

github.com/SL-A-SH/AI-Native-Cohort

---

## Notes

* Post 1 carries the PDF. Post 9 carries the repo link.
* Course preprint in IJCAI format — do not describe it as submitted to or accepted by
  IJCAI anywhere in the thread or the replies.
* If someone answers the question in post 9, that reply is a discussion for
  `discussion-record.md` — log it with whatever design change it causes, or "no change,
  with the reason".
