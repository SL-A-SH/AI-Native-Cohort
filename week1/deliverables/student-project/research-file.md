# Problem Statement

The agent observes a validated in-app purchase receipt together with the buying account's history, device signals, and current session behaviour. It must approve, question, stop, or examine the entitlement grant because the purchase's true nature is not known.

Hidden states: legitimate purchase; stolen card on the store account; account takeover; friendly fraud (family member with real access); refund abuse.

# Project Objective

Design, build, and test a minimal entitlement-decision agent that maintains a belief over five hidden states and selects the lowest-expected-cost action then publish the results as an IJCAI-style preprint within one week.

## Technical terms

- POMDP (Partially Observable Markov Decision Process) — the standard formal model for "act when you can't see the true state." It means choosing actions based on often imperfect observations, with unknown outcomes.

- Belief state — the agent's probability distribution over hidden states, updated as evidence arrives (via Bayesian updating / Bayes filters).

- Value of information (VoI) — the expected benefit of gathering more evidence before acting. This is exactly your "examine" action.

- Optimal stopping — when to stop gathering evidence and commit to a decision.

- Cost-sensitive classification — false positives (blocking a good customer) and false negatives (letting fraud through) have different costs.

- Selective prediction / classification with a reject option — a model that's allowed to say "I'm not sure, escalate" instead of forcing approve/deny.

- Human-in-the-loop (HITL) / escalation policy — your "question" and "examine" actions route to a human or a step-up check.

- Step-up authentication / friction — the fraud-industry term for "question the user" (e.g., OTP, 3DS challenge).

- Related: anomaly detection, class imbalance, concept drift (fraud patterns change over time), expected utility / decision theory, calibration (do your probabilities mean what they say?), sequential hypothesis testing (SPRT) — a classic method for approve/reject/keep-collecting-evidence.

## Useful Search Queries

- "POMDP fraud detection"
- "transaction fraud detection reject option classification"
- "cost-sensitive learning class imbalance fraud"
- "value of information decision agent"
- "sequential probability ratio test transaction approval"
- "step-up authentication risk-based decisioning"
- "selective prediction abstention machine learning"
- "human-in-the-loop fraud review escalation"
- "belief state Bayesian filtering tutorial"
- "concept drift credit card fraud"
- "calibration classifier probability fraud"
- "LLM agent tool use decision making uncertainty"
- "in-app purchase fraud detection" 
- "app store receipt validation server side" 
- "IAP refund abuse games" 
- "chargeback digital goods entitlement", "friendly fraud in-app purchases"

## Reddit communities

- r/learnmachinelearning
- r/datascience
- r/MLQuestions
- r/statistics
- r/gamedev
- r/iosProgramming
- r/androiddev
- r/reactnative

## Relevant X accounts

NIL

## 5 Useful papers, articles, repositories or datasets

1. Kochenderfer's Decision Making Under Uncertainty
2. The Kaggle ULB credit card fraud dataset
3. Wald's SPRT

## Questions that you want to answer

### Hidden state

What exactly is hidden? Just fraud/legit, or richer states (stolen card, account takeover, friendly fraud, merchant error)?
Does the hidden state change during the episode, or is it fixed per transaction?
Is one transaction independent, or does the hidden state live at the account/user level across transactions?

### Evidence

What observations arrive for free (amount, merchant, time, device) vs at a cost (calling the bank, asking the user)?
How noisy is each evidence source? Can a fraudster manipulate it (adversarial evidence)?
How do I combine evidence into a belief — Bayesian update, a trained classifier score, or an LLM's judgment?
Does evidence go stale? (A device check from 5 minutes ago vs 5 days ago.)

### Actions

What does "question" cost? (Customer friction, abandonment rate.) What does "examine" cost? (Analyst time, delay.)
Are actions reversible? Can I approve now and claw back later?
Is "stop" terminal, or can the user retry?
Can "examine" loop forever? What's my stopping rule?

### Errors

What's the cost ratio of a false positive vs a false negative, in currency?
Which errors are observable? (You learn about fraud you approved via chargeback, weeks later; you may never learn a blocked transaction was legitimate — this is selective label bias.)
How do I measure error rates when ground truth is delayed and partial?
What happens to error rates when fraud patterns drift?

## AI prompts and important AI Errors

1. Give me the technical terms for this problem.
2. Give me useful search queries.
3. Find 5 to 10 relevant Reddit communities.
4. Tell me why each community is relevant.
 
5. Find relevant researchers and engineers on X.
- AI suggested POMDP academics as X follows, but verification showed the field's key academics are inactive or absent on X

6. Give me questions about hidden states, evidence, actions, and errors.
7. Identify each claim that needs a source or a test.
8. Tell me which parts of my problem are not clear.