# research-file.md: NPC Tactical Belief Agent

Prepared per assignment §4. Nothing in this file is a finished citation: entries marked
**[UNVERIFIED]** must be checked by me before they are used, and papers must be read before
they are cited. The mistake that cost the most on my previous agent was accepting a
plausible AI claim without checking it, so this file separates *what I know* from *what I
have been told*.

---

## Problem statement

**The agent observes a stream of noisy perception events (a partial sighting, a footstep, a
door state change, a gunshot bearing, a teammate's report) and the time elapsed since each.
It must search, hold position, flank, call reinforcements, or retreat because the player's
actual position and intent are not known.**

Two features distinguish this from a static classification problem, and both are
load-bearing:

1. **The hidden state moves.** The player changes position between observations, so the
   belief must be *predicted forward* (diffused along walkable space) as well as updated on
   evidence. This is a Bayes filter over a dynamic state, not a one-shot posterior.
2. **The agent's actions change what it will observe next.** Searching a room reveals that
   room. Holding position reveals nothing but preserves cover. Evidence is therefore not
   passively received, and information gathering is an action with a tactical price.

## Project objective

Design, build and test a minimal NPC tactical agent that maintains a probability
distribution over the player's location on a discrete grid, updates it from noisy perception
events and a movement model, and selects a tactical action by expected cost, where the cost
function encodes **believability rather than accuracy**. Publish the result as an
IJCAI-style preprint within one week.

**The objective is deliberately not "find the player efficiently".** A perfect posterior
produces an NPC that walks straight to the player every time, which is a defect, not a
feature. The design question is how to specify a cost function whose optimum is a
*plausible-looking* opponent that loses at an acceptable rate. That framing is the intended
contribution, and it is the part most likely to interest both game developers and a
decision-theory audience.

## Scope decisions (settled)

Four decisions were taken before any code, and all four share one reason: **the follow-on
brief asks for an expansion of this agent, so complexity spent now is expansion budget
destroyed.** Each decision therefore names what it defers.

| Decision | Choice | Why | Deferred to the expansion |
|---|---|---|---|
| Is intent a second hidden variable? | **No.** Position only. | A joint position-and-intent state multiplies the state space, and every added cell needs an invented likelihood. The previous agent was criticised heavily for the number of assumed parameters; doubling them weakens the paper. | Promote intent to a proper hidden variable with its own transition model. |
| Then how do the tactical actions get any notion of intent? | **A derived proxy, not a variable.** Read intent from the movement of the belief itself: mass drifting toward the agent reads as advancing, away as disengaging, static and concentrated as camping. | It is a function of the filter's existing output, so it costs no new parameters and nothing that has to be defended as an assumption. It still gives the action rule something real to act on. | Replace the proxy with an inferred variable and measure whether inference beats the heuristic. |
| One NPC or a squad? | **One NPC**, with the belief built as an object that has a single owner but does not assume a single reader. `call reinforcements` is the action that would spawn a second subscriber. | A single agent gives a testable experiment inside the week. Squad coordination is a substantial piece of work, and it is the most natural expansion available. | Shared versus private beliefs, and whether disagreement between squad members is what makes a squad look alive. |
| Toy grid or imported level? | **Hand-made toy grid.** | Reproducible, legible as a figure, and no reviewer can claim the result is an artifact of one level's geometry. An imported level is a better screenshot and a worse experiment. | A real level, once the result on the toy grid is established. |

**What this protects:** the believability objective is the contribution, it needs most of
the experiment budget, and it is where review pressure will land. A single NPC on a small
grid with a position-only filter and a carefully argued cost function is a stronger paper
than a squad with intent modelling and a hand-waved evaluation.

## Technical terms

- **Bayes filter / recursive state estimation.** Alternate *predict* (apply a motion model)
  and *update* (apply an observation) to maintain a belief over a changing state. The
  general form of what this agent does.
- **Occupancy grid.** The world discretised into cells, each holding a probability that the
  target is there. The belief representation this project starts with.
- **Particle filter (sequential Monte Carlo).** Represent the belief as weighted samples
  rather than a full grid. The alternative representation; cheaper in large maps, and the
  usual choice when the state space is continuous.
- **Motion / transition model.** How probability mass spreads between observations.
  Diffusion restricted by walls and reachability, not a Gaussian blur.
- **Observation model / sensor likelihood.** `P(observation | player at cell)`. Must include
  false positives (a noise that was not the player) and false negatives (looked and saw
  nothing, which is *evidence*, not absence of evidence).
- **Negative information.** Searching a cell and finding nothing must reduce that cell's
  probability rather than leave it unchanged. Frequently omitted, and the source of NPCs
  that re-check the same room.
- **Information gain / expected entropy reduction.** Ranking which cell to search next.
- **Value of information.** Whether a search is worth its tactical exposure.
- **Coordinated multi-agent belief.** Whether squad members share one belief or hold private
  ones. Deferred, but the belief object is built so it can be added.
- **Believability / perceived intelligence.** The actual objective. An NPC is judged by
  observers, not by estimation error.
- **Artificial stupidity.** The deliberate degradation of an optimal policy to produce fair
  play: reaction delays, aim error, deliberately imperfect search. [UNVERIFIED as a standard
  term of art. Check whether the literature uses it, or whether it is folklore.]
- **Rubber-banding / dynamic difficulty adjustment.** The adjacent design mechanism, worth
  distinguishing from belief degradation.
- **POMDP (partially observable Markov decision process).** The formal frame: act on a
  belief over hidden state, where actions affect both the world and future observations.

## Useful search queries

- "occupancy grid player position game AI"
- "Bayesian belief NPC search behaviour"
- "particle filter target tracking game AI"
- "negative information search theory" / "searching where you have not looked"
- "probabilistic search theory coast guard" (the mature applied literature is search and
  rescue, not games)
- "game AI perception model last known position"
- "believable NPC artificial stupidity"
- "perceived intelligence game AI evaluation"
- "dynamic difficulty adjustment player experience"
- "squad AI coordination shared blackboard"
- "influence map tactical position evaluation"
- "entropy based search planning"
- "hidden information asymmetry game AI fairness"
- "playtesting metrics believability NPC"

## Reddit communities: candidates, all [UNVERIFIED]

Rule to repeat: **check each one is active, read its rules, and record the date and why it
is relevant before posting.** On my previous agent, two cold posts drew zero replies, most
likely because they were posted into communities I had not sized up first.

| Community | Why it may be relevant | Verified? |
|---|---|---|
| r/gamedev | Largest general developer community; shipped-game war stories | No |
| r/GameAI or r/gameai | Nominally the exact topic. Alive: posted 2026-09-03, three replies, one verified citation lead and one new baseline. See `discussion-record.md` R1 to R3 | **Yes** |
| r/Unity3D | Practitioners who have implemented NPC perception | No |
| r/unrealengine | Same, plus UE's built-in perception system as a concrete reference point | No |
| r/roguelikedev | Unusually thoughtful about deterministic AI, FOV and known-versus-unknown state. Alive: posted 2026-09-03, three replies, the most useful of the week. Corrected one of my claims and supplied a missing action. See `discussion-record.md` R4 to R6 | **Yes** |
| r/truegaming | Players rather than developers, which covers the "users and critics" quota. The believability question belongs to them | No |
| r/MachineLearning or r/learnmachinelearning | Bayes filter correctness | No |
| r/statistics or r/AskStatistics | Evaluation under a non-accuracy objective | No |
| r/robotics | Occupancy grids as practised where they originated | No |

**Deliberate change of tactic:** the strongest replies I have had came from *one* thread
where I answered fast and asked a narrow follow-up. Post fewer, follow up harder, and
comment inside existing active threads rather than only starting new ones.

## Relevant X accounts

**TODO: 15 to 25 accounts.**

A specific AI error is already on record here. An assistant suggested POMDP academics whose
accounts turned out to be dormant or non-existent. **Do not accept a handle that has not
been opened and checked for recent, relevant posts.**

Better-shaped search: game AI practitioners and conference speakers rather than probability
academics. Candidate seeds to check by hand are GDC AI Summit speakers, the Game AI Pro
contributor list, the `#gameai` tag, and AI programmers at studios shipping stealth or
shooter AI. Follow researchers, engineers, users and critics, per §6.

## Five useful papers, articles, repositories or datasets

**None of these have been read yet. Read before recording, and read before citing.**

1. **Thrun, Burgard & Fox, *Probabilistic Robotics*.** The canonical treatment of Bayes
   filters, occupancy grids and particle filters. Most likely the methodological backbone.
   [UNVERIFIED that a legitimately free copy exists; check before linking.]
2. **Orkin, "Three States and a Plan: The AI of F.E.A.R."** (GDC 2006). The standard
   reference for the game most often named in this context. See the AI-errors section below:
   this paper is about goal-oriented action planning, **not** probabilistic belief, and that
   distinction is now a feature of the project rather than an inconvenience.
3. **Isla, "Handling Complexity in the Halo 2 AI"** (GDC 2005). Describes a *discrete* belief
   model: fallible perception, remembered target positions, and awareness levels (unaware,
   aware, in visual contact). Directly relevant as the shipped alternative this project is
   measured against.
3b. **"Revisiting the AI of Alien: Isolation"**, gamedeveloper.com. **READ, 2026-09-03.**
   Surfaced by a commenter on r/GameAI. Describes a two-system design where an AI director
   "always knows not only the location of the xenomorph is, but where the player is as well"
   and periodically advises the alien to head towards the player's general area, never telling
   it exactly where the player is. The clearest published case of belief derived
   *subtractively* from ground truth, which is now the paper's motivating contrast. **Do not
   cite it** for a probability distribution in the director, or for negative information in the
   alien: the article addresses neither, and presents the shrinking stalking radius as a
   parameter rather than a response to what has been searched.
4. **Search theory literature (search and rescue / optimal search).** Where "distribute
   effort over a probability map, and account for having looked and found nothing" is a
   mature applied field. Find one canonical text or survey. [Candidate to identify.]
5. **A believability or perceived-intelligence evaluation paper.** Needed because the
   objective is not accuracy and the evaluation must not be accuracy either. [Candidate to
   identify. This is the reference most likely to be missing, and the one that most
   determines whether the experiment is meaningful.]

Also consider, if a slot frees up: Kochenderfer, *Decision Making Under Uncertainty*, and an
open-source particle filter implementation to compare against rather than to copy.

## Questions I want to answer

### Hidden state
- How fast should the belief diffuse per tick, and is diffusion uniform over walkable
  neighbours or weighted by how a fleeing player actually moves?
- At what grid size does a full grid stop being viable, and where is the crossover to
  particles?
- Does the intent proxy (mass drifting toward or away from the agent) actually track what a
  player is doing, or does it mostly track the diffusion model? This is testable and it is
  the assumption the scope decision rests on.

### Evidence
- What are the real sensor events, and what are their false-positive and false-negative
  rates? A footstep heard through a wall is not the same evidence as a muzzle flash seen.
- How is **negative information** handled when the agent searched a room and found nothing?
- How fast should belief decay toward uniform when nothing is observed? Too fast and the NPC
  forgets instantly; too slow and it is omniscient.
- Can the player *manipulate* the evidence with a thrown object or a noise decoy? An
  adversarial observation model is the interesting case and the one players actually
  exploit.

### Actions
- What does each action cost in *player experience* rather than in units of loss?
- Does searching change the belief only through observation, or also by pushing the player?
- Is "call reinforcements" a belief action or a difficulty action?
- ~~What stops the agent oscillating between two equally probable rooms? That failure reads as
  "the AI is broken" faster than any other.~~
  **REFRAMED, 2026-09-04, by a developer on r/roguelikedev.** They animate exactly this on
  purpose: monsters enter a "where did they go?" state, "look around and pace back and forth to
  show their consternation", then return to patrol. Same motion, read as confusion rather than
  as a bug. So the oscillation is not the failure. The failure is oscillation that is
  **unbounded** (nothing ends it) and **illegible** (nothing tells the player it is indecision).
  The question becomes: what bounds it, and what makes the indecision visible? The
  `resume_post` action is the bound. See `discussion-record.md` R7.

### Errors
- **Which error is worse: the NPC that finds the player too easily, or the one that walks
  past them?** This is the inverted-objective question, and the whole project turns on it.
- Perfect play is a defect. How is that written into a cost function without simply capping
  accuracy?
- Which errors are observable? A player never reports "the AI searched suboptimally". They
  report "that felt cheap" or "that felt stupid".
- How do I evaluate believability without a user study I do not have time to run? A stated,
  honest proxy is acceptable. Pretending it measures believability is not.

## AI prompts and important AI errors

### Prompts used so far

1. Recommend candidate agent problems suited to my background (MSc Computer Game
   Engineering, React Native engineer).
2. Give the technical terms for a probabilistic NPC belief agent.
3. Give useful search queries.
4. Find 5 to 10 relevant Reddit communities, and say why each is relevant.
5. Find relevant researchers and engineers on X.
6. Give questions about hidden states, evidence, actions and errors.
7. Identify each claim that needs a source or a test.
8. Tell me which parts of the problem are not clear.
9. Given a seven-day clock and a follow-on brief that asks for an expansion, recommend how
   to scope the hidden state, the agent count and the map.

### AI errors caught

**Error 1: an overstated claim about shipped games, caught before it entered the work.**
The problem recommendation asserted that *"F.E.A.R. and Halo both shipped versions of
this"*, referring to a probabilistic occupancy grid or particle filter. Checking the primary
sources does not support that:

- F.E.A.R.'s published AI work is goal-oriented action planning and squad coordination
  (Orkin, GDC 2006). It is famous for planning, not for probabilistic belief.
- Halo 2's published AI work describes a *discrete* belief model, with remembered target
  position and orientation plus awareness levels, explicitly noting that "the actor can
  believe things that are not true". That is fallible discrete belief, not a probability
  distribution over space.

**Why this matters, and why it improves the project:** the honest version of the claim is
that shipped game AI represents belief *discretely* (a last-known position and an awareness
level), while the probabilistic machinery lives in robotics and search theory. The gap
between those two worlds is a better motivation for the paper than "F.E.A.R. did this
already". Cited correctly it becomes the related-work section. Cited as given it would have
been a factual error in a published preprint.

**Error 2, recorded on my previous agent and listed here so it is not repeated.** An
assistant proposed X accounts for POMDP academics; verification showed the accounts were
dormant or did not exist. Rule: open every handle before recording it.

### Claims in this file that still need a source or a test

- That a perfect posterior produces an unplayable opponent. Plausible, widely repeated, and
  currently unsourced. Needs either a citation or an experiment.
- ~~That negative information is commonly omitted in game AI search behaviour.~~
  **WRONG, corrected 2026-09-03 by a practitioner on r/roguelikedev.** They described a
  shipped give-up rule in their own words: "if that room has two doors and i exit, it will
  give up after it sees the room empty." That is negative information, implemented. The
  corrected claim, which still needs a source of its own, is that negative information is
  usually applied at **room scope as a terminating give-up rule**, whereas this project
  applies it at **cell scope as a likelihood** that redistributes mass to the cells still
  live. The difference to argue in the paper is not presence versus absence, it is what
  happens to the belief that was there. See `discussion-record.md` R5.
- That the belief-drift intent proxy tracks player intent rather than the diffusion model.
  This is the assumption the scope decision rests on, and it is testable in the simulation.
- That squad members holding disagreeing beliefs looks more alive than a shared belief.
  Currently a hypothesis, and deferred to the expansion.
- The false-positive and false-negative rates of every sensor event. These will be
  [ASSUMED] and must be labelled as such everywhere.

### Parts of the problem that are not yet clear

- **What "believable" is measured against, given no user study.** ~~The single largest open
  question.~~ **PARTLY SETTLED, 2026-09-10.** The proxy is chosen, implemented and measured;
  what remains open is narrower and better understood.

  The original candidates were capture rate held inside a target band, time-to-detection
  distribution, path predictability, and how often the agent re-searches a cleared cell. All
  four are still worth logging, but none is the primary measure, because each is a statement
  about outcomes rather than about whether the agent's behaviour is *readable*.

  **The chosen proxy is the evidence ratio** (`BeliefGrid.evidence_ratio`, `src/belief.py`),
  which came out of R6 and is the computable half of that answer. A second belief runs in
  parallel receiving no positive observations, only diffusion and the results of failed
  searches. The ratio between the two at a given cell says how much of the agent's confidence
  there is owed to something the player actually did. Above 1 is attributable; at or below 1
  means the agent would have drifted to that belief regardless, so an action taken on it cannot
  be traced to any event.

  Measured behaviour, from a run on 2026-09-10: 1.00 before any observation, 11.05 the moment a
  sound is heard, 4.41 after 33 ticks of diffusion, and after a failed search 157 of 201 cells
  sit below 1.0 with a median of 0.64. So it discriminates, it decays with staleness the way it
  should, and it is cheap to compute.

  **It must be argued for and labelled a proxy everywhere, and there are two specific reasons
  it is not a measurement of believability.** First, a design choice: the parallel belief does
  receive failed searches, so the ratio isolates what the *player* caused rather than what the
  agent did to itself, and a different choice would give a different measure. Second, and
  larger, R9: shipped practice includes choreography, where ground truth is spent deliberately
  on staging a moment. Nothing in a belief-based cost function can represent that, so the ratio
  captures at most the *fairness* half of believability and none of the *drama* half. Both
  points go in Limitations.
- How many episodes count as a "case" for the 30 to 50 requirement: one episode per case, or
  one map-and-start-position configuration per case with several episodes each.
