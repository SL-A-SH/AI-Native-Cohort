# discussion-record.md: NPC Tactical Belief Agent

Per assignment §7. Every useful human answer records **exactly one** outcome: a new
assumption, a new failure condition, a new test, a change to the agent, a change to the
probability model, or no change with the reason.

**Status, 2026-09-04:** 2 of 5 required communities engaged. 2 of 10 required contributions
posted as opening posts, plus 2 replies posted inside r/roguelikedev, so 4 contributions total.
2 of 5 required discussions with two or more replies **complete**: r/GameAI drew three replies,
r/roguelikedev drew five.

**Outstanding before this file is submittable:**

- Thread URLs. Every `[LINK: fill in]` below must be replaced with the real permalink. A link
  without an explanation does not complete the task, and an explanation without a link does
  not either.
- Three more communities, at two contributions each. Drafts are in
  `social/reddit-questions.md`: r/gamedev A, r/truegaming A, r/unrealengine A are the next
  three.
- Six of the eight replies in `social/reddit-replies.md` are **still unposted** (both roguelikedev
  R4 and R5 replies have gone out). Post the rest, then update the "My next answer" column with
  what actually went out and anything it drew.

---

## Summary table

| # | Platform | Community | Link | My first contribution | Human answer (compressed) | My next answer | Outcome recorded |
|---|---|---|---|---|---|---|---|
| R1 | Reddit | r/GameAI | [LINK: fill in] | Post: do any shipped games keep a probability distribution over player position, or is it always last-known-position plus awareness levels? | Pointed me at the Alien: Isolation writeup as adjacent territory | Read it, reported what it actually says, asked whether the shrinking search radius is tied to where the alien has already been | **New assumption:** shipped game AI derives belief *subtractively* from ground truth, not additively from evidence |
| R2 | Reddit | r/GameAI | [LINK: fill in] | as R1 | Game AI has no sensor noise by default because the engine knows the pixel-level position; what ships is artificial reduction of accuracy, via vision cones and sensor malfunction | Named the subtractive/additive split, adopted his description as a second baseline, asked whether a failed search leaves any trace | **New test:** second baseline, "degraded omniscience" |
| R3 | Reddit | r/GameAI | [LINK: fill in] | as R1 | Computationally suicidal to solve for noise; you have a clean room, optimise for the clean room; simulate the concept instead; realism is rarely healthy for a game | Agreed on realism from the other direction, conceded the cost point, asked which cheap approximation he considers the fair one to beat | **New test:** indistinguishability against a cheap scripted approximation |
| R4 | Reddit | r/roguelikedev | [LINK: fill in] | Post: how long should a monster remember, and what does it do while it forgets? Uniform diffusion or weighted toward exits? | Send the monster to points of interest (exits, heal sources), or to nearby sound sources | Mapped it onto a weighted transition model, made it a switchable policy, asked whether POI weighting ever read as *too* smart | **Change to the probability model:** POI-weighted diffusion as policy B |
| R5 | Reddit | r/roguelikedev | [LINK: fill in] | as R4 | Last tile seen plus timestamp, then follow a heat trail (Brogue's scent); timers configurable per NPC type; hunter follows longer, grunt gives up; "if that room has two doors and I exit, it will give up after it sees the room empty" | Retracted my claim, proposed that a scent trail is a lossy form of the same belief field, asked whether the give-up rule uses the room's exits | **New assumption** (replacing a wrong one): negative information *is* implemented in practice, but scoped to a room as a give-up rule, not to a cell as a likelihood |
| R6 | Reddit | r/roguelikedev | [LINK: fill in] | as R4 | Unerring chase is annoying; omniscience is fine for a psionic because it is thematically justified; tiered behaviour: engage on sight, move relative to last known on losing LOS, and on fully losing, warn others, call reinforcements, or return to previous tasks | Named legibility as the real axis, admitted my action set has no way to stop, asked what makes giving up read as reasonable rather than broken | **Change to the agent:** add a sixth action, `resume_post` |
| R7 | Reddit | r/roguelikedev | [LINK: fill in] | as R4 | Challenged whether a probability field is what a person actually does; a hot/cold scent trail is enough; you need a hierarchy of senses (sight beats a noise next door) and a trail can be broken by water; monsters enter a "where did they go?" state where they look around and **pace back and forth to show their consternation**, then return to patrol | Agreed the grid is bookkeeping not cognition, mapped the sense hierarchy onto likelihood weights, said the pacing had caught me out, asked how long the state runs before it reads as a bug | **New failure condition, redefined:** the failure is not oscillation, it is *unbounded and illegible* oscillation |
| R8 | Reddit | r/roguelikedev | [LINK: fill in] | as R4 | On losing LOS, record the last position **and a vector towards it**; move to the position, then continue along the vector; after a few turns return to idle or patrol, and back to the patrol zone if it has one | Pointed out the vector is a one-step prediction where everyone else stores only a point, asked whether it updates after LOS breaks and whether it bends round corners | **New assumption:** shipped memory models form a ladder of increasingly lossy approximations to a Bayes filter |

---

## R1. r/GameAI, commenter 1: the Alien: Isolation lead

**Link:** [LINK: fill in]

**What they said.** That they would need to double check what other games have done with
probability fields, but that I was getting close to this territory, with a link to the Game
Developer article revisiting the AI of Alien: Isolation.

**What I did with it.** Read the article before using it, per the rule that cost me on the
previous agent. It says the game runs two systems. The second is an AI director, in the
Left 4 Dead sense, that manages pacing. In the article's own words, the director "always
knows not only the location of the xenomorph is, but where the player is as well", and its
job is "to periodically tell the xenomorph to head towards the player throughout a given
level. It never tells the alien exactly where you are, but advises it to head towards your
general, and the behaviour tree will take it from there."

**What the article does not say.** It does not describe the director's model as a
distribution or attach any notion of confidence to it, and it does not describe how the alien
represents last known position internally. It notes that the stalking search radius "shrinks"
on subsequent passes, but presents that as a parameter rather than as a response to having
searched and found nothing. Do not cite it for either of those points.

**Why this matters more than the correction it replaced.** I expected this lead to be a
counterexample to my claim that nobody ships a probabilistic belief. It is not. It is a third
data point for the claim, and it sharpens it: the director holds ground truth and
deliberately degrades it on the way out. Knowledge starts complete and is *coarsened*. My
agent never holds ground truth at all.

**Outcome, exactly one: new assumption.**

> **[A-07]** Shipped game AI derives NPC belief subtractively, by degrading engine-side ground
> truth, rather than additively, by accumulating evidence from noisy observations. Sources so
> far: Isla on Halo 2 (published), the Alien: Isolation director (published, verified above),
> and one practitioner report (R2 below). This is now the paper's motivation, and it is
> falsifiable: one counterexample kills it.

**Still to verify:** whether the shrinking search radius is a fixed parameter or is tied to
swept areas. The article does not resolve it, and the two are indistinguishable to a player
while being completely different systems. Do not claim either way until a primary source
settles it.

---

## R2. r/GameAI, commenter 2: subtractive belief, and where my baseline came from

**Link:** [LINK: fill in]

**What they said.** That in-game AI has no sensor noise by default because the engine knows
the pixel-level position, and that what is actually implemented is an artificial reduction of
accuracy: vision cones, and sensor malfunction from low energy, to make the NPC more
realistic.

**Why this is the most useful answer in the set.** It names the industry pattern in one
sentence, independently of R1, and it hands me a baseline I did not have. My test plan had
one baseline, a Halo-2-style last-known-position model. That baseline is weak, because it is
obviously less informed than my agent, so beating it proves very little. Degraded omniscience
is the opposite: it is *better* informed than my agent, and it is what practitioners actually
ship. If the filter cannot beat it, the filter is not earning its place, and that is the
result I should report.

**Outcome, exactly one: new test.**

> **[T-02] Baseline B, "degraded omniscience".** The NPC receives the player's true cell, gated
> by a line-of-sight and vision-cone check, with a reaction delay and a positional error term.
> No belief, no memory beyond the last gated reading. Every parameter `[ASSUMED]` and labelled.
> Runs on the same 30 to 50 cases, the same seeds, and the same believability proxy as the
> agent and as Baseline A.

**Note against lesson 6 in the handover:** report this comparison over the seed sweep, never
from a single seed. A single-seed win here would be exactly the shape of the failure that
already cost me once.

---

## R3. r/GameAI, commenter 3: the cost objection

**Link:** [LINK: fill in]

**What they said.** That it is computationally suicidal to solve for noise the way the real
world forces you to, that I already have a clean room and should optimise for the clean room,
that I should simulate whatever concept I am trying to make feel realistic rather than
importing the real-world solve, and that my assumption that realism is healthy for a game is
usually wrong.

**The half that misreads the project.** The premise is that realism is *not* the objective. An
accurate estimator is the failure case here, because an NPC that solves for the player's
position walks straight to them. We agree on the conclusion and arrived from opposite
directions.

**The half that lands, and that I had not planned to test.** Whether the machinery earns its
place. Compute is not the issue at this size: the grid is 24 by 14, so 336 cells and a few
array operations per tick. The real objection is that a cheap scripted approximation might
produce behaviour that is indistinguishable on screen. If that is true, the honest paper says
so.

**Outcome, exactly one: new test.**

> **[T-03] Indistinguishability.** Compare the belief agent against a cheap scripted
> approximation (candidate: move to last known position, then sweep adjacent rooms on a timer,
> with a give-up counter) on the believability proxy and on the observable behaviour traces,
> not on estimation error. A null result is a publishable finding and belongs in the paper
> whichever way it goes.

**Scope explicitly did not change, and the reason:** the grid stays. At 336 cells the cost
argument does not bind, and the grid is what makes the figure legible. The crossover to
particles remains an honest limitation to state, not a change to make now.

---

## R4. r/roguelikedev, commenter 1: points of interest as a prior

**Link:** [LINK: fill in]

**What they said.** Make the monster visit points of interest, such as possible exits or heal
sources, so it feels like it is hunting the player. Alternatively, have it randomly visit
nearby sources of sound, such as other NPCs or the sounds of battle.

**Translation into my model.** This is my own open question answered from the design side.
My transition model currently spreads probability uniformly into every walkable neighbour.
What they are describing is the same spread weighted toward exits and points of interest, so
mass accumulates where a player would plausibly go instead of smearing evenly. It costs no
new free parameters beyond the weighting itself, and it is a strictly better prior than
uniform if players do in fact head for exits.

**Outcome, exactly one: change to the probability model.**

> **[M-03] POI-weighted diffusion.** The transition model gains a weighting toward exits and
> points of interest, switchable at run time. Uniform diffusion becomes **policy A**,
> POI-weighted becomes **policy B**. This satisfies the §9 requirement for at least two
> policies with a difference that is motivated rather than invented, and it makes my own
> open question ("uniform or weighted?") into a measured result instead of a decision.

**Open, and asked in my reply:** whether POI weighting makes the agent read as *too* smart.
Uniform spreading looks like searching. Weighting toward exits looks like prediction, and
prediction is the thing players call cheap.

---

## R5. r/roguelikedev, commenter 2: the answer that corrected me

**Link:** [LINK: fill in]

**What they said.** A sensor-based approach: on line of sight, mark the last tile seen and
the timestamp. On losing line of sight, path to that last known tile and then follow a heat
trail, which they believe Brogue implements as a scent trail. Detection quality varies by NPC
type, all timers configurable, so hunters follow the trail longer and grunts give up and
wander. After giving up, NPCs return to their assigned post, patrol, or job. And the closing
detail: "if that room has two doors and i exit, it will give up after it sees the room empty."

**The claim of mine this kills.** `research-file.md` records, under claims needing a source,
that negative information is commonly omitted in game AI search behaviour. That last sentence
is negative information, working, in a real project. The claim as written is wrong and I am
rewriting it rather than defending it.

**Outcome, exactly one: new assumption, replacing a wrong one.**

> **[A-08]** Negative information is implemented in practice, but at **room scope, as a
> terminating give-up rule**: the room is seen to be empty, so the pursuit ends. My model
> applies it at **cell scope, as a likelihood**: the cell is seen to be empty, so its mass
> drops by the miss rate and is redistributed to the cells that are still live. The claim in
> `research-file.md` must be edited to say this, and the difference to be argued in the paper
> is not presence versus absence, it is **what happens to the belief that was there**.

**A framing I owe them, and asked about in my reply.** A scent trail and my probability field
may be closer than they look. A decaying trail is a record of where the target has been,
which is much of what my grid holds. The difference is that my grid also pushes mass forward
into where the target could be *now*, including cells the target never touched. If that
framing survives contact, it is a clean sentence for the related work section: the shipped
technique is a lossy special case of the filter, missing the predict step.

**Also confirmed:** their per-NPC configurable timers are my decay rate. Their hunter and
their grunt are two settings of the same parameter. This supports, without proving, my
position that the decay rate is where NPC personality actually lives.

---

## R6. r/roguelikedev, commenter 3: legibility, and the action I was missing

**Link:** [LINK: fill in]

**What they said.** As a player, they find it annoying when enemies chase unerringly. It is
acceptable for an enemy that can reasonably track the player, such as a psionic which for all
purposes is omniscient, but it makes little sense for a regular guard to pathfind exactly to
the player after several corners. They then gave a tiered structure: on sight, engage. On
losing line of sight, remember the player exists and move relative to the last known location,
such as flanking round a pillar. On completely losing the player, thematically resume: warn
other enemies, call reinforcements, continue previous tasks, or guard points of interest.

**The reframe, which is the most valuable thing anyone said this week.** The psionic is
allowed to be omniscient because the fiction explains it. The guard is not. So the player's
complaint was never about how much the AI knows. It is about whether the player can
**attribute** the knowledge to something. That moves my open evaluation question off
accuracy entirely: the thing worth measuring is whether an action is traceable to evidence
the player knows the agent could have had.

Logged as a **candidate believability proxy**, new and untested, alongside the four already in
`research-file.md`: the fraction of agent actions whose triggering evidence lies inside the
player's own observable history. It is still a proxy, it is not yet defined precisely enough
to compute, and it must never be described as a measurement of believability.

**The gap in my design.** My action set was search, hold position, flank, call reinforcements,
retreat. Every option in their "completely lost the player" tier is a version of *stopping*,
and I have no such action. Without one the agent has no graceful way to end an episode, which
is almost certainly the mechanism behind the oscillation failure I already expect: with no
terminal action, two near-equal cells leave the policy switching between them forever.

**Outcome, exactly one: change to the agent.**

> **[G-04] Sixth action, `resume_post`.** Disengage and return to a patrol or post. Costs
> nothing in exposure, forfeits the episode, and is the only terminal action available. Its
> cost must be derived from the cost matrix rather than eye-tuned, per handover lesson 5. This
> is also the predicted fix for the oscillation failure condition, so the failure analysis
> should test it with the action removed.

---

## R7. r/roguelikedev, commenter 4: the failure condition I had backwards

**Link:** [LINK: fill in]

**What they said.** Quoted "probability field" and "mass diffuses along walkable cells" back at
me and asked whether that is really what goes through the average person's head when following
someone, since a simple smell trail would be enough: a tile is hot when the player stands on it
and goes cold when left untouched. They have also done sound tracking, but say you need a
hierarchy of senses for it to make sense, since following someone you have seen outranks the
noise in the room next door, another monster might rank a scent trail above everything, and
that trail could be interrupted by passing through a stream of water. On memory, their monsters
enter a "where did they go?" state, which has them look around and **pace back and forth to
show their consternation**, before returning to the patrol route. Doubling down on memory would
mean getting more annoyed on the next lead, staying in the state longer, and trying different
nearby paths to pick the trail back up.

**The reversal, and it is the most useful thing anyone has said this week.**
`research-file.md` lists oscillation between two equally probable rooms as the failure that
"reads as the AI is broken faster than any other". This person deliberately animates that exact
behaviour to communicate uncertainty to the player. Same motion, opposite reading.

The difference is not the oscillation. It is that theirs is **framed** (a named state, a
visible look-around, a deliberate animation) and **bounded** (it ends, and it ends by returning
to patrol). Mine would be neither: silent, unexplained, and with no exit, since I have no
terminal action until [G-04] lands.

**Outcome, exactly one: new failure condition, replacing the one I had.**

> **[F-01, rewritten]** The failure is not that the agent alternates between two near-equal
> cells. It is that the alternation is **unbounded** and **illegible**: nothing caps its
> duration, and nothing on screen tells the player it is indecision rather than a stuck script.
> The measurement changes with it. Do not count oscillation events. Count oscillations that
> exceed a bound, and log the belief state alongside each so the failure analysis can say
> whether the agent was genuinely torn or just broken. This also gives the `resume_post` action
> from [G-04] a job: it is the bound.

**Two other points from this answer, deliberately not recorded as the outcome:**

- **Sense hierarchy: no change needed, and here is the reason.** Their priority ordering is
  already what a likelihood does. "Sight beats a noise through a wall" is not a rule to add, it
  is a large likelihood ratio for a sighting and a small one for a muffled sound. Worth saying
  in the paper, because it shows the shipped heuristic and the probabilistic version agreeing,
  which strengthens rather than weakens the framing.
- **Water breaking a scent trail: deferred, with the reason.** This is evidence being
  *invalidated after the fact*, which nothing in my model can express. Every update I have
  moves belief forward in time; none retracts an earlier observation. It is a genuinely new
  mechanism, it is close cousin to the decoy problem already noted in `research-file.md`, and
  the scope is locked. Both go to the expansion brief as a pair.

---

## R8. r/roguelikedev, commenter 5: the vector, and a ladder for the related work

**Link:** [LINK: fill in]

**What they said.** In simple cases, when a hostile NPC loses line of sight it records the
player's last position **and a vector towards it**. It moves to the recorded position, and if it
still cannot see the player it continues in the direction of the vector. After a few turns it
returns to idle or patrol behaviour, and back to its patrol zone if it has one.

**What is actually new here.** Everyone else in the thread stores a point. This stores a point
plus a direction, which is a one-step prediction rather than pure memory. That is the smallest
possible version of my predict step, and it is the first answer in either thread that has one
at all.

Lining the answers up gives a ladder, and it is a much better related work section than the
one I had:

| What is stored | What it can do | Who described it |
|---|---|---|
| Last known tile plus timestamp | Update only. No forward guess | R5, and Isla's Halo 2 model |
| Last known tile plus a vector | Update, plus one step of prediction | R8 |
| Decaying scent or heat trail | Update, plus decay. A history, not a forecast | R5, R7, Brogue |
| Ground truth behind a vision cone | Perfect state, degraded on read | R2, and the Alien: Isolation director |
| Full occupancy grid | Update, plus prediction over every reachable cell | this project |

**Outcome, exactly one: new assumption.**

> **[A-09]** The memory models shipped in practice are lossy special cases of a Bayes filter,
> not different approaches to it. A last known point is the update step with the predict step
> deleted. A vector is a rank-one predict step. A scent trail is a decaying record of the past
> with no forward push. This is falsifiable and it is the claim the related work section will
> rest on, so it needs checking against a real implementation before it goes in the paper, not
> just against four Reddit comments.

**Also worth noting:** this is the fourth independent description of "then it goes back to
patrol" across the two threads. [G-04] `resume_post` is now the best-evidenced change in the
record.

---

## Reading across all eight

Three things the two threads did that I could not have done alone.

1. **The motivation got stronger and changed direction.** I went in expecting to find that
   games do this probabilistically somewhere. Three independent sources say the opposite, and
   say it consistently: games start from omniscience and remove information. That is a better
   gap than the one I thought I had.
2. **I gained the baseline that can actually beat me.** Degraded omniscience is more informed
   than my agent and is what ships. Comparing against last-known-position alone would have
   been a soft test, and a reviewer would have said so.
3. **One of my written claims was wrong and a practitioner corrected it in passing.** The
   negative-information claim was already flagged as needing a source. It got one, pointing
   the other way.
4. **My worst-named failure turned out to be someone's deliberate feature.** I had oscillation
   between two rooms down as the thing that most obviously reads as broken. R7 animates it on
   purpose to show a monster is confused. The failure condition is rewritten and it is now
   about bounds and legibility, not about the behaviour itself. I would not have found this by
   reading, because nobody writes up the animation they added to make indecision readable.
5. **The related work section arrived as a by-product.** R5, R7 and R8 between them describe
   three different memory models, and R2 and the Alien: Isolation director describe a fourth.
   Lined up ([A-09], R8) they form a ladder of lossy approximations to the same filter, which
   is a far better frame than "here are some things games do".

**What this does not settle.** The believability proxy is still undecided, and it is still the
thing that blocks the experiment design. R6 gave a better *framing* for it, not a computable
measure. The three unposted questions (r/gamedev A on the proxy, r/truegaming A on which error
costs more, r/unrealengine A on perception systems) are all aimed at it, and r/gamedev A should
go up first for that reason.
