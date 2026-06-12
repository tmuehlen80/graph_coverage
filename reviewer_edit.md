# Reviewer Edits — SafeComp Paper

Tracking changes made in response to reviewer comments (no rebuttal possible; changes baked into the paper).

## Reviewer comment 1 — Coverage definition / denominator not formalized

> Coverage definition is not fully formalized. The paper does not clearly define the exact
> coverage denominator, for example whether coverage is computed per scene, per actor node,
> per graph component, or per matched archetype.

### Response strategy (woven into the text, not a rebuttal)

The unit of coverage is the **complete traffic scene**, represented as a whole graph: nodes
carry the parameters describing individual actors, edges carry the parameters describing the
relations between them. A coverage argument asks whether the observed scene graphs cover the
scenes that can occur in reality — assessed against *whole scenes*, not isolated factors or
low-order factor interactions. This is the central advantage of the graph representation and
is now stated up front, before either coverage method is introduced. The two methods then
operationalize it: subgraph isomorphism reports per-archetype occurrence within scene graphs,
the embedding approach places whole scenes in a latent space.

### Edits made

**1. `sections_safecomp/02_introduction.tex`** (early framing)

Added as the second sentence, defining what a coverage argument asks:

> At its core, a coverage argument asks whether the scenes observed so far have exhausted the
> space of traffic scenes that can occur in the real world.

And where the graph framework is introduced, defining the unit of coverage and the graph
advantage:

> The framework takes the *complete traffic scene* as the unit of coverage rather than an
> individual factor: each scene is represented as a single graph whose nodes carry the
> parameters describing the individual actors and whose edges carry the parameters describing
> the relations between them. The key advantage of this representation is that arbitrarily
> complex scenes — with varying numbers of actors and the higher-order interactions between
> them — are captured naturally within one object, instead of being decomposed into isolated
> coverage factors or low-order factor combinations. Coverage is therefore assessed against
> whole scenes: we ask how completely the observed scene graphs cover those that can occur in
> reality.

---

## Reviewer comment 2 — Archetype completeness / safety relevance

> The archetype set is manually defined. The 18 scenario archetypes are intuitive, but
> the paper does not sufficiently justify their completeness or safety relevance. It is
> unclear whether important rare or safety-critical scenarios are missing.

### Response strategy (woven into the text, not a rebuttal)

The 18 archetypes are not a completeness claim — they are a **user-defined input** to the
framework. The methodological contribution is "given a set of archetypes, measure coverage
against them," not "these 18 are complete." The complementary **embedding approach** is the
safeguard against missing rare/safety-critical scenarios: it needs no predefined archetypes
and surfaces real-world scenes that match *no* archetype (candidates for extending the set).
Automatic archetype discovery from those unmatched dense regions is framed as future work —
we do **not** claim to already close that loop.

### Edits made

**1. `sections_safecomp/05_create_subgraphs.tex`** (archetype definition paragraph)

Added after the archetype list, before "The last step in the analysis…":

> These archetypes are drawn from common scenario primitives in the literature
> (lead--follow, cut-in, cut-out, opposite traffic, platoons) and constitute a
> *user-defined input* to the framework rather than a complete or safety-certified
> taxonomy: practitioners can substitute archetypes specific to their operational design
> domain or safety case. Completeness of this particular set is therefore not claimed and
> not required for the method itself. Crucially, the complementary embedding approach
> (Section~\ref{chapter:implementation_of_graph_embeddings_for_traffic_scene_analysis})
> operates *without* predefined archetypes and can surface real-world scenes that match no
> archetype, indicating candidates for extending the set and thus guarding against
> omissions of rare or safety-critical situations.

**2. `sections_safecomp/09_summary.tex`** (future work)

Changed:

> ~~Future work will incorporate temporal information through multi-timestep graph
> structures and investigate automatic archetype extraction from real-world data.~~

To:

> Future work will incorporate temporal information through multi-timestep graph structures
> and close the loop on archetype completeness by automatically extracting new archetype
> candidates from real-world scenes that populate dense embedding regions yet match no
> predefined archetype.

### Open items

- [ ] Verify the paper still fits the 12-page limit after the Section 5 addition
      (~4 extra lines). If it overflows, trim the literature-primitives clause and keep
      only the "user-defined input / completeness not required" + embedding-safeguard
      sentences.

---

## Reviewer comment 3 — Continuous parameter coverage under-specified (binning)

> Continuous parameter coverage is under-specified. The paper shows speed-distribution gaps
> between CARLA and Argoverse, but does not clearly define the binning strategy, density
> thresholds, or formal criteria used to identify parameter-level coverage gaps. Or it doesn't
> need to discretise the continuous parameters at all to calculate coverage?

### Response strategy (woven into the text, not a rebuttal)

Two distinct cases. (a) The **archetype-level** parameter analysis (role-specific speed
distributions) *does* bin — the exact binning, density normalization, and gap criterion are
now stated in the figure caption, taken directly from the plotting code. (b) The **embedding**
analysis needs *no* binning: the full scene graph (continuous node/edge features included) is
embedded as a whole, so gaps appear as sparsely populated regions of the embedding space
rather than empty parameter bins. Separately, we make explicit that the graph properties
(parameters, actor types, relations) are a user-configurable example and the method is
agnostic to them.

### Edits made

**1. `sections_safecomp/08_discussion.tex`** (caption of Fig. `role_speed_comparison`)

Added the formal binning / density / gap criterion (values verified against
`notebooks/shared/subgraph_isomorphism.ipynb`):

> Only actors with speed ≥ 2 m/s are considered, binned into fixed 1 m/s intervals, and the
> histograms are density-normalised. Green rectangles mark identified coverage gaps, defined
> per bin as a normalised Argoverse density ≥ 0.005 together with a CARLA density below 15% of
> it — i.e. Argoverse shows sufficient density while CARLA is nearly empty.

**2. `sections_safecomp/08_discussion.tex`** (embedding coverage-gap paragraph)

Added a short clause clarifying the embedding path needs no discretization:

> No binning is required here, as the full graph is embedded.

**3. `sections_safecomp/04_defining_traffic_scene_graph.tex`** (after the relation-type list)

Made explicit that the graph model is a user-configurable example and the method is agnostic
to it:

> The attributes and relation types defined above are one example instantiation of the graph
> model. The parameters of interest, the actor types, and the actor relations are free to be
> chosen by the user. The coverage methodology is unaffected by these modelling choices.

### Open items

- [ ] Re-check the 12-page limit after this session's additions (intro framing, caption
      expansion, §4 user-configurability note).

---

## Reviewer comment 4 — Static graph / temporal semantics

> The current graph representation is largely static. The actor graph is constructed at a
> specific timestep, which limits its ability to capture temporal semantics. Scenarios such as
> cut-in, cut-out, braking, yielding, and merging are dynamic processes, not merely static
> spatial configurations.

### Response strategy (woven into the text, not a rebuttal)

Partly conceded, partly clarified. (a) The single-timestep graph is not purely static: each
actor node carries a `lane_change` indicator computed by comparison to the previous timestep
(Δt = 1 s), flagging lateral lane changes. This is exactly what lets dynamic maneuvers such as
cut-in and cut-out be represented and matched as archetypes (their templates set
`lane_change = True` on the maneuvering actor). Verified in
`src/graph_creator/ActorGraph.py` (lane is compared to the following-continuation of the
previous timestep's lane; the first graph is dropped for lack of a predecessor). (b) Full
temporal modelling of speed-profile dynamics (braking, yielding) via multi-timestep graphs is
kept in the **outlook** as future work, not claimed here.

### Edits made

**1. `sections_safecomp/04_defining_traffic_scene_graph.tex`** (after the actor-graph attributes) — factual rebuttal only:

> Although each actor graph is built at a single timestep, the lane change indicator already
> encodes a one-step temporal signal: it is set by comparing an actor's lane to its lane one
> timestep earlier (Δt = 1 s), so dynamic maneuvers such as cut-in and cut-out are captured
> even within a single-timestep graph.

**2. `sections_safecomp/09_summary.tex`** (outlook) — time-based-graph future work made explicit about the reviewer's dynamics:

> Future work will incorporate temporal information through multi-timestep graph structures,
> capturing the full dynamics of maneuvers such as cut-in, cut-out, braking, and yielding more
> faithfully than a single-timestep graph, …

(Time-based-graph work lives in the outlook only; §4 keeps just the `lane_change` clarification.)

---

## Reviewer comment 6 — Coverage gaps not directly linked to safety risk

> Coverage gaps are not directly linked to safety risk. The method identifies structural and
> parameter-level differences between CARLA and Argoverse, but it does not show whether these
> gaps correspond to higher failure probability, hazard exposure, or safety-critical system
> behavior.

### Response strategy (woven into the text, not a rebuttal)

Scope clarification, building on the coverage-model definition added for comment 1: a coverage
model is a *completeness* argument, not a safety analysis. Linking gaps to failure probability
or hazard exposure is a separate (downstream) safety analysis — which itself can only be
conducted on top of a completeness argument. The comment is therefore out of scope for the
method, and we state the scope explicitly rather than over-claiming.

### Edits made

**1. `sections_safecomp/02_introduction.tex`** (right after the coverage-argument definition)

> This is a completeness argument, not a safety analysis, though any safety analysis can in
> turn only be carried out on top of such a completeness argument.

---

## Reviewer comment 7 — Missing archetypes → missed hazards / weak safety evidence

> The paper should explain how missing archetypes may lead to missed hazards or weak safety
> evidence in scenario-based testing.

### Response strategy (woven into the text, not a rebuttal)

Add the explicit risk statement exactly where archetypes are introduced as a user-defined,
free-to-choose input (the comment-2 paragraph in §5): a hazard reachable only via an archetype
absent from the set goes untested, weakening the safety evidence. This is immediately followed
by the existing mitigation — the embedding approach surfaces real-world scenes matching *no*
archetype, so the risk and its mitigation sit in one risk → mitigation flow.

### Edits made

**1. `sections_safecomp/05_create_subgraphs.tex`** (archetype paragraph, between "completeness not required" and the "Crucially, the embedding approach…" mitigation):

> In scenario-based testing this choice is safety-relevant: a hazard that manifests only
> through an archetype absent from the set would go untested, weakening the safety evidence the
> testing provides.

(Mitigation sentence already present from comment 2: the embedding approach guards against such
omissions of rare or safety-critical situations.)

---

## Reviewer comment 8 — Justify hierarchical edge selection (dense traffic)

> The paper should provide a stronger justification for the hierarchical edge selection rules,
> because removing edges may remove useful safety-relevant relations in dense traffic.

### Response strategy (woven into the text, not a rebuttal)

We disagree, and strengthen the justification rather than concede the premise. Key point: a
skipped edge is *not* lost information — it is only skipped when the relation is already encoded
transitively by a retained path (length ≤ `max_node_distance`), so the multi-actor interaction
is fully preserved, just implicit rather than explicit. This is in fact *especially* beneficial
in dense traffic, where pairwise relations grow quadratically and pruning keeps the graph and
the isomorphism checks tractable. We then open up to the reviewer's concern: pruning can be
gated by a configurable lower bound (implicit path length / local-density threshold) so edge
deletion is disabled in dense neighborhoods, keeping the closest, most safety-relevant relations
explicit.

### Edits made

**1. `sections_safecomp/04_defining_traffic_scene_graph.tex`** (appended to the "redundancy prevention … preserving connectivity" paragraph):

> This is especially beneficial in dense traffic, where the number of pairwise relations grows
> quadratically with the number of actors: pruning the redundant edges keeps both the graph and
> the subsequent isomorphism checks tractable, while the complex multi-actor interaction remains
> fully represented — each skipped relation is still encoded transitively through the retained
> paths, so no safety-relevant relation is lost, only made implicit rather than explicit. Where
> an application nonetheless requires every pairwise relation to be explicit, the pruning can be
> gated by a lower bound on the implicit path length (equivalently, a local-density threshold):
> edges are then removed only above this bound, so that the closest and most safety-relevant
> relations in dense traffic are always retained.
