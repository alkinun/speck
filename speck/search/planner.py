"""budgeted posterior action planning without product thresholds."""

import math
from dataclasses import asdict, dataclass

import numpy as np

from speck.search.protocol import content_digest
from speck.search.study_v3 import V3Study


@dataclass(frozen=True)
class ActionProposal:
    kind: str
    architecture_digest: str
    estimated_cost: float
    frontier_probability: float
    expected_information: float
    novelty: float
    payload: dict

    def __post_init__(self):
        if not self.kind or self.kind.lower() != self.kind:
            raise ValueError("proposal kinds must be lowercase")
        if not self.architecture_digest:
            raise ValueError("proposals need an architecture identity")
        values = (
            self.estimated_cost,
            self.frontier_probability,
            self.expected_information,
            self.novelty,
        )
        if any(not math.isfinite(value) for value in values):
            raise ValueError("proposal values must be finite")
        if self.estimated_cost <= 0:
            raise ValueError("proposal cost must be positive")
        if not 0 <= self.frontier_probability <= 1:
            raise ValueError("frontier probability must be between zero and one")
        if self.expected_information < 0 or self.novelty < 0:
            raise ValueError("proposal information and novelty cannot be negative")

    @property
    def digest(self):
        return content_digest(self)


@dataclass(frozen=True)
class PlannedAction:
    proposal: ActionProposal
    priority: float


@dataclass(frozen=True)
class PlanningDecision:
    seed: int
    available_cost: float
    committed_cost: float
    criterion_weights: tuple[float, float, float]
    eligible_digests: tuple[str, ...]
    selected: tuple[PlannedAction, ...]

    @property
    def digest(self):
        return content_digest(self)

    def export(self):
        return asdict(self)


def _rank_score(values):
    if len(values) == 1:
        return np.ones(1, dtype=np.float64)
    order = sorted(range(len(values)), key=lambda index: (values[index], index))
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        rank = (start + end - 1) / 2 / (len(values) - 1)
        for position in range(start, end):
            ranks[order[position]] = rank
        start = end
    return ranks


def plan_actions(proposals, available_cost, max_actions, seed):
    proposals = tuple(proposals)
    if not proposals:
        raise ValueError("planning needs at least one proposal")
    if available_cost <= 0 or max_actions < 1:
        raise ValueError("planning budget and action count must be positive")
    if len({proposal.digest for proposal in proposals}) != len(proposals):
        raise ValueError("planning proposals must be unique")
    rng = np.random.default_rng(seed)
    weights = rng.dirichlet(np.ones(3))
    frontier = _rank_score(
        np.asarray([proposal.frontier_probability for proposal in proposals])
    )
    information = _rank_score(
        np.asarray([proposal.expected_information for proposal in proposals])
    )
    novelty = _rank_score(np.asarray([proposal.novelty for proposal in proposals]))
    utility = weights[0] * frontier + weights[1] * information + weights[2] * novelty
    priorities = utility / np.asarray(
        [proposal.estimated_cost for proposal in proposals]
    )
    order = sorted(
        range(len(proposals)),
        key=lambda index: (-priorities[index], proposals[index].digest),
    )
    selected = []
    committed = 0.0
    for index in order:
        proposal = proposals[index]
        if committed + proposal.estimated_cost > available_cost:
            continue
        selected.append(PlannedAction(proposal, float(priorities[index])))
        committed += proposal.estimated_cost
        if len(selected) == max_actions:
            break
    return PlanningDecision(
        seed=seed,
        available_cost=float(available_cost),
        committed_cost=committed,
        criterion_weights=tuple(float(value) for value in weights),
        eligible_digests=tuple(sorted(proposal.digest for proposal in proposals)),
        selected=tuple(selected),
    )


def commit_plan(study, decision):
    if not isinstance(study, V3Study) or not isinstance(
        decision, PlanningDecision
    ):
        raise TypeError("committing a plan needs a v3 study and planning decision")
    actions = []
    for action in decision.selected:
        payload = {
            **action.proposal.payload,
            "architecture_digest": action.proposal.architecture_digest,
            "proposal_digest": action.proposal.digest,
        }
        actions.append(
            {
                "estimated_cost": action.proposal.estimated_cost,
                "kind": action.proposal.kind,
                "payload": payload,
                "priority": action.priority,
            }
        )
    return study.commit_planning_decision(
        decision.digest,
        decision.export(),
        actions,
    )


def posterior_information(covariance, reduction):
    covariance = np.asarray(covariance, dtype=np.float64)
    if covariance.ndim != 2 or covariance.shape[0] != covariance.shape[1]:
        raise ValueError("posterior covariance must be square")
    if not 0 <= reduction <= 1:
        raise ValueError("posterior reduction must be between zero and one")
    if np.linalg.eigvalsh(covariance).min() < -1e-10:
        raise ValueError("posterior covariance must be positive semidefinite")
    return float(math.sqrt(max(0.0, np.trace(covariance))) * reduction)
