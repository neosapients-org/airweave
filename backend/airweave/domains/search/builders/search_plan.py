"""Builder for the complete search plan.

Combines the LLM-generated plan with user-supplied deterministic filters.
User filters are AND'd into every LLM filter group — they are constraints
that must always be satisfied, not optional additions.
"""

from typing import List, Optional

from airweave.core.config import settings
from airweave.domains.search.types.filters import FilterGroup
from airweave.domains.search.types.plan import SearchPlan


class SearchPlanBuilder:
    """Builds the complete plan by AND'ing user filters into LLM filters.

    User filters are leading — they restrict all results. If the user
    filters on source_name="github", no results from other sources can
    ever be returned, regardless of what the LLM generates.

    Merge logic:
    - If the LLM has filter groups: inject user filter conditions into
      each LLM group (AND'd together within each group).
    - If the LLM has no filter groups: create a single group from the
      user filter conditions.
    - If no user filters: return the plan unchanged.

    With ``SEARCH_FILTER_GROUPS_OR`` (contract C8) the user's groups keep their OR: the
    result is (U1 OR U2 ...) AND (L1 OR L2 ...), as one group per pair (Ui AND Lj), or the
    user's groups as they are when the LLM has none. With a single user group both ways are
    the same.
    """

    @staticmethod
    def build(
        plan: SearchPlan,
        user_filter: List[FilterGroup],
        keep_or_groups: Optional[bool] = None,
    ) -> SearchPlan:
        """Build the complete plan by AND'ing user filters into LLM filters.

        Args:
            plan: The LLM-generated search plan.
            user_filter: User-supplied deterministic filter groups.
                Each group's conditions are AND'd into every LLM filter group.
            keep_or_groups: Keep OR between the user's groups; None reads
                ``SEARCH_FILTER_GROUPS_OR`` (False: upstream behaviour).

        Returns:
            The complete plan with user filters AND'd in.
        """
        if not user_filter:
            return plan

        if keep_or_groups is None:
            keep_or_groups = settings.SEARCH_FILTER_GROUPS_OR
        if keep_or_groups:
            return SearchPlanBuilder._build_keeping_or(plan, user_filter)

        # Collect all user filter conditions (flattened from all user groups)
        user_conditions = []
        for uf_group in user_filter:
            user_conditions.extend(uf_group.conditions)

        if not user_conditions:
            return plan

        complete_plan = plan.model_copy(deep=True)

        if complete_plan.filter_groups:
            # AND user conditions into each existing LLM group
            for group in complete_plan.filter_groups:
                group.conditions.extend(user_conditions)
        else:
            # No LLM groups — create one from user conditions alone
            complete_plan.filter_groups = [FilterGroup(conditions=list(user_conditions))]

        return complete_plan

    @staticmethod
    def _build_keeping_or(plan: SearchPlan, user_filter: List[FilterGroup]) -> SearchPlan:
        """AND the user's groups into the LLM's, keeping OR between the user's groups."""
        user_groups = [g for g in user_filter if g.conditions]
        if not user_groups:
            return plan

        complete_plan = plan.model_copy(deep=True)
        if complete_plan.filter_groups:
            complete_plan.filter_groups = [
                FilterGroup(conditions=[*llm.conditions, *user.conditions])
                for llm in complete_plan.filter_groups
                for user in user_groups
            ]
        else:
            complete_plan.filter_groups = [g.model_copy(deep=True) for g in user_groups]
        return complete_plan
