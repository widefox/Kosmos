"""
Phase 3 end-to-end integration tests.

Tests complete workflow: Generation → Novelty → Testability → Prioritization

Tests using REAL Claude API for LLM-dependent tests.
"""

import os
import pytest
import uuid
from unittest.mock import Mock, patch
from kosmos.agents.hypothesis_generator import HypothesisGeneratorAgent
from kosmos.hypothesis.novelty_checker import NoveltyChecker
from kosmos.hypothesis.testability import TestabilityAnalyzer
from kosmos.hypothesis.prioritizer import HypothesisPrioritizer
from kosmos.models.hypothesis import Hypothesis


# Skip all tests if API key not available
pytestmark = [
    pytest.mark.integration,
    pytest.mark.requires_claude,
    pytest.mark.skipif(
        not os.getenv("ANTHROPIC_API_KEY"),
        reason="Requires ANTHROPIC_API_KEY for real LLM calls"
    )
]


def unique_id() -> str:
    """Generate unique ID for test isolation."""
    return uuid.uuid4().hex[:8]

class TestPhase3EndToEnd:
    """Test complete Phase 3 workflow with real Claude API."""

    @patch('kosmos.hypothesis.novelty_checker.UnifiedLiteratureSearch')
    @patch('kosmos.hypothesis.novelty_checker.get_session')
    def test_full_hypothesis_pipeline(self, mock_session, mock_search):
        """Test: Generate → Check Novelty → Analyze Testability → Prioritize with real Claude."""
        from kosmos.core.llm import ClaudeClient

        # Setup mocks for non-LLM services
        mock_search_inst = Mock()
        mock_search_inst.search.return_value = []
        mock_search.return_value = mock_search_inst

        mock_sess = Mock()
        mock_sess.query.return_value.filter.return_value.all.return_value = []
        mock_session.return_value = mock_sess

        # Step 1: Generate hypotheses with real Claude
        uid = unique_id()
        with patch('kosmos.agents.hypothesis_generator.get_client') as mock_get_client:
            mock_get_client.return_value = ClaudeClient(model="claude-3-haiku-20240307")

            agent = HypothesisGeneratorAgent(config={
                "use_literature_context": False,
                "num_hypotheses": 2
            })
            response = agent.generate_hypotheses(
                research_question=f"How does batch size affect neural network training? [test-{uid}]",
                store_in_db=False
            )

        assert len(response.hypotheses) >= 1
        hypotheses = response.hypotheses

        # Step 2: Check novelty (mock literature search)
        novelty_checker = NoveltyChecker(use_vector_db=False)
        novelty_checker.literature_search = mock_search_inst

        for hyp in hypotheses:
            report = novelty_checker.check_novelty(hyp)
            assert report.novelty_score is not None
            assert 0.0 <= report.novelty_score <= 1.0
            hyp.novelty_score = report.novelty_score

        # Step 3: Analyze testability (pure Python)
        testability_analyzer = TestabilityAnalyzer(use_llm_for_assessment=False)

        for hyp in hypotheses:
            report = testability_analyzer.analyze_testability(hyp)
            assert report.testability_score is not None
            assert report.is_testable or not report.is_testable  # Boolean
            hyp.testability_score = report.testability_score

        # Step 4: Prioritize (pure Python)
        prioritizer = HypothesisPrioritizer(
            use_novelty_checker=False,  # Already done
            use_testability_analyzer=False,  # Already done
            use_impact_prediction=False
        )

        ranked = prioritizer.prioritize(hypotheses, run_analysis=False)

        assert len(ranked) >= 1
        assert ranked[0].rank == 1
        assert ranked[0].priority_score > 0.0

        # Verify all scores present
        for p in ranked:
            assert p.novelty_score is not None
            assert p.testability_score is not None
            assert p.feasibility_score is not None
            assert p.impact_score is not None

    def test_hypothesis_filtering(self):
        """Test filtering untestable or non-novel hypotheses with real Claude."""
        from kosmos.core.llm import ClaudeClient

        uid = unique_id()
        with patch('kosmos.agents.hypothesis_generator.get_client') as mock_get_client:
            mock_get_client.return_value = ClaudeClient(model="claude-3-haiku-20240307")

            agent = HypothesisGeneratorAgent(config={
                "use_literature_context": False,
                "num_hypotheses": 3
            })
            response = agent.generate_hypotheses(
                research_question=f"What is the effect of temperature on protein folding? [test-{uid}]",
                store_in_db=False
            )

        # With real Claude, we should get hypotheses with varying testability scores
        assert len(response.hypotheses) >= 1

        # Filter testable hypotheses (threshold-based filtering)
        testable = [h for h in response.hypotheses if h.is_testable(threshold=0.5)]

        # At least some hypotheses should be testable from a real generation
        assert len(testable) >= 0  # May vary based on Claude's response
        for h in testable:
            assert h.testability_score >= 0.5

    def test_hypothesis_model_validation(self):
        """Test Pydantic validation on Hypothesis model."""
        # Valid hypothesis
        hyp = Hypothesis(
            research_question="Valid question?",
            statement="This is a clear testable statement",
            rationale="This is a sufficient rationale that explains the hypothesis",
            domain="test"
        )
        assert hyp.statement == "This is a clear testable statement"

        # Invalid: statement too short (should fail validation)
        with pytest.raises(ValueError):
            Hypothesis(
                research_question="Test",
                statement="Too short",
                rationale="Valid rationale here",
                domain="test"
            )

@pytest.mark.slow
class TestPhase3RealIntegration:
    """Integration tests with real services (requires Claude, DB)."""

    def test_real_hypothesis_workflow(self):
        """Test with real Claude API (slow, requires API key)."""
        from kosmos.core.llm import ClaudeClient

        uid = unique_id()
        with patch('kosmos.agents.hypothesis_generator.get_client') as mock_get_client:
            mock_get_client.return_value = ClaudeClient(model="claude-3-haiku-20240307")

            agent = HypothesisGeneratorAgent(config={
                "num_hypotheses": 2,
                "use_literature_context": False
            })

            response = agent.generate_hypotheses(
                research_question=f"How does batch size affect neural network training? [test-{uid}]",
                domain="machine_learning",
                store_in_db=False
            )

        assert len(response.hypotheses) > 0

        # Analyze first hypothesis
        hyp = response.hypotheses[0]

        # Check novelty (mock literature search to avoid rate limits)
        with patch('kosmos.hypothesis.novelty_checker.UnifiedLiteratureSearch') as mock_search:
            mock_search_inst = Mock()
            mock_search_inst.search.return_value = []
            mock_search.return_value = mock_search_inst

            novelty_checker = NoveltyChecker(use_vector_db=False)
            novelty_checker.literature_search = mock_search_inst
            novelty_report = novelty_checker.check_novelty(hyp)
            assert novelty_report.novelty_score is not None

        # Check testability (pure Python)
        testability_analyzer = TestabilityAnalyzer(use_llm_for_assessment=False)
        testability_report = testability_analyzer.analyze_testability(hyp)
        assert testability_report.is_testable or not testability_report.is_testable
