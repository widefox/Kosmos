"""
Real data tests for statistical validation module.

Tests StatisticalValidator methods using actual numerical data without mocking.
These tests verify that the statistical calculations produce correct results
with real-world data distributions.
"""

import pytest
import numpy as np
from scipy import stats as scipy_stats

from kosmos.execution.statistics import StatisticalValidator


class TestRealDataEffectSizes:
    """Test effect sizes with real generated data."""

    def test_cohens_d_with_known_effect(self):
        """Test Cohen's d produces expected effect size with known difference."""
        np.random.seed(42)
        # Two groups with known mean difference
        # Group 1: mean=100, SD=15
        # Group 2: mean=115, SD=15
        # Expected Cohen's d ≈ -1.0 (group1 < group2)
        n = 1000  # Large n for stable estimate
        group1 = np.random.normal(100, 15, n)
        group2 = np.random.normal(115, 15, n)

        d = StatisticalValidator.calculate_cohens_d(group1, group2)

        # With large n, should be close to -1.0
        assert -1.2 <= d <= -0.8, f"Expected d ≈ -1.0, got {d}"
        assert StatisticalValidator.interpret_effect_size(d, 'cohens_d') == 'large'

    def test_cohens_d_matches_formula(self):
        """Verify Cohen's d matches the mathematical formula."""
        group1 = np.array([2, 4, 6, 8, 10])
        group2 = np.array([3, 5, 7, 9, 11])

        d = StatisticalValidator.calculate_cohens_d(group1, group2)

        # Manual calculation
        mean_diff = np.mean(group1) - np.mean(group2)  # -1.0
        pooled_var = ((4 * np.var(group1, ddof=1) + 4 * np.var(group2, ddof=1)) / 8)
        expected_d = mean_diff / np.sqrt(pooled_var)

        assert abs(d - expected_d) < 0.001, f"Formula mismatch: {d} vs {expected_d}"

    def test_eta_squared_with_anova_groups(self):
        """Test eta-squared with groups that have known differences."""
        np.random.seed(123)
        # Three groups with different means
        n_per_group = 100
        group_a = np.random.normal(50, 10, n_per_group)
        group_b = np.random.normal(55, 10, n_per_group)
        group_c = np.random.normal(60, 10, n_per_group)

        eta_sq = StatisticalValidator.calculate_eta_squared([group_a, group_b, group_c])

        # With clear group differences, should detect medium-large effect
        assert eta_sq > 0.05, f"Expected detectable effect, got eta²={eta_sq}"
        assert eta_sq < 0.5, f"Effect size unrealistically large: {eta_sq}"

    def test_cramers_v_with_contingency_table(self):
        """Test Cramér's V with a contingency table showing association."""
        # Strong association table
        strong_association = [
            [90, 10],
            [10, 90]
        ]
        v_strong = StatisticalValidator.calculate_cramers_v(strong_association)
        assert v_strong > 0.5, f"Expected strong association, got V={v_strong}"

        # No association table
        no_association = [
            [50, 50],
            [50, 50]
        ]
        v_none = StatisticalValidator.calculate_cramers_v(no_association)
        assert v_none < 0.1, f"Expected no association, got V={v_none}"


class TestRealDataHypothesisTesting:
    """Test hypothesis testing with real data."""

    def test_ttest_detects_real_difference(self):
        """Verify t-test correctly detects a real mean difference."""
        np.random.seed(42)
        # Groups with different means
        group1 = np.random.normal(100, 10, 50)
        group2 = np.random.normal(110, 10, 50)

        t_stat, p_value = scipy_stats.ttest_ind(group1, group2)
        sig_result = StatisticalValidator.apply_significance_threshold(p_value)

        assert sig_result['significant_0.05'] is True
        assert sig_result['significance_label'] in ['*', '**', '***']

    def test_ttest_no_false_positive(self):
        """Verify t-test does not produce false positives with same distribution."""
        np.random.seed(42)
        # Same distribution
        all_data = np.random.normal(100, 10, 100)
        group1 = all_data[:50]
        group2 = all_data[50:]  # Random split - should be similar

        # Run multiple times and check false positive rate
        false_positives = 0
        n_tests = 100
        for i in range(n_tests):
            np.random.seed(i)
            g1 = np.random.normal(100, 10, 30)
            g2 = np.random.normal(100, 10, 30)  # Same distribution
            _, p = scipy_stats.ttest_ind(g1, g2)
            if p < 0.05:
                false_positives += 1

        # False positive rate should be around 5%
        fp_rate = false_positives / n_tests
        assert 0.01 <= fp_rate <= 0.15, f"False positive rate {fp_rate} unexpected"

    def test_mann_whitney_with_skewed_data(self):
        """Test Mann-Whitney U with non-normal (skewed) data."""
        np.random.seed(42)
        # Exponential distributions - good case for non-parametric test
        group1 = np.random.exponential(scale=2, size=50)
        group2 = np.random.exponential(scale=4, size=50)  # Different scale

        result = StatisticalValidator.mann_whitney_u_test(group1, group2)

        # Should detect difference
        assert result['p_value'] < 0.05
        assert result['median1'] < result['median2']

    def test_chi_square_with_categorical_data(self):
        """Test chi-square with realistic categorical data."""
        # Simulating treatment effect: more success in treatment group
        contingency = [
            [70, 30],  # Treatment: 70 success, 30 failure
            [40, 60]   # Control: 40 success, 60 failure
        ]

        result = StatisticalValidator.chi_square_test(contingency)

        assert result['p_value'] < 0.001  # Strong association
        assert result['cramers_v'] > 0.2  # Medium effect
        assert result['degrees_of_freedom'] == 1


class TestRealDataConfidenceIntervals:
    """Test confidence intervals with real data."""

    def test_ci_covers_true_mean(self):
        """Verify 95% CI covers true mean in ~95% of samples."""
        true_mean = 100
        true_std = 15
        n_samples = 500
        coverage_count = 0

        for seed in range(n_samples):
            np.random.seed(seed)
            sample = np.random.normal(true_mean, true_std, 50)
            lower, upper = StatisticalValidator.parametric_confidence_interval(sample, 0.95)
            if lower <= true_mean <= upper:
                coverage_count += 1

        coverage_rate = coverage_count / n_samples
        # Should be approximately 95% (allow 92-98% for random variation)
        assert 0.92 <= coverage_rate <= 0.98, f"Coverage {coverage_rate} outside expected range"

    def test_bootstrap_ci_comparable_to_parametric(self):
        """Bootstrap CI should be similar to parametric for normal data."""
        np.random.seed(42)
        normal_data = np.random.normal(50, 10, 100)

        param_lower, param_upper = StatisticalValidator.parametric_confidence_interval(normal_data)
        boot_lower, boot_upper = StatisticalValidator.bootstrap_confidence_interval(
            normal_data, n_iterations=5000
        )

        # Bootstrap and parametric should be within ~10% for normal data
        param_width = param_upper - param_lower
        boot_width = boot_upper - boot_lower

        assert abs(param_width - boot_width) / param_width < 0.15, \
            f"CI widths differ too much: {param_width} vs {boot_width}"


class TestRealDataMultipleComparisons:
    """Test multiple comparison corrections with real p-values."""

    def test_bonferroni_controls_fwer(self):
        """Verify Bonferroni controls family-wise error rate."""
        # Simulate 20 null hypothesis tests (all from same distribution)
        n_simulations = 200
        fwer_bonf = 0
        fwer_uncorrected = 0

        for sim in range(n_simulations):
            np.random.seed(sim)
            p_values = []
            for _ in range(20):
                g1 = np.random.normal(0, 1, 30)
                g2 = np.random.normal(0, 1, 30)  # Same distribution
                _, p = scipy_stats.ttest_ind(g1, g2)
                p_values.append(p)

            # Check uncorrected
            if any(p < 0.05 for p in p_values):
                fwer_uncorrected += 1

            # Check Bonferroni
            bonf = StatisticalValidator.bonferroni_correction(p_values)
            if bonf['n_significant'] > 0:
                fwer_bonf += 1

        fwer_rate_uncorrected = fwer_uncorrected / n_simulations
        fwer_rate_bonf = fwer_bonf / n_simulations

        # Uncorrected should have high FWER (~64% for 20 tests at α=0.05)
        assert fwer_rate_uncorrected > 0.4, f"Uncorrected FWER unexpectedly low: {fwer_rate_uncorrected}"

        # Bonferroni should control at ~5%
        assert fwer_rate_bonf < 0.10, f"Bonferroni FWER too high: {fwer_rate_bonf}"

    def test_bh_fdr_more_powerful(self):
        """BH-FDR should be more powerful than Bonferroni for real effects."""
        np.random.seed(42)

        # Generate 10 tests: 3 with real effects, 7 null
        p_values = []
        for i in range(10):
            if i < 3:
                # Real effect
                g1 = np.random.normal(0, 1, 50)
                g2 = np.random.normal(0.8, 1, 50)  # Different mean
            else:
                # No effect
                g1 = np.random.normal(0, 1, 50)
                g2 = np.random.normal(0, 1, 50)
            _, p = scipy_stats.ttest_ind(g1, g2)
            p_values.append(p)

        bonf = StatisticalValidator.bonferroni_correction(p_values)
        bh = StatisticalValidator.benjamini_hochberg_fdr(p_values)

        # BH should detect at least as many as Bonferroni
        assert bh['n_significant'] >= bonf['n_significant']


class TestRealDataAssumptions:
    """Test assumption checking with real data distributions."""

    def test_normality_detection_accuracy(self):
        """Test that normality detection is accurate for different distributions."""
        np.random.seed(42)

        # Generate different distributions
        distributions = {
            'normal': np.random.normal(0, 1, 100),
            'uniform': np.random.uniform(-1, 1, 100),
            'exponential': np.random.exponential(1, 100),
            'bimodal': np.concatenate([np.random.normal(-2, 0.5, 50), np.random.normal(2, 0.5, 50)])
        }

        results = {}
        for name, data in distributions.items():
            result = StatisticalValidator.check_assumptions(data)
            results[name] = result['normality_assumption_met']

        # Normal should pass
        assert results['normal'] is True, "Normal data should pass normality test"

        # Non-normal distributions should fail (most of the time)
        # Note: Small samples may pass even for non-normal data
        non_normal_passed = sum(results[d] for d in ['uniform', 'exponential', 'bimodal'])
        assert non_normal_passed <= 1, "Too many non-normal distributions passed"

    def test_small_sample_warnings(self):
        """Test that small samples produce appropriate warnings."""
        small_sample = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

        result = StatisticalValidator.check_assumptions(small_sample, test_type='t-test')

        assert any('small sample' in w.lower() for w in result['warnings']), \
            "Should warn about small sample size"


class TestStatisticalReportGeneration:
    """Test report generation with real test results."""

    def test_complete_analysis_report(self):
        """Test generating a complete analysis report from real data."""
        np.random.seed(42)

        # Perform actual t-test
        group1 = np.random.normal(100, 15, 50)
        group2 = np.random.normal(115, 15, 50)

        t_stat, p_value = scipy_stats.ttest_ind(group1, group2)
        cohens_d = StatisticalValidator.calculate_cohens_d(group1, group2)
        ci = StatisticalValidator.parametric_confidence_interval(group1 - group2[:len(group1)])

        test_results = {
            't_statistic': t_stat,
            'p_value': p_value,
            **StatisticalValidator.apply_significance_threshold(p_value),
            'n_group1': len(group1),
            'n_group2': len(group2)
        }

        report = StatisticalValidator.generate_statistical_report(
            test_type='t-test',
            test_results=test_results,
            effect_size=cohens_d,
            confidence_interval=ci
        )

        # Verify report contains expected components
        assert 'T-TEST' in report
        assert 't-statistic' in report
        assert 'p-value' in report
        assert 'Effect size' in report
        assert 'Confidence interval' in report
        assert 'Sample size' in report
        assert 'SIGNIFICANT' in report  # Should be significant


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
