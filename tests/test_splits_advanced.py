"""
tests/test_splits_advanced.py
============================
Tests for WalkForwardSplit and CombinatorialPurgedCV classes.
"""
import pytest
from Backtest.scoring.splits import (
    TimeRange,
    PurgedKFold,
    WalkForwardSplit,
    CombinatorialPurgedCV,
    DAY_NS,
    HOUR_NS,
)


class TestTimeRange:
    """Tests for TimeRange dataclass."""
    
    def test_duration_ns(self):
        """Duration calculation is correct."""
        tr = TimeRange(start_ns=0, end_ns=DAY_NS)
        assert tr.duration_ns == DAY_NS
    
    def test_duration_days(self):
        """Duration in days is correct."""
        tr = TimeRange(start_ns=0, end_ns=7 * DAY_NS)
        assert tr.duration_days == 7.0
    
    def test_overlaps(self):
        """Overlap detection works."""
        tr1 = TimeRange(start_ns=0, end_ns=DAY_NS)
        tr2 = TimeRange(start_ns=12 * HOUR_NS, end_ns=2 * DAY_NS)
        tr3 = TimeRange(start_ns=2 * DAY_NS, end_ns=3 * DAY_NS)
        
        assert tr1.overlaps(tr2)  # 12h overlap
        assert tr2.overlaps(tr1)  # Symmetric
        assert not tr1.overlaps(tr3)  # No overlap
    
    def test_contains(self):
        """Containment check works."""
        tr = TimeRange(start_ns=DAY_NS, end_ns=2 * DAY_NS)
        
        assert tr.contains(DAY_NS)  # Start is included
        assert tr.contains(DAY_NS + HOUR_NS)  # Middle
        assert not tr.contains(2 * DAY_NS)  # End is excluded
        assert not tr.contains(0)  # Before


class TestWalkForwardSplit:
    """Tests for WalkForwardSplit class."""
    
    def test_sliding_window_basic(self):
        """Sliding window produces correct splits."""
        start = 0
        end = 30 * DAY_NS  # 30 days
        
        splitter = WalkForwardSplit(
            start_ns=start,
            end_ns=end,
            train_duration_ns=7 * DAY_NS,
            test_duration_ns=3 * DAY_NS,
            step_ns=3 * DAY_NS,
            embargo_pct=0.0,
            expanding=False,
        )
        
        splits = list(splitter.split())
        
        # Should have multiple splits
        assert len(splits) >= 1
        
        # Check first split
        train_ranges, test_range = splits[0]
        assert len(train_ranges) == 1  # Single contiguous training range
        assert train_ranges[0].duration_ns == 7 * DAY_NS
        assert test_range.duration_ns == 3 * DAY_NS
        
        # Training should end at test start (no embargo)
        assert train_ranges[0].end_ns == test_range.start_ns
    
    def test_expanding_window_basic(self):
        """Expanding window produces correct splits."""
        start = 0
        end = 30 * DAY_NS
        
        splitter = WalkForwardSplit(
            start_ns=start,
            end_ns=end,
            train_duration_ns=7 * DAY_NS,  # Initial
            test_duration_ns=3 * DAY_NS,
            step_ns=3 * DAY_NS,
            embargo_pct=0.0,
            expanding=True,
        )
        
        splits = list(splitter.split())
        
        # All training starts at beginning
        for train_ranges, _ in splits:
            assert train_ranges[0].start_ns == start
        
        # Training window grows
        if len(splits) >= 2:
            assert splits[1][0][0].duration_ns > splits[0][0][0].duration_ns
    
    def test_embargo_applied(self):
        """Embargo creates gap between train and test."""
        start = 0
        end = 100 * DAY_NS
        
        splitter = WalkForwardSplit(
            start_ns=start,
            end_ns=end,
            train_duration_ns=20 * DAY_NS,
            test_duration_ns=10 * DAY_NS,
            embargo_pct=0.01,  # 1% embargo
            expanding=False,
        )
        
        embargo_ns = int((end - start) * 0.01)  # 1 day
        
        for train_ranges, test_range in splitter.split():
            # There should be a gap
            assert test_range.start_ns > train_ranges[0].end_ns
            assert test_range.start_ns - train_ranges[0].end_ns == embargo_ns
    
    def test_no_overlap_train_test(self):
        """Train and test ranges never overlap."""
        splitter = WalkForwardSplit(
            start_ns=0,
            end_ns=60 * DAY_NS,
            train_duration_ns=14 * DAY_NS,
            test_duration_ns=7 * DAY_NS,
            embargo_pct=0.02,
        )
        
        for train_ranges, test_range in splitter.split():
            for train_range in train_ranges:
                assert not train_range.overlaps(test_range)
    
    def test_get_n_splits(self):
        """get_n_splits returns correct count."""
        splitter = WalkForwardSplit(
            start_ns=0,
            end_ns=30 * DAY_NS,
            train_duration_ns=7 * DAY_NS,
            test_duration_ns=3 * DAY_NS,
            step_ns=3 * DAY_NS,
            embargo_pct=0.0,
        )
        
        n = splitter.get_n_splits()
        actual = len(list(splitter.split()))
        assert n == actual
    
    def test_info(self):
        """info() returns expected keys."""
        splitter = WalkForwardSplit(
            start_ns=0,
            end_ns=30 * DAY_NS,
            train_duration_ns=7 * DAY_NS,
            test_duration_ns=3 * DAY_NS,
        )
        
        info = splitter.info()
        assert "mode" in info
        assert "n_splits" in info
        assert "train_duration_days" in info
        assert info["mode"] == "sliding"
    
    def test_consistent_with_purged_kfold_interface(self):
        """WalkForwardSplit has same interface as PurgedKFold."""
        splitter = WalkForwardSplit(
            start_ns=0,
            end_ns=30 * DAY_NS,
            train_duration_ns=7 * DAY_NS,
            test_duration_ns=3 * DAY_NS,
        )
        
        for train_ranges, test_range in splitter.split():
            # train_ranges is a list
            assert isinstance(train_ranges, list)
            # test_range is a TimeRange
            assert isinstance(test_range, TimeRange)


class TestCombinatorialPurgedCV:
    """Tests for CombinatorialPurgedCV class."""
    
    def test_basic_splits(self):
        """CPCV produces expected number of combinations."""
        from math import comb
        
        splitter = CombinatorialPurgedCV(
            start_ns=0,
            end_ns=60 * DAY_NS,
            n_splits=6,
            n_test_splits=2,
            embargo_pct=0.0,
        )
        
        splits = list(splitter.split())
        
        # C(6, 2) = 15 combinations
        assert len(splits) == comb(6, 2)
    
    def test_test_ranges_are_lists(self):
        """CPCV returns test_ranges as a list (multiple folds)."""
        splitter = CombinatorialPurgedCV(
            start_ns=0,
            end_ns=60 * DAY_NS,
            n_splits=6,
            n_test_splits=2,
        )
        
        for train_ranges, test_ranges in splitter.split():
            assert isinstance(test_ranges, list)
            assert len(test_ranges) == 2  # n_test_splits
    
    def test_no_overlap_train_test(self):
        """Train and test ranges never overlap."""
        splitter = CombinatorialPurgedCV(
            start_ns=0,
            end_ns=60 * DAY_NS,
            n_splits=6,
            n_test_splits=2,
            embargo_pct=0.01,
        )
        
        for train_ranges, test_ranges in splitter.split():
            for train_range in train_ranges:
                for test_range in test_ranges:
                    assert not train_range.overlaps(test_range), \
                        f"Overlap: train={train_range}, test={test_range}"
    
    def test_embargo_applied_after_test(self):
        """Embargo is applied to training folds after test folds."""
        splitter = CombinatorialPurgedCV(
            start_ns=0,
            end_ns=100 * DAY_NS,
            n_splits=5,
            n_test_splits=1,
            embargo_pct=0.02,  # 2% embargo
        )
        
        embargo_ns = int(100 * DAY_NS * 0.02)
        
        for train_ranges, test_ranges in splitter.split():
            for test_range in test_ranges:
                for train_range in train_ranges:
                    # If train is after test, should have embargo gap
                    if train_range.start_ns >= test_range.end_ns:
                        gap = train_range.start_ns - test_range.end_ns
                        assert gap >= embargo_ns, \
                            f"Insufficient embargo: gap={gap}, required={embargo_ns}"
    
    def test_get_n_splits(self):
        """get_n_splits returns correct count."""
        splitter = CombinatorialPurgedCV(
            start_ns=0,
            end_ns=60 * DAY_NS,
            n_splits=5,
            n_test_splits=2,
        )
        
        n = splitter.get_n_splits()
        actual = len(list(splitter.split()))
        assert n == actual
    
    def test_info(self):
        """info() returns expected keys."""
        splitter = CombinatorialPurgedCV(
            start_ns=0,
            end_ns=60 * DAY_NS,
            n_splits=6,
            n_test_splits=2,
        )
        
        info = splitter.info()
        assert "n_folds" in info
        assert "n_test_folds" in info
        assert "n_combinations" in info
        assert info["n_combinations"] == 15
    
    def test_validation_errors(self):
        """Invalid parameters raise ValueError."""
        with pytest.raises(ValueError):
            CombinatorialPurgedCV(
                start_ns=0,
                end_ns=DAY_NS,
                n_splits=2,  # Too few
                n_test_splits=1,
            )
        
        with pytest.raises(ValueError):
            CombinatorialPurgedCV(
                start_ns=0,
                end_ns=DAY_NS,
                n_splits=5,
                n_test_splits=5,  # n_test_splits >= n_splits
            )


class TestSplitterCompatibility:
    """Test that all splitters work with BatchBacktest.run_with_cv."""
    
    def test_all_splitters_have_split_method(self):
        """All splitters have .split() method."""
        splitters = [
            PurgedKFold(start_ns=0, end_ns=30 * DAY_NS, n_splits=5),
            WalkForwardSplit(
                start_ns=0, end_ns=30 * DAY_NS,
                train_duration_ns=7 * DAY_NS, test_duration_ns=3 * DAY_NS,
            ),
            CombinatorialPurgedCV(
                start_ns=0, end_ns=30 * DAY_NS,
                n_splits=5, n_test_splits=2,
            ),
        ]
        
        for splitter in splitters:
            assert hasattr(splitter, 'split')
            assert callable(splitter.split)
    
    def test_all_splitters_yield_correct_structure(self):
        """All splitters yield (list, TimeRange or list) tuples."""
        splitters = [
            PurgedKFold(start_ns=0, end_ns=30 * DAY_NS, n_splits=5),
            WalkForwardSplit(
                start_ns=0, end_ns=30 * DAY_NS,
                train_duration_ns=7 * DAY_NS, test_duration_ns=3 * DAY_NS,
            ),
            CombinatorialPurgedCV(
                start_ns=0, end_ns=30 * DAY_NS,
                n_splits=5, n_test_splits=2,
            ),
        ]
        
        for splitter in splitters:
            for train_ranges, test_data in splitter.split():
                # train_ranges should be a list
                assert isinstance(train_ranges, list)
                assert all(isinstance(r, TimeRange) for r in train_ranges)
                
                # test_data is TimeRange or list of TimeRange
                if isinstance(test_data, list):
                    assert all(isinstance(r, TimeRange) for r in test_data)
                else:
                    assert isinstance(test_data, TimeRange)
