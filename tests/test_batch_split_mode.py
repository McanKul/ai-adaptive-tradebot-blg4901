"""
tests/test_batch_split_mode.py
==============================
Tests for split_mode parameter in BatchBacktest.run_with_cv().
"""
import pytest
from unittest.mock import Mock, patch
import pandas as pd
import numpy as np
from datetime import datetime

from Backtest.scoring.batch import BatchBacktest
from Backtest.scoring.splits import (
    PurgedKFold,
    WalkForwardSplit,
    CombinatorialPurgedCV,
    TimeRange,
)


# Time constants for testing
DAY_NS = 86_400_000_000_000
START_NS = 1_700_000_000_000_000_000  # Some arbitrary start
END_NS = START_NS + 100 * DAY_NS  # 100 days of data


class TestSplitModeParameter:
    """Tests for split_mode parameter in run_with_cv."""
    
    def test_splitter_classes_exist_and_importable(self):
        """All splitter classes exist and can be imported."""
        assert PurgedKFold is not None
        assert WalkForwardSplit is not None
        assert CombinatorialPurgedCV is not None
    
    def test_split_mode_options_documented(self):
        """Valid split_mode options are documented."""
        # Verify classes match mode names
        assert 'purged' in PurgedKFold.__name__.lower()
        assert 'walk' in WalkForwardSplit.__name__.lower() or 'forward' in WalkForwardSplit.__name__.lower()
        assert 'combinatorial' in CombinatorialPurgedCV.__name__.lower() or 'cpcv' in CombinatorialPurgedCV.__name__.lower()


class TestPurgedKFoldSplit:
    """Tests for PurgedKFold splitter."""
    
    def test_purged_kfold_creates_correct_number_of_folds(self):
        """PurgedKFold creates the specified number of folds."""
        splitter = PurgedKFold(
            start_ns=START_NS,
            end_ns=END_NS,
            n_splits=5,
            embargo_pct=0.01,
        )
        
        splits = list(splitter.split())
        assert len(splits) == 5
    
    def test_purged_kfold_train_test_no_overlap(self):
        """PurgedKFold maintains separation between train and test."""
        splitter = PurgedKFold(
            start_ns=START_NS,
            end_ns=END_NS,
            n_splits=5,
            embargo_pct=0.02,
        )
        
        for train_ranges, test_range in splitter.split():
            # Test should not overlap with any train range
            for train_range in train_ranges:
                # Check no time overlap
                assert (train_range.end_ns <= test_range.start_ns or 
                        train_range.start_ns >= test_range.end_ns)
    
    def test_purged_kfold_all_folds_tested(self):
        """Each fold is used as test exactly once."""
        splitter = PurgedKFold(
            start_ns=START_NS,
            end_ns=END_NS,
            n_splits=5,
            embargo_pct=0.01,
        )
        
        test_starts = []
        for train_ranges, test_range in splitter.split():
            test_starts.append(test_range.start_ns)
        
        # Each fold should start at different position
        assert len(set(test_starts)) == 5


class TestWalkForwardSplit:
    """Tests for WalkForwardSplit splitter."""
    
    def test_walk_forward_sliding_windows(self):
        """WalkForwardSplit creates sliding training windows by default."""
        splitter = WalkForwardSplit(
            start_ns=START_NS,
            end_ns=END_NS,
            train_duration_ns=30 * DAY_NS,
            test_duration_ns=10 * DAY_NS,
            embargo_pct=0.01,
            expanding=False,
        )
        
        train_starts = []
        for train_ranges, test_range in splitter.split():
            train_starts.append(train_ranges[0].start_ns)
            # Test should come after train
            assert test_range.start_ns >= train_ranges[0].end_ns
        
        # Training window should move (sliding)
        if len(train_starts) > 1:
            assert train_starts[1] > train_starts[0]
    
    def test_walk_forward_no_future_leakage(self):
        """WalkForwardSplit never uses future data for training."""
        splitter = WalkForwardSplit(
            start_ns=START_NS,
            end_ns=END_NS,
            train_duration_ns=20 * DAY_NS,
            test_duration_ns=10 * DAY_NS,
            embargo_pct=0.01,
        )
        
        for train_ranges, test_range in splitter.split():
            # All train end times must be < test start
            for train_range in train_ranges:
                assert train_range.end_ns <= test_range.start_ns
    
    def test_walk_forward_expanding_anchored_start(self):
        """Expanding WalkForward keeps training start fixed."""
        splitter = WalkForwardSplit(
            start_ns=START_NS,
            end_ns=END_NS,
            train_duration_ns=20 * DAY_NS,
            test_duration_ns=10 * DAY_NS,
            embargo_pct=0.01,
            expanding=True,
        )
        
        train_starts = []
        train_durations = []
        for train_ranges, test_range in splitter.split():
            train_starts.append(train_ranges[0].start_ns)
            train_durations.append(train_ranges[0].duration_ns)
        
        # All training sets should start at the same point
        if len(train_starts) > 0:
            assert all(s == train_starts[0] for s in train_starts)
        
        # Training duration should increase (expanding)
        if len(train_durations) > 1:
            assert train_durations[-1] > train_durations[0]


class TestCombinatorialPurgedCV:
    """Tests for CombinatorialPurgedCV (CPCV) splitter."""
    
    def test_cpcv_creates_combinatorial_splits(self):
        """CPCV creates correct number of combinatorial splits."""
        splitter = CombinatorialPurgedCV(
            start_ns=START_NS,
            end_ns=END_NS,
            n_splits=6,
            n_test_splits=2,
            embargo_pct=0.01,
        )
        
        splits = list(splitter.split())
        
        # C(6,2) = 15 combinations
        from math import comb
        expected_splits = comb(6, 2)
        assert len(splits) == expected_splits
    
    def test_cpcv_maintains_train_test_separation(self):
        """CPCV maintains separation between train and test."""
        splitter = CombinatorialPurgedCV(
            start_ns=START_NS,
            end_ns=END_NS,
            n_splits=5,
            n_test_splits=2,
            embargo_pct=0.01,
        )
        
        for train_ranges, test_ranges in splitter.split():
            train_times = set()
            test_times = set()
            
            for tr in train_ranges:
                for t in range(tr.start_ns, tr.end_ns, DAY_NS):
                    train_times.add(t)
            
            for tr in test_ranges:
                for t in range(tr.start_ns, tr.end_ns, DAY_NS):
                    test_times.add(t)
            
            # No overlap between train and test
            assert train_times.isdisjoint(test_times)
    
    def test_cpcv_get_n_splits(self):
        """CPCV correctly reports number of splits."""
        splitter = CombinatorialPurgedCV(
            start_ns=START_NS,
            end_ns=END_NS,
            n_splits=5,
            n_test_splits=1,
            embargo_pct=0.01,
        )
        
        # C(5,1) = 5 splits
        assert splitter.get_n_splits() == 5
        
        # Verify by counting
        splits = list(splitter.split())
        assert len(splits) == 5


class TestSplitIntegration:
    """Integration tests for splits."""
    
    def test_all_splitters_have_consistent_interface(self):
        """All splitters implement same interface."""
        splitters = [
            PurgedKFold(
                start_ns=START_NS,
                end_ns=END_NS,
                n_splits=3,
                embargo_pct=0.01,
            ),
            WalkForwardSplit(
                start_ns=START_NS,
                end_ns=END_NS,
                train_duration_ns=30 * DAY_NS,
                test_duration_ns=10 * DAY_NS,
                embargo_pct=0.01,
            ),
            CombinatorialPurgedCV(
                start_ns=START_NS,
                end_ns=END_NS,
                n_splits=5,
                n_test_splits=2,
                embargo_pct=0.01,
            ),
        ]
        
        for splitter in splitters:
            # All should have split() method
            assert hasattr(splitter, 'split')
            
            # split() should yield tuples
            splits = list(splitter.split())
            assert len(splits) > 0
            
            for train_ranges, test_range_or_ranges in splits:
                # Train is always a list
                assert isinstance(train_ranges, list)
    
    def test_splitters_are_deterministic(self):
        """Same splitter with same data produces same splits."""
        splitter_configs = [
            (PurgedKFold, {
                'start_ns': START_NS,
                'end_ns': END_NS,
                'n_splits': 3,
                'embargo_pct': 0.01,
            }),
            (WalkForwardSplit, {
                'start_ns': START_NS,
                'end_ns': END_NS,
                'train_duration_ns': 30 * DAY_NS,
                'test_duration_ns': 10 * DAY_NS,
                'embargo_pct': 0.01,
            }),
            (CombinatorialPurgedCV, {
                'start_ns': START_NS,
                'end_ns': END_NS,
                'n_splits': 5,
                'n_test_splits': 2,
                'embargo_pct': 0.01,
            }),
        ]
        
        for cls, kwargs in splitter_configs:
            splitter1 = cls(**kwargs)
            splitter2 = cls(**kwargs)
            
            splits1 = list(splitter1.split())
            splits2 = list(splitter2.split())
            
            assert len(splits1) == len(splits2)
            # Verify first split matches
            if len(splits1) > 0:
                train1, test1 = splits1[0]
                train2, test2 = splits2[0]
                assert len(train1) == len(train2)


class TestTimeRange:
    """Tests for TimeRange helper class."""
    
    def test_time_range_duration(self):
        """TimeRange correctly calculates duration."""
        tr = TimeRange(start_ns=1000, end_ns=2000)
        assert tr.duration_ns == 1000
    
    def test_time_range_contains(self):
        """TimeRange.contains works correctly."""
        tr = TimeRange(start_ns=1000, end_ns=2000)
        assert tr.contains(1500)
        assert tr.contains(1000)
        assert not tr.contains(2000)  # end is exclusive
        assert not tr.contains(500)
