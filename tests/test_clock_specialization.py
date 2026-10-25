import importlib.util
from pathlib import Path
import unittest

import torch
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('clock_specialization', ROOT / 'scripts/analyze-clock-specialization.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class ClockSelectionTests(unittest.TestCase):
    def test_validation_choice_is_fixed_when_test_prefers_opposite(self):
        dates = pd.to_datetime(['2024-01-02 09:30', '2024-01-02 10:00', '2024-02-02 09:30', '2024-02-02 10:00'])
        def frame(values):
            return pd.DataFrame({'ic': values, 'rank_ic': values}, index=dates)
        validation = {'base': frame([.1, .3, .1, .3]), 'other': frame([.4, .2, .4, .2])}
        choices = module.choose_by_validation(validation, 'base')
        self.assertEqual(choices, {'09:30': 'other', '10:00': 'base'})
        test = {'base': frame([.9, .1, .9, .1]), 'other': frame([.2, .8, .2, .8])}
        result = module.apply_choice(test, choices)
        self.assertAlmostEqual(result.ic.mean(), .15)
        with self.assertRaises(ValueError):
            module.apply_choice(test, {'09:30': 'other'})

    def test_held_month_does_not_select_its_own_winner_and_misalignment_rejected(self):
        dates = pd.to_datetime(['2024-01-02 09:30', '2024-02-02 09:30', '2024-03-02 09:30'])
        frames = {'base': pd.DataFrame({'ic': [.2, .2, .2], 'rank_ic': [.2, .2, .2]}, index=dates),
                  'other': pd.DataFrame({'ic': [.9, .1, .1], 'rank_ic': [.9, .1, .1]}, index=dates)}
        result = module.held_months(frames, 'base')
        self.assertEqual(result['months'][0]['choices_from_other_validation_months'], {'09:30': 'base'})
        self.assertAlmostEqual(result['summary']['ic_mean'], (.2 + .1 + .1) / 3)
        frames['other'] = frames['other'].iloc[1:]
        with self.assertRaises(ValueError):
            module.choose_by_validation(frames, 'base')


if __name__ == '__main__':
    unittest.main()
