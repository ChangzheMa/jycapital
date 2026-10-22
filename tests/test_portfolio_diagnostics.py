"""可手算的持仓、缺失标签、成本单位和日边界测试。"""

from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'evaluation'))
from portfolio_diagnostics import portfolio_diagnostics


class PortfolioDiagnosticsTests(unittest.TestCase):
    def example(self):
        keys=pd.DataFrame([(pd.Timestamp(day+' '+clock),stock) for day in ('2024-01-02','2024-01-03')
                           for clock in ('10:00','10:30') for stock in ('A','B')],columns=['date','instrument'])
        predictions=keys.assign(factor=[1.,-1.,-1.,1.,1.,-1.,1.,-1.])
        labels=keys.assign(vwap_return=[.02,-.02,-.01,.01,.01,-.01,.005,-.005])
        return predictions,labels

    def test_turnover_cost_and_daily_open_close_are_not_confused(self):
        predictions,labels=self.example()
        summary,_,sections,days=portfolio_diagnostics(predictions,labels,costs_bps=(0.,10.))
        np.testing.assert_allclose(days.gross_label_return,[.03,.015])
        np.testing.assert_allclose(days.rebalance_notional,[2.,0.])
        np.testing.assert_allclose(days.traded_notional,[4.,2.])
        self.assertAlmostEqual(summary['one_way_intraday_turnover_mean'],.5)
        self.assertAlmostEqual(summary['cost_scenarios'][1]['daily_net_label_return_mean'],(.026+.013)/2)
        self.assertAlmostEqual(summary['zero_mean_cost_threshold_bps'],75.)
        self.assertEqual(sections.one_way_intraday_turnover.isna().sum(),2)

    def test_missing_future_labels_do_not_change_portfolios_or_reweight(self):
        predictions,labels=self.example()
        _,_,original,days=portfolio_diagnostics(predictions,labels)
        labels.loc[0,'vwap_return']=np.nan
        summary,_,missing,changed=portfolio_diagnostics(predictions,labels)
        np.testing.assert_array_equal(original.rebalance_notional,missing.rebalance_notional)
        np.testing.assert_array_equal(days.traded_notional,changed.traded_notional)
        self.assertAlmostEqual(missing.iloc[0].gross_label_return,.01)
        self.assertAlmostEqual(missing.iloc[0].missing_label_mass,.5)
        self.assertAlmostEqual(summary['known_held_label_mass_minimum'],.5)

    def test_order_invariance_and_coverage_errors(self):
        predictions,labels=self.example()
        expected=portfolio_diagnostics(predictions,labels)
        actual=portfolio_diagnostics(predictions.sample(frac=1,random_state=3),labels.sample(frac=1,random_state=5))
        pd.testing.assert_frame_equal(expected[2],actual[2])
        pd.testing.assert_frame_equal(expected[3],actual[3])
        with self.assertRaises(ValueError):
            portfolio_diagnostics(predictions.iloc[:-1],labels)

    def test_constant_scores_do_not_report_zero_risk_success(self):
        predictions,labels=self.example()
        predictions['factor']=0.
        summary,_,_,_=portfolio_diagnostics(predictions,labels)
        self.assertEqual(summary['undefined_sections'],4)
        self.assertEqual(summary['defined_days'],0)
        self.assertIsNone(summary['daily_label_return_ir'])
        self.assertIsNone(summary['zero_mean_cost_threshold_bps'])

    def test_complete_labels_are_not_reported_missing_due_to_fraction_rounding(self):
        keys=pd.DataFrame([(pd.Timestamp(day),str(stock)) for day in ('2024-01-02','2024-01-03')
                           for stock in range(20)],columns=['date','instrument'])
        predictions=keys.assign(factor=([-1.]*3+[0.]*14+[1.]*3)*2)
        labels=keys.assign(vwap_return=.01)
        summary,_,sections,_=portfolio_diagnostics(predictions,labels)
        self.assertEqual(summary['missing_held_label_sections'],0)
        np.testing.assert_array_equal(sections.missing_label_mass,np.zeros(2))
        np.testing.assert_array_equal(sections.known_label_mass,np.ones(2))


if __name__=='__main__':
    unittest.main()
