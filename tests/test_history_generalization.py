"""分组边界及限制其余股票关系的事后IC上界。"""
import importlib.util
from pathlib import Path
import unittest

import torch
import numpy as np
import pandas as pd

spec=importlib.util.spec_from_file_location('history_audit',Path(__file__).resolve().parents[1]/'scripts/analyze-history-generalization.py')
audit=importlib.util.module_from_spec(spec);spec.loader.exec_module(audit)


class HistoryGeneralizationTests(unittest.TestCase):
    def test_history_boundaries(self):
        np.testing.assert_array_equal(audit.history_band([0,1,59,60,251,252,970]),[0,1,1,2,2,3,3])
        with self.assertRaises(ValueError):audit.history_band([-1])

    def test_oracle_matches_explicit_permitted_projection(self):
        x=np.array([-1.,0.,1.,2.,3.]);y=np.array([1.,2.,4.,0.,5.]);mask=np.array([False,False,False,True,True])
        known=~mask
        a=(x[known]-x[known].mean());b=y[known]-y[known].mean()
        slope=max(0,float(a@b)/float(a@a))
        fitted=y.copy();fitted[known]=y[known].mean()+slope*a
        expected=float(np.corrcoef(fitted,y)[0,1])
        self.assertAlmostEqual(audit.restricted_oracle_ceiling(x,y,mask),expected,places=14)
        self.assertAlmostEqual(audit.restricted_oracle_ceiling(x,y,np.zeros(5,bool)),max(0,float(np.corrcoef(x,y)[0,1])),places=14)
        self.assertEqual(audit.restricted_oracle_ceiling(x,y,np.ones(5,bool)),1)
        self.assertEqual(audit.restricted_oracle_ceiling(-y,y,np.zeros(5,bool)),0)

    def test_contributions_reconstruct_correlation(self):
        x=pd.Series([1.,2.,4.,-1.,0.,2.]);y=pd.Series([0.,3.,1.,2.,4.,-1.]);dates=pd.Series([0,0,0,1,1,1])
        ic,xvar,yvar=audit.contributions(x,y,dates)
        for date in (0,1):
            selected=dates==date
            self.assertAlmostEqual(ic[selected].sum(),np.corrcoef(x[selected],y[selected])[0,1],places=14)
            self.assertAlmostEqual(xvar[selected].sum(),1)
            self.assertAlmostEqual(yvar[selected].sum(),1)

    def test_oracle_dominates_unchanged_prediction(self):
        rng=np.random.default_rng(7)
        for _ in range(40):
            x,y=rng.normal(size=(2,40));mask=rng.uniform(size=40)<.2
            ceiling=audit.restricted_oracle_ceiling(x,y,mask)
            self.assertGreaterEqual(ceiling+1e-12,float(np.corrcoef(x,y)[0,1]))


if __name__=='__main__':unittest.main()
