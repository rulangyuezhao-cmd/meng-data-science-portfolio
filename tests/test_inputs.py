"""Data-contract checks catch silent label, schema and time-axis errors."""
import hashlib
import json
import unittest
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT/'data'

class InputContracts(unittest.TestCase):
    def test_snapshots_match_manifest(self):
        for item in json.loads((DATA/'manifest.json').read_text()):
            with self.subTest(file=item['file']):
                p=DATA/item['file']
                self.assertEqual(hashlib.sha256(p.read_bytes()).hexdigest(),item['sha256'])
                d=pd.read_csv(p)
                self.assertEqual(len(d),item['rows'])
                self.assertEqual(list(d),item['columns'])

    def test_anomaly_labels_are_binary_and_present(self):
        d=pd.read_csv(DATA/'landsat.csv')
        self.assertEqual(set(d.isanomaly.unique()),{0.,1.})
        # The supplied snapshot has only 71 anomalies; preserve the actual imbalance.
        self.assertEqual(d.isanomaly.value_counts().to_dict(),{0.:5732,1.:71})
        self.assertEqual(d.drop(columns='isanomaly').shape[1],36)

    def test_sales_dates_are_columns(self):
        d=pd.read_csv(DATA/'sales-by-family.csv')
        self.assertTrue(d.family.is_unique)
        dates=pd.to_datetime(d.columns[1:])
        self.assertTrue(dates.is_monotonic_increasing)
        self.assertGreater(len(dates),100)
        self.assertEqual(len(d),33)

    def test_housing_months_are_contiguous(self):
        d=pd.read_csv(DATA/'toronto-housing.csv',parse_dates=['date'])
        self.assertEqual(len(d),192)
        self.assertTrue(d.date.is_unique)
        self.assertEqual(list(d.date),list(pd.date_range(d.date.min(),periods=len(d),freq='MS')))
        self.assertTrue((d.price>0).all())

    def test_classification_excludes_target_proxies(self):
        d=pd.read_csv(DATA/'salary-2022.csv')
        self.assertNotIn('Q29',d)
        self.assertNotIn('Q29_Encoded',d)
        self.assertEqual(d.Q29_buckets.nunique(),15)
        self.assertFalse(d.Q29_buckets.isna().any())

    def test_recovered_skill_table_is_complete(self):
        d=pd.read_csv(DATA/'skill-features.csv')
        self.assertEqual(d.shape,(50,11))
        self.assertTrue(d.Skill.is_unique)
        self.assertFalse(d.isna().any().any())
        self.assertTrue(d.frequency.between(0,1000).all())
        self.assertTrue(d.iloc[:,2:].isin([0,1]).all().all())

    def test_synthetic_retail_has_no_customer_identifiers(self):
        d=pd.read_csv(DATA/'retail-annual.csv')
        self.assertEqual(len(d),1600)
        self.assertEqual(set(d.response),{0,1})
        self.assertFalse(any('id' in c.lower() or 'name' in c.lower() for c in d))
        self.assertTrue(d.drop(columns='response').isna().any().any())

if __name__=='__main__': unittest.main()
