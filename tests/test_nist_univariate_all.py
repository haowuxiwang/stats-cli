"""Complete NIST StRD Univariate Statistics Coverage.

All 9 NIST StRD univariate datasets with certified mean and standard deviation.
Data: https://www.itl.nist.gov/div898/strd/univ/data/{name}.dat
"""

import json
import os

from stats_engine.descriptive import descriptive

DATA_FILE = os.path.join(os.path.dirname(__file__), "nist_univ_data.json")


def load_data():
    with open(DATA_FILE) as f:
        return json.load(f)


NIST = load_data()


class TestNISTUnivariateAll:
    """All 9 NIST StRD univariate statistics datasets."""

    def test_numacc1(self):
        """NumAcc1: 3 obs, mean=10000002 (exact), std=1 (exact)"""
        d = NIST["NumAcc1"]
        result = descriptive(values=d["values"])
        assert abs(result["mean"] - 10000002.0) < 1e-4
        assert abs(result["std"] - 1.0) < 1e-4

    def test_numacc2(self):
        """NumAcc2: 1001 obs, mean=1.2, std=0.1"""
        d = NIST["NumAcc2"]
        result = descriptive(values=d["values"])
        assert abs(result["mean"] - 1.2) < 1e-4
        assert abs(result["std"] - 0.1) < 1e-3

    def test_numacc3(self):
        """NumAcc3: 1001 obs, mean=1000000.2, std=0.1"""
        d = NIST["NumAcc3"]
        result = descriptive(values=d["values"])
        assert abs(result["mean"] - 1000000.2) < 1e-3
        assert abs(result["std"] - 0.1) < 1e-3

    def test_numacc4(self):
        """NumAcc4: 1001 obs, mean=10000000.2, std=0.1"""
        d = NIST["NumAcc4"]
        result = descriptive(values=d["values"])
        assert abs(result["mean"] - 10000000.2) < 1e-3
        assert abs(result["std"] - 0.1) < 0.01

    def test_lew(self):
        """Lew: 200 obs, mean=-177.435, std=277.332"""
        d = NIST["Lew"]
        result = descriptive(values=d["values"])
        assert abs(result["mean"] - (-177.435)) < 0.01
        assert abs(result["std"] - 277.332) < 0.1

    def test_mavro(self):
        """Mavro: 50 obs, mean=2.001856, std=0.000429"""
        d = NIST["Mavro"]
        result = descriptive(values=d["values"])
        assert abs(result["mean"] - 2.001856) < 1e-5
        assert abs(result["std"] - 0.000429) < 1e-4

    def test_michelso(self):
        """Michelso: 100 obs, mean=299.8524, std=0.07901"""
        d = NIST["Michelso"]
        result = descriptive(values=d["values"])
        assert abs(result["mean"] - 299.8524) < 0.01
        assert abs(result["std"] - 0.07901) < 0.001

    def test_pidigits(self):
        """PiDigits: 5000 obs, mean=4.5348, std=2.8673"""
        d = NIST["PiDigits"]
        result = descriptive(values=d["values"])
        assert abs(result["mean"] - 4.5348) < 0.01
        assert abs(result["std"] - 2.8673) < 0.01

    def test_lottery(self):
        """Lottery: 218 obs, mean=518.959, std=291.700"""
        d = NIST["Lottery"]
        result = descriptive(values=d["values"])
        assert abs(result["mean"] - 518.959) < 0.1
        assert abs(result["std"] - 291.700) < 0.1
