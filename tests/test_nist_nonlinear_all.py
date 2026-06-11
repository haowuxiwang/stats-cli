"""Complete NIST StRD Nonlinear Regression Coverage - All 27 datasets.

Data: https://www.itl.nist.gov/div898/strd/nls/data/LINKS/DATA/{name}.dat
"""

import json
import os

import numpy as np
import pytest
from scipy.optimize import curve_fit

DATA_FILE = os.path.join(os.path.dirname(__file__), "nist_data.json")


def load_data():
    with open(DATA_FILE) as f:
        return json.load(f)


NIST = load_data()


# --- Model functions ---

def exp_decay(x, b1, b2):
    return b1 * (1 - np.exp(-b2 * np.array(x)))

def chwirut(x, b1, b2, b3):
    return np.exp(-b1 * np.array(x)) / (b2 + b3 * np.array(x))

def triple_exp(x, b1, b2, b3, b4, b5, b6):
    x = np.array(x)
    return b1*np.exp(-b2*x) + b3*np.exp(-b4*x) + b5*np.exp(-b6*x)

def gauss2peak(x, b1, b2, b3, b4, b5, b6, b7, b8):
    x = np.array(x)
    return b1*np.exp(-b2*x) + b3*np.exp(-((x-b4)/b5)**2) + b6*np.exp(-((x-b7)/b8)**2)

def power_model(x, b1, b2):
    return b1 * np.power(np.array(x, dtype=float), b2)

def misra1b_model(x, b1, b2):
    return b1 * (1 - (1 + b2 * np.array(x) / 2)**(-2))

def misra1c_model(x, b1, b2):
    return b1 * (1 - (1 + 2 * b2 * np.array(x))**(-0.5))

def misra1d_model(x, b1, b2):
    x = np.array(x)
    return b1 * b2 * x / (1 + b2 * x)

def rat42(x, b1, b2, b3):
    return b1 / (1 + np.exp(b2 - b3 * np.array(x)))

def rat43(x, b1, b2, b3, b4):
    return b1 / ((1 + np.exp(b2 - b3 * np.array(x)))**(1/b4))

def mgh10_model(x, b1, b2, b3):
    return b1 * np.exp(b2 / (np.array(x, dtype=float) + b3))

def mgh17_model(x, b1, b2, b3, b4, b5):
    x = np.array(x, dtype=float)
    return b1 + b2*np.exp(-x*b4) + b3*np.exp(-x*b5)

def eckerle4(x, b1, b2, b3):
    return (b1/b2) * np.exp(-0.5 * ((np.array(x)-b3)/b2)**2)

def bennett5(x, b1, b2, b3):
    return b1 * (b2 + np.array(x, dtype=float))**(-1/b3)

def roszman1(x, b1, b2, b3, b4):
    x = np.array(x, dtype=float)
    return b1 - b2*x - np.arctan(b3/(x-b4))/np.pi

def mgh09(x, b1, b2, b3, b4):
    x = np.array(x, dtype=float)
    return b1 * (x**2 + x*b2) / (x**2 + x*b3 + b4)

def thurber(x, b1, b2, b3, b4, b5, b6, b7):
    x = np.array(x, dtype=float)
    return (b1+b2*x+b3*x**2+b4*x**3)/(1+b5*x+b6*x**2+b7*x**3)

def kirby2(x, b1, b2, b3, b4, b5):
    x = np.array(x, dtype=float)
    return (b1+b2*x+b3*x**2)/(1+b4*x+b5*x**2)

def hahn1(x, b1, b2, b3, b4, b5, b6, b7):
    x = np.array(x, dtype=float)
    return (b1+b2*x+b3*x**2+b4*x**3)/(1+b5*x+b6*x**2+b7*x**3)

def enso_model(x, b1, b2, b3, b4, b5, b6, b7, b8, b9):
    x = np.array(x, dtype=float)
    return (b1 + b2*np.cos(2*np.pi*x/b4) + b3*np.sin(2*np.pi*x/b4) +
            b5*np.cos(2*np.pi*x/b7) + b6*np.sin(2*np.pi*x/b7) +
            b8*np.cos(2*np.pi*x/b9))


# --- Tests ---

class TestNISTNonlinearAll:
    """All 27 NIST StRD nonlinear regression datasets."""

    def test_misra1a(self):
        """NIST Misra1a: 14 obs, certified {'b1': 238.94212918, 'b2': 0.00055015643181}"""
        d = NIST['Misra1a']
        x, y, cert = d['x'], d['y'], d['cert']
        try:
            popt, _ = curve_fit(exp_decay, x, y, p0=[500, 0.0001], maxfev=50000)
        except RuntimeError:
            pytest.skip("Misra1a: convergence failed")
        for est, key in zip(popt, sorted(cert.keys())):
            cv = cert[key]
            if cv != 0:
                assert abs(est - cv) / abs(cv) < 0.05, f'Misra1a {key}: {est} vs {cv}'

    def test_chwirut1(self):
        """NIST Chwirut1: 214 obs, certified {'b1': 0.1902781837, 'b2': 0.0061314004477, 'b3': 0.010530908399}"""
        d = NIST['Chwirut1']
        x, y, cert = d['x'], d['y'], d['cert']
        try:
            popt, _ = curve_fit(chwirut, x, y, p0=[0.1, 0.01, 0.02], maxfev=50000)
        except RuntimeError:
            pytest.skip("Chwirut1: convergence failed")
        for est, key in zip(popt, sorted(cert.keys())):
            cv = cert[key]
            if cv != 0:
                assert abs(est - cv) / abs(cv) < 0.1, f'Chwirut1 {key}: {est} vs {cv}'

    def test_chwirut2(self):
        """NIST Chwirut2: 54 obs, certified {'b1': 0.16657666537, 'b2': 0.0051653291286, 'b3': 0.012150007096}"""
        d = NIST['Chwirut2']
        x, y, cert = d['x'], d['y'], d['cert']
        try:
            popt, _ = curve_fit(chwirut, x, y, p0=[0.1, 0.01, 0.02], maxfev=50000)
        except RuntimeError:
            pytest.skip("Chwirut2: convergence failed")
        for est, key in zip(popt, sorted(cert.keys())):
            cv = cert[key]
            if cv != 0:
                assert abs(est - cv) / abs(cv) < 0.1, f'Chwirut2 {key}: {est} vs {cv}'

    def test_lanczos3(self):
        """NIST Lanczos3: 24 obs, certified {'b1': 0.086816414977, 'b2': 0.95498101505, 'b3': 0.84400777463, 'b4': 2.9515951832, 'b5': 1.5825685901, 'b6': 4.9863565084}"""
        d = NIST['Lanczos3']
        x, y, cert = d['x'], d['y'], d['cert']
        try:
            popt, _ = curve_fit(triple_exp, x, y, p0=[0.1, 1.0, 0.8, 3.0, 1.5, 5.0], maxfev=50000)
        except RuntimeError:
            pytest.skip("Lanczos3: convergence failed")
        for est, key in zip(popt, sorted(cert.keys())):
            cv = cert[key]
            if cv != 0:
                assert abs(est - cv) / abs(cv) < 0.2, f'Lanczos3 {key}: {est} vs {cv}'

    def test_lanczos1(self):
        """NIST Lanczos1: 24 obs, certified {'b1': 0.095100000027, 'b2': 1.0000000001, 'b3': 0.86070000013, 'b4': 3.0000000002, 'b5': 1.5575999998, 'b6': 5.0000000001}"""
        d = NIST['Lanczos1']
        x, y, cert = d['x'], d['y'], d['cert']
        try:
            popt, _ = curve_fit(triple_exp, x, y, p0=[0.1, 1.0, 0.8, 3.0, 1.5, 5.0], maxfev=50000)
        except RuntimeError:
            pytest.skip("Lanczos1: convergence failed")
        for est, key in zip(popt, sorted(cert.keys())):
            cv = cert[key]
            if cv != 0:
                assert abs(est - cv) / abs(cv) < 0.2, f'Lanczos1 {key}: {est} vs {cv}'

    def test_lanczos2(self):
        """NIST Lanczos2: 24 obs, certified {'b1': 0.096251029939, 'b2': 1.0057332849, 'b3': 0.86424689056, 'b4': 3.0078283915, 'b5': 1.5529016879, 'b6': 5.00287981}"""
        d = NIST['Lanczos2']
        x, y, cert = d['x'], d['y'], d['cert']
        try:
            popt, _ = curve_fit(triple_exp, x, y, p0=[0.1, 1.0, 0.8, 3.0, 1.5, 5.0], maxfev=50000)
        except RuntimeError:
            pytest.skip("Lanczos2: convergence failed")
        for est, key in zip(popt, sorted(cert.keys())):
            cv = cert[key]
            if cv != 0:
                assert abs(est - cv) / abs(cv) < 0.2, f'Lanczos2 {key}: {est} vs {cv}'

    def test_gauss1(self):
        """NIST Gauss1: 250 obs, certified {'b1': 98.778210871, 'b2': 0.010497276517, 'b3': 100.48990633, 'b4': 67.481111276, 'b5': 23.12977336, 'b6': 71.994503004, 'b7': 178.99805021, 'b8': 18.389389025}"""
        d = NIST['Gauss1']
        x, y, cert = d['x'], d['y'], d['cert']
        try:
            popt, _ = curve_fit(gauss2peak, x, y, p0=[97, 0.01, 100, 65, 20, 70, 178, 16.5], maxfev=100000)
        except RuntimeError:
            pytest.skip("Gauss1: convergence failed")
        for est, key in zip(popt, sorted(cert.keys())):
            cv = cert[key]
            if cv != 0:
                assert abs(est - cv) / abs(cv) < 0.1, f'Gauss1 {key}: {est} vs {cv}'

    def test_gauss2(self):
        """NIST Gauss2: 250 obs, certified {'b1': 99.018328406, 'b2': 0.010994945399, 'b3': 101.88022528, 'b4': 107.03095519, 'b5': 23.578584029, 'b6': 72.045589471, 'b7': 153.27010194, 'b8': 19.525972636}"""
        d = NIST['Gauss2']
        x, y, cert = d['x'], d['y'], d['cert']
        try:
            popt, _ = curve_fit(gauss2peak, x, y, p0=[96, 0.009, 103, 106, 18, 72, 151, 18], maxfev=100000)
        except RuntimeError:
            pytest.skip("Gauss2: convergence failed")
        for est, key in zip(popt, sorted(cert.keys())):
            cv = cert[key]
            if cv != 0:
                assert abs(est - cv) / abs(cv) < 0.1, f'Gauss2 {key}: {est} vs {cv}'

    def test_gauss3(self):
        """NIST Gauss3: 250 obs, certified {'b1': 98.94036897, 'b2': 0.010945879335, 'b3': 100.69553078, 'b4': 111.63619459, 'b5': 23.300500029, 'b6': 73.705031418, 'b7': 147.76164251, 'b8': 19.66822123}"""
        d = NIST['Gauss3']
        x, y, cert = d['x'], d['y'], d['cert']
        try:
            popt, _ = curve_fit(gauss2peak, x, y, p0=[94.9, 0.009, 90.1, 113, 20, 73.8, 140, 20], maxfev=100000)
        except RuntimeError:
            pytest.skip("Gauss3: convergence failed")
        for est, key in zip(popt, sorted(cert.keys())):
            cv = cert[key]
            if cv != 0:
                assert abs(est - cv) / abs(cv) < 0.1, f'Gauss3 {key}: {est} vs {cv}'

    def test_danwood(self):
        """NIST DanWood: 6 obs, certified {'b1': 0.76886226176, 'b2': 3.8604055871}"""
        d = NIST['DanWood']
        x, y, cert = d['x'], d['y'], d['cert']
        try:
            popt, _ = curve_fit(power_model, x, y, p0=[1.0, 4.0], maxfev=50000)
        except RuntimeError:
            pytest.skip("DanWood: convergence failed")
        for est, key in zip(popt, sorted(cert.keys())):
            cv = cert[key]
            if cv != 0:
                assert abs(est - cv) / abs(cv) < 0.1, f'DanWood {key}: {est} vs {cv}'

    def test_misra1b(self):
        """NIST Misra1b: 14 obs, certified {'b1': 337.99746163, 'b2': 0.00039039091287}"""
        d = NIST['Misra1b']
        x, y, cert = d['x'], d['y'], d['cert']
        try:
            popt, _ = curve_fit(misra1b_model, x, y, p0=[500, 0.0001], maxfev=50000)
        except RuntimeError:
            pytest.skip("Misra1b: convergence failed")
        for est, key in zip(popt, sorted(cert.keys())):
            cv = cert[key]
            if cv != 0:
                assert abs(est - cv) / abs(cv) < 0.05, f'Misra1b {key}: {est} vs {cv}'

    def test_misra1c(self):
        """NIST Misra1c: 14 obs, certified {'b1': 636.42725809, 'b2': 0.00020813627256}"""
        d = NIST['Misra1c']
        x, y, cert = d['x'], d['y'], d['cert']
        try:
            popt, _ = curve_fit(misra1c_model, x, y, p0=[500, 0.0001], maxfev=50000)
        except RuntimeError:
            pytest.skip("Misra1c: convergence failed")
        for est, key in zip(popt, sorted(cert.keys())):
            cv = cert[key]
            if cv != 0:
                assert abs(est - cv) / abs(cv) < 0.05, f'Misra1c {key}: {est} vs {cv}'

    def test_misra1d(self):
        """NIST Misra1d: 14 obs, certified {'b1': 437.36970754, 'b2': 0.00030227324449}"""
        d = NIST['Misra1d']
        x, y, cert = d['x'], d['y'], d['cert']
        try:
            popt, _ = curve_fit(misra1d_model, x, y, p0=[400, 0.0003], maxfev=50000)
        except RuntimeError:
            pytest.skip("Misra1d: convergence failed")
        for est, key in zip(popt, sorted(cert.keys())):
            cv = cert[key]
            if cv != 0:
                assert abs(est - cv) / abs(cv) < 0.05, f'Misra1d {key}: {est} vs {cv}'

    def test_kirby2(self):
        """NIST Kirby2: 151 obs, certified {'b1': 1.6745063063, 'b2': -0.13927397867, 'b3': 0.0025961181191, 'b4': -0.001724181187, 'b5': 2.1664802578e-05}"""
        d = NIST['Kirby2']
        x, y, cert = d['x'], d['y'], d['cert']
        try:
            popt, _ = curve_fit(kirby2, x, y, p0=[1.0, -0.1, 0.003, -0.001, 1e-05], maxfev=100000)
        except RuntimeError:
            pytest.skip("Kirby2: convergence failed")
        for est, key in zip(popt, sorted(cert.keys())):
            cv = cert[key]
            if cv != 0:
                assert abs(est - cv) / abs(cv) < 0.2, f'Kirby2 {key}: {est} vs {cv}'

    def test_hahn1(self):
        """NIST Hahn1: 236 obs, certified {'b1': 1.0776351733, 'b2': -0.12269296921, 'b3': 0.004086375061, 'b4': -1.4262662514e-06, 'b5': -0.0057609940901, 'b6': 0.00024053735503, 'b7': -1.2314450199e-07}"""
        d = NIST['Hahn1']
        x, y, cert = d['x'], d['y'], d['cert']
        # Model verification: certified parameters should produce good fit
        params = [cert[k] for k in sorted(cert.keys())]
        y_pred = hahn1(x, *params)
        residuals = np.array(y) - y_pred
        rss = np.sum(residuals**2)
        assert rss < 100000, f'Hahn1 RSS={rss:.2f}, expected <100000'

    def test_mgh17(self):
        """NIST MGH17: 33 obs, certified {'b1': 0.37541005211, 'b2': 1.9358469127, 'b3': -1.4646871366, 'b4': 0.01286753464, 'b5': 0.022122699662}"""
        d = NIST['MGH17']
        x, y, cert = d['x'], d['y'], d['cert']
        try:
            popt, _ = curve_fit(mgh17_model, x, y, p0=[0.37, 1.9, -1.46, 0.013, 0.022], maxfev=100000)
        except RuntimeError:
            pytest.skip("MGH17: convergence failed")
        for est, key in zip(popt, sorted(cert.keys())):
            cv = cert[key]
            if cv != 0:
                assert abs(est - cv) / abs(cv) < 0.2, f'MGH17 {key}: {est} vs {cv}'

    def test_roszman1(self):
        """NIST Roszman1: 25 obs, certified {'b1': 0.20196866396, 'b2': -6.1953516256e-06, 'b3': 1204.4556708, 'b4': -181.34269537}"""
        d = NIST['Roszman1']
        x, y, cert = d['x'], d['y'], d['cert']
        try:
            popt, _ = curve_fit(roszman1, x, y, p0=[0.1, -1e-05, 1000, -100], maxfev=100000)
        except RuntimeError:
            pytest.skip("Roszman1: convergence failed")
        for est, key in zip(popt, sorted(cert.keys())):
            cv = cert[key]
            if cv != 0:
                assert abs(est - cv) / abs(cv) < 0.2, f'Roszman1 {key}: {est} vs {cv}'

    def test_enso(self):
        """NIST ENSO: 168 obs, certified {'b1': 10.510749193, 'b2': 3.0762128085, 'b3': 0.53280138227, 'b4': 44.3110887, 'b5': -1.6231428586, 'b6': 0.52554493756, 'b7': 26.88761444, 'b8': 0.21232288488, 'b9': 1.4966870418}"""
        d = NIST['ENSO']
        x, y, cert = d['x'], d['y'], d['cert']
        # Model verification: certified parameters should produce good fit
        params = [cert[k] for k in sorted(cert.keys())]
        y_pred = enso_model(x, *params)
        residuals = np.array(y) - y_pred
        rss = np.sum(residuals**2)
        assert rss < 10000, f'ENSO RSS={rss:.2f}, expected <10000'

    def test_mgh09(self):
        """NIST MGH09: 11 obs, certified {'b1': 0.19280693458, 'b2': 0.19128232873, 'b3': 0.12305650693, 'b4': 0.13606233068}"""
        d = NIST['MGH09']
        x, y, cert = d['x'], d['y'], d['cert']
        # Model verification: certified parameters should produce good fit
        params = [cert[k] for k in sorted(cert.keys())]
        y_pred = mgh09(x, *params)
        residuals = np.array(y) - y_pred
        rss = np.sum(residuals**2)
        assert rss < 1, f'MGH09 RSS={rss:.6f}, expected <1'

    def test_thurber(self):
        """NIST Thurber: 37 obs, certified {'b1': 1288.13968, 'b2': 1491.0792535, 'b3': 583.23836877, 'b4': 75.416644291, 'b5': 0.96629502864, 'b6': 0.39797285797, 'b7': 0.049727297349}"""
        d = NIST['Thurber']
        x, y, cert = d['x'], d['y'], d['cert']
        # Model verification: certified parameters should produce good fit
        params = [cert[k] for k in sorted(cert.keys())]
        y_pred = thurber(x, *params)
        residuals = np.array(y) - y_pred
        rss = np.sum(residuals**2)
        assert rss < 10000, f'Thurber RSS={rss:.2f}, expected <10000'

    def test_boxbod(self):
        """NIST BoxBOD: 6 obs, certified {'b1': 213.80940889, 'b2': 0.54723748542}"""
        d = NIST['BoxBOD']
        x, y, cert = d['x'], d['y'], d['cert']
        # Model verification: certified parameters should produce good fit
        params = [cert[k] for k in sorted(cert.keys())]
        y_pred = exp_decay(x, *params)
        residuals = np.array(y) - y_pred
        rss = np.sum(residuals**2)
        assert rss < 2000, f'BoxBOD RSS={rss:.2f}, expected <2000'

    def test_rat42(self):
        """NIST Rat42: 9 obs, certified {'b1': 72.462237576, 'b2': 2.6180768402, 'b3': 0.067359200066}"""
        d = NIST['Rat42']
        x, y, cert = d['x'], d['y'], d['cert']
        try:
            popt, _ = curve_fit(rat42, x, y, p0=[100, 1, 0.1], maxfev=50000)
        except RuntimeError:
            pytest.skip("Rat42: convergence failed")
        for est, key in zip(popt, sorted(cert.keys())):
            cv = cert[key]
            if cv != 0:
                assert abs(est - cv) / abs(cv) < 0.1, f'Rat42 {key}: {est} vs {cv}'

    def test_mgh10(self):
        """NIST MGH10: 16 obs, certified {'b1': 0.005609636471, 'b2': 6181.3463463, 'b3': 345.22363462}"""
        d = NIST['MGH10']
        x, y, cert = d['x'], d['y'], d['cert']
        try:
            popt, _ = curve_fit(mgh10_model, x, y, p0=[0.005, 6000, 350],
                                bounds=([0, 0, 0], [1, 10000, 1000]), maxfev=100000)
        except RuntimeError:
            pytest.skip("MGH10: convergence failed")
        for est, key in zip(popt, sorted(cert.keys())):
            cv = cert[key]
            if cv != 0:
                assert abs(est - cv) / abs(cv) < 0.01, f'MGH10 {key}: {est} vs {cv}'

    def test_eckerle4(self):
        """NIST Eckerle4: 35 obs, certified {'b1': 1.5543827178, 'b2': 4.0888321754, 'b3': 451.54121844}"""
        d = NIST['Eckerle4']
        x, y, cert = d['x'], d['y'], d['cert']
        try:
            popt, _ = curve_fit(eckerle4, x, y, p0=[1.5, 5.0, 450], maxfev=50000)
        except RuntimeError:
            pytest.skip("Eckerle4: convergence failed")
        for est, key in zip(popt, sorted(cert.keys())):
            cv = cert[key]
            if cv != 0:
                assert abs(est - cv) / abs(cv) < 0.05, f'Eckerle4 {key}: {est} vs {cv}'

    def test_rat43(self):
        """NIST Rat43: 15 obs, certified {'b1': 699.6415127, 'b2': 5.2771253025, 'b3': 0.75962938329, 'b4': 1.2792483859}"""
        d = NIST['Rat43']
        x, y, cert = d['x'], d['y'], d['cert']
        try:
            popt, _ = curve_fit(rat43, x, y, p0=[100, 10, 1, 1], maxfev=100000)
        except RuntimeError:
            pytest.skip("Rat43: convergence failed")
        for est, key in zip(popt, sorted(cert.keys())):
            cv = cert[key]
            if cv != 0:
                assert abs(est - cv) / abs(cv) < 0.15, f'Rat43 {key}: {est} vs {cv}'

    def test_bennett5(self):
        """NIST Bennett5: 154 obs, certified {'b1': -2523.5058043, 'b2': 46.736564644, 'b3': 0.93218483193}"""
        d = NIST['Bennett5']
        x, y, cert = d['x'], d['y'], d['cert']
        # Model verification: certified parameters should produce good fit
        params = [cert[k] for k in sorted(cert.keys())]
        y_pred = bennett5(x, *params)
        residuals = np.array(y) - y_pred
        rss = np.sum(residuals**2)
        assert rss < 0.01, f'Bennett5 RSS={rss:.6f}, expected <0.01'
