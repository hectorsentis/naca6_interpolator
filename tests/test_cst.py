"""python -B -m unittest discover -s tests -v"""

import contextlib
import csv
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import cst
from naca6 import naca6


class TestCst(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.case = cst.generate_naca_camber_thickness("63-412", 150)
        cls.cc = cst.fit_cst(cls.case["xi"], cls.case["yc"], 5, 1, 1)
        cls.ct = cst.fit_cst(cls.case["xi"], cls.case["t"], 5)
        cls.result = cst.build_cst_result(cls.case, cls.cc, cls.ct)

    def test_exact_endpoints_and_nominal_chord(self):
        for data in (self.case, self.result):
            np.testing.assert_array_equal(data["yc"][[0,-1]], [0,0])
            np.testing.assert_array_equal(data["t"][[0,-1]], [0,0])
            xu, yu, xl, yl = data["surfaces"]
            for xx, yy in ((xu,yu), (xl,yl)):
                np.testing.assert_array_equal(xx[[0,-1]], [0,1])
                np.testing.assert_array_equal(yy[[0,-1]], [0,0])
                self.assertEqual(float(np.arctan2(yy[-1]-yy[0], xx[-1]-xx[0])), 0)

    def test_normal_composition(self):
        for data in (self.case, self.result):
            xu,yu,xl,yl=data["surfaces"]
            xi=self.case["xi"]
            np.testing.assert_allclose((xu+xl)/2, xi, atol=1e-16)
            np.testing.assert_allclose((yu+yl)/2, data["yc"], atol=1e-16)
            np.testing.assert_allclose(np.hypot(xl-xu,yl-yu)/2,data["t"],atol=1e-16)
            # Normal vector is perpendicular to (1,dyc).
            np.testing.assert_allclose((xl-xu)+(yl-yu)*data["dyc"],0,atol=3e-16)

    def test_naca_63412_regression_and_no_clipping(self):
        expected=naca6("63-412",150)
        for actual, reference in zip(self.case["surfaces"], expected):
            np.testing.assert_allclose(actual,reference,atol=2e-14,rtol=0)
        self.assertLess(self.case["surfaces"][0].min(),-0.0003)
        self.assertLess(self.result["surfaces"][0].min(),0)
        # Direct reconstruction also preserves negative abscissas for CST-like data.
        surfaces=cst.reconstruct_surfaces([0,.0001,1],[0,.001,0],[0,.003,0],[0,.2,0])
        self.assertLess(surfaces[0][1],0)
        np.testing.assert_array_equal(self.case["xi"][[0,-1]],[0,1])

    def test_fit_and_analytic_derivative_consistency(self):
        xi=np.linspace(0,1,150)
        coeff=np.array([.12,.17,.23,.19,.10,.05])
        for n1,n2 in ((1,1),(.5,1)):
            y=cst.evaluate_cst(xi,coeff,n1,n2)
            np.testing.assert_allclose(cst.fit_cst(xi,y,5,n1,n2),coeff,atol=1e-13)
            x=np.linspace(.01,.99,40);h=1e-6
            fd=(cst.evaluate_cst(x+h,coeff,n1,n2)-cst.evaluate_cst(x-h,coeff,n1,n2))/(2*h)
            np.testing.assert_allclose(cst.evaluate_cst(x,coeff,n1,n2,derivative=True),fd,atol=1e-8)
        np.testing.assert_allclose(cst.bernstein_basis(xi,5).sum(axis=1),1,atol=1e-14)

    def test_geometric_errors_are_euclidean(self):
        zeros=np.zeros(3)
        errors=cst.calculate_geometric_errors((zeros,zeros,zeros,zeros),
                                              (np.full(3,3),np.full(3,4),zeros,zeros))
        np.testing.assert_array_equal(errors["upper"]["error"],[5,5,5])
        self.assertEqual(errors["upper"]["rms"],5)
        self.assertEqual(errors["lower"]["max_abs"],0)
        for stats in self.result["geometric_errors"].values():
            self.assertTrue(np.isfinite(stats["error"]).all())
            self.assertLess(stats["rms"],1e-3)
            self.assertLess(stats["max_abs"],2e-3)

    def test_symmetric_alpha_reference(self):
        case=cst.generate_naca_camber_thickness("63-012",150)
        np.testing.assert_array_equal(case["yc"],np.zeros(150))
        xu,yu,xl,yl=case["surfaces"]
        np.testing.assert_array_equal(xu,case["xi"])
        np.testing.assert_array_equal(xl,case["xi"])
        np.testing.assert_array_equal(yu,-yl)

    def test_single_csv_and_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            output=Path(directory)/"output"
            with patch("cst.OUTPUT_DIR",output),contextlib.redirect_stdout(io.StringIO()):
                cst.main(["63-412","--order-camber","5","--order-thickness","5","--points","150"])
            self.assertEqual([p.name for p in output.iterdir()],["NACA_63-412_CST.csv"])
            text=(output/"NACA_63-412_CST.csv").read_text(encoding="utf-8")
            for obsolete in ("frame origin","chord vector","inverse transform"):
                self.assertNotIn(obsolete,text.lower())
            rows=list(csv.reader(io.StringIO("\n".join(line for line in text.splitlines() if not line.startswith('#')))))
            self.assertEqual(rows[0],list(cst.COLUMNS))
            self.assertEqual(len(rows),151)
            data=np.array(rows[1:],dtype=float)
            np.testing.assert_allclose(data[:,13],np.hypot(data[:,9]-data[:,5],data[:,10]-data[:,6]),atol=1e-16)
            np.testing.assert_allclose(data[:,14],np.hypot(data[:,11]-data[:,7],data[:,12]-data[:,8]),atol=1e-16)

    def test_distributions_and_loading(self):
        for distribution in ("lineal","coseno","seno","seno_salida"):
            for a in (0,.5,1):
                case=cst.generate_naca_camber_thickness("63-412",150,distribution,a=a)
                for actual,expected in zip(case["surfaces"],naca6("63-412",150,distribution,a=a)):
                    np.testing.assert_allclose(actual,expected,atol=2e-14,rtol=0)

    def test_bad_inputs(self):
        for options in (dict(points=2),dict(a=-1),dict(a=float('nan')),dict(distribution='unknown')):
            with self.assertRaises(ValueError):
                cst.generate_naca_camber_thickness('63-412',**options)
        with self.assertRaises(ValueError):
            cst.fit_cst([0,.5,1],[0,.1,0],5)
        with self.assertRaises(ValueError):
            cst.reconstruct_surfaces([0,.5,1],[0,0,0],[0,-.1,0],[0,0,0])

    def test_plot_has_three_panels_and_saves_nothing(self):
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
        except ImportError:
            self.skipTest('matplotlib opcional no instalado')
        def inspect_plot():
            self.assertEqual(len(plt.gcf().axes),3)
            self.assertEqual(len(plt.gcf().axes[1].lines),2)
            self.assertLess(min(plt.gcf().axes[0].lines[0].get_xdata()),0)
        with patch.object(plt,'show',side_effect=inspect_plot) as show, patch('matplotlib.figure.Figure.savefig') as save:
            cst.plot_comparison(self.case,self.result)
            show.assert_called_once()
            save.assert_not_called()
        self.assertEqual(plt.get_fignums(),[])


if __name__ == '__main__':
    unittest.main()
