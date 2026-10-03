"""Validación con ordenadas publicadas por PDAS/NACA456, no con el propio código.

Datos: https://www.pdas.com/packages/naca456.zip, samples.zip (*.out).
Se usa el t/c del encabezado: 67-021.out contiene realmente un espesor 15%.
La tolerancia 5e-5 sólo corresponde a estos casos de referencia.
Ejecutar: python -m unittest discover -s tests -v
"""

import contextlib
import csv
import io
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from naca6 import _mean_line, _thickness_at, _thickness_splines, main, naca6


class TestNaca6(unittest.TestCase):
    def test_published_reference_ordinates(self):
        cases = json.loads(Path(__file__).with_name("naca6_reference.json").read_text())
        for case in cases:
            with self.subTest(code=case["code"]):
                sx, sy = _thickness_splines(case["family"], case["tc"])
                for x, expected in case["ordinates"]:
                    self.assertLess(abs(_thickness_at(x, sx, sy) - expected), 5e-5)

    def test_published_cambered_profile(self):
        result = naca6("63-206", 21, "lineal")
        for index, expected in [(1, (0.049315, 0.017770, 0.050685, -0.011451)),
                                (5, (0.249503, 0.037370, 0.250497, -0.019470)),
                                (10, (0.500000, 0.038250, 0.500000, -0.016186))]:
            for values, ref in zip(result, expected):
                self.assertLess(abs(values[index] - ref), 1e-5)

    def test_symmetry_endpoints_and_thickness_range(self):
        for family in range(63, 68):
            for thickness in (1, 6, 12, 21, 30):
                code = f"{family}0{thickness:02d}"
                with self.subTest(code=code):
                    xu, yu, xl, yl = naca6(code, 101)
                    self.assertEqual(xu, xl)
                    self.assertEqual(yu, [-v for v in yl])
                    self.assertEqual((xu[0], xu[-1], yu[0], yu[-1]), (0, 1, 0, 0))
                    self.assertTrue(all(math.isfinite(v) for v in yu))
                    sx, sy = _thickness_splines(family, thickness / 100)
                    self.assertAlmostEqual(2 * sy.maximum(200), thickness / 100, places=9)

    def test_designations_and_subscript(self):
        ref = naca6("632215", 11)
        for code in ("NACA 63(2)-215", "63_2-215", "63-215", "63215", 632215):
            self.assertEqual(naca6(code, 11), ref)

    def test_spacing(self):
        for name, fn in [("lineal", lambda u: u),
                         ("coseno", lambda u: (1 - math.cos(math.pi * u)) / 2),
                         ("seno", lambda u: 1 - math.cos(math.pi * u / 2)),
                         ("seno_salida", lambda u: math.sin(math.pi * u / 2))]:
            xu, _, xl, _ = naca6("632215", 21, name)
            for i in range(21):
                self.assertAlmostEqual((xu[i] + xl[i]) / 2, fn(i / 20), places=14)

    def test_a_parameter(self):
        for a in (0.0, 0.5, 0.8, 1 - 1e-12, 1.0):
            self.assertTrue(all(math.isfinite(v) for values in naca6("63-215", 101, a=a) for v in values))
        # Relación diferencial entre pendiente y ordenada de línea media.
        for a in (0.0, 0.5, 0.8, 1.0):
            for x in (0.1, 0.3, 0.5, 0.9):
                step = 1e-6
                numerical = (_mean_line(x + step, 0.2, a)[0]
                             - _mean_line(x - step, 0.2, a)[0]) / (2 * step)
                self.assertAlmostEqual(numerical, _mean_line(x, 0.2, a)[1], places=8)
        self.assertNotEqual(naca6("63-215", a=0.5), naca6("63-215", a=1))

    def test_bad_inputs(self):
        for code in ("0012", "62-215", "63A215", "63(015)-215", "6321500", "632200", "63-299"):
            with self.assertRaises(ValueError):
                naca6(code)
        for count in (True, 0, 2, 3.5, "101"):
            with self.assertRaises(ValueError):
                naca6("632215", count)
        for a in (-0.1, 1.1, float("nan"), float("inf"), None):
            with self.assertRaises(ValueError):
                naca6("632215", a=a)
        with self.assertRaises(ValueError):
            naca6("632215", distribution="unknown")

    def test_cli_csv_in_output_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory) / "output"
            with patch("naca6.OUTPUT_DIR", output_dir), contextlib.redirect_stdout(io.StringIO()):
                for options, filename in [([], "naca_632215.csv"), (["-o", "profile.csv"], "profile.csv")]:
                    main(["632215", "11", *options])
                    with (output_dir / filename).open(newline="") as stream:
                        rows = list(csv.reader(stream))
                    self.assertEqual(len(rows), 12)
                    self.assertEqual(rows[0], ["x_upper", "y_upper", "x_lower", "y_lower"])

    def test_cli_errors_have_correct_accents(self):
        errors = io.StringIO()
        with contextlib.redirect_stderr(errors), self.assertRaises(SystemExit) as exc:
            main(["0012", "11"])
        self.assertEqual(exc.exception.code, 2)
        self.assertIn("Código no válido", errors.getvalue())
        for filename in ("../profile.csv", "output/profile.csv", "profile.json"):
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                main(["632215", "11", "-o", filename])

    def test_cli_plot(self):
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory) / "output"
            with patch("naca6.OUTPUT_DIR", output_dir), patch("naca6._plot") as plot, contextlib.redirect_stdout(io.StringIO()):
                main(["632215", "11", "--plot", "--no-show"])
            self.assertTrue((output_dir / "naca_632215.csv").exists())
            self.assertEqual(plot.call_args.args[2], output_dir / "naca_632215.png")
            self.assertFalse(plot.call_args.kwargs["show"])


if __name__ == "__main__":
    unittest.main()
