"""Generador de perfiles NACA de la serie 6 (63, 64, 65, 66 y 67).

Uso Python::

    from naca6 import naca6
    x_upper, y_upper, x_lower, y_lower = naca6("632215", 101, "coseno")

Uso CLI::

    python naca6.py 632215 101 --distribution coseno --plot
    python naca6.py 632215 101 --output perfil.csv --plot --no-show

El CLI guarda siempre el CSV en output/, junto a este script. --output
permite elegir el nombre del archivo dentro de esa carpeta. --plot guarda
también un PNG y muestra la figura; --no-show permite generar el PNG sin
abrir ventanas. Sólo el plot requiere matplotlib (pip install matplotlib).

Las cuatro salidas son listas de floats, con n_points POR SUPERFICIE,
ordenadas del borde de ataque al borde de salida. Coordenadas x/c, y/c.
La distribución se aplica a las estaciones de la línea media; al añadir
espesor en dirección normal, x_upper y x_lower pueden ser diferentes.

Notación: 632215 = 63(2)-215: familia 63, intervalo de baja resistencia
de +/-0.2 alrededor de Cl de diseño 0.2, espesor 15%. El subíndice es
descriptivo del intervalo aerodinámico, no otro parámetro geométrico.
También se admiten "NACA 63(2)-215", "63_2-215", "63-215" y "63215".
El parámetro a (extensión de carga uniforme) NO está codificado en esos
dígitos: se especifica aparte, con a=1 por defecto. No incluye la serie 6A
ni las designaciones con distribuciones de espesor modificadas entre
paréntesis, como 63(015)-215.

Referencias oficiales:
  NASA TM X-3069 (1974), Development of a Computer Program to Obtain
  Ordinates for NACA 6- and 6A-Series Airfoils:
  https://ntrs.nasa.gov/citations/19740025318
  NASA TM 4741 (1996), Computer Program To Obtain Ordinates for NACA
  Airfoils, transformación conforme, línea media y composición normal:
  https://ntrs.nasa.gov/citations/19970008124

Tablas epsilon/psi: epspsi.f90 de NACA456, Ralph L. Carmichael / PDAS,
código de dominio público basado en los programas NASA anteriores:
https://www.pdas.com/packages/naca456.zip (versión descargada 2026-10-03).
Las tablas de 201 estaciones están incorporadas: no se requiere Internet,
NumPy, SciPy ni compilador Fortran. Se usan splines cúbicos naturales
paramétricos en longitud de arco e inversión por bisección; los extremos
se fuerzan a (0,0), (1,0). Esto difiere del spline FMM de PDAS y del ajuste
elíptico local del programa NASA original. No se afirma igualdad bit a bit
ni una tolerancia universal respecto a tablas históricas publicadas.
"""

from __future__ import annotations

import argparse
from bisect import bisect_right
import cmath
import csv
from functools import lru_cache
import math
from numbers import Integral
from pathlib import Path
import re
import sys


# Tablas EPS/PSI incorporadas al final del archivo desde la fuente PDAS.


def _parse(code: str | int) -> tuple[int, float, float]:
    text = re.sub(r"^NACA\s*", "", str(code).strip(), flags=re.IGNORECASE)
    text = re.sub(r"\s+", "", text)
    # El subíndice opcional tiene exactamente un dígito.
    match = re.fullmatch(r"(6[3-7])(?:\((\d)\)|_(\d)|(\d))?-(\d)(\d{2})", text)
    if match:
        family, _, _, _, cl, thickness = match.groups()
    elif re.fullmatch(r"6[3-7]\d{3,4}", text):
        family, cl, thickness = text[:2], text[-3], text[-2:]
    else:
        raise ValueError("Código no válido. Usa 632215, 63(2)-215 o 63-215; familias 63 a 67.")
    tc = int(thickness) / 100.0
    if not 0.01 <= tc <= 0.30:
        raise ValueError("Este generador admite espesores del 1% al 30% de la cuerda.")
    return int(family), int(cl) / 10.0, tc


class _Spline:
    """Spline cúbico natural, con solución tridiagonal O(n)."""

    def __init__(self, knots, values):
        self.knots = knots
        self.values = values
        n = len(knots)
        h = [knots[i + 1] - knots[i] for i in range(n - 1)]
        diag, rhs = [1.0] * n, [0.0] * n
        upper, lower = [0.0] * n, [0.0] * n
        for i in range(1, n - 1):
            lower[i], diag[i], upper[i] = h[i - 1], 2 * (h[i - 1] + h[i]), h[i]
            rhs[i] = 6 * ((values[i + 1] - values[i]) / h[i]
                          - (values[i] - values[i - 1]) / h[i - 1])
        for i in range(1, n):
            factor = lower[i] / diag[i - 1]
            diag[i] -= factor * upper[i - 1]
            rhs[i] -= factor * rhs[i - 1]
        second = [0.0] * n
        second[-1] = rhs[-1] / diag[-1]
        for i in range(n - 2, -1, -1):
            second[i] = (rhs[i] - upper[i] * second[i + 1]) / diag[i]
        self.coefficients = []
        for i, step in enumerate(h):
            b = (values[i + 1] - values[i]) / step - step * (2 * second[i] + second[i + 1]) / 6
            self.coefficients.append((values[i], b, second[i] / 2,
                                      (second[i + 1] - second[i]) / (6 * step)))

    def __call__(self, s):
        i = min(max(bisect_right(self.knots, s) - 1, 0), len(self.knots) - 2)
        u = s - self.knots[i]
        a, b, c, d = self.coefficients[i]
        return a + u * (b + u * (c + u * d))

    def maximum(self, start):
        """Máximo exacto de los segmentos cúbicos desde el índice start."""
        best = max(self.values[start:])
        for i in range(start, len(self.coefficients)):
            a, b, c, d = self.coefficients[i]
            h = self.knots[i + 1] - self.knots[i]
            roots = []
            if abs(d) < 1e-20:
                if c:
                    roots = [-b / (2 * c)]
            else:
                disc = 4 * c * c - 12 * d * b
                if disc >= 0:
                    roots = [(-2 * c + math.sqrt(disc)) / (6 * d),
                             (-2 * c - math.sqrt(disc)) / (6 * d)]
            for u in roots:
                if 0 < u < h:
                    best = max(best, a + u * (b + u * (c + u * d)))
        return best


def _mapped_splines(family, scale):
    eps, psi = _TABLES[family]
    zeta = []
    for i in range(201):
        z = cmath.exp(complex(scale * psi[i], math.pi * i / 200 - scale * eps[i]))
        zeta.append(z + 1 / z)
    chord = abs(zeta[-1] - zeta[0])
    points = [(zeta[0] - z) / chord for z in zeta]
    xt, yt = [z.real for z in points], [-z.imag for z in points]
    xt[0], xt[-1], yt[0], yt[-1] = 0.0, 1.0, 0.0, 0.0
    if any(x1 <= x0 for x0, x1 in zip(xt, xt[1:])) or min(yt) < -1e-12:
        raise ValueError("La transformación produjo una distribución no válida para este espesor.")
    # Contorno inferior TE->LE y superior LE->TE: continuidad de tangente en LE.
    xx = xt[:0:-1] + xt
    yy = [-v for v in yt[:0:-1]] + yt
    s = [0.0]
    for i in range(1, len(xx)):
        s.append(s[-1] + math.hypot(xx[i] - xx[i - 1], yy[i] - yy[i - 1]))
    return _Spline(s, xx), _Spline(s, yy)


@lru_cache(maxsize=64)
def _thickness_splines(family, tc):
    # NASA: escalar conjuntamente epsilon y psi hasta alcanzar el espesor.
    # No se escalan sólo las ordenadas y, pues eso cambia el perfil de la serie.
    low, high = 0.0, 1.0
    while True:
        sx, sy = _mapped_splines(family, high)
        if 2 * sy.maximum(200) >= tc:
            break
        high *= 2
        if high > 16:
            raise ValueError("No se pudo acotar el espesor solicitado.")
    for _ in range(50):
        scale = (low + high) / 2
        sx, sy = _mapped_splines(family, scale)
        error = 2 * sy.maximum(200) - tc
        if abs(error) < 1e-11:
            return sx, sy
        if error > 0:
            high = scale
        else:
            low = scale
    raise RuntimeError("La iteración del espesor no convergió.")


def _thickness_at(x, sx, sy):
    if x == 0 or x == 1:
        return 0.0
    # Localizar el segmento por x y después invertir x(s).
    i = min(max(bisect_right(sx.values, x, lo=200) - 1, 200), 399)
    low, high = sx.knots[i], sx.knots[i + 1]
    for _ in range(48):
        mid = (low + high) / 2
        if sx(mid) < x:
            low = mid
        else:
            high = mid
    return sy((low + high) / 2)


def _mean_line(x, cl, a):
    if x == 0 or x == 1 or cl == 0:
        return 0.0, 0.0
    omx = 1 - x
    # Límite a->1: evita cancelación en la expresión general (NASA usa 1e-7).
    if 1 - a < 1e-7:
        factor = cl / (4 * math.pi)
        return (-factor * (omx * math.log(omx) + x * math.log(x)),
                factor * (math.log(omx) - math.log(x)))
    oma = 1 - a
    if a == 0:
        g, h = -0.25, -0.5
    else:
        g = -(a * a * (0.5 * math.log(a) - 0.25) + 0.25) / oma
        h = g + oma * (0.5 * math.log(oma) - 0.25)
    amx = a - x
    term1 = amx * amx * (2 * math.log(abs(amx)) - 1) if amx else 0.0
    term1p = -amx * math.log(abs(amx)) if amx else 0.0
    term2 = omx * omx * (1 - 2 * math.log(omx))
    yc = 0.25 * (term1 + term2) / oma - x * math.log(x) + g - h * x
    slope = (term1p + omx * math.log(omx)) / oma - 1 - math.log(x) - h
    factor = cl / (2 * math.pi * (a + 1))
    return factor * yc, factor * slope


def naca6(naca: str | int, n_points: int = 101, distribution: str = "coseno",
          *, a: float = 1.0) -> tuple[list[float], list[float], list[float], list[float]]:
    """Devuelve (x_upper, y_upper, x_lower, y_lower), normalizados por cuerda.

    naca: seis dígitos con subíndice, o cinco sin él (ver docstring del módulo).
    n_points: número de estaciones por superficie, incluyendo ambos bordes.
    distribution: lineal/linear, coseno/cosine (concentración en ambos bordes),
                  seno/sine (en borde de ataque), seno_salida/sine_te (en salida).
    a: extensión de carga uniforme en fracción de cuerda, 0 <= a <= 1.
    El espesor se mide normal a la línea media; no es y_upper - y_lower a x fijo.
    """
    family, cl, tc = _parse(naca)
    if isinstance(n_points, bool) or not isinstance(n_points, Integral) or n_points < 3:
        raise ValueError("n_points debe ser un entero >= 3 (puntos por superficie).")
    try:
        a = float(a)
    except (TypeError, ValueError) as exc:
        raise ValueError("a debe ser un número finito entre 0 y 1.") from exc
    if not math.isfinite(a) or not 0 <= a <= 1:
        raise ValueError("a debe ser un número finito entre 0 y 1.")
    if not isinstance(distribution, str):
        raise ValueError("distribution debe ser una cadena.")
    distributions = {
        "lineal": lambda u: u, "linear": lambda u: u,
        "coseno": lambda u: (1 - math.cos(math.pi * u)) / 2,
        "cosine": lambda u: (1 - math.cos(math.pi * u)) / 2,
        "seno": lambda u: 1 - math.cos(math.pi * u / 2),
        "sine": lambda u: 1 - math.cos(math.pi * u / 2),
        "seno_salida": lambda u: math.sin(math.pi * u / 2),
        "sine_te": lambda u: math.sin(math.pi * u / 2),
    }
    try:
        spacing = distributions[distribution.strip().lower()]
    except KeyError as exc:
        raise ValueError("Distribución no válida: lineal, coseno, seno o seno_salida.") from exc
    sx, sy = _thickness_splines(family, tc)
    xu, yu, xl, yl = [], [], [], []
    for i in range(n_points):
        x = 0.0 if i == 0 else 1.0 if i == n_points - 1 else spacing(i / (n_points - 1))
        yt = _thickness_at(x, sx, sy)
        yc, slope = _mean_line(x, cl, a)
        theta = math.atan(slope)
        dx, dy = yt * math.sin(theta), yt * math.cos(theta)
        xu.append(x - dx)
        yu.append(yc + dy)
        xl.append(x + dx)
        yl.append(yc - dy)
    return xu, yu, xl, yl


generate_naca6 = naca6


OUTPUT_DIR = Path(__file__).resolve().parent / "output"


def _output_filename(value):
    path = Path(value)
    if path.name != value or value in (".", "..") or "/" in value or "\\" in value:
        raise argparse.ArgumentTypeError("Indica sólo un nombre de archivo; los resultados se guardan en output.")
    if path.suffix.lower() != ".csv":
        raise argparse.ArgumentTypeError("El archivo de salida debe tener extensión .csv.")
    return value


def _plot(result, title, path, *, show=True):
    try:
        import matplotlib
        if not show:
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ValueError("Para usar --plot instala matplotlib: python -m pip install matplotlib") from exc
    xu, yu, xl, yl = result
    fig, ax = plt.subplots(figsize=(10, 4))
    try:
        ax.plot(xu, yu, "o-", markersize=2, linewidth=1.2, label="Extradós")
        ax.plot(xl, yl, "o-", markersize=2, linewidth=1.2, label="Intradós")
        ax.set(xlabel="x/c", ylabel="y/c", title=title)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(path, dpi=200)
        if show:
            plt.show()
    finally:
        plt.close(fig)


def main(argv=None):
    # UTF-8 también al redirigir la salida en Windows (evita usar CP1252).
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="backslashreplace")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("naca", help="Ejemplo: 632215 o '63(2)-215'")
    parser.add_argument("n_points", type=int, help="Puntos por superficie, incluyendo los bordes")
    parser.add_argument("--distribution", "--distribucion", default="coseno")
    parser.add_argument("--a", type=float, default=1.0, help="Extensión de carga uniforme (0..1), defecto 1")
    parser.add_argument("--output", "-o", type=_output_filename, help="Nombre del CSV en output/ (por defecto: naca_<código>.csv)")
    parser.add_argument("--plot", action="store_true", help="Guardar PNG y mostrar el perfil (requiere matplotlib)")
    parser.add_argument("--no-show", action="store_true", help="Con --plot, guardar PNG sin abrir ventanas")
    args = parser.parse_args(argv)
    if args.no_show and not args.plot:
        parser.error("--no-show requiere --plot.")
    try:
        result = naca6(args.naca, args.n_points, args.distribution, a=args.a)
    except (ValueError, RuntimeError) as exc:
        parser.error(str(exc))
    names = ("x_upper", "y_upper", "x_lower", "y_lower")
    code = re.sub(r"[^0-9]", "", args.naca)
    output_path = OUTPUT_DIR / (args.output or f"naca_{code}.csv")
    try:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        if args.plot:
            _plot(result, f"NACA {args.naca} · a={args.a:g}", output_path.with_suffix(".png"), show=not args.no_show)
        with output_path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(names)
            writer.writerows(zip(*result))
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    print(f"CSV guardado: {output_path}")
    if args.plot:
        print(f"Gráfico guardado: {output_path.with_suffix('.png')}")


_TABLES = {
    63: (
        (
            0.00000, 0.00164, 0.00327, 0.00487, 0.00641, 0.00790, 0.00928, 0.01057,
            0.01174, 0.01278, 0.01367, 0.01439, 0.01497, 0.01542, 0.01576, 0.01601,
            0.01619, 0.01632, 0.01642, 0.01651, 0.01661, 0.01673, 0.01688, 0.01705,
            0.01725, 0.01747, 0.01771, 0.01797, 0.01824, 0.01853, 0.01884, 0.01916,
            0.01949, 0.01984, 0.02020, 0.02058, 0.02097, 0.02137, 0.02179, 0.02223,
            0.02268, 0.02315, 0.02363, 0.02413, 0.02464, 0.02517, 0.02571, 0.02626,
            0.02683, 0.02741, 0.02801, 0.02862, 0.02924, 0.02988, 0.03052, 0.03118,
            0.03185, 0.03253, 0.03323, 0.03393, 0.03465, 0.03538, 0.03611, 0.03686,
            0.03762, 0.03839, 0.03917, 0.03995, 0.04075, 0.04156, 0.04237, 0.04319,
            0.04402, 0.04486, 0.04571, 0.04657, 0.04743, 0.04831, 0.04919, 0.05008,
            0.05098, 0.05189, 0.05280, 0.05372, 0.05464, 0.05556, 0.05648, 0.05740,
            0.05831, 0.05921, 0.06011, 0.06099, 0.06187, 0.06273, 0.06357, 0.06440,
            0.06522, 0.06602, 0.06681, 0.06757, 0.06832, 0.06905, 0.06976, 0.07044,
            0.07111, 0.07176, 0.07238, 0.07298, 0.07356, 0.07411, 0.07464, 0.07514,
            0.07562, 0.07607, 0.07650, 0.07690, 0.07727, 0.07761, 0.07793, 0.07822,
            0.07848, 0.07871, 0.07891, 0.07908, 0.07922, 0.07933, 0.07941, 0.07945,
            0.07946, 0.07944, 0.07938, 0.07929, 0.07916, 0.07900, 0.07880, 0.07856,
            0.07829, 0.07799, 0.07764, 0.07726, 0.07685, 0.07640, 0.07591, 0.07539,
            0.07483, 0.07423, 0.07359, 0.07293, 0.07222, 0.07148, 0.07070, 0.06989,
            0.06904, 0.06815, 0.06723, 0.06628, 0.06529, 0.06427, 0.06322, 0.06214,
            0.06103, 0.05989, 0.05871, 0.05751, 0.05628, 0.05502, 0.05374, 0.05243,
            0.05109, 0.04973, 0.04834, 0.04693, 0.04549, 0.04404, 0.04256, 0.04106,
            0.03955, 0.03802, 0.03647, 0.03491, 0.03333, 0.03174, 0.03014, 0.02853,
            0.02690, 0.02527, 0.02363, 0.02198, 0.02032, 0.01865, 0.01698, 0.01530,
            0.01361, 0.01192, 0.01023, 0.00853, 0.00683, 0.00512, 0.00342, 0.00171,
            0.00000,
        ),
        (
            0.15066, 0.15058, 0.15035, 0.14999, 0.14950, 0.14891, 0.14823, 0.14748,
            0.14668, 0.14583, 0.14497, 0.14410, 0.14323, 0.14238, 0.14155, 0.14074,
            0.13998, 0.13927, 0.13862, 0.13804, 0.13753, 0.13711, 0.13676, 0.13648,
            0.13627, 0.13610, 0.13598, 0.13590, 0.13584, 0.13579, 0.13576, 0.13573,
            0.13570, 0.13567, 0.13564, 0.13561, 0.13558, 0.13555, 0.13552, 0.13550,
            0.13547, 0.13544, 0.13542, 0.13539, 0.13536, 0.13533, 0.13529, 0.13525,
            0.13521, 0.13516, 0.13511, 0.13505, 0.13499, 0.13491, 0.13483, 0.13475,
            0.13465, 0.13454, 0.13442, 0.13428, 0.13414, 0.13398, 0.13381, 0.13363,
            0.13343, 0.13321, 0.13299, 0.13275, 0.13249, 0.13222, 0.13194, 0.13164,
            0.13133, 0.13100, 0.13065, 0.13028, 0.12988, 0.12947, 0.12903, 0.12857,
            0.12808, 0.12756, 0.12702, 0.12644, 0.12584, 0.12521, 0.12455, 0.12385,
            0.12313, 0.12238, 0.12160, 0.12079, 0.11994, 0.11907, 0.11817, 0.11724,
            0.11628, 0.11529, 0.11428, 0.11324, 0.11218, 0.11109, 0.10998, 0.10884,
            0.10768, 0.10650, 0.10530, 0.10407, 0.10283, 0.10157, 0.10029, 0.09899,
            0.09767, 0.09634, 0.09499, 0.09363, 0.09224, 0.09085, 0.08944, 0.08801,
            0.08657, 0.08512, 0.08365, 0.08217, 0.08068, 0.07917, 0.07766, 0.07614,
            0.07461, 0.07307, 0.07153, 0.06998, 0.06842, 0.06687, 0.06530, 0.06374,
            0.06217, 0.06060, 0.05904, 0.05747, 0.05591, 0.05435, 0.05280, 0.05125,
            0.04970, 0.04817, 0.04664, 0.04512, 0.04362, 0.04213, 0.04065, 0.03919,
            0.03774, 0.03631, 0.03490, 0.03350, 0.03213, 0.03077, 0.02943, 0.02811,
            0.02682, 0.02555, 0.02430, 0.02308, 0.02188, 0.02071, 0.01956, 0.01844,
            0.01735, 0.01630, 0.01527, 0.01428, 0.01331, 0.01239, 0.01149, 0.01062,
            0.00979, 0.00899, 0.00823, 0.00750, 0.00680, 0.00614, 0.00551, 0.00491,
            0.00435, 0.00382, 0.00332, 0.00286, 0.00244, 0.00205, 0.00169, 0.00137,
            0.00108, 0.00083, 0.00061, 0.00042, 0.00027, 0.00015, 0.00007, 0.00002,
            0.00000,
        ),
    ),
    64: (
        (
            0.00000, 0.00233, 0.00465, 0.00693, 0.00915, 0.01130, 0.01337, 0.01532,
            0.01715, 0.01884, 0.02035, 0.02169, 0.02288, 0.02391, 0.02481, 0.02557,
            0.02624, 0.02682, 0.02731, 0.02774, 0.02812, 0.02846, 0.02877, 0.02905,
            0.02931, 0.02957, 0.02982, 0.03007, 0.03033, 0.03060, 0.03090, 0.03122,
            0.03158, 0.03196, 0.03236, 0.03280, 0.03326, 0.03375, 0.03427, 0.03481,
            0.03538, 0.03598, 0.03660, 0.03725, 0.03792, 0.03862, 0.03935, 0.04010,
            0.04087, 0.04167, 0.04250, 0.04335, 0.04423, 0.04512, 0.04605, 0.04699,
            0.04796, 0.04896, 0.04998, 0.05102, 0.05208, 0.05317, 0.05428, 0.05541,
            0.05657, 0.05774, 0.05894, 0.06016, 0.06140, 0.06267, 0.06395, 0.06526,
            0.06658, 0.06793, 0.06931, 0.07070, 0.07213, 0.07357, 0.07505, 0.07655,
            0.07808, 0.07964, 0.08123, 0.08284, 0.08447, 0.08613, 0.08780, 0.08949,
            0.09119, 0.09290, 0.09462, 0.09635, 0.09808, 0.09980, 0.10151, 0.10321,
            0.10488, 0.10653, 0.10815, 0.10972, 0.11125, 0.11273, 0.11415, 0.11553,
            0.11686, 0.11814, 0.11938, 0.12057, 0.12171, 0.12281, 0.12386, 0.12487,
            0.12583, 0.12675, 0.12762, 0.12844, 0.12922, 0.12994, 0.13062, 0.13125,
            0.13182, 0.13234, 0.13281, 0.13322, 0.13358, 0.13389, 0.13414, 0.13434,
            0.13448, 0.13456, 0.13459, 0.13456, 0.13447, 0.13433, 0.13413, 0.13387,
            0.13354, 0.13316, 0.13272, 0.13222, 0.13166, 0.13104, 0.13035, 0.12960,
            0.12879, 0.12792, 0.12698, 0.12598, 0.12492, 0.12380, 0.12261, 0.12136,
            0.12004, 0.11866, 0.11723, 0.11573, 0.11417, 0.11255, 0.11087, 0.10914,
            0.10735, 0.10550, 0.10361, 0.10165, 0.09964, 0.09758, 0.09546, 0.09329,
            0.09106, 0.08878, 0.08645, 0.08406, 0.08163, 0.07914, 0.07661, 0.07403,
            0.07140, 0.06874, 0.06604, 0.06329, 0.06052, 0.05770, 0.05487, 0.05199,
            0.04908, 0.04615, 0.04319, 0.04021, 0.03721, 0.03417, 0.03113, 0.02807,
            0.02499, 0.02189, 0.01879, 0.01567, 0.01255, 0.00942, 0.00628, 0.00314,
            0.00000,
        ),
        (
            0.25269, 0.25265, 0.25251, 0.25227, 0.25193, 0.25147, 0.25090, 0.25020,
            0.24937, 0.24841, 0.24730, 0.24605, 0.24467, 0.24321, 0.24170, 0.24016,
            0.23864, 0.23715, 0.23573, 0.23442, 0.23325, 0.23224, 0.23138, 0.23066,
            0.23006, 0.22956, 0.22916, 0.22884, 0.22858, 0.22836, 0.22818, 0.22802,
            0.22788, 0.22775, 0.22764, 0.22755, 0.22747, 0.22740, 0.22736, 0.22732,
            0.22730, 0.22729, 0.22730, 0.22731, 0.22733, 0.22736, 0.22739, 0.22742,
            0.22745, 0.22748, 0.22751, 0.22753, 0.22755, 0.22756, 0.22756, 0.22755,
            0.22753, 0.22751, 0.22747, 0.22742, 0.22736, 0.22729, 0.22720, 0.22709,
            0.22697, 0.22683, 0.22668, 0.22650, 0.22630, 0.22608, 0.22584, 0.22557,
            0.22528, 0.22497, 0.22462, 0.22426, 0.22386, 0.22345, 0.22300, 0.22253,
            0.22203, 0.22150, 0.22094, 0.22034, 0.21969, 0.21899, 0.21823, 0.21741,
            0.21652, 0.21554, 0.21449, 0.21334, 0.21211, 0.21081, 0.20941, 0.20795,
            0.20642, 0.20482, 0.20316, 0.20143, 0.19966, 0.19784, 0.19597, 0.19406,
            0.19210, 0.19009, 0.18805, 0.18597, 0.18385, 0.18169, 0.17950, 0.17727,
            0.17502, 0.17274, 0.17042, 0.16807, 0.16570, 0.16330, 0.16087, 0.15843,
            0.15596, 0.15347, 0.15095, 0.14842, 0.14588, 0.14331, 0.14073, 0.13812,
            0.13551, 0.13288, 0.13024, 0.12759, 0.12492, 0.12225, 0.11957, 0.11688,
            0.11418, 0.11149, 0.10878, 0.10608, 0.10338, 0.10068, 0.09798, 0.09529,
            0.09260, 0.08992, 0.08725, 0.08459, 0.08195, 0.07932, 0.07671, 0.07412,
            0.07155, 0.06900, 0.06648, 0.06398, 0.06151, 0.05907, 0.05666, 0.05428,
            0.05193, 0.04962, 0.04733, 0.04509, 0.04288, 0.04072, 0.03859, 0.03650,
            0.03446, 0.03247, 0.03051, 0.02861, 0.02674, 0.02493, 0.02317, 0.02147,
            0.01982, 0.01824, 0.01672, 0.01525, 0.01385, 0.01252, 0.01125, 0.01006,
            0.00892, 0.00786, 0.00686, 0.00593, 0.00506, 0.00426, 0.00353, 0.00287,
            0.00227, 0.00174, 0.00128, 0.00089, 0.00057, 0.00032, 0.00014, 0.00004,
            0.00000,
        ),
    ),
    65: (
        (
            0.00000, 0.00165, 0.00330, 0.00493, 0.00653, 0.00810, 0.00963, 0.01110,
            0.01253, 0.01387, 0.01515, 0.01625, 0.01702, 0.01746, 0.01771, 0.01795,
            0.01824, 0.01858, 0.01893, 0.01922, 0.01943, 0.01951, 0.01945, 0.01926,
            0.01899, 0.01865, 0.01830, 0.01794, 0.01760, 0.01733, 0.01715, 0.01704,
            0.01702, 0.01706, 0.01715, 0.01727, 0.01742, 0.01759, 0.01777, 0.01798,
            0.01821, 0.01846, 0.01876, 0.01908, 0.01944, 0.01983, 0.02024, 0.02068,
            0.02114, 0.02161, 0.02211, 0.02261, 0.02313, 0.02365, 0.02419, 0.02474,
            0.02530, 0.02587, 0.02647, 0.02708, 0.02772, 0.02837, 0.02904, 0.02974,
            0.03046, 0.03119, 0.03195, 0.03271, 0.03350, 0.03429, 0.03510, 0.03593,
            0.03677, 0.03763, 0.03849, 0.03938, 0.04028, 0.04119, 0.04212, 0.04307,
            0.04404, 0.04502, 0.04602, 0.04704, 0.04808, 0.04914, 0.05021, 0.05130,
            0.05241, 0.05353, 0.05467, 0.05582, 0.05699, 0.05817, 0.05935, 0.06055,
            0.06174, 0.06294, 0.06414, 0.06534, 0.06653, 0.06772, 0.06890, 0.07006,
            0.07121, 0.07234, 0.07346, 0.07456, 0.07564, 0.07668, 0.07771, 0.07872,
            0.07970, 0.08064, 0.08155, 0.08244, 0.08327, 0.08406, 0.08480, 0.08550,
            0.08614, 0.08673, 0.08728, 0.08778, 0.08825, 0.08866, 0.08905, 0.08939,
            0.08968, 0.08995, 0.09017, 0.09034, 0.09048, 0.09057, 0.09062, 0.09061,
            0.09057, 0.09046, 0.09031, 0.09009, 0.08982, 0.08950, 0.08911, 0.08868,
            0.08819, 0.08765, 0.08707, 0.08643, 0.08576, 0.08504, 0.08427, 0.08346,
            0.08260, 0.08170, 0.08075, 0.07976, 0.07871, 0.07759, 0.07640, 0.07510,
            0.07368, 0.07216, 0.07047, 0.06864, 0.06664, 0.06448, 0.06220, 0.05982,
            0.05735, 0.05483, 0.05228, 0.04972, 0.04718, 0.04467, 0.04225, 0.03991,
            0.03766, 0.03548, 0.03338, 0.03135, 0.02939, 0.02752, 0.02569, 0.02394,
            0.02224, 0.02059, 0.01899, 0.01744, 0.01592, 0.01446, 0.01302, 0.01163,
            0.01026, 0.00892, 0.00761, 0.00630, 0.00502, 0.00376, 0.00250, 0.00124,
            0.00000,
        ),
        (
            0.17464, 0.17456, 0.17434, 0.17397, 0.17348, 0.17285, 0.17211, 0.17125,
            0.17030, 0.16923, 0.16808, 0.16682, 0.16542, 0.16391, 0.16236, 0.16086,
            0.15949, 0.15823, 0.15711, 0.15610, 0.15523, 0.15451, 0.15390, 0.15343,
            0.15305, 0.15278, 0.15258, 0.15246, 0.15239, 0.15235, 0.15235, 0.15237,
            0.15241, 0.15249, 0.15258, 0.15269, 0.15282, 0.15297, 0.15314, 0.15331,
            0.15350, 0.15370, 0.15389, 0.15408, 0.15427, 0.15445, 0.15464, 0.15482,
            0.15501, 0.15518, 0.15536, 0.15553, 0.15569, 0.15585, 0.15600, 0.15615,
            0.15629, 0.15643, 0.15655, 0.15667, 0.15678, 0.15688, 0.15697, 0.15705,
            0.15711, 0.15718, 0.15723, 0.15726, 0.15729, 0.15731, 0.15731, 0.15730,
            0.15728, 0.15724, 0.15719, 0.15713, 0.15704, 0.15694, 0.15682, 0.15669,
            0.15653, 0.15637, 0.15618, 0.15598, 0.15576, 0.15552, 0.15526, 0.15498,
            0.15466, 0.15431, 0.15393, 0.15351, 0.15305, 0.15255, 0.15200, 0.15141,
            0.15078, 0.15011, 0.14938, 0.14862, 0.14779, 0.14693, 0.14602, 0.14505,
            0.14402, 0.14295, 0.14181, 0.14063, 0.13940, 0.13812, 0.13680, 0.13544,
            0.13403, 0.13259, 0.13110, 0.12958, 0.12804, 0.12646, 0.12484, 0.12320,
            0.12154, 0.11984, 0.11813, 0.11638, 0.11461, 0.11281, 0.11100, 0.10915,
            0.10730, 0.10541, 0.10353, 0.10161, 0.09970, 0.09778, 0.09583, 0.09389,
            0.09194, 0.08998, 0.08800, 0.08601, 0.08401, 0.08201, 0.08000, 0.07798,
            0.07596, 0.07393, 0.07191, 0.06989, 0.06788, 0.06586, 0.06385, 0.06185,
            0.05986, 0.05787, 0.05589, 0.05392, 0.05196, 0.05001, 0.04807, 0.04615,
            0.04422, 0.04232, 0.04042, 0.03854, 0.03667, 0.03481, 0.03297, 0.03115,
            0.02937, 0.02762, 0.02590, 0.02423, 0.02260, 0.02101, 0.01949, 0.01802,
            0.01661, 0.01526, 0.01396, 0.01272, 0.01154, 0.01042, 0.00935, 0.00834,
            0.00739, 0.00650, 0.00566, 0.00489, 0.00416, 0.00350, 0.00289, 0.00234,
            0.00185, 0.00141, 0.00104, 0.00072, 0.00046, 0.00026, 0.00012, 0.00003,
            0.00000,
        ),
    ),
    66: (
        (
            0.00000, 0.00145, 0.00290, 0.00433, 0.00574, 0.00712, 0.00847, 0.00978,
            0.01105, 0.01225, 0.01340, 0.01447, 0.01547, 0.01638, 0.01719, 0.01789,
            0.01847, 0.01893, 0.01924, 0.01940, 0.01940, 0.01924, 0.01893, 0.01850,
            0.01799, 0.01741, 0.01679, 0.01616, 0.01556, 0.01499, 0.01450, 0.01410,
            0.01379, 0.01356, 0.01340, 0.01331, 0.01327, 0.01328, 0.01333, 0.01340,
            0.01350, 0.01361, 0.01373, 0.01387, 0.01402, 0.01419, 0.01438, 0.01458,
            0.01480, 0.01504, 0.01530, 0.01558, 0.01588, 0.01620, 0.01654, 0.01689,
            0.01726, 0.01765, 0.01805, 0.01847, 0.01890, 0.01934, 0.01980, 0.02026,
            0.02074, 0.02124, 0.02174, 0.02226, 0.02279, 0.02334, 0.02390, 0.02447,
            0.02506, 0.02566, 0.02627, 0.02690, 0.02754, 0.02819, 0.02885, 0.02952,
            0.03020, 0.03089, 0.03160, 0.03231, 0.03304, 0.03378, 0.03453, 0.03530,
            0.03608, 0.03688, 0.03770, 0.03853, 0.03938, 0.04025, 0.04113, 0.04202,
            0.04293, 0.04386, 0.04479, 0.04574, 0.04670, 0.04767, 0.04866, 0.04966,
            0.05067, 0.05171, 0.05277, 0.05386, 0.05498, 0.05612, 0.05730, 0.05851,
            0.05976, 0.06103, 0.06231, 0.06362, 0.06493, 0.06625, 0.06758, 0.06889,
            0.07020, 0.07149, 0.07277, 0.07402, 0.07524, 0.07644, 0.07760, 0.07872,
            0.07979, 0.08082, 0.08180, 0.08272, 0.08359, 0.08440, 0.08515, 0.08585,
            0.08649, 0.08708, 0.08761, 0.08808, 0.08850, 0.08886, 0.08916, 0.08941,
            0.08959, 0.08972, 0.08978, 0.08978, 0.08972, 0.08959, 0.08940, 0.08914,
            0.08882, 0.08843, 0.08797, 0.08745, 0.08687, 0.08622, 0.08551, 0.08474,
            0.08390, 0.08300, 0.08203, 0.08101, 0.07991, 0.07876, 0.07752, 0.07622,
            0.07485, 0.07341, 0.07190, 0.07031, 0.06865, 0.06692, 0.06511, 0.06324,
            0.06130, 0.05929, 0.05723, 0.05509, 0.05290, 0.05064, 0.04834, 0.04597,
            0.04355, 0.04109, 0.03857, 0.03602, 0.03342, 0.03077, 0.02810, 0.02539,
            0.02264, 0.01988, 0.01708, 0.01427, 0.01144, 0.00858, 0.00573, 0.00287,
            0.00000,
        ),
        (
            0.16457, 0.16455, 0.16449, 0.16437, 0.16416, 0.16386, 0.16345, 0.16292,
            0.16223, 0.16139, 0.16037, 0.15916, 0.15779, 0.15631, 0.15475, 0.15316,
            0.15157, 0.15002, 0.14856, 0.14722, 0.14604, 0.14506, 0.14427, 0.14364,
            0.14316, 0.14281, 0.14257, 0.14242, 0.14235, 0.14233, 0.14236, 0.14241,
            0.14248, 0.14257, 0.14267, 0.14280, 0.14294, 0.14310, 0.14327, 0.14346,
            0.14366, 0.14387, 0.14410, 0.14433, 0.14457, 0.14481, 0.14506, 0.14530,
            0.14554, 0.14578, 0.14601, 0.14623, 0.14645, 0.14665, 0.14685, 0.14704,
            0.14722, 0.14740, 0.14757, 0.14774, 0.14790, 0.14806, 0.14821, 0.14835,
            0.14849, 0.14862, 0.14875, 0.14886, 0.14897, 0.14908, 0.14917, 0.14925,
            0.14933, 0.14940, 0.14945, 0.14950, 0.14954, 0.14957, 0.14959, 0.14961,
            0.14961, 0.14960, 0.14959, 0.14956, 0.14953, 0.14948, 0.14943, 0.14936,
            0.14928, 0.14918, 0.14908, 0.14896, 0.14883, 0.14869, 0.14853, 0.14835,
            0.14816, 0.14796, 0.14774, 0.14750, 0.14725, 0.14698, 0.14669, 0.14638,
            0.14606, 0.14571, 0.14533, 0.14494, 0.14452, 0.14407, 0.14360, 0.14310,
            0.14256, 0.14198, 0.14135, 0.14067, 0.13992, 0.13910, 0.13820, 0.13722,
            0.13615, 0.13498, 0.13371, 0.13237, 0.13094, 0.12943, 0.12787, 0.12624,
            0.12457, 0.12284, 0.12108, 0.11929, 0.11746, 0.11561, 0.11374, 0.11183,
            0.10988, 0.10790, 0.10590, 0.10386, 0.10180, 0.09970, 0.09758, 0.09542,
            0.09325, 0.09106, 0.08885, 0.08662, 0.08439, 0.08214, 0.07989, 0.07763,
            0.07537, 0.07311, 0.07085, 0.06859, 0.06633, 0.06407, 0.06182, 0.05957,
            0.05733, 0.05510, 0.05287, 0.05066, 0.04845, 0.04627, 0.04411, 0.04196,
            0.03983, 0.03773, 0.03566, 0.03362, 0.03160, 0.02963, 0.02769, 0.02579,
            0.02394, 0.02214, 0.02038, 0.01869, 0.01705, 0.01547, 0.01396, 0.01249,
            0.01111, 0.00980, 0.00856, 0.00740, 0.00631, 0.00531, 0.00439, 0.00356,
            0.00281, 0.00216, 0.00159, 0.00110, 0.00071, 0.00040, 0.00018, 0.00004,
            0.00000,
        ),
    ),
    67: (
        (
            0.00000, 0.00169, 0.00338, 0.00506, 0.00672, 0.00835, 0.00995, 0.01152,
            0.01304, 0.01450, 0.01591, 0.01719, 0.01822, 0.01896, 0.01952, 0.02004,
            0.02060, 0.02116, 0.02170, 0.02220, 0.02259, 0.02287, 0.02300, 0.02300,
            0.02289, 0.02270, 0.02245, 0.02215, 0.02184, 0.02154, 0.02126, 0.02101,
            0.02079, 0.02061, 0.02045, 0.02030, 0.02017, 0.02005, 0.01994, 0.01985,
            0.01979, 0.01972, 0.01969, 0.01966, 0.01966, 0.01968, 0.01971, 0.01975,
            0.01982, 0.01989, 0.01997, 0.02006, 0.02016, 0.02028, 0.02040, 0.02054,
            0.02067, 0.02083, 0.02099, 0.02116, 0.02135, 0.02155, 0.02175, 0.02198,
            0.02221, 0.02245, 0.02271, 0.02298, 0.02325, 0.02355, 0.02386, 0.02417,
            0.02450, 0.02484, 0.02519, 0.02555, 0.02592, 0.02630, 0.02670, 0.02710,
            0.02752, 0.02795, 0.02839, 0.02884, 0.02929, 0.02976, 0.03025, 0.03075,
            0.03126, 0.03178, 0.03232, 0.03287, 0.03343, 0.03401, 0.03461, 0.03522,
            0.03585, 0.03648, 0.03714, 0.03782, 0.03849, 0.03920, 0.03991, 0.04063,
            0.04136, 0.04213, 0.04290, 0.04369, 0.04451, 0.04535, 0.04622, 0.04712,
            0.04803, 0.04898, 0.04994, 0.05092, 0.05192, 0.05294, 0.05397, 0.05503,
            0.05610, 0.05721, 0.05833, 0.05949, 0.06067, 0.06187, 0.06309, 0.06434,
            0.06559, 0.06687, 0.06814, 0.06944, 0.07073, 0.07201, 0.07327, 0.07450,
            0.07571, 0.07689, 0.07801, 0.07908, 0.08009, 0.08104, 0.08194, 0.08277,
            0.08353, 0.08422, 0.08483, 0.08538, 0.08584, 0.08625, 0.08659, 0.08685,
            0.08706, 0.08720, 0.08729, 0.08732, 0.08728, 0.08716, 0.08691, 0.08654,
            0.08602, 0.08530, 0.08440, 0.08327, 0.08189, 0.08025, 0.07838, 0.07630,
            0.07403, 0.07162, 0.06908, 0.06647, 0.06378, 0.06107, 0.05836, 0.05568,
            0.05306, 0.05048, 0.04794, 0.04546, 0.04301, 0.04061, 0.03825, 0.03592,
            0.03363, 0.03137, 0.02914, 0.02694, 0.02477, 0.02262, 0.02049, 0.01838,
            0.01630, 0.01422, 0.01216, 0.01011, 0.00808, 0.00605, 0.00403, 0.00201,
            0.00000,
        ),
        (
            0.18028, 0.18026, 0.18017, 0.17999, 0.17970, 0.17927, 0.17866, 0.17786,
            0.17684, 0.17558, 0.17403, 0.17233, 0.17077, 0.16944, 0.16824, 0.16708,
            0.16586, 0.16461, 0.16334, 0.16208, 0.16085, 0.15968, 0.15859, 0.15759,
            0.15667, 0.15583, 0.15506, 0.15438, 0.15377, 0.15323, 0.15278, 0.15240,
            0.15207, 0.15179, 0.15156, 0.15135, 0.15115, 0.15097, 0.15079, 0.15063,
            0.15048, 0.15035, 0.15023, 0.15012, 0.15004, 0.14997, 0.14992, 0.14988,
            0.14985, 0.14983, 0.14982, 0.14982, 0.14982, 0.14983, 0.14985, 0.14986,
            0.14989, 0.14992, 0.14995, 0.14999, 0.15002, 0.15006, 0.15010, 0.15014,
            0.15019, 0.15023, 0.15027, 0.15032, 0.15036, 0.15041, 0.15046, 0.15050,
            0.15056, 0.15060, 0.15066, 0.15070, 0.15076, 0.15081, 0.15085, 0.15090,
            0.15095, 0.15098, 0.15101, 0.15103, 0.15106, 0.15108, 0.15108, 0.15109,
            0.15110, 0.15110, 0.15110, 0.15110, 0.15109, 0.15108, 0.15106, 0.15103,
            0.15100, 0.15096, 0.15090, 0.15084, 0.15076, 0.15068, 0.15059, 0.15050,
            0.15038, 0.15026, 0.15014, 0.14999, 0.14983, 0.14965, 0.14945, 0.14924,
            0.14900, 0.14875, 0.14847, 0.14817, 0.14784, 0.14750, 0.14712, 0.14671,
            0.14628, 0.14582, 0.14532, 0.14479, 0.14422, 0.14362, 0.14296, 0.14226,
            0.14149, 0.14065, 0.13974, 0.13875, 0.13768, 0.13652, 0.13526, 0.13391,
            0.13246, 0.13090, 0.12925, 0.12748, 0.12560, 0.12366, 0.12163, 0.11954,
            0.11738, 0.11520, 0.11298, 0.11074, 0.10848, 0.10618, 0.10386, 0.10150,
            0.09911, 0.09667, 0.09420, 0.09168, 0.08911, 0.08650, 0.08384, 0.08115,
            0.07840, 0.07562, 0.07280, 0.06994, 0.06704, 0.06412, 0.06117, 0.05820,
            0.05524, 0.05228, 0.04935, 0.04644, 0.04358, 0.04078, 0.03803, 0.03536,
            0.03277, 0.03026, 0.02785, 0.02551, 0.02327, 0.02112, 0.01906, 0.01710,
            0.01522, 0.01345, 0.01178, 0.01021, 0.00875, 0.00740, 0.00615, 0.00501,
            0.00397, 0.00306, 0.00226, 0.00158, 0.00101, 0.00057, 0.00025, 0.00006,
            0.00000,
        ),
    ),
}


if __name__ == "__main__":
    main()
