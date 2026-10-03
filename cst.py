"""TH-X01: generar NACA 6-series y ajustar CST en un único CSV.

    python cst.py 63-412 --order 5 --points 150
    python cst.py 632215 --distribution lineal --a 0.8

Dependencia: numpy. No necesita scipy ni genera plots, JSON o temporales.
Resultado: output/NACA_<designación>_CST.csv, junto a este script.

Convención para SpaceClaim/ANSYS (x e y normalizados por cuerda):
    y(x) = x**0.5 * (1-x) * sum(A[i] * B[i,n](x), i=0..n)
    B[i,n](x) = comb(n,i) * x**i * (1-x)**(n-i)
AU y AL se utilizan con su signo, sin negar AL al reconstruir el intradós.
El modelo impone y(0)=y(1)=0; no incorpora un término de espesor de salida.
Referencia CST: B. M. Kulfan, doi:10.2514/1.29958.

El generador NACA devuelve abscisas upper/lower distintas por la composición
normal del espesor. Se toma su estación de cuerda x=(xu+xl)/2 y se remuestrea
cada superficie por interpolación LINEAL para obtener una sola columna x.
Se conservan los extremos nominales (0,0) y (1,0). La pequeña prolongación
anterior a x=0 que puede aparecer en perfiles combados se excluye, pues
este modelo CST representa sólo el intervalo de cuerda 0<=x<=1.
Los errores se calculan frente al NACA remuestreado que aparece en el CSV,
en las estaciones elegidas y con igual peso por punto (no por longitud).
Los splines propios del generador NACA no se usan para ajustar el CST.
"""

from __future__ import annotations

import argparse
import csv
import math
from numbers import Integral
from pathlib import Path
import re
import sys

# La ejecución no debe crear archivos auxiliares __pycache__ al importar NACA.
sys.dont_write_bytecode = True

try:
    import numpy as np
except ImportError as exc:
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="backslashreplace")
    raise SystemExit("Falta numpy. Instálalo con: python -m pip install numpy") from exc

from naca6 import naca6

N1 = 0.5
N2 = 1.0
OUTPUT_DIR = Path(__file__).resolve().parent / "output"
COLUMNS = ("x", "yu_naca", "yl_naca", "yu_cst", "yl_cst", "error_upper", "error_lower")


def _vector(values, name):
    try:
        vector = np.asarray(values, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} debe contener números reales.") from exc
    if vector.ndim != 1 or vector.size == 0 or not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} debe ser un vector no vacío de números finitos.")
    return vector


def _x_vector(x):
    x = _vector(x, "x")
    if np.any((x < 0) | (x > 1)):
        raise ValueError("Las coordenadas x deben estar entre 0 y 1.")
    return x


def _order(order):
    if isinstance(order, bool) or not isinstance(order, Integral) or order < 0:
        raise ValueError("El orden CST debe ser un entero >= 0.")
    return int(order)


def class_function(x, n1=N1, n2=N2):
    """C(x) = x**n1 * (1-x)**n2, evaluada en un vector de x."""
    x = _x_vector(x)
    if not (math.isfinite(n1) and math.isfinite(n2) and n1 > 0 and n2 > 0):
        raise ValueError("N1 y N2 deben ser positivos y finitos.")
    return x**n1 * (1.0 - x)**n2


def bernstein_basis(x, order):
    """Matriz (n_puntos, order+1), columnas B0..Bn en ese orden."""
    x = _x_vector(x)
    order = _order(order)
    return np.column_stack([
        math.comb(order, i) * x**i * (1.0 - x)**(order - i)
        for i in range(order + 1)
    ])


def evaluate_cst(x, coefficients, n1=N1, n2=N2):
    """Reconstruye y(x); usa los coeficientes upper/lower con su signo."""
    x = _x_vector(x)
    coefficients = _vector(coefficients, "coefficients")
    return class_function(x, n1, n2) * (bernstein_basis(x, len(coefficients) - 1) @ coefficients)


def fit_cst(x, y, order=5, n1=N1, n2=N2):
    """Ajuste independiente de una superficie por mínimos cuadrados en y.

    Devuelve order+1 coeficientes A0..An. No divide y por C(x), para
    evitar singularidades en los bordes y conservar mínimos cuadrados
    sin ponderación. Rechaza mallas insuficientes y matrices sin rango.
    """
    x, y, order = _x_vector(x), _vector(y, "y"), _order(order)
    if x.shape != y.shape:
        raise ValueError("x e y deben tener el mismo número de puntos.")
    if np.any((x == 0) | (x == 1)) and np.any(np.abs(y[(x == 0) | (x == 1)]) > 1e-12):
        raise ValueError("Este modelo CST requiere y=0 en los bordes x=0 y x=1.")
    if np.unique(x[(x > 0) & (x < 1)]).size < order + 1:
        raise ValueError("Faltan estaciones interiores distintas para el orden CST solicitado.")
    matrix = class_function(x, n1, n2)[:, None] * bernstein_basis(x, order)
    coefficients, _, rank, _ = np.linalg.lstsq(matrix, y, rcond=None)
    if rank != order + 1 or not np.all(np.isfinite(coefficients)):
        raise ValueError("El ajuste CST no tiene rango completo; reduce el orden o aumenta los puntos.")
    return coefficients


def calculate_errors(y_naca, y_cst):
    """Devuelve dict con error firmado CST-NACA, RMS y máximo absoluto."""
    y_naca, y_cst = _vector(y_naca, "y_naca"), _vector(y_cst, "y_cst")
    if y_naca.shape != y_cst.shape:
        raise ValueError("Las superficies comparadas deben tener el mismo número de puntos.")
    error = y_cst - y_naca
    return {"error": error, "rms": float(np.sqrt(np.mean(error**2))),
            "max_abs": float(np.max(np.abs(error)))}


def _profile_name(profile):
    # Se llama después de que naca6 haya validado la designación.
    digits = re.sub(r"\D", "", str(profile))
    if len(digits) == 6:
        return f"{digits[:2]}({digits[2]})-{digits[3:]}"
    return f"{digits[:2]}-{digits[2:]}"


def generate_common_coordinates(profile, points=150, distribution="coseno", *, a=1.0):
    """Importa NACA y remuestrea linealmente upper/lower en x común.

    Devuelve x, yu_naca, yl_naca. Mantiene el número de puntos y las
    estaciones de cuerda de la distribución solicitada al generador.
    """
    xu, yu, xl, yl = (np.asarray(v, dtype=float) for v in naca6(profile, points, distribution, a=a))
    x = (xu + xl) / 2.0
    x[0], x[-1] = 0.0, 1.0

    def resample(xs, ys):
        inside = (xs > 0) & (xs < 1)
        xx = np.concatenate(([0.0], xs[inside], [1.0]))
        yy = np.concatenate(([0.0], ys[inside], [0.0]))
        if np.any(np.diff(xx) <= 0):
            raise ValueError("La superficie NACA no es univaluada en 0<x<1; no admite este ajuste CST.")
        return np.interp(x, xx, yy)

    return x, resample(xu, yu), resample(xl, yl)


def write_combined_csv(path, profile, x, yu_naca, yl_naca, upper_coefficients,
                       lower_coefficients, *, distribution="coseno", a=1.0):
    """Escribe el único CSV con metadatos, reconstrucción y errores.

    Las ordenadas CST y errores se calculan aquí desde los coeficientes,
    para que el encabezado y las filas representen exactamente el mismo caso.
    Devuelve (estadísticas_upper, estadísticas_lower).
    """
    x = _x_vector(x)
    yu_naca, yl_naca = _vector(yu_naca, "yu_naca"), _vector(yl_naca, "yl_naca")
    au, al = _vector(upper_coefficients, "AU"), _vector(lower_coefficients, "AL")
    if au.size != al.size or yu_naca.shape != x.shape or yl_naca.shape != x.shape:
        raise ValueError("Tamaños incompatibles de coordenadas o coeficientes.")
    if np.any(np.diff(x) <= 0) or x[0] != 0 or x[-1] != 1:
        raise ValueError("x debe crecer estrictamente de 0 a 1, incluyendo ambos bordes.")
    yu_cst, yl_cst = evaluate_cst(x, au), evaluate_cst(x, al)
    upper, lower = calculate_errors(yu_naca, yu_cst), calculate_errors(yl_naca, yl_cst)
    fmt = lambda values: ", ".join(format(float(v), ".17g") for v in values)
    comments = [
        "TH-X01 AIRFOIL CST FIT", f"Profile: NACA {profile}",
        f"CST order: {au.size - 1}", f"N1: {N1}", f"N2: {N2}",
        "Upper coefficients: " + fmt(au), "Lower coefficients: " + fmt(al),
        f"RMS error upper: {upper['rms']:.17g}", f"RMS error lower: {lower['rms']:.17g}",
        f"Max abs error upper: {upper['max_abs']:.17g}",
        f"Max abs error lower: {lower['max_abs']:.17g}", f"Number of points: {x.size}",
        f"Distribution: {distribution}", f"NACA loading a: {a:.17g}",
        "Coefficient order: A0 through An; use lower coefficients with their signed values",
        "CST formula: y=x^0.5*(1-x)*sum(Ai*binomial(n,i)*x^i*(1-x)^(n-i))",
        "Trailing edge term: 0; nominal endpoints (0,0) and (1,0)",
        "NACA resampling: linear onto common chordwise x; points outside [0,1] excluded",
        "Errors: CST minus resampled NACA; unweighted statistics on the output stations",
        "Columns below are normalized by chord",
    ]
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        for comment in comments:
            stream.write(f"# {comment}\n")
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(COLUMNS)
        for row in zip(x, yu_naca, yl_naca, yu_cst, yl_cst, upper["error"], lower["error"]):
            writer.writerow(format(float(v), ".17g") for v in row)
    return upper, lower


def main(argv=None):
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="backslashreplace")
    parser = argparse.ArgumentParser(description="TH-X01: NACA 6-series y ajuste CST en un único CSV.")
    parser.add_argument("profile", help="Perfil NACA, por ejemplo 63-412 o 632215")
    parser.add_argument("--order", type=int, default=5, help="Orden Bernstein, defecto 5 (6 coeficientes)")
    parser.add_argument("--points", type=int, default=150, help="Estaciones por superficie, defecto 150")
    parser.add_argument("--distribution", "--distribucion", default="coseno", help="lineal, coseno, seno o seno_salida")
    parser.add_argument("--a", type=float, default=1.0, help="Extensión de carga NACA, defecto 1")
    args = parser.parse_args(argv)
    try:
        order = _order(args.order)
        if args.points < order + 3:
            raise ValueError("--points debe ser >= --order + 3, contando los dos bordes.")
        x, yu, yl = generate_common_coordinates(args.profile, args.points, args.distribution, a=args.a)
        au, al = fit_cst(x, yu, order), fit_cst(x, yl, order)
        profile = _profile_name(args.profile)
        upper, lower = write_combined_csv(OUTPUT_DIR / f"NACA_{profile}_CST.csv", profile,
                                         x, yu, yl, au, al, distribution=args.distribution, a=args.a)
    except (ValueError, OSError, np.linalg.LinAlgError) as exc:
        parser.error(str(exc))
    print(f"Perfil: NACA {profile} | orden CST: {order} | puntos: {x.size}")
    print("AU: " + ", ".join(f"{v:.17g}" for v in au))
    print("AL: " + ", ".join(f"{v:.17g}" for v in al))
    print(f"Upper: RMS={upper['rms']:.6e} | error máximo={upper['max_abs']:.6e}")
    print(f"Lower: RMS={lower['rms']:.6e} | error máximo={lower['max_abs']:.6e}")


if __name__ == "__main__":
    main()
