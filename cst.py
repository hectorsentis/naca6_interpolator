"""TH-X01: generar NACA 6-series y ajustar CST en un único CSV.

    python cst.py 63-412 --order 5 --points 150
    python cst.py 632215 --distribution lineal --a 0.8

Dependencia: numpy; --plot requiere matplotlib y muestra una figura sin guardarla.
No necesita scipy ni genera JSON o temporales.
Resultado: output/NACA_<designación>_CST.csv, junto a este script.

Convención para SpaceClaim/ANSYS (x e y normalizados por cuerda):
    y(x) = x**0.5 * (1-x) * sum(A[i] * B[i,n](x), i=0..n)
    B[i,n](x) = comb(n,i) * x**i * (1-x)**(n-i)
AU y AL se utilizan con su signo, sin negar AL al reconstruir el intradós.
El modelo impone y(0)=y(1)=0; no incorpora un término de espesor de salida.
Referencia CST: B. M. Kulfan, doi:10.2514/1.29958.

El generador NACA devuelve abscisas upper/lower distintas por la composición
normal del espesor. Por defecto se evalúa un contorno denso, se localiza
el punto más alejado del borde de salida, y se traslada, rota y normaliza
su cuerda geométrica. Esto conserva el contorno sin recortar la nariz.
El marco y su transformación inversa se documentan en el CSV para CAD.
Se remuestrea LINEALMENTE en una x común con la distribución solicitada.
--frame nominal reproduce el marco anterior, que recorta puntos x<0.
Los errores se calculan frente al NACA remuestreado que aparece en el CSV,
en las estaciones elegidas. Las estadísticas no tienen ponderación; el
ajuste prioriza el primer 2% de cuerda con peso cuadrático 10 por defecto.
El objetivo LE de 1e-5 se comprueba, no se garantiza para todo orden/perfil.
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


def fit_cst(x, y, order=5, n1=N1, n2=N2, *, le_region=0.02, le_weight=10.0):
    """Ajuste independiente de una superficie por mínimos cuadrados en y.

    Devuelve order+1 coeficientes A0..An. No divide y por C(x), para
    evitar singularidades en los bordes y conservar mínimos cuadrados
    con ponderación local en el borde de ataque. le_weight multiplica el
    peso del error cuadrático (su raíz multiplica las filas de la matriz).
    Usa le_weight=1 para mínimos cuadrados sin ponderación.
    """
    x, y, order = _x_vector(x), _vector(y, "y"), _order(order)
    if x.shape != y.shape:
        raise ValueError("x e y deben tener el mismo número de puntos.")
    if np.any((x == 0) | (x == 1)) and np.any(np.abs(y[(x == 0) | (x == 1)]) > 1e-12):
        raise ValueError("Este modelo CST requiere y=0 en los bordes x=0 y x=1.")
    if np.unique(x[(x > 0) & (x < 1)]).size < order + 1:
        raise ValueError("Faltan estaciones interiores distintas para el orden CST solicitado.")
    matrix = class_function(x, n1, n2)[:, None] * bernstein_basis(x, order)
    if not math.isfinite(le_region) or not 0 < le_region <= 1:
        raise ValueError("le_region debe estar entre 0 y 1.")
    if not math.isfinite(le_weight) or le_weight < 1:
        raise ValueError("le_weight debe ser finito y >= 1.")
    weights = np.where(x <= le_region, math.sqrt(le_weight), 1.0)
    coefficients, _, rank, _ = np.linalg.lstsq(matrix * weights[:, None], y * weights, rcond=None)
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


def generate_common_coordinates(profile, points=150, distribution="coseno", *, a=1.0,
                                frame="leading-edge", return_frame=False):
    """Importa NACA, normaliza el marco y remuestrea en x común.

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

    metadata = {"frame": frame, "origin": (0.0, 0.0), "chord_vector": (1.0, 0.0)}
    if frame == "nominal":
        result = x, resample(xu, yu), resample(xl, yl)
    elif frame == "leading-edge":
        # Malla densa sólo para evaluar el contorno NACA, no para ajustar CST.
        ux, uy, lx, ly = (np.asarray(v) for v in naca6(profile, max(8193, points * 16), "coseno", a=a))
        contour = np.vstack((np.column_stack((ux, uy))[::-1], np.column_stack((lx, ly))[1:]))
        # Punto más alejado de TE: tangente perpendicular a la cuerda real.
        index = int(np.argmax(np.sum((contour - [1.0, 0.0])**2, axis=1)))
        origin = contour[index]
        direction = np.array([1.0, 0.0]) - origin
        chord_squared = float(direction @ direction)
        translated = contour - origin
        xx = translated @ direction / chord_squared
        yy = (direction[0] * translated[:, 1] - direction[1] * translated[:, 0]) / chord_squared
        upper_x, upper_y = xx[:index + 1][::-1], yy[:index + 1][::-1]
        lower_x, lower_y = xx[index:], yy[index:]
        if np.any(np.diff(upper_x) <= 0) or np.any(np.diff(lower_x) <= 0):
            raise ValueError("El contorno no es univaluado en la cuerda geométrica; prueba --frame nominal.")
        upper_y, lower_y = np.interp(x, upper_x, upper_y), np.interp(x, lower_x, lower_y)
        upper_y[[0, -1]], lower_y[[0, -1]] = 0.0, 0.0
        metadata.update(origin=tuple(origin), chord_vector=tuple(direction))
        result = x, upper_y, lower_y
    else:
        raise ValueError("frame debe ser leading-edge o nominal.")
    return (*result, metadata) if return_frame else result


def write_combined_csv(path, profile, x, yu_naca, yl_naca, upper_coefficients,
                       lower_coefficients, *, distribution="coseno", a=1.0,
                       frame_metadata=None, le_region=0.02, le_weight=10.0):
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
    metadata = frame_metadata or {"frame": "nominal", "origin": (0.0, 0.0), "chord_vector": (1.0, 0.0)}
    comments[comments.index("NACA resampling: linear onto common chordwise x; points outside [0,1] excluded")] = (
        "NACA resampling: dense contour, linear interpolation in geometric chord frame; no nose clipping"
        if metadata["frame"] == "leading-edge" else "NACA resampling: linear in nominal frame; points outside [0,1] excluded")
    comments += [f"Coordinate frame: {metadata['frame']}",
                 "Frame origin in original NACA coordinates: " + fmt(metadata["origin"]),
                 "Frame chord vector in original NACA coordinates: " + fmt(metadata["chord_vector"]),
                 "Inverse transform: original_point=origin+x*chord_vector+y*(-chord_vector_y,chord_vector_x)",
                 f"Leading edge region: 0 <= x <= {le_region:.17g}", f"Leading edge least-squares weight: {le_weight:.17g}",
                 f"Max abs LE error upper: {np.max(np.abs(upper['error'][x <= le_region])):.17g}",
                 f"Max abs LE error lower: {np.max(np.abs(lower['error'][x <= le_region])):.17g}"]
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


def plot_comparison(x, yu, yl, au, al, profile):
    """Figura interactiva: perfil y barras superpuestas de error por x/c.

    Las barras representan residuos locales, no frecuencias de residuos.
    No guarda ningún archivo persistente.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ValueError("Para --plot instala matplotlib: python -m pip install matplotlib") from exc
    uc, lc = evaluate_cst(x, au), evaluate_cst(x, al)
    fig, (ax, error_ax) = plt.subplots(2, 1, figsize=(11, 7), sharex=True,
                                     gridspec_kw={"height_ratios": [2, 1]})
    ax.scatter(x, yu, s=12, facecolors="none", edgecolors="tab:blue", label="NACA upper")
    ax.scatter(x, yl, s=12, facecolors="none", edgecolors="tab:orange", label="NACA lower")
    # Curvas CST densas para que también se vean entre las estaciones.
    dense_x = (1 - np.cos(np.linspace(0, np.pi, 2001))) / 2
    ax.plot(dense_x, evaluate_cst(dense_x, au), color="tab:blue", label="CST upper")
    ax.plot(dense_x, evaluate_cst(dense_x, al), color="tab:orange", label="CST lower")
    ax.set(ylabel="y/c", title=f"NACA {profile} — ajuste CST de orden {len(au)-1}")
    ax.set_aspect("equal", adjustable="box")
    ax.legend(ncol=2)
    edges = np.r_[0.0, (x[:-1] + x[1:]) / 2, 1.0]
    for error, color, name in ((uc-yu, "tab:blue", "Error upper"), (lc-yl, "tab:orange", "Error lower")):
        error_ax.bar(x, error, width=np.diff(edges), alpha=0.5, color=color, label=name)
    error_ax.axhline(0, color="black", linewidth=0.7)
    for threshold in (-1e-5, 1e-5):
        error_ax.axhline(threshold, color="gray", linestyle="--", linewidth=0.8)
    error_ax.set(xlabel="x/c", ylabel="Error CST − NACA [y/c]", title="Errores superpuestos a lo largo de la cuerda")
    error_ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    error_ax.legend()
    for axis in (ax, error_ax):
        axis.grid(alpha=0.25)
    fig.tight_layout()
    try:
        plt.show()
    finally:
        plt.close(fig)


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
    parser.add_argument("--plot", action="store_true", help="Mostrar comparación y barras de error; no guardar imagen")
    parser.add_argument("--frame", choices=("leading-edge", "nominal"), default="leading-edge", help="Marco geométrico del borde de ataque (defecto) o marco nominal anterior")
    parser.add_argument("--le-region", type=float, default=0.02, help="Zona LE en x/c, defecto primer 2%% de cuerda")
    parser.add_argument("--le-weight", type=float, default=10.0, help="Peso del error cuadrático en LE, defecto 10")
    args = parser.parse_args(argv)
    try:
        order = _order(args.order)
        if args.points < order + 3:
            raise ValueError("--points debe ser >= --order + 3, contando los dos bordes.")
        x, yu, yl, frame = generate_common_coordinates(args.profile, args.points, args.distribution, a=args.a, frame=args.frame, return_frame=True)
        au, al = (fit_cst(x, y, order, le_region=args.le_region, le_weight=args.le_weight) for y in (yu, yl))
        profile = _profile_name(args.profile)
        upper, lower = write_combined_csv(OUTPUT_DIR / f"NACA_{profile}_CST.csv", profile,
                                         x, yu, yl, au, al, distribution=args.distribution, a=args.a,
                                         frame_metadata=frame, le_region=args.le_region, le_weight=args.le_weight)
    except (ValueError, OSError, np.linalg.LinAlgError) as exc:
        parser.error(str(exc))
    print(f"Perfil: NACA {profile} | orden CST: {order} | puntos: {x.size}")
    print("AU: " + ", ".join(f"{v:.17g}" for v in au))
    print("AL: " + ", ".join(f"{v:.17g}" for v in al))
    print(f"Upper: RMS={upper['rms']:.6e} | error máximo={upper['max_abs']:.6e}")
    print(f"Lower: RMS={lower['rms']:.6e} | error máximo={lower['max_abs']:.6e}")
    le_error = max(np.max(np.abs(stats["error"][x <= args.le_region])) for stats in (upper, lower))
    print(f"LE (x/c <= {args.le_region:g}): error máximo={le_error:.6e}; objetivo 1e-5 {'alcanzado' if le_error <= 1e-5 else 'no alcanzado con este orden'}")
    if args.plot:
        try:
            plot_comparison(x, yu, yl, au, al, profile)
        except ValueError as exc:
            parser.error(str(exc))


if __name__ == "__main__":
    main()
