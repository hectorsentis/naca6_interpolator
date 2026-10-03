"""TH-X01: CST de camber y semiespesor normal en la cuerda nominal NACA.

    python cst.py 63-412 --order-camber 5 --order-thickness 5 --points 150 --plot

Dependencias: numpy; matplotlib sólo para --plot. Un único CSV persistente:
output/NACA_<perfil>_CST.csv. El plot se muestra, no se guarda.

Camber: yc=xi*(1-xi)*sum(Ci*B_i,n(xi)); clase (1,1).
Espesor normal: t=sqrt(xi)*(1-xi)*sum(Ti*B_i,n(xi)); clase (0.5,1).
t significa SEMIESPESOR, igual que _thickness_at de naca6.py; no espesor total.
La clase camber (1,1) impone extremos nulos sin imponer una nariz redondeada
a la línea media. La línea media NACA tiene términos logarítmicos; un CST
polinómico finito no reproduce exactamente sus pendientes en los bordes.
Se usan derivadas analíticas CST, nunca diferencias finitas ni splines de
ajuste. En los extremos se adopta pendiente cero, como _mean_line de NACA;
la geometría no depende de esa convención porque allí yc=t=0.

theta=atan(dyc/dxi); xu=xi-t*sin(theta); yu=yc+t*cos(theta);
xl=xi+t*sin(theta); yl=yc-t*cos(theta). Sin rotaciones ni clipping de xu/xl.
LE nominal=(0,0), TE nominal=(1,0), alpha=0 referido al eje x original.
Para CAD, escalar todas las coordenadas por la cuerda dimensional deseada.
Referencia de la representación CST: B. M. Kulfan, doi:10.2514/1.29958.
"""

from __future__ import annotations

import argparse
import csv
import math
from numbers import Integral
from pathlib import Path
import re
import sys

sys.dont_write_bytecode = True
try:
    import numpy as np
except ImportError as exc:
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="backslashreplace")
    raise SystemExit("Falta numpy: python -m pip install numpy") from exc

from naca6 import _parse, _mean_line, _thickness_splines, _thickness_at

CAMBER_N1, CAMBER_N2 = 0.9, 0.8
THICKNESS_N1, THICKNESS_N2 = 0.5, 0.8
OUTPUT_DIR = Path(__file__).resolve().parent / "output"
COLUMNS = ("xi", "yc_naca", "t_naca", "yc_cst", "t_cst", "xu_naca", "yu_naca",
           "xl_naca", "yl_naca", "xu_cst", "yu_cst", "xl_cst", "yl_cst", "error_upper", "error_lower")


def _vector(values, name):
    try:
        result = np.asarray(values, dtype=float)
    except (ValueError, TypeError) as exc:
        raise ValueError(f"{name} debe contener números reales.") from exc
    if result.ndim != 1 or not result.size or not np.isfinite(result).all():
        raise ValueError(f"{name} debe ser un vector no vacío de números finitos.")
    return result


def _xi(values):
    values = _vector(values, "xi")
    if np.any((values < 0) | (values > 1)):
        raise ValueError("xi debe estar entre 0 y 1.")
    return values


def _order(value):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
        raise ValueError("El orden CST debe ser un entero >= 0.")
    return int(value)


def class_function(xi, n1=THICKNESS_N1, n2=THICKNESS_N2):
    """Clase xi**n1 * (1-xi)**n2; ambos exponentes positivos."""
    xi = _xi(xi)
    if not (math.isfinite(n1) and math.isfinite(n2) and n1 > 0 and n2 > 0):
        raise ValueError("Los exponentes de clase deben ser positivos y finitos.")
    return xi**n1 * (1 - xi)**n2


def bernstein_basis(xi, order):
    """Matriz de Bernstein (n_puntos, order+1), columnas B0..Bn."""
    xi, order = _xi(xi), _order(order)
    return np.column_stack([math.comb(order, i) * xi**i * (1-xi)**(order-i)
                            for i in range(order+1)])


def evaluate_cst(xi, coefficients, n1=THICKNESS_N1, n2=THICKNESS_N2,
                 *, derivative=False):
    """Evalúa CST o su derivada analítica. Extremos de derivada: convenio 0.

    Para camber se deben pasar n1=1,n2=1. Coeficientes C0..Cn o T0..Tn.
    """
    xi, coefficients = _xi(xi), _vector(coefficients, "coefficients")
    order = len(coefficients)-1
    cls = class_function(xi, n1, n2)
    shape = bernstein_basis(xi, order) @ coefficients
    if not derivative:
        return cls * shape
    result = np.zeros_like(xi)
    interior = (xi > 0) & (xi < 1)
    x = xi[interior]
    if x.size:
        shape_prime = (order * (bernstein_basis(x, order-1) @ np.diff(coefficients))
                       if order else np.zeros_like(x))
        class_prime = cls[interior] * (n1/x - n2/(1-x))
        result[interior] = class_prime * shape[interior] + cls[interior] * shape_prime
    return result


def fit_cst(xi, values, order=5, n1=THICKNESS_N1, n2=THICKNESS_N2):
    """Mínimos cuadrados sin ponderación en las ordenadas, independiente.

    Sin dividir por la clase: extremos no singulares y pesos iguales.
    """
    xi, values, order = _xi(xi), _vector(values, "values"), _order(order)
    if xi.shape != values.shape:
        raise ValueError("xi y values deben tener el mismo tamaño.")
    endpoints = (xi == 0) | (xi == 1)
    if np.any(np.abs(values[endpoints]) > 1e-12):
        raise ValueError("La clase CST requiere valores nulos en xi=0 y xi=1.")
    if np.unique(xi[~endpoints]).size < order+1:
        raise ValueError("Faltan estaciones interiores distintas para este orden.")
    matrix = class_function(xi, n1, n2)[:, None] * bernstein_basis(xi, order)
    coefficients, _, rank, _ = np.linalg.lstsq(matrix, values, rcond=None)
    if rank != order+1 or not np.isfinite(coefficients).all():
        raise ValueError("El ajuste no tiene rango completo; reduce el orden.")
    return coefficients


def reconstruct_surfaces(xi, yc, t, dyc):
    """Composición normal; devuelve xu,yu,xl,yl sin recortar abscisas."""
    xi = _xi(xi)
    yc, t, dyc = (_vector(v, name) for v, name in ((yc, "yc"), (t, "t"), (dyc, "dyc")))
    if any(v.shape != xi.shape for v in (yc, t, dyc)):
        raise ValueError("xi, yc, t y dyc deben tener el mismo tamaño.")
    if np.any(t < 0):
        raise ValueError("El semiespesor normal no puede ser negativo.")
    theta = np.arctan(dyc)
    dx, dy = t * np.sin(theta), t * np.cos(theta)
    return xi-dx, yc+dy, xi+dx, yc-dy


def _profile_name(profile):
    digits = re.sub(r"\D", "", str(profile))
    return (f"{digits[:2]}({digits[2]})-{digits[3:]}" if len(digits) == 6
            else f"{digits[:2]}-{digits[2:]}")


def generate_naca_camber_thickness(profile, points=150, distribution="coseno", *, a=1.0):
    """Obtiene yc, dyc y SEMIESPESOR directamente de los auxiliares NACA.

    Devuelve dict con xi, yc, t, dyc, surfaces y metadatos del caso.
    No deduce línea media/espesor a partir de las superficies.
    """
    family, cl, tc = _parse(profile)
    if isinstance(points, bool) or not isinstance(points, Integral) or points < 3:
        raise ValueError("points debe ser un entero >= 3.")
    try:
        a = float(a)
    except (TypeError, ValueError) as exc:
        raise ValueError("a debe estar entre 0 y 1.") from exc
    if not math.isfinite(a) or not 0 <= a <= 1:
        raise ValueError("a debe ser finito y estar entre 0 y 1.")
    u = np.linspace(0, 1, points)
    distributions = {"lineal": lambda: u, "linear": lambda: u,
                     "coseno": lambda: (1-np.cos(np.pi*u))/2,
                     "cosine": lambda: (1-np.cos(np.pi*u))/2,
                     "seno": lambda: 1-np.cos(np.pi*u/2),
                     "sine": lambda: 1-np.cos(np.pi*u/2),
                     "seno_salida": lambda: np.sin(np.pi*u/2),
                     "sine_te": lambda: np.sin(np.pi*u/2)}
    try:
        xi = distributions[distribution.strip().lower()]().copy()
    except (KeyError, AttributeError) as exc:
        raise ValueError("Distribución no válida: lineal, coseno, seno o seno_salida.") from exc
    xi[0], xi[-1] = 0.0, 1.0
    yc, dyc = np.array([_mean_line(float(x), cl, a) for x in xi]).T
    sx, sy = _thickness_splines(family, tc)
    t = np.array([_thickness_at(float(x), sx, sy) for x in xi])
    return dict(profile=_profile_name(profile), family=family, cl=cl, tc=tc, a=a,
                distribution=distribution, xi=xi, yc=yc, t=t, dyc=dyc,
                surfaces=reconstruct_surfaces(xi, yc, t, dyc))


def _statistics(error):
    error = _vector(error, "error")
    return dict(error=error, rms=float(np.sqrt(np.mean(error**2))),
                max_abs=float(np.max(np.abs(error))))


def calculate_geometric_errors(naca_surfaces, cst_surfaces):
    """Distancias euclídeas upper/lower en la MISMA estación xi (no firmadas).

    Entradas: tuplas (xu,yu,xl,yl). Devuelve estadísticas upper/lower.
    """
    if len(naca_surfaces) != 4 or len(cst_surfaces) != 4:
        raise ValueError("Cada geometría debe contener xu,yu,xl,yl.")
    original = tuple(_vector(v, "NACA") for v in naca_surfaces)
    fitted = tuple(_vector(v, "CST") for v in cst_surfaces)
    if any(v.shape != original[0].shape for v in original+fitted):
        raise ValueError("Las coordenadas deben tener el mismo tamaño.")
    return dict(upper=_statistics(np.hypot(fitted[0]-original[0], fitted[1]-original[1])),
                lower=_statistics(np.hypot(fitted[2]-original[2], fitted[3]-original[3])))


def build_cst_result(case, camber_coefficients, thickness_coefficients):
    """Reconstruye desde coeficientes, con derivada analítica de camber."""
    xi = case["xi"]
    cc, ct = _vector(camber_coefficients, "C"), _vector(thickness_coefficients, "T")
    yc = evaluate_cst(xi, cc, CAMBER_N1, CAMBER_N2)
    dyc = evaluate_cst(xi, cc, CAMBER_N1, CAMBER_N2, derivative=True)
    t = evaluate_cst(xi, ct, THICKNESS_N1, THICKNESS_N2)
    surfaces = reconstruct_surfaces(xi, yc, t, dyc)
    return dict(yc=yc, t=t, dyc=dyc, surfaces=surfaces,
                camber_coefficients=cc, thickness_coefficients=ct,
                camber_errors=_statistics(yc-case["yc"]), thickness_errors=_statistics(t-case["t"]),
                geometric_errors=calculate_geometric_errors(case["surfaces"], surfaces))


def write_combined_csv(path, case, camber_coefficients, thickness_coefficients):
    """Escribe un único CSV de 15 columnas; devuelve la reconstrucción CST."""
    result = build_cst_result(case, camber_coefficients, thickness_coefficients)
    fmt = lambda values: ", ".join(f"{v:.17g}" for v in values)
    comments = ["TH-X01 AIRFOIL CST FIT", f"Profile: NACA {case['profile']}",
                f"NACA family: {case['family']}", f"Cl design: {case['cl']:.17g}",
                f"t/c: {case['tc']:.17g}", f"a: {case['a']:.17g}",
                f"CST camber order: {len(camber_coefficients)-1}",
                f"CST thickness order: {len(thickness_coefficients)-1}",
                "Camber coefficients C0..Cn: " + fmt(result["camber_coefficients"]),
                "Thickness coefficients T0..Tn: " + fmt(result["thickness_coefficients"]),
                "Camber class N1,N2: 1,1; finite polynomial approximation of logarithmic NACA mean line",
                "Thickness class N1,N2: 0.5,1; t is NORMAL HALF-THICKNESS",
                "CST formula: f(xi)=xi^N1*(1-xi)^N2*sum(Ai*binomial(n,i)*xi^i*(1-xi)^(n-i))",
                "Camber derivative: analytic derivative of class times Bernstein shape",
                "Normal reconstruction: theta=atan(dyc/dxi); xu=xi-t*sin(theta); yu=yc+t*cos(theta); xl=xi+t*sin(theta); yl=yc-t*cos(theta)",
                "Nominal chord: LE=(0,0); TE=(1,0); chord axis=x; alpha=0 unchanged; no rotation or clipping",
                "Endpoint slope convention: 0 as in NACA generator; geometry independent since yc=t=0",
                "Geometry errors: Euclidean distance between corresponding NACA and CST points at the same xi",
                "Least squares: independent unweighted fits of camber and half-thickness; no surface fit",
                f"Number of points: {len(case['xi'])}", f"Distribution: {case['distribution']}"]
    for label, stats in (("camber", result["camber_errors"]), ("thickness", result["thickness_errors"]),
                         ("geometry upper", result["geometric_errors"]["upper"]),
                         ("geometry lower", result["geometric_errors"]["lower"])):
        comments += [f"RMS error {label}: {stats['rms']:.17g}", f"Max abs error {label}: {stats['max_abs']:.17g}"]
    comments.append("Columns below are normalized by nominal NACA chord")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        for comment in comments:
            stream.write(f"# {comment}\n")
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(COLUMNS)
        arrays = (case["xi"], case["yc"], case["t"], result["yc"], result["t"],
                  *case["surfaces"], *result["surfaces"],
                  result["geometric_errors"]["upper"]["error"], result["geometric_errors"]["lower"]["error"])
        for row in zip(*arrays):
            writer.writerow(f"{value:.17g}" for value in row)
    return result


def plot_comparison(case, result):
    """Perfil paramétrico, distancias por xi y componentes; no guarda PNG."""
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ValueError("Para --plot: python -m pip install matplotlib") from exc
    fig, axes = plt.subplots(3, 1, figsize=(11, 9))
    ax, error_ax, components = axes
    for geometry, style, label in ((case["surfaces"], "o", "NACA"), (result["surfaces"], "-", "CST")):
        xu, yu, xl, yl = geometry
        ax.plot(xu, yu, style, color="tab:blue", markersize=3, label=label+" upper")
        ax.plot(xl, yl, style, color="tab:orange", markersize=3, label=label+" lower")
    ax.axhline(0, color="gray", linewidth=0.7)
    ax.set(xlabel="x/c nominal", ylabel="y/c", title=f"NACA {case['profile']} — CST camber + semiespesor normal")
    ax.set_aspect("equal", adjustable="box")
    xi = case["xi"]
    for side, color in (("upper", "tab:blue"), ("lower", "tab:orange")):
        error_ax.plot(xi, result["geometric_errors"][side]["error"], color=color, label=side)
    error_ax.set(xlabel="xi", ylabel="Distancia / c", title="Error geométrico en estaciones correspondientes")
    for key, label, color in (("yc", "Camber", "tab:green"), ("t", "Semiespesor", "tab:purple")):
        components.plot(xi, case[key], "o", markersize=3, color=color, label=label+" NACA")
        components.plot(xi, result[key], "-", color=color, label=label+" CST")
    components.set(xlabel="xi", ylabel="yc/c, t/c", title="Componentes paramétricas")
    for axis in axes:
        axis.legend(ncol=2)
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
    parser = argparse.ArgumentParser(description="TH-X01: CST de camber y semiespesor normal NACA en un único CSV.")
    parser.add_argument("profile", help="Por ejemplo 63-412 o 632215")
    parser.add_argument("--order-camber", type=int, default=5)
    parser.add_argument("--order-thickness", type=int, default=5)
    parser.add_argument("--points", type=int, default=150)
    parser.add_argument("--distribution", "--distribucion", default="coseno")
    parser.add_argument("--a", type=float, default=1.0)
    parser.add_argument("--plot", action="store_true", help="Mostrar comparación sin guardar imágenes")
    args = parser.parse_args(argv)
    try:
        nc, nt = _order(args.order_camber), _order(args.order_thickness)
        if args.points < max(nc, nt)+3:
            raise ValueError("--points debe ser >= máximo de los órdenes + 3.")
        case = generate_naca_camber_thickness(args.profile, args.points, args.distribution, a=args.a)
        cc = fit_cst(case["xi"], case["yc"], nc, CAMBER_N1, CAMBER_N2)
        ct = fit_cst(case["xi"], case["t"], nt, THICKNESS_N1, THICKNESS_N2)
        result = write_combined_csv(OUTPUT_DIR / f"NACA_{case['profile']}_CST.csv", case, cc, ct)
    except (ValueError, OSError, np.linalg.LinAlgError) as exc:
        parser.error(str(exc))
    print(f"Perfil: NACA {case['profile']} | cuerda nominal | puntos: {args.points}")
    print("C: " + ", ".join(f"{v:.17g}" for v in cc))
    print("T: " + ", ".join(f"{v:.17g}" for v in ct))
    for label, stats in (("Camber", result["camber_errors"]), ("Semiespesor", result["thickness_errors"]),
                         ("Upper", result["geometric_errors"]["upper"]), ("Lower", result["geometric_errors"]["lower"])):
        print(f"{label}: RMS={stats['rms']:.6e} | máximo={stats['max_abs']:.6e}")
    if args.plot:
        try:
            plot_comparison(case, result)
        except ValueError as exc:
            parser.error(str(exc))


if __name__ == "__main__":
    main()
