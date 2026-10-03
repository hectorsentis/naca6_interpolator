"""Barrido de clases CST reutilizando cst.py, sin modificar sus valores globales.

    python cst_sweep.py 63-412
    python cst_sweep.py 63-412 --target thickness --n1 0.3,0.4,0.5,0.6 --n2 0.8,1,1.2

Por defecto varía la clase CAMBER; la clase thickness se mantiene (0.5,1).
--target thickness varía sólo espesor; --target both aplica la MISMA pareja
a ambas componentes (no es una búsqueda de cuatro exponentes independientes).
Filas=N1, columnas=N2, celdas=RMS geométrico combinado upper/lower.
Cada ajuste usa --points estaciones. La comparación usa otra malla densa
--validation-points para comprobar también el error entre estaciones.
Un único archivo de resultados: output/NACA_<perfil>_CST_<target>_matrix.csv.
No genera perfiles CSV individuales, plots, JSON ni archivos temporales.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

sys.dont_write_bytecode = True
import cst
import numpy as np


def _grid(values, name):
    grid = np.asarray(values, dtype=float)
    if grid.ndim != 1 or not grid.size or not np.isfinite(grid).all() or np.any(grid <= 0):
        raise ValueError(f"{name} debe contener valores positivos y finitos.")
    if np.unique(grid).size != grid.size:
        raise ValueError(f"{name} contiene valores repetidos.")
    return grid


def _parse_grid(text):
    try:
        return _grid([float(v.strip()) for v in text.split(',')], 'N1/N2')
    except (ValueError, TypeError) as exc:
        raise argparse.ArgumentTypeError('Usa una lista de números positivos separados por comas.') from exc


def sweep_classes(profile, n1_values, n2_values, *, target="camber", order_camber=5,
                  order_thickness=5, points=150, validation_points=1001,
                  distribution="coseno", a=1.0):
    """Devuelve dict con matriz RMS, mejor caso y métricas de validación.

    RMS combinado = sqrt(mean((dist_upper**2 + dist_lower**2)/2)).
    Una celda NaN indica ajuste inválido (p. ej., semiespesor negativo).
    """
    n1_values, n2_values = _grid(n1_values, 'N1'), _grid(n2_values, 'N2')
    if target not in ('camber', 'thickness', 'both'):
        raise ValueError('target debe ser camber, thickness o both.')
    nc, nt = cst._order(order_camber), cst._order(order_thickness)
    if points < max(nc, nt)+3:
        raise ValueError('Faltan puntos para los órdenes solicitados.')
    case = cst.generate_naca_camber_thickness(profile, points, distribution, a=a)
    validation = cst.generate_naca_camber_thickness(profile, validation_points, 'coseno', a=a)
    matrix = np.full((len(n1_values), len(n2_values)), np.nan)
    best = None
    for i, n1 in enumerate(n1_values):
        for j, n2 in enumerate(n2_values):
            camber_class = (float(n1), float(n2)) if target in ('camber','both') else (cst.CAMBER_N1,cst.CAMBER_N2)
            thickness_class = (float(n1), float(n2)) if target in ('thickness','both') else (cst.THICKNESS_N1,cst.THICKNESS_N2)
            try:
                cc = cst.fit_cst(case['xi'],case['yc'],nc,*camber_class)
                ct = cst.fit_cst(case['xi'],case['t'],nt,*thickness_class)
                xi = validation['xi']
                yc = cst.evaluate_cst(xi,cc,*camber_class)
                dyc = cst.evaluate_cst(xi,cc,*camber_class,derivative=True)
                t = cst.evaluate_cst(xi,ct,*thickness_class)
                surfaces = cst.reconstruct_surfaces(xi,yc,t,dyc)
                errors = cst.calculate_geometric_errors(validation['surfaces'],surfaces)
                rms = float(np.sqrt((errors['upper']['rms']**2+errors['lower']['rms']**2)/2))
            except (ValueError, np.linalg.LinAlgError, FloatingPointError):
                continue
            matrix[i,j] = rms
            if best is None or rms < best['rms']:
                best = dict(n1=float(n1),n2=float(n2),rms=rms,camber_class=camber_class,
                            thickness_class=thickness_class,camber_coefficients=cc,
                            thickness_coefficients=ct,errors=errors)
    if best is None:
        raise ValueError('Ninguna combinación produjo un ajuste válido.')
    return dict(profile=case['profile'],target=target,n1=n1_values,n2=n2_values,matrix=matrix,
                best=best,points=points,validation_points=validation_points,
                order_camber=nc,order_thickness=nt,distribution=distribution,a=float(a))


def write_matrix_csv(path, result):
    """Escribe la matriz y documenta clases y coeficientes de la mejor celda."""
    best = result['best']
    path = Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',encoding='utf-8',newline='') as stream:
        comments = ['TH-X01 CST CLASS SWEEP',f"Profile: NACA {result['profile']}",
                    f"Target: {result['target']}",f"a: {result['a']}",
                    f"Camber order: {result['order_camber']}",f"Thickness order: {result['order_thickness']}",
                    f"Fit points: {result['points']}",f"Fit distribution: {result['distribution']}",
                    f"Validation points: {result['validation_points']}; cosine distribution",
                    'Cells: combined upper/lower geometric RMS, normalized by nominal chord',
                    'Rows: N1; columns: N2; NaN: invalid fit',
                    f"Best N1: {best['n1']:.17g}",f"Best N2: {best['n2']:.17g}",
                    f"Best RMS: {best['rms']:.17g}",f"Best camber class: {best['camber_class']}",
                    f"Best thickness class: {best['thickness_class']}",
                    'Best camber coefficients: '+', '.join(f'{v:.17g}' for v in best['camber_coefficients']),
                    'Best thickness coefficients: '+', '.join(f'{v:.17g}' for v in best['thickness_coefficients']),
                    'Minimum is only among the tested pairs, not a continuous global optimum']
        for line in comments:
            stream.write('# '+line+'\n')
        writer = csv.writer(stream,lineterminator='\n')
        writer.writerow(['N1 / N2',*(f'{v:.17g}' for v in result['n2'])])
        for n1,row in zip(result['n1'],result['matrix']):
            writer.writerow([f'{n1:.17g}',*(f'{v:.17g}' for v in row)])


def main(argv=None):
    for stream in (sys.stdout,sys.stderr):
        if hasattr(stream,'reconfigure'):
            stream.reconfigure(encoding='utf-8',errors='backslashreplace')
    parser = argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('profile')
    parser.add_argument('--target',choices=('camber','thickness','both'),default='camber')
    parser.add_argument('--n1',type=_parse_grid,default=_parse_grid('0.3,0.5,0.7,0.9,1,1.1,1.3'))
    parser.add_argument('--n2',type=_parse_grid,default=_parse_grid('0.6,0.8,1,1.2,1.4,1.6'))
    parser.add_argument('--order-camber',type=int,default=5)
    parser.add_argument('--order-thickness',type=int,default=5)
    parser.add_argument('--points',type=int,default=150)
    parser.add_argument('--validation-points',type=int,default=1001)
    parser.add_argument('--distribution','--distribucion',default='coseno')
    parser.add_argument('--a',type=float,default=1.0)
    args = parser.parse_args(argv)
    try:
        result = sweep_classes(args.profile,args.n1,args.n2,target=args.target,
                               order_camber=args.order_camber,order_thickness=args.order_thickness,
                               points=args.points,validation_points=args.validation_points,
                               distribution=args.distribution,a=args.a)
        path = cst.OUTPUT_DIR/f"NACA_{result['profile']}_CST_{args.target}_matrix.csv"
        write_matrix_csv(path,result)
    except (ValueError,OSError) as exc:
        parser.error(str(exc))
    print(f"NACA {result['profile']} | clase variable: {args.target} | RMS geométrico combinado")
    print(' N1 / N2 '+''.join(f'{v:>12g}' for v in result['n2']))
    for n1,row in zip(result['n1'],result['matrix']):
        print(f'{n1:8g} '+''.join(f'{v:12.4e}' for v in row))
    best = result['best']
    print(f"Mejor combinación probada: N1={best['n1']:g}, N2={best['n2']:g}; RMS={best['rms']:.6e}")
    print(f"Matriz: {path}")


if __name__ == '__main__':
    main()
