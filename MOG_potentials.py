import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar

plt.rcParams.update({
    "font.size": 11,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "figure.dpi": 120,
})


# ----------------------------------------------------------------------
# Geometría de fondo (2M = 1, G_N = c = 1)
# ----------------------------------------------------------------------
def f(r, alpha=0.0):
    """Función lapso de Schwarzschild-MOG, Ec. (37) del preprint."""
    return 1.0 - (1.0 + alpha) / r + alpha * (1.0 + alpha) / (4.0 * r**2)


def df_dr(r, alpha=0.0):
    """Derivada radial f'(r), necesaria para el potencial escalar."""
    return (1.0 + alpha) / r**2 - alpha * (1.0 + alpha) / (2.0 * r**3)


def horizons(alpha=0.0):
    """Horizontes interno y externo, r_- y r_+ (analíticos)."""
    s = np.sqrt(1.0 + alpha)
    return 0.5 * ((1.0 + alpha) - s), 0.5 * ((1.0 + alpha) + s)


def surface_gravity(alpha=0.0):
    """Gravedad superficial del horizonte externo, kappa_+ = f'(r_+)/2."""
    _, r_p = horizons(alpha)
    return 0.5 * df_dr(r_p, alpha)


def tortoise(r, alpha=0.0):
    """
    Coordenada tortuga analítica para f(r) = (r - r_+)(r - r_-)/r^2:

        r_* = r + [r_+^2 ln(r/r_+ - 1) - r_-^2 ln(r/r_- - 1)] / (r_+ - r_-)

    En el límite alpha -> 0 (r_- = 0) se recupera Schwarzschild con 2M = 1.
    """
    r_m, r_p = horizons(alpha)
    rs = r + r_p**2 / (r_p - r_m) * np.log(r / r_p - 1.0)
    if r_m > 0:
        rs -= r_m**2 / (r_p - r_m) * np.log(r / r_m - 1.0)
    return rs


# ----------------------------------------------------------------------
# Potenciales efectivos
# ----------------------------------------------------------------------
def V_scalar(r, ell=2, mu=0.0, alpha=0.0):
    """Potencial escalar masivo, Ec. (15). Incluye el acople de curvatura f'(r)/r."""
    return f(r, alpha) * (ell * (ell + 1) / r**2 + df_dr(r, alpha) / r + mu**2)


def V_em(r, ell=1, alpha=0.0):
    """Potencial electromagnético de prueba (s = 1)

    La invariancia conforme del campo de Maxwell en D = 4 elimina el
    término f'(r)/r: el potencial solo contiene la barrera centrífuga.
    """
    return f(r, alpha) * ell * (ell + 1) / r**2


def V_coupled(r, ell=2, alpha=0.0, sector=2):
    """
    Potencial axial del sistema acoplado vector-tensorial

    sector = 1 : modo dominado por la perturbación vectorial (Z_1, usa q_2)
    sector = 2 : modo dominado por la perturbación gravitacional (Z_2, usa q_1)

    Nota: n2 = l(l+1) - 2 es el parámetro multipolar de Chandrasekhar
    (denotado mu^2 en el preprint); se renombra para no confundirlo con
    la masa del campo escalar.
    """
    A = ell * (ell + 1)
    n2 = A - 2.0
    Delta = r**2 - (1.0 + alpha) * r + alpha * (1.0 + alpha) / 4.0
    root = np.sqrt(9.0 * (1.0 + alpha)**2 + 4.0 * alpha * (1.0 + alpha) * n2)
    q1 = 0.5 * (3.0 * (1.0 + alpha) + root)
    q2 = 0.5 * (3.0 * (1.0 + alpha) - root)
    qj = q2 if sector == 1 else q1           # V_i usa q_j con j != i
    charge_term = alpha * (1.0 + alpha) / r  # 4 alpha(1+alpha) M^2 / r, con M = 1/2
    return (Delta / r**5) * (A * r - qj + charge_term)


# ----------------------------------------------------------------------
# Análisis de la barrera (justificación del método WKB)
# ----------------------------------------------------------------------
def barrier_maximum(V_func, alpha=0.0, r_search=200.0, N=200000, **kwargs):
    """
    Localiza la CIMA LOCAL de la barrera fuera del horizonte externo.

    Estrategia robusta en dos pasos:
      1. Malla densa en (r_+, r_search]: se identifican los máximos locales
         como cambios de signo + -> - de dV/dr. Esto evita el fallo de la
         optimización acotada cuando el paisaje no es unimodal (caso escalar
         masivo: cima local seguida de la subida asintótica hacia mu^2).
      2. Refinamiento con minimize_scalar restringido al entorno inmediato
         del primer máximo detectado (el más cercano al horizonte).

    Devuelve (r_max, V_max), o (None, None) si no existe cima local
    (potencial monótono: fuera del dominio de validez del WKB).
    """
    _, r_p = horizons(alpha)
    r = np.linspace(r_p * (1.0 + 1e-6), r_search, N)
    V = V_func(r, alpha=alpha, **kwargs)
    dV = np.gradient(V, r)
    sign_changes = np.where(np.diff(np.sign(dV)) != 0)[0]
    maxima = [i for i in sign_changes if dV[i] > 0 and dV[i + 1] < 0]
    if not maxima:
        return None, None
    i0 = maxima[0]
    a = r[max(i0 - 2, 0)]
    b = r[min(i0 + 3, N - 1)]
    res = minimize_scalar(
        lambda x: -V_func(x, alpha=alpha, **kwargs),
        bounds=(a, b), method="bounded",
    )
    return res.x, -res.fun


def d2V_drstar2(V_func, r_max, alpha=0.0, h=1e-5, **kwargs):
    """
    Curvatura de la barrera respecto a la coordenada tortuga, evaluada
    en r_max:  d^2V/dr_*^2 = f(r) d/dr [ f(r) dV/dr ].
    Es la cantidad que entra en la fórmula WKB de primer orden.
    """
    Vr = lambda x: V_func(x, alpha=alpha, **kwargs)
    dVdr = lambda x: (Vr(x + h) - Vr(x - h)) / (2.0 * h)
    g = lambda x: f(x, alpha) * dVdr(x)
    dg = (g(r_max + h) - g(r_max - h)) / (2.0 * h)
    return f(r_max, alpha) * dg


def check_single_barrier(V_func, alpha=0.0, r_search=200.0, N=200000, **kwargs):
    """Cuenta los máximos locales del potencial en el exterior (debe ser 1)."""
    _, r_p = horizons(alpha)
    r = np.linspace(r_p * (1.0 + 1e-5), r_search, N)
    V = V_func(r, alpha=alpha, **kwargs)
    dV = np.gradient(V, r)
    sign_changes = np.where(np.diff(np.sign(dV)) != 0)[0]
    return len([i for i in sign_changes if dV[i] > 0 and dV[i + 1] < 0])


SECTORES = {
    "Escalar (l=2)": (V_scalar, dict(ell=2, mu=0.0)),
    "EM (l=1)":      (V_em,      dict(ell=1)),
    "Z1 (l=2)":      (V_coupled, dict(ell=2, sector=1)),
    "Z2 (l=2)":      (V_coupled, dict(ell=2, sector=2)),
}


def tabla_barrera(alphas=(0, 1, 4, 9), latex=False):
    """
    Tabla (r_max, V_max) por sector y por alpha. Con latex=True imprime
    las filas en formato booktabs, listas para pegar en la tesis.
    """
    nombres = list(SECTORES.keys())
    if not latex:
        ancho = 22
        encabezado = f"{'alpha':>5} | " + " | ".join(f"{n:>{ancho}}" for n in nombres)
        print(encabezado)
        print("-" * len(encabezado))
    for a in alphas:
        celdas = []
        for nombre in nombres:
            Vf, kw = SECTORES[nombre]
            rm, vm = barrier_maximum(Vf, alpha=a, **kw)
            celdas.append((rm, vm))
        if latex:
            fila = f"        {a} & " + " & ".join(
                f"{rm:.3f}  & {vm:.4f}" for rm, vm in celdas) + r" \\"
            print(fila)
        else:
            print(f"{a:>5} | " + " | ".join(
                f"{rm:9.4f}, {vm:9.5f}" for rm, vm in celdas))


def tabla_curvatura(alphas=(0, 1, 4, 9), sector_nombre="Z2 (l=2)"):
    """Curvatura en la cima y escala eikonal sqrt(V_max) para un sector."""
    Vf, kw = SECTORES[sector_nombre]
    print(f"Sector {sector_nombre}:")
    print(f"{'alpha':>5} | {'r_max':>9} | {'V_max':>9} | {'sqrt(V_max)':>11} | {'d2V/dr*2':>12}")
    for a in alphas:
        rm, vm = barrier_maximum(Vf, alpha=a, **kw)
        c = d2V_drstar2(Vf, rm, alpha=a, **kw)
        print(f"{a:>5} | {rm:9.4f} | {vm:9.5f} | {np.sqrt(vm):11.4f} | {c:12.3e}")


def margen_masivo(alpha=1.0, ell=2, mu_values=(0, 0.2, 0.4, 0.6)):
    """
    Margen V_max - mu^2 del sector escalar masivo (frontera de validez WKB).
    Devuelve None en r_max si la cima local ha desaparecido.
    """
    print(f"Escalar masivo, alpha={alpha}, l={ell}:")
    print(f"{'mu':>5} | {'r_max':>9} | {'V_max':>9} | {'mu^2':>7} | {'margen':>9}")
    for mu in mu_values:
        rm, vm = barrier_maximum(V_scalar, alpha=alpha, ell=ell, mu=mu)
        if rm is None:
            print(f"{mu:>5} |  sin cima local: potencial monotono (WKB no aplicable)")
        else:
            print(f"{mu:>5} | {rm:9.4f} | {vm:9.5f} | {mu**2:7.4f} | {vm - mu**2:9.5f}")


# ----------------------------------------------------------------------
# Graficadores
# ----------------------------------------------------------------------
def _r_grid(alpha, r_max=60.0, N=40000, eps=1e-4):
    _, r_p = horizons(alpha)
    return np.linspace(r_p * (1.0 + eps), r_max, N)


def plot_scalar(alphas=(0, 1, 4, 9), ell=2, mu=0.0,
                coord="rstar", xlim=(-40, 60), savename=None):
    """Potencial escalar para distintos valores de alpha (mu fijo)."""
    fig, ax = plt.subplots(figsize=(7.0, 4.4))
    for a in alphas:
        r = _r_grid(a)
        x = tortoise(r, a) if coord == "rstar" else r
        ax.plot(x, V_scalar(r, ell=ell, mu=mu, alpha=a), label=rf"$\alpha = {a}$")
    ax.set_xlabel(r"$r_*$" if coord == "rstar" else r"$r$")
    ax.set_ylabel(r"$V_\ell(r)$")
    ax.set_title(rf"Potencial escalar ($\ell = {ell}$, $\mu = {mu}$)")
    if coord == "rstar":
        ax.set_xlim(*xlim)
    ax.legend()
    fig.tight_layout()
    if savename:
        fig.savefig(savename, bbox_inches="tight")
    return fig


def plot_scalar_mass(alpha=1.0, ell=2, mu_values=(0, 0.2, 0.4, 0.6),
                     coord="rstar", xlim=(-40, 80), savename=None):
    """Potencial escalar para distintas masas del campo (alpha fijo)."""
    fig, ax = plt.subplots(figsize=(7.0, 4.4))
    for mu in mu_values:
        r = _r_grid(alpha, r_max=120.0)
        x = tortoise(r, alpha) if coord == "rstar" else r
        ax.plot(x, V_scalar(r, ell=ell, mu=mu, alpha=alpha), label=rf"$\mu = {mu}$")
    ax.set_xlabel(r"$r_*$" if coord == "rstar" else r"$r$")
    ax.set_ylabel(r"$V_\ell(r)$")
    ax.set_title(rf"Potencial escalar masivo ($\ell = {ell}$, $\alpha = {alpha}$)")
    if coord == "rstar":
        ax.set_xlim(*xlim)
    ax.legend()
    fig.tight_layout()
    if savename:
        fig.savefig(savename, bbox_inches="tight")
    return fig


def plot_em(alphas=(0, 1, 4, 9), ell=1, coord="rstar",
            xlim=(-40, 60), savename=None):
    """Potencial electromagnético de prueba para distintos alpha."""
    fig, ax = plt.subplots(figsize=(7.0, 4.4))
    for a in alphas:
        r = _r_grid(a)
        x = tortoise(r, a) if coord == "rstar" else r
        ax.plot(x, V_em(r, ell=ell, alpha=a), label=rf"$\alpha = {a}$")
    ax.set_xlabel(r"$r_*$" if coord == "rstar" else r"$r$")
    ax.set_ylabel(r"$V^{\mathrm{EM}}_\ell(r)$")
    ax.set_title(rf"Potencial electromagnético ($\ell = {ell}$)")
    if coord == "rstar":
        ax.set_xlim(*xlim)
    ax.legend()
    fig.tight_layout()
    if savename:
        fig.savefig(savename, bbox_inches="tight")
    return fig


def plot_coupled(alphas=(0, 1, 4, 9), ell=2, sector=2, coord="rstar",
                 xlim=(-40, 60), savename=None):
    """Potencial axial acoplado V_i^(-) para distintos alpha."""
    fig, ax = plt.subplots(figsize=(7.0, 4.4))
    for a in alphas:
        r = _r_grid(a)
        x = tortoise(r, a) if coord == "rstar" else r
        ax.plot(x, V_coupled(r, ell=ell, alpha=a, sector=sector),
                label=rf"$\alpha = {a}$")
    ax.set_xlabel(r"$r_*$" if coord == "rstar" else r"$r$")
    ax.set_ylabel(rf"$V^{{(-)}}_{{{sector}}}(r)$")
    name = "electromagnético" if sector == 1 else "gravitacional"
    ax.set_title(rf"Sector acoplado {name} $Z_{sector}$ ($\ell = {ell}$)")
    if coord == "rstar":
        ax.set_xlim(*xlim)
    ax.legend()
    fig.tight_layout()
    if savename:
        fig.savefig(savename, bbox_inches="tight")
    return fig


def plot_comparison_sectors(alpha=1.0, savename=None):
    """Comparación de los tres sectores a l = 2 (análogo a Fig. 3-10 de Ladino)."""
    fig, ax = plt.subplots(figsize=(7.0, 4.4))
    r = _r_grid(alpha)
    x = tortoise(r, alpha)
    ax.plot(x, V_scalar(r, ell=2, mu=0.0, alpha=alpha), label="Escalar ($s=0$)")
    ax.plot(x, V_em(r, ell=2, alpha=alpha), label="Electromagnético ($s=1$)")
    ax.plot(x, V_coupled(r, ell=2, alpha=alpha, sector=2),
            label="Gravitacional $Z_2$ ($s=2$)")
    ax.set_xlabel(r"$r_*$")
    ax.set_ylabel(r"$V(r)$")
    ax.set_title(rf"Potenciales efectivos, $\ell = 2$, $\alpha = {alpha}$")
    ax.set_xlim(-40, 60)
    ax.legend()
    fig.tight_layout()
    if savename:
        fig.savefig(savename, bbox_inches="tight")
    return fig


# ----------------------------------------------------------------------
# Verificaciones, tablas y figuras
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # --- Chequeos de consistencia ---
    r_test = np.array([1.5, 2.0, 5.0, 10.0])
    assert np.allclose(f(r_test, 0.0), 1.0 - 1.0 / r_test), "f(r) no recupera Schwarzschild"
    rm, rp = horizons(0.0)
    assert np.isclose(rp, 1.0) and np.isclose(rm, 0.0), "horizontes incorrectos en alpha=0"
    V_rw = f(r_test, 0.0) * (6.0 / r_test**2 - 3.0 / r_test**3)
    assert np.allclose(V_coupled(r_test, ell=2, alpha=0.0, sector=2), V_rw), \
        "Z_2 no recupera Regge-Wheeler en alpha=0"
    V_em0 = f(r_test, 0.0) * 6.0 / r_test**2
    assert np.allclose(V_coupled(r_test, ell=2, alpha=0.0, sector=1), V_em0), \
        "Z_1 no recupera el potencial EM en alpha=0"
    print("Chequeos de consistencia: OK\n")

    # --- Barrera única (condición WKB) ---
    print("Número de máximos locales (debe ser 1):")
    for a in (0, 1, 4, 9):
        n_s = check_single_barrier(V_scalar, alpha=a, ell=2, mu=0.0)
        n_e = check_single_barrier(V_em, alpha=a, ell=1)
        n_g = check_single_barrier(V_coupled, alpha=a, ell=2, sector=2)
        print(f"  alpha={a}:  escalar={n_s}, EM={n_e}, grav Z2={n_g}")
    print()

    # --- Tablas para la tesis ---
    tabla_barrera()
    print("\nFilas en formato LaTeX (booktabs):")
    tabla_barrera(latex=True)
    print()
    tabla_curvatura()
    print()
    margen_masivo(alpha=1.0)

    # --- Figuras ---
    plot_scalar(savename="V_escalar_alpha.png")
    plot_scalar_mass(savename="V_escalar_masa.png")
    plot_em(savename="V_em_alpha.png")
    plot_coupled(sector=1, ell=2, savename="V_acoplado_Z1.png")
    plot_coupled(sector=2, ell=2, savename="V_acoplado_Z2.png")
    plot_comparison_sectors(alpha=1.0, savename="V_comparacion_sectores.png")
    print("\nFiguras generadas.")