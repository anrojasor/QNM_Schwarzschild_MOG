import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar


def calculate_wkb_guess(l, alpha, perturbation_type="GRAVITATIONAL", M=0.5):
    """
    Calcula la frecuencia QNM aproximada (WKB) para MOG.
    
    Parámetros:
        l (int): Número cuántico de momento angular.
        alpha (float): Parámetro de gravedad modificada.
        perturbation_type (str): "GRAVITATIONAL" o "ELECTROMAGNETIC".
        M (float): Masa del agujero negro.
    
    Retorna:
        (omega_real, omega_imag): Tupla con la semilla para la PINN.
    """
    
    # 1. Definición de Constantes MOG 
    M_eff = M * (1.0 + alpha)
    Qg_sq = alpha * (1.0 + alpha) * (M**2) 
    
    # Parámetros geométricos para f(r)
    Coeff_B = 2.0 * M_eff
    Coeff_C = Qg_sq
    
    # Horizonte de eventos r+
    r_plus = M * (1.0 + alpha + np.sqrt(1.0 + alpha))
    
    # 2. Selección de q_j 
    mu_sq = (l - 1.0) * (l + 2.0)
    root_term = np.sqrt(9.0 * (M_eff**2) + 4.0 * Qg_sq * mu_sq)
    
    if perturbation_type == "GRAVITATIONAL":
        # Rama Positiva (+)
        q_j = 3.0 * M_eff + root_term
    elif perturbation_type == "ELECTROMAGNETIC":
        # Rama Negativa (-)
        q_j = 3.0 * M_eff - root_term
    else:
        raise ValueError("Tipo de perturbación no válido")

    # 3. Definición del Potencial V(r) y sus derivadas
    Lambda = l * (l + 1.0)
    
    def get_functions(r):
        inv_r = 1.0 / r
        inv_r2 = inv_r * inv_r
        inv_r3 = inv_r2 * inv_r
        inv_r4 = inv_r3 * inv_r
        inv_r5 = inv_r4 * inv_r
        
        # Métrica f(r) y derivadas
        f = 1.0 - Coeff_B * inv_r + Coeff_C * inv_r2
        f_p = Coeff_B * inv_r2 - 2.0 * Coeff_C * inv_r3
        f_pp = -2.0 * Coeff_B * inv_r3 + 6.0 * Coeff_C * inv_r4
        
        # Parte 'U' del potencial y derivadas (usando Qg_sq = Q_eff^2)
        U = Lambda * inv_r2 - q_j * inv_r3 + 4.0 * Qg_sq * inv_r4
        U_p = -2.0 * Lambda * inv_r3 + 3.0 * q_j * inv_r4 - 16.0 * Qg_sq * inv_r5
        U_pp = 6.0 * Lambda * inv_r4 - 12.0 * q_j * inv_r5 + 80.0 * Qg_sq * (inv_r**6)
        
        # Potencial V = f * U
        V = f * U
        V_p = f_p * U + f * U_p
        V_pp = f_pp * U + 2.0 * f_p * U_p + f * U_pp
        
        return V, V_p, V_pp, f

    # 4. Encontrar el máximo del potencial (r_max)
    def neg_V(r):
        val, _, _, _ = get_functions(r)
        return -val

    res = minimize_scalar(neg_V, bounds=(r_plus * 1.01, r_plus * 10), method='bounded')
    r_peak = res.x
    
    # 5. Calcular valores en el pico para WKB
    V_max, V_p_max, V_pp_max, f_peak = get_functions(r_peak)
    
    # Segunda derivada respecto a la coordenada tortuga (r*)
    K = (f_peak**2) * V_pp_max
    
    # 6. Fórmulas WKB (Primer Orden / Eikonal)
    n_mode = 0
    w_real = np.sqrt(V_max)
    w_imag = - (n_mode + 0.5) * np.sqrt(-2.0 * K) / (2.0 * w_real)
    
    return w_real, w_imag



# Configuración de dispositivo 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_dtype(torch.float64)
print(f"Usando dispositivo: {device}")

# ==========================================
# 1. Configuración del Problema (MOG Black Hole)
# ==========================================


alpha_param = 1 # Cambiar a 0, 1, 4, 9 según se necesite
l_param = 1    # Número cuántico de momento angular (l=2 gravitacional)

perturbation_type_parameter = "ELECTROMAGNETIC" # Opciones: "GRAVITATIONAL" o "ELECTROMAGNETIC"

wr_init, wi_init = calculate_wkb_guess(l_param, alpha_param, perturbation_type_parameter, M=1.0)

# Nota: En unidades 2M=1, r+ y r- se calculan así:
r_plus = 0.5 * (1 + alpha_param + np.sqrt(1 + alpha_param))
r_minus = 0.5 * (1 + alpha_param - np.sqrt(1 + alpha_param))
Qg_sq = alpha_param / 4.0  # Q_g^2 = (sqrt(alpha)/2)^2 = alpha/4

print(f"Parámetros: Alpha={alpha_param}, l={l_param}")
print(f"Horizontes: r+ = {r_plus:.4f}, r- = {r_minus:.4f}")

print(f"Omegas de inicialización: w_r = {wr_init:.4f}, w_i = {wi_init:.4f}")

# ==========================================
# 2. Arquitectura de la Red Neuronal (PINN)
# ==========================================
class ComplexPINN(nn.Module):
    def __init__(self):
        super(ComplexPINN, self).__init__()
        # Entrada: xi (1 dimensión)
        # Salida: [Real(chi), Imag(chi)]
        self.net = nn.Sequential(
            nn.Linear(1, 50),
            nn.Tanh(),
            nn.Linear(50, 50),
            nn.Tanh(),
            nn.Linear(50, 50),
            nn.Tanh(),
            nn.Linear(50, 50),
            nn.Tanh(),
            nn.Linear(50, 50),
            nn.Tanh(),
            nn.Linear(50, 2) 
        )
        
        # Inicialización de pesos tipo Xavier
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

        # La frecuencia compleja omega es un parámetro entrenable
        self.omega_real = nn.Parameter(torch.tensor([wr_init*2], device=device))
        self.omega_imag = nn.Parameter(torch.tensor([wi_init*2], device=device))

    def forward(self, x):
        return self.net(x)

    def get_omega(self):
        return self.omega_real, self.omega_imag

# ==========================================
# 3. Definición de la Física (Ecuaciones AIM)
# ==========================================
def compute_loss(model, xi_batch):
    xi = xi_batch.requires_grad_(True)
    
    # Obtener omega actual
    w_r, w_i = model.get_omega()
    rho_r = w_i  # rho = -i * omega => rho_real = omega_imag
    rho_i = -w_r # rho_imag = -omega_real
    
    # Salida de la red: Chi = u + i*v
    out = model(xi)
    u = out[:, 0:1] # Parte Real
    v = out[:, 1:2] # Parte Imaginaria
    
    # --- Derivadas Automáticas (Autograd) ---
    grads_u = torch.autograd.grad(u, xi, torch.ones_like(u), create_graph=True)[0]
    grads_v = torch.autograd.grad(v, xi, torch.ones_like(v), create_graph=True)[0]
    
    u_xx = torch.autograd.grad(grads_u, xi, torch.ones_like(grads_u), create_graph=True)[0]
    v_xx = torch.autograd.grad(grads_v, xi, torch.ones_like(grads_v), create_graph=True)[0]

    # --- Construcción de Coeficientes ---
    xi_minus_1 = xi - 1.0
    one_minus_xi = 1.0 - xi
    
    num_delta = r_plus * xi * (r_plus - r_minus * one_minus_xi)
    den_delta = one_minus_xi ** 2
    Delta = num_delta / (den_delta + 1e-8)
    
    # --- Gamma_Z ---
    denom_gamma_3 = (r_plus - r_minus) * (r_plus - r_minus * one_minus_xi)
    denom_gamma_4 = (r_plus - r_minus) * xi
    
    C1 = (r_plus - r_minus) * one_minus_xi
    C2 = (r_plus - r_minus + r_plus**2) * one_minus_xi
    
    gamma_3_real = (C1 - rho_r * C2) / (denom_gamma_3 + 1e-8)
    gamma_3_imag = (-rho_i * C2) / (denom_gamma_3 + 1e-8)
    
    gamma_4_real = (rho_r * r_plus * one_minus_xi) / (denom_gamma_4 + 1e-8)
    gamma_4_imag = (rho_i * r_plus * one_minus_xi) / (denom_gamma_4 + 1e-8)
    
    GZ_real = -rho_r - (one_minus_xi / r_plus) + gamma_3_real + gamma_4_real
    GZ_imag = -rho_i + gamma_3_imag + gamma_4_imag

    GZ_real_sum = GZ_real.sum()
    GZ_imag_sum = GZ_imag.sum()
    GZ_xi_real = torch.autograd.grad(GZ_real_sum, xi, create_graph=True)[0]
    GZ_xi_imag = torch.autograd.grad(GZ_imag_sum, xi, create_graph=True)[0]
    
    # ==========================================
    # SELECCIÓN DEL TIPO DE PERTURBACIÓN 
    # ==========================================
    PERTURBATION_TYPE = perturbation_type_parameter 
    r_val = r_plus / (one_minus_xi + 1e-8)
    
    A = l_param * (l_param + 1.0)
    mu_sq = (l_param - 1.0) * (l_param + 2.0)
    
    # 1. Definiciones físicas generales 
    M_eff = 0.5 * (1.0 + alpha_param)
    Qg_sq = alpha_param * (1.0 + alpha_param) * (0.5**2) 
    
    # 2. Raíz para q_j usando torch.sqrt 
    interior_raiz = 9.0 * (M_eff**2) + 4.0 * Qg_sq * mu_sq
    # Convertimos a tensor flotante y lo enviamos al mismo dispositivo que xi
    interior_tensor = torch.tensor(interior_raiz, dtype=torch.float32, device=xi.device) 
    
    root_term = torch.sqrt(interior_tensor)
    
    if PERTURBATION_TYPE == "GRAVITATIONAL":
        q_j = 3.0 * M_eff + root_term
    elif PERTURBATION_TYPE == "ELECTROMAGNETIC":
        q_j = 3.0 * M_eff - root_term
    else:
        raise ValueError("Tipo de perturbación no válido")

    # 3. Potencial V_i^{(-)} corregido
    # Sabemos que r = r_plus / (1 - xi), por ende 1/r = (1 - xi) / r_plus
    bracket_V = (A * r_plus / (one_minus_xi + 1e-8)) - q_j + (4.0 * Qg_sq * one_minus_xi / r_plus)
    
    V_factor = Delta * (one_minus_xi**5) / (r_plus**5)
    V_real = V_factor * bracket_V
    V_imag = torch.zeros_like(V_real)

    # --- Coeficiente Lambda ---
    lam_term1 = (alpha_param*(alpha_param+1.0) + 4.0*Delta) / (2.0 * Delta * one_minus_xi + 1e-8)
    lam_num2_real = r_plus * (alpha_param + 1.0 + 2.0 * (GZ_real * Delta))
    lam_num2_imag = r_plus * (2.0 * GZ_imag * Delta)
    lam_den2 = Delta * (xi_minus_1**2) + 1e-8
    
    lam_real = lam_term1 - (lam_num2_real / lam_den2)
    lam_imag = - (lam_num2_imag / lam_den2)
    
    # --- Coeficiente s ---
    rho_sq_real = rho_r**2 - rho_i**2
    rho_sq_imag = 2.0 * rho_r * rho_i
    
    s_den_common = Delta**2 * (xi_minus_1**8) + 1e-10 
    T1_real = (r_plus**6) * (rho_sq_real + V_real) / s_den_common
    T1_imag = (r_plus**6) * (rho_sq_imag + V_imag) / s_den_common
    
    DG_real = Delta * GZ_real
    DG_imag = Delta * GZ_imag
    bra_real = 1.0 + alpha_param + DG_real
    bra_imag = DG_imag
    
    GB_real = GZ_real * bra_real - GZ_imag * bra_imag
    GB_imag = GZ_real * bra_imag + GZ_imag * bra_real
    
    den_T2 = Delta * (xi_minus_1**4) + 1e-8
    T2_real = (r_plus**2) * GB_real / den_T2
    T2_imag = (r_plus**2) * GB_imag / den_T2
    
    den_T3 = 2.0 * Delta * (xi_minus_1**3) + 1e-8
    const_T3 = alpha_param * (alpha_param + 1.0) * r_plus
    T3_real = const_T3 * GZ_real / den_T3
    T3_imag = const_T3 * GZ_imag / den_T3
    
    den_T4 = (xi_minus_1**2) + 1e-8
    T4_real = r_plus * GZ_xi_real / den_T4
    T4_imag = r_plus * GZ_xi_imag / den_T4
    
    s_real = T1_real - T2_real - T3_real - T4_real
    s_imag = T1_imag - T2_imag - T3_imag - T4_imag

    # ==========================================
    # 4. Cálculo del Residuo Regularizado
    # ==========================================
    Reg = (Delta**2) * (xi_minus_1**8)
    
    T_chi_pp_real = u_xx
    T_chi_pp_imag = v_xx
    
    LC_real = lam_real * grads_u - lam_imag * grads_v
    LC_imag = lam_real * grads_v + lam_imag * grads_u
    
    SC_real = s_real * u - s_imag * v
    SC_imag = s_real * v + s_imag * u
    
    Raw_Res_real = T_chi_pp_real - LC_real - SC_real
    Raw_Res_imag = T_chi_pp_imag - LC_imag - SC_imag
    
    scale_fix = 1.0 / (r_plus**4)

    if alpha_param > 1.0:
        Loss_ODE_real = Reg * Raw_Res_real * scale_fix
        Loss_ODE_imag = Reg * Raw_Res_imag * scale_fix
    else:
        Loss_ODE_real = Reg * Raw_Res_real
        Loss_ODE_imag = Reg * Raw_Res_imag
    
    L_ode = torch.mean(Loss_ODE_real**2 + Loss_ODE_imag**2)
    
    # 5. Condiciones de Frontera y Trivialidad
    mid_idx = xi.shape[0] // 2
    norm_mid = u[mid_idx]**2 + v[mid_idx]**2
    L_triv = (norm_mid - 1.0)**2
    
    return L_ode + L_triv

import torch
import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# 6. Entrenamiento y Visualización
# ==========================================
def train_pinn():
    model = ComplexPINN().to(device)
    # Fase 1: Adam para aproximación rápida
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    # Puntos de colocación: más densos cerca de los extremos 0 y 1
    # Usamos Chebyshev nodes reescalados a [0,1] para mejor estabilidad espectral
    N_points = 500
    cheb_nodes = np.cos(np.pi * (2 * np.arange(N_points) + 1) / (2 * N_points)) # [-1, 1]
    x_col = 0.5 * (cheb_nodes + 1) # [0, 1]
    # Recortar extremos muy cercanos para evitar NaNs numéricos extremos aunque esté regularizado
    x_col = x_col[(x_col > 1e-4) & (x_col < 1.0 - 1e-4)]
    
    xi_tensor = torch.tensor(x_col, dtype=torch.float64, device=device).unsqueeze(1)
    
    print("Iniciando entrenamiento con Adam...")
    for epoch in range(5000):
        optimizer.zero_grad()
        loss = compute_loss(model, xi_tensor)
        loss.backward()
        optimizer.step()
        
        if epoch % 500 == 0:
            w_r, w_i = model.get_omega()
            # Convertir a unidades M=1 (dividir por 2)
            final_wr = w_r.item() / 2.0
            final_wi = w_i.item() / 2.0
            print(f"Epoch {epoch}, Loss: {loss.item():.6f}, Omega(M=1): {final_wr:.4f} {final_wi:+.4f}i")

    # Fase 2: L-BFGS para precisión fina
    print("\nRefinando con L-BFGS...")
    optimizer_lbfgs = torch.optim.LBFGS(model.parameters(), 
                                        history_size=50, 
                                        max_iter=1000, 
                                        line_search_fn="strong_wolfe")
    
    def closure():
        optimizer_lbfgs.zero_grad()
        loss = compute_loss(model, xi_tensor)
        loss.backward()
        return loss

    optimizer_lbfgs.step(closure)
    
    # Resultado Final
    w_r, w_i = model.get_omega()
    final_wr = w_r.item() / 2.0
    final_wi = w_i.item() / 2.0
    print("\n" + "="*30)
    print(f"RESULTADO FINAL (MOG alpha={alpha_param}, l={l_param}):")
    print(f"Frecuencia calculada : {final_wr:.5f} {final_wi:+.5f}i")
    print("="*30)

   # ==========================================
    # 7. Evaluación y Ploteo de la Autofunción y Onda Física
    # ==========================================
    print("\nGenerando gráficos de las funciones...")
    
    # ---------------------------------------------------------
    # Gráfico 1: Autofunción regularizada chi(xi) en el dominio de la PINN
    # ---------------------------------------------------------
    xi_plot = np.linspace(1e-4, 1.0 - 1e-4, 500)
    xi_plot_tensor = torch.tensor(xi_plot, dtype=torch.float64, device=device).unsqueeze(1)
    
    model.eval()
    with torch.no_grad():
        salida_xi = model(xi_plot_tensor)
        chi_r_xi = salida_xi[:, 0].cpu().numpy().flatten()
        chi_i_xi = salida_xi[:, 1].cpu().numpy().flatten()

    fig1, ax1 = plt.subplots(figsize=(10, 6))
    ax1.plot(xi_plot, chi_r_xi, label=r'$\mathrm{Re}[\chi(\xi)]$', color='#1f77b4', linewidth=2)
    ax1.plot(xi_plot, chi_i_xi, label=r'$\mathrm{Im}[\chi(\xi)]$', color='#d62728', linestyle='--', linewidth=2)
    ax1.set_title(r"Autofunción Regularizada $\chi(\xi)$ (MOG $\alpha=$" + f"{alpha_param})", fontsize=15, pad=15)
    ax1.set_xlabel(r"Coordenada compactificada $\xi = 1 - r_+ / r$", fontsize=14)
    ax1.set_ylabel(r"Amplitud", fontsize=14)
    ax1.axhline(0, color='black', linewidth=1, linestyle=':')
    ax1.set_xlim(0, 1)
    ax1.grid(True, which='both', linestyle='--', alpha=0.6)
    ax1.legend(loc='upper right', fontsize=13)
    plt.tight_layout()
    plt.show()

    # ---------------------------------------------------------
    # Gráfico 2: Reconstrucción de la Onda Física Z(r)
    # ---------------------------------------------------------
    def reconstruir_onda_fisica(r_vals, omega_complex, r_plus, r_minus, model):
        """
        Calcula Z(r) evaluando la red neuronal y aplicando los factores asintóticos.
        Se utilizan las unidades internas del código (donde omega está en escala 2M=1).
        """
        # Mapear r a la coordenada xi de la red
        xi_vals = 1.0 - (r_plus / r_vals)
        xi_tensor = torch.tensor(xi_vals, dtype=torch.float64, device=device).unsqueeze(1)
        
        # Obtener chi(xi) de la red
        with torch.no_grad():
            salida_r = model(xi_tensor)
            chi_r = salida_r[:, 0].cpu().numpy().flatten()
            chi_i = salida_r[:, 1].cpu().numpy().flatten()
            chi_complex = chi_r + 1j * chi_i
            
        # Parámetro rho = -i * omega
        rho = -1j * omega_complex
        
        # Diferencia de horizontes
        delta_r = r_plus - r_minus
        
        # Exponentes de la
        exp_minus = 1.0 - rho - (rho * r_plus**2) / delta_r
        exp_plus = (rho * r_plus**2) / delta_r
        
        # Construcción del pre-factor analítico
        # Z_i = e^(-rho*r) * r^(-1) * (r - r_1)^(exp_minus) * (r - r_+)^(exp_plus) * chi
        prefactor = np.exp(-rho * r_vals) * (1.0 / r_vals) * \
                    ((r_vals - r_minus)**exp_minus) * \
                    ((r_vals - r_plus)**exp_plus)
                    
        Z_complex = prefactor * chi_complex
        return Z_complex

    # Generamos un grid uniforme en r, desde un poco afuera del horizonte hasta r_+ + 25
    r_fisico = np.linspace(r_plus + 0.05, r_plus + 25.0, 1000)
    
    # Recuperamos el omega INTERNO (escala 2M=1) directamente
    omega_interno = w_r.item() + 1j * w_i.item()
    
    Z_fisica = reconstruir_onda_fisica(r_fisico, omega_interno, r_plus, r_minus, model)

    # Physical wave plot configuration
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    ax2.plot(r_fisico, np.real(Z_fisica), label=r'$\mathrm{Re}[Z(r)]$', color='purple', linewidth=2)
    ax2.plot(r_fisico, np.imag(Z_fisica), label=r'$\mathrm{Im}[Z(r)]$', color='orange', linestyle='--', linewidth=2)
    
    # Plotting the envelope (absolute value)
    ax2.plot(r_fisico, np.abs(Z_fisica), label=r'$|Z(r)|$ (Envelope)', color='black', linestyle=':', linewidth=1.5)

    # Strict formatting
    ax2.set_title(r"Physical Perturbation Wave $Z(r)$ near Black Hole", fontsize=15, pad=15)
    ax2.set_xlabel(r"Radial coordinate $r$", fontsize=14)
    ax2.set_ylabel(r"Wave amplitude $Z(r)$", fontsize=14)
    ax2.axhline(0, color='black', linewidth=1, alpha=0.5)
    
    # Vertical line for event horizon position
    #ax2.axvline(r_plus + 1, color='red', linestyle='-.', alpha=0.6, label=r'Horizon $r_+$')
    
    ax2.set_xlim(r_plus, r_plus + 25.0)
    ax2.grid(True, which='both', linestyle='--', alpha=0.6)
    
    # Text box with wave info
    textstr2 = '\n'.join((
        r'$\mathbf{Mode\ Physics}$',
        f'$\\alpha = {alpha_param}$',
        f'$l = {l_param}$',
        f'Type: {perturbation_type_parameter[:4]}.'
    ))
    props2 = dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='gray', alpha=0.9)
    ax2.text(0.85, 0.15, textstr2, transform=ax2.transAxes, fontsize=12,
            verticalalignment='bottom', bbox=props2)
            
    ax2.legend(loc='upper left', fontsize=13)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    train_pinn()