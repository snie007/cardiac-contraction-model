"""
Interactive Web GUI for the Lewalle 2024 Cardiac Muscle Contraction Model
=========================================================================
Streamlit app with parameter sliders and real-time visualisation.

Run with:  streamlit run app.py
"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from Lewalle2024 import Lewalle2024

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Cardiac Contraction Model – Lewalle 2024",
    page_icon="❤️",
    layout="wide",
)

st.title("Cardiac Muscle Contraction Model")
st.markdown(
    "**Lewalle 2024** — Interactive explorer for the biophysical model of cardiac "
    "muscle contraction ([paper/code](https://github.com/CEMRG-publications/Lewalle_2025_BiophysJ)). "
    "Adjust parameters in the sidebar and the plots update in real time."
)

# ---------------------------------------------------------------------------
# Sidebar – parameter controls
# ---------------------------------------------------------------------------
st.sidebar.header("Model Parameters")

# --- Sarcomere geometry ---
st.sidebar.subheader("Sarcomere Geometry")
SL0 = st.sidebar.slider("SL₀ (resting sarcomere length, μm)", 1.6, 2.4, 1.8, 0.01)
Lambda_short = st.sidebar.slider(
    "λ  short SL (stretch ratio)", 0.90, 1.30, 1.05, 0.01,
    help="Stretch ratio for the SHORT sarcomere length curve"
)
Lambda_long = st.sidebar.slider(
    "λ  long SL (stretch ratio)", 0.90, 1.30, 1.25, 0.01,
    help="Stretch ratio for the LONG sarcomere length curve"
)

# --- Calcium ---
st.sidebar.subheader("Calcium Handling")
pCa_value = st.sidebar.slider(
    "pCa (for dynamic sims)", 3.5, 7.0, 4.5, 0.05,
    help="pCa = -log₁₀[Ca²⁺]. Lower = more calcium = stronger contraction"
)

# --- Active tension ---
st.sidebar.subheader("Active Tension")
Tref = st.sidebar.slider("T_ref (reference tension, kPa)", 5.0, 50.0, 23.0, 0.5)
pCa50ref = st.sidebar.slider("pCa50_ref (Ca sensitivity)", 4.5, 6.0, 5.25, 0.01)
ntrpn = st.sidebar.slider("n_trpn (troponin Hill coeff)", 1.0, 5.0, 2.58, 0.1)
ku = st.sidebar.slider("k_u (unblocking rate, s⁻¹)", 100.0, 3000.0, 1000.0, 50.0)
nTm = st.sidebar.slider("n_Tm (tropomyosin cooperativity)", 1.0, 5.0, 2.2, 0.1)
kuw = st.sidebar.slider("k_uw (U→W rate, s⁻¹)", 0.5, 20.0, 4.98, 0.1)
kws = st.sidebar.slider("k_ws (W→S rate, s⁻¹)", 1.0, 50.0, 19.10, 0.5)
rw = st.sidebar.slider("r_w (W-state duty ratio)", 0.1, 0.9, 0.5, 0.01)
rs = st.sidebar.slider("r_s (S-state duty ratio)", 0.05, 0.5, 0.25, 0.01)

# --- Cross-bridge distortion ---
st.sidebar.subheader("Cross-Bridge Distortion")
gs_val = st.sidebar.slider("g_s (S distortion decay, s⁻¹)", 5.0, 100.0, 42.1, 1.0)
gw_val = st.sidebar.slider("g_w (W distortion decay, s⁻¹)", 5.0, 100.0, 28.3, 1.0)
phi = st.sidebar.slider("φ (distortion coupling)", 0.01, 0.5, 0.1498, 0.005)
Aeff = st.sidebar.slider("A_eff (distortion sensitivity)", 10.0, 300.0, 125.0, 5.0)

# --- Passive tension ---
st.sidebar.subheader("Passive Tension")
a_pas = st.sidebar.slider("a (passive stiffness, Pa)", 50.0, 500.0, 241.0, 10.0)
b_pas = st.sidebar.slider("b (passive nonlinearity)", 1.0, 20.0, 9.1, 0.1)
k_pas = st.sidebar.slider("k (dashpot stiffness)", 1.0, 20.0, 8.86, 0.1)
eta_l = st.sidebar.slider("η_l (long dashpot, s)", 0.01, 1.0, 0.2, 0.01)
eta_s = st.sidebar.slider("η_s (short dashpot, ms)", 1.0, 100.0, 20.0, 1.0)

# --- Myosin OFF/ON ---
st.sidebar.subheader("Myosin OFF/ON State")
k1 = st.sidebar.slider("k₁ (OFF→ON rate, s⁻¹)", 0.1, 5.0, 0.877, 0.01)
k2 = st.sidebar.slider("k₂ (ON→OFF rate, s⁻¹)", 1.0, 30.0, 12.6, 0.1)

# --- Length-dependent activation ---
st.sidebar.subheader("Length-Dependent Activation")
beta0 = st.sidebar.slider("β₀ (force-length gain)", -5.0, 5.0, 0.0, 0.1)
beta1 = st.sidebar.slider("β₁ (Ca-sensitivity shift)", -5.0, 5.0, 0.0, 0.1)

# --- Dynamic experiment settings ---
st.sidebar.subheader("Dynamic Experiment")
exp_type = st.sidebar.selectbox("Experiment type", ["Step length change", "Sinusoidal oscillation"])
if exp_type == "Step length change":
    DLambda_step = st.sidebar.slider("Δλ step magnitude", -0.05, 0.05, 0.01, 0.001)
    t_duration = st.sidebar.slider("Duration (s)", 0.5, 20.0, 5.0, 0.5)
else:
    freq = st.sidebar.slider("Frequency (Hz)", 0.5, 50.0, 10.0, 0.5)
    n_cycles = st.sidebar.slider("Number of cycles", 5, 40, 20, 1)
    amp = st.sidebar.slider("Amplitude (Δλ)", 0.00001, 0.001, 0.0001, 0.00001, format="%.5f")


# ---------------------------------------------------------------------------
# Build model with current parameters
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def run_fpca(Tref, pCa50ref, ntrpn, ku, nTm, kuw, kws, rw, rs,
             gs_val, gw_val, phi, Aeff, a_pas, b_pas, k_pas, eta_l, eta_s_ms,
             k1, k2, beta0, beta1, Lambda_short, Lambda_long, SL0):
    """Compute steady-state F-pCa curves for two sarcomere lengths."""
    PSet = pd.Series({
        'Tref': Tref / 23.0,
        'pCa50ref': pCa50ref / 5.25,
        'ntrpn': ntrpn / 2.58,
        'ku': ku / 1000.0,
        'nTm': nTm / 2.2,
        'kuw': kuw / 4.98,
        'kws': kws / 19.10,
        'rw': rw / 0.5,
        'rs': rs / 0.25,
        'gs': gs_val / 42.1,
        'gw': gw_val / 28.3,
        'phi': phi / 0.1498,
        'Aeff': Aeff / 125.0,
        'a': a_pas / 241.0,
        'b': b_pas / 9.1,
        'k': k_pas / 8.86,
        'eta_l': eta_l / 0.2,
        'eta_s': (eta_s_ms * 1e-3) / 20e-3,
        'k1': k1 / 0.877,
        'k2': k2 / 12.6,
        'beta0': 1.0 if beta0 == 0.0 else beta0 / 0.0,  # handled below
        'beta1': 1.0 if beta1 == 0.0 else beta1 / 0.0,   # handled below
    })

    # beta0/beta1 defaults are 0, so multiplicative scaling doesn't work.
    # We override directly after construction.
    # Set them to 1.0 in PSet so the constructor doesn't break.
    PSet['beta0'] = 1.0
    PSet['beta1'] = 1.0

    model = Lewalle2024(PSet1=PSet)
    model.SL0 = SL0
    # Override beta0, beta1 directly (additive, not multiplicative)
    model.beta0 = beta0
    model.beta1 = beta1

    pCa_list = np.linspace(7.0, 4.0, 80)

    results = {}
    for lam_label, lam_val in [("short", Lambda_short), ("long", Lambda_long)]:
        model.Lambda_ext = lam_val
        Ta = model.Ta_ss(pCa_list)
        Tp = model.Tp_ss(lam_val)
        Ftotal = Tp + Ta

        # Get steady-state populations
        ss = model.Get_ss(pCa_list)  # shape (10, N)
        CaTRPN, B, S, W, Zs, Zw, Lambda, Cd, BE, UE = ss
        U = 1.0 - B - S - W - BE - UE

        results[lam_label] = {
            'pCa': pCa_list,
            'Ta': Ta,
            'Tp': np.full_like(pCa_list, Tp),
            'Ftotal': Ftotal,
            'SL': SL0 * lam_val,
            'Lambda': lam_val,
            'U': U, 'B': B, 'S': S, 'W': W, 'BE': BE, 'UE': UE,
            'CaTRPN': CaTRPN,
        }

    return results


@st.cache_data(show_spinner=False)
def run_dynamic_step(Tref, pCa50ref, ntrpn, ku, nTm, kuw, kws, rw, rs,
                     gs_val, gw_val, phi, Aeff, a_pas, b_pas, k_pas, eta_l, eta_s_ms,
                     k1, k2, beta0, beta1, pCa_value, DLambda_step, t_duration):
    """Run a step-length-change dynamic simulation."""
    PSet = pd.Series({
        'Tref': Tref / 23.0, 'pCa50ref': pCa50ref / 5.25,
        'ntrpn': ntrpn / 2.58, 'ku': ku / 1000.0, 'nTm': nTm / 2.2,
        'kuw': kuw / 4.98, 'kws': kws / 19.10, 'rw': rw / 0.5, 'rs': rs / 0.25,
        'gs': gs_val / 42.1, 'gw': gw_val / 28.3, 'phi': phi / 0.1498,
        'Aeff': Aeff / 125.0, 'a': a_pas / 241.0, 'b': b_pas / 9.1,
        'k': k_pas / 8.86, 'eta_l': eta_l / 0.2, 'eta_s': (eta_s_ms * 1e-3) / 20e-3,
        'k1': k1 / 0.877, 'k2': k2 / 12.6,
        'beta0': 1.0, 'beta1': 1.0,
    })
    model = Lewalle2024(PSet1=PSet)
    model.beta0 = beta0
    model.beta1 = beta1
    model.pCai = lambda t: pCa_value
    model.Lambda_ext = 1.0

    t = np.linspace(0, t_duration, 1000)
    model.DoDynamic(dLambdadt_imposed=lambda t: 0, t=t, DLambda_init=DLambda_step)

    res = model.ExpResults['Dynamic']
    Ysol = res['Ysol']
    Tasol = res['Tasol']
    Ta_ss_init = res['Ta_ss_init']
    F1 = model.F1(Ysol)
    F2 = model.F2(Ysol)
    Ttotal = model.Ttotal(Ysol)

    CaTRPN, B, S, W, Zs, Zw, Lambda, Cd, BE, UE = Ysol.T
    U = 1.0 - B - S - W - BE - UE

    return {
        't': t, 'Tasol': Tasol, 'Ttotal': Ttotal, 'F1': F1, 'F2': F2,
        'Lambda': Lambda, 'Ta_ss_init': Ta_ss_init,
        'U': U, 'B': B, 'S': S, 'W': W, 'BE': BE, 'UE': UE, 'CaTRPN': CaTRPN,
        'Zs': Zs, 'Zw': Zw,
    }


@st.cache_data(show_spinner=False)
def run_dynamic_sin(Tref, pCa50ref, ntrpn, ku, nTm, kuw, kws, rw, rs,
                    gs_val, gw_val, phi, Aeff, a_pas, b_pas, k_pas, eta_l, eta_s_ms,
                    k1, k2, beta0, beta1, pCa_value, freq, n_cycles, amp):
    """Run a sinusoidal-oscillation dynamic simulation."""
    PSet = pd.Series({
        'Tref': Tref / 23.0, 'pCa50ref': pCa50ref / 5.25,
        'ntrpn': ntrpn / 2.58, 'ku': ku / 1000.0, 'nTm': nTm / 2.2,
        'kuw': kuw / 4.98, 'kws': kws / 19.10, 'rw': rw / 0.5, 'rs': rs / 0.25,
        'gs': gs_val / 42.1, 'gw': gw_val / 28.3, 'phi': phi / 0.1498,
        'Aeff': Aeff / 125.0, 'a': a_pas / 241.0, 'b': b_pas / 9.1,
        'k': k_pas / 8.86, 'eta_l': eta_l / 0.2, 'eta_s': (eta_s_ms * 1e-3) / 20e-3,
        'k1': k1 / 0.877, 'k2': k2 / 12.6,
        'beta0': 1.0, 'beta1': 1.0,
    })
    model = Lewalle2024(PSet1=PSet)
    model.beta0 = beta0
    model.beta1 = beta1
    model.pCai = lambda t: pCa_value
    model.Lambda_ext = 1.0

    ppc = 30
    t = np.linspace(0, n_cycles / freq, n_cycles * ppc)
    dLdt = lambda t: amp * np.cos(2 * np.pi * freq * t) * 2 * np.pi * freq
    model.DoDynamic(dLambdadt_imposed=dLdt, t=t)

    res = model.ExpResults['Dynamic']
    Ysol = res['Ysol']
    Tasol = res['Tasol']
    CaTRPN, B, S, W, Zs, Zw, Lambda, Cd, BE, UE = Ysol.T
    U = 1.0 - B - S - W - BE - UE

    return {
        't': t, 'Tasol': Tasol, 'Lambda': Lambda,
        'U': U, 'B': B, 'S': S, 'W': W, 'BE': BE, 'UE': UE,
    }


# ---------------------------------------------------------------------------
# Run computations
# ---------------------------------------------------------------------------
with st.spinner("Computing steady-state F-pCa..."):
    fpca = run_fpca(Tref, pCa50ref, ntrpn, ku, nTm, kuw, kws, rw, rs,
                    gs_val, gw_val, phi, Aeff, a_pas, b_pas, k_pas, eta_l, eta_s,
                    k1, k2, beta0, beta1, Lambda_short, Lambda_long, SL0)

with st.spinner("Computing dynamic response..."):
    if exp_type == "Step length change":
        dyn = run_dynamic_step(Tref, pCa50ref, ntrpn, ku, nTm, kuw, kws, rw, rs,
                               gs_val, gw_val, phi, Aeff, a_pas, b_pas, k_pas, eta_l, eta_s,
                               k1, k2, beta0, beta1, pCa_value, DLambda_step, t_duration)
    else:
        dyn = run_dynamic_sin(Tref, pCa50ref, ntrpn, ku, nTm, kuw, kws, rw, rs,
                              gs_val, gw_val, phi, Aeff, a_pas, b_pas, k_pas, eta_l, eta_s,
                              k1, k2, beta0, beta1, pCa_value, freq, n_cycles, amp)


# ---------------------------------------------------------------------------
# PLOTS
# ---------------------------------------------------------------------------
COLORS = {
    'short': '#1f77b4',
    'long': '#ff7f0e',
    'Ta': '#e74c3c',
    'Tp': '#3498db',
    'Ftotal': '#2ecc71',
    'U': '#636EFA', 'B': '#EF553B', 'W': '#00CC96',
    'S': '#AB63FA', 'BE': '#FFA15A', 'UE': '#19D3F3',
}

# ===== Row 1: F-pCa curves =====
st.header("Steady-State Force–pCa Relationship")

col1, col2 = st.columns(2)

with col1:
    fig_fpca = go.Figure()
    for key, label_prefix in [("short", "Short"), ("long", "Long")]:
        d = fpca[key]
        sl_str = f"SL = {d['SL']:.2f} μm"
        fig_fpca.add_trace(go.Scatter(
            x=d['pCa'], y=d['Ftotal'] / 1000,
            name=f"Total ({sl_str})", mode='lines',
            line=dict(color=COLORS[key], width=2),
        ))
        fig_fpca.add_trace(go.Scatter(
            x=d['pCa'], y=d['Ta'] / 1000,
            name=f"Active ({sl_str})", mode='lines',
            line=dict(color=COLORS[key], width=2, dash='dash'),
        ))
        fig_fpca.add_trace(go.Scatter(
            x=d['pCa'], y=d['Tp'] / 1000,
            name=f"Passive ({sl_str})", mode='lines',
            line=dict(color=COLORS[key], width=2, dash='dot'),
        ))
    fig_fpca.update_layout(
        title="Force vs pCa",
        xaxis_title="pCa", yaxis_title="Force (kPa)",
        xaxis=dict(autorange='reversed'),
        height=420, template='plotly_white',
        legend=dict(font=dict(size=10)),
    )
    st.plotly_chart(fig_fpca, use_container_width=True)

with col2:
    fig_pop = go.Figure()
    d = fpca["long"]
    for state, label in [('U', 'U (unblocked-ON)'), ('B', 'B (blocked-ON)'),
                         ('W', 'W (weakly bound)'), ('S', 'S (strongly bound)'),
                         ('BE', 'BE (blocked-OFF)'), ('UE', 'UE (unblocked-OFF)')]:
        fig_pop.add_trace(go.Scatter(
            x=d['pCa'], y=d[state],
            name=label, mode='lines',
            line=dict(color=COLORS[state], width=2),
        ))
    fig_pop.update_layout(
        title=f"State Populations vs pCa  (SL = {d['SL']:.2f} μm)",
        xaxis_title="pCa", yaxis_title="Population fraction",
        xaxis=dict(autorange='reversed'),
        yaxis=dict(range=[0, 1]),
        height=420, template='plotly_white',
        legend=dict(font=dict(size=10)),
    )
    st.plotly_chart(fig_pop, use_container_width=True)


# ===== Row 2: Dynamic response =====
st.header("Dynamic Response")

if exp_type == "Step length change":
    col3, col4 = st.columns(2)

    with col3:
        fig_dyn = make_subplots(rows=2, cols=1, shared_xaxes=True,
                                subplot_titles=["Imposed Stretch (λ)", "Tension (kPa)"],
                                vertical_spacing=0.12)
        fig_dyn.add_trace(go.Scatter(
            x=dyn['t'], y=dyn['Lambda'], mode='lines',
            line=dict(color='#2ecc71', width=2), name='λ', showlegend=False,
        ), row=1, col=1)
        fig_dyn.add_trace(go.Scatter(
            x=dyn['t'], y=dyn['Tasol'] / 1000, mode='lines',
            line=dict(color='#e74c3c', width=2), name='Active (Tₐ)',
        ), row=2, col=1)
        fig_dyn.add_trace(go.Scatter(
            x=dyn['t'], y=dyn['Ttotal'] / 1000, mode='lines',
            line=dict(color='#2ecc71', width=2), name='Total',
        ), row=2, col=1)
        fig_dyn.update_layout(
            height=480, template='plotly_white',
            xaxis2_title="Time (s)",
        )
        st.plotly_chart(fig_dyn, use_container_width=True)

    with col4:
        fig_states = go.Figure()
        for state, label in [('U', 'U (unblocked-ON)'), ('B', 'B (blocked-ON)'),
                             ('W', 'W (weakly bound)'), ('S', 'S (strongly bound)'),
                             ('BE', 'BE (blocked-OFF)'), ('UE', 'UE (unblocked-OFF)')]:
            fig_states.add_trace(go.Scatter(
                x=dyn['t'], y=dyn[state], mode='lines',
                name=label, line=dict(color=COLORS[state], width=2),
            ))
        fig_states.update_layout(
            title="State Populations During Step Response",
            xaxis_title="Time (s)", yaxis_title="Population fraction",
            yaxis=dict(range=[0, 1]),
            height=480, template='plotly_white',
            legend=dict(font=dict(size=10)),
        )
        st.plotly_chart(fig_states, use_container_width=True)

    # Distortion plots
    st.subheader("Cross-Bridge Distortions")
    col5, col6 = st.columns(2)
    with col5:
        fig_z = go.Figure()
        fig_z.add_trace(go.Scatter(x=dyn['t'], y=dyn['Zs'], name='Zs (strong)', mode='lines',
                                   line=dict(color='#AB63FA', width=2)))
        fig_z.add_trace(go.Scatter(x=dyn['t'], y=dyn['Zw'], name='Zw (weak)', mode='lines',
                                   line=dict(color='#00CC96', width=2)))
        fig_z.update_layout(title="Cross-Bridge Distortions", xaxis_title="Time (s)",
                            yaxis_title="Distortion", height=350, template='plotly_white')
        st.plotly_chart(fig_z, use_container_width=True)
    with col6:
        fig_trpn = go.Figure()
        fig_trpn.add_trace(go.Scatter(x=dyn['t'], y=dyn['CaTRPN'], name='CaTRPN', mode='lines',
                                      line=dict(color='#e74c3c', width=2)))
        fig_trpn.update_layout(title="Troponin Calcium Binding", xaxis_title="Time (s)",
                               yaxis_title="CaTRPN", yaxis=dict(range=[0, 1]),
                               height=350, template='plotly_white')
        st.plotly_chart(fig_trpn, use_container_width=True)

else:
    # Sinusoidal
    col3, col4 = st.columns(2)
    with col3:
        fig_sin = make_subplots(rows=2, cols=1, shared_xaxes=True,
                                subplot_titles=["Imposed Stretch (λ)", "Active Tension (kPa)"],
                                vertical_spacing=0.12)
        fig_sin.add_trace(go.Scatter(
            x=dyn['t'], y=dyn['Lambda'], mode='lines',
            line=dict(color='#2ecc71', width=1.5), showlegend=False,
        ), row=1, col=1)
        fig_sin.add_trace(go.Scatter(
            x=dyn['t'], y=dyn['Tasol'] / 1000, mode='lines',
            line=dict(color='#e74c3c', width=1.5), showlegend=False,
        ), row=2, col=1)
        fig_sin.update_layout(height=480, template='plotly_white',
                              xaxis2_title="Time (s)")
        st.plotly_chart(fig_sin, use_container_width=True)

    with col4:
        fig_states = go.Figure()
        for state, label in [('U', 'U (unblocked-ON)'), ('B', 'B (blocked-ON)'),
                             ('W', 'W (weakly bound)'), ('S', 'S (strongly bound)'),
                             ('BE', 'BE (blocked-OFF)'), ('UE', 'UE (unblocked-OFF)')]:
            fig_states.add_trace(go.Scatter(
                x=dyn['t'], y=dyn[state], mode='lines',
                name=label, line=dict(color=COLORS[state], width=2),
            ))
        fig_states.update_layout(
            title="State Populations During Oscillation",
            xaxis_title="Time (s)", yaxis_title="Population fraction",
            yaxis=dict(range=[0, 1]),
            height=480, template='plotly_white',
            legend=dict(font=dict(size=10)),
        )
        st.plotly_chart(fig_states, use_container_width=True)


# ===== Row 3: Passive tension curve =====
st.header("Passive Tension vs Stretch")

@st.cache_data(show_spinner=False)
def compute_passive(a_pas, b_pas, SL0):
    lam_arr = np.linspace(0.9, 1.3, 100)
    Tp = a_pas * (np.exp(b_pas * (lam_arr - 1)) - 1)
    return lam_arr, Tp

lam_arr, Tp_curve = compute_passive(a_pas, b_pas, SL0)

col7, col8 = st.columns(2)
with col7:
    fig_pas = go.Figure()
    fig_pas.add_trace(go.Scatter(
        x=lam_arr * SL0, y=Tp_curve / 1000, mode='lines',
        line=dict(color='#3498db', width=2.5), name='Passive tension',
    ))
    fig_pas.update_layout(
        title="Passive Tension (F₁ spring element)",
        xaxis_title="Sarcomere Length (μm)", yaxis_title="Passive Tension (kPa)",
        height=380, template='plotly_white',
    )
    st.plotly_chart(fig_pas, use_container_width=True)

with col8:
    st.subheader("Calcium Troponin Binding (CaTRPN)")
    fig_trpn2 = go.Figure()
    d = fpca["long"]
    fig_trpn2.add_trace(go.Scatter(
        x=d['pCa'], y=d['CaTRPN'], mode='lines',
        line=dict(color='#e74c3c', width=2.5),
        name=f"SL = {d['SL']:.2f} μm",
    ))
    d2 = fpca["short"]
    fig_trpn2.add_trace(go.Scatter(
        x=d2['pCa'], y=d2['CaTRPN'], mode='lines',
        line=dict(color='#1f77b4', width=2.5),
        name=f"SL = {d2['SL']:.2f} μm",
    ))
    fig_trpn2.update_layout(
        title="Troponin Ca²⁺ Binding vs pCa",
        xaxis_title="pCa", yaxis_title="CaTRPN",
        xaxis=dict(autorange='reversed'),
        yaxis=dict(range=[0, 1]),
        height=380, template='plotly_white',
    )
    st.plotly_chart(fig_trpn2, use_container_width=True)


# ---------------------------------------------------------------------------
# Model explanation
# ---------------------------------------------------------------------------
with st.expander("About the Model"):
    st.markdown("""
### Lewalle 2024 – Biophysical Model of Cardiac Muscle Contraction

This model describes the mechanics of cardiac muscle at the sarcomere level, coupling:

**Myosin cross-bridge cycling** (active tension):
- **U** (Unblocked, ON): Myosin heads available for binding
- **W** (Weakly bound): Initial attachment to actin
- **S** (Strongly bound): Force-generating power-stroke state
- **B** (Blocked, ON): Tropomyosin blocks binding sites
- **UE** (Unblocked, OFF): Myosin in the super-relaxed OFF state
- **BE** (Blocked, OFF): Blocked + OFF state

**Thin filament regulation**:
- Calcium binds troponin (CaTRPN), shifting tropomyosin to unblock binding sites
- Cooperative transitions (nTm) amplify the calcium signal

**Passive viscoelastic elements**:
- Exponential spring (F₁) + dashpot element (F₂)

**Myosin thick filament regulation**:
- OFF/ON state transition (k₁, k₂) with force-dependent feedback
- Mechano-sensing stabilises the ON state under load

**Length-dependent activation (LDA)**:
- Longer sarcomeres → increased calcium sensitivity (β₁) and force (β₀)
- Explains the Frank-Starling mechanism of the heart

**Key reference**: Lewalle et al. 2025, Biophysical Journal.
[GitHub repository](https://github.com/CEMRG-publications/Lewalle_2025_BiophysJ)
""")
