# Lewalle 2024 - Cardiac Muscle Contraction Model
# From: https://github.com/CEMRG-publications/Lewalle_2025_BiophysJ
#
# Biophysical model of cardiac muscle contraction with:
# - Passive tension (viscoelastic elements)
# - Active tension (cross-bridge cycling with thin filament regulation)
# - Myosin OFF/ON state dynamics
# - Length-dependent activation (LDA)

import numpy as np
from scipy.integrate import odeint
import pandas as pd


AllParams = ('a', 'b', 'k', 'eta_l', 'eta_s',
             'k_trpn_on', 'k_trpn_off', 'ntrpn', 'pCa50ref',
             'ku', 'nTm', 'trpn50', 'kuw', 'kws',
             'rw', 'rs', 'gs', 'gw', 'phi',
             'Aeff', 'beta0', 'beta1', 'Tref',
             'k2', 'k1', 'koffon')


def HillFn(x, ymax, n, ca50):
    return ymax * x**n / (x**n + ca50**n)


class Lewalle2024:

    # Defaults
    SL0 = 1.8               # um
    pCai = lambda self, t: 4.5

    # Passive tension parameters
    a = 241                 # Pa
    b = 9.1                 # dimensionless
    k = 8.86                # dimensionless
    eta_l = 0.2             # s
    eta_s = 20e-3           # s

    # Active tension parameters
    k_trpn_on = 0.1e3       # s-1
    k_trpn_off = 0.1e3      # s-1
    ntrpn = 2.58            # dimensionless
    pCa50ref = 5.25         # M
    ku = 1000               # s-1
    nTm = 2.2               # dimensionless
    trpn50 = 0.35           # dimensionless
    kuw = 4.98              # s-1
    kws = 19.10             # s-1
    rw = 0.5                # dimensionless
    rs = 0.25               # dimensionless
    gs = 42.1               # s-1
    es = 1.0                # Controls asymmetry in gsu
    gw = 28.3               # s-1
    phi = 0.1498            # dimensionless
    Aeff = 125              # dimensionless
    beta0 = 0.0             # dimensionless
    beta1 = 0.0             # dimensionless
    Tref = 23.e3            # Pa

    # OFF-state parameters
    k1 = 0.877              # s-1
    k2 = 12.6               # s-1

    ra = 1.
    rb = 1.

    koffon = None
    Dep_k1ork2 = None
    koffon_ref = {'force': None,
                  'totalforce': 0.00144064,
                  'passiveforce': None,
                  'Lambda': None,
                  'C': None}

    Lambda_ext = 1.0

    def kwu(self):
        return self.kuw * (1 / self.rw - 1) - self.kws

    def ksu(self):
        return self.kws * self.rw * (1 / self.rs - 1)

    def kb(self):
        return self.ku * self.trpn50**self.nTm / (1 - self.rs - (1 - self.rs) * self.rw)

    def dLambdadt_fun(self, t):
        return 0

    def Aw(self):
        return self.Aeff * self.rs / ((1 - self.rs) * self.rw + self.rs)

    def As(self):
        return self.Aw()

    def __init__(self, PSet1=None, WhichDep='totalforce', Dep_k1ork2='k1'):
        self.WhichDep = WhichDep
        self.Dep_k1ork2 = Dep_k1ork2
        self.koffon = self.koffon_ref[WhichDep]

        if type(PSet1) == type(None):
            PSet1 = pd.Series({par1: 1. for par1 in AllParams})

        self.PSet = pd.Series({par1: 1. for par1 in AllParams})

        if type(PSet1) == pd.core.series.Series:
            for par1 in PSet1.index:
                assert par1 in self.PSet.index, f'Unknown parameter specified: {par1}'
                self.PSet[par1] = PSet1[par1]

        for param in PSet1.index:
            if param == 'koffon':
                setattr(self, param, self.koffon * PSet1[param])
            else:
                setattr(self, param, getattr(Lewalle2024, param) * PSet1[param])

        self.ExpResults = {}
        self.Features = {}

    def pCa50(self, Lambda):
        Ca50ref = 10**-self.pCa50ref
        Ca50 = Ca50ref + self.beta1 * (np.minimum(Lambda, 1.2) - 1)
        if np.size(Ca50) > 1:
            if any(np.array(Ca50) < 0):
                for j in range(len(Ca50)):
                    if Ca50[j] <= 0:
                        Ca50[j] = np.nan
        return -np.log10(Ca50)

    def h(self, Lambda=None):
        if Lambda is None:
            Lambda = self.Lambda_ext
        def hh(Lambda):
            return 1 + self.beta0 * (Lambda + np.minimum(Lambda, 0.87) - 1.87)
        return np.maximum(0, hh(np.minimum(Lambda, 1.2)))

    def Ta(self, Y):
        CaTRPN, B, S, W, Zs, Zw, Lambda, Cd, BE, UE = Y.transpose()
        return self.h(Lambda) * self.Tref / self.rs * (S * (Zs + 1) + W * Zw)

    def Ta_S(self, Y):
        CaTRPN, B, S, W, Zs, Zw, Lambda, Cd, BE, UE = Y.transpose()
        return self.h(Lambda) * self.Tref / self.rs * (S * (Zs + 1))

    def Ta_W(self, Y):
        CaTRPN, B, S, W, Zs, Zw, Lambda, Cd, BE, UE = Y.transpose()
        return self.h(Lambda) * self.Tref / self.rs * (W * Zw)

    def F1(self, Y):
        CaTRPN, B, S, W, Zs, Zw, Lambda, Cd, BE, UE = Y.transpose()
        C = Lambda - 1
        return self.a * (np.exp(self.b * C) - 1)

    def F2(self, Y):
        CaTRPN, B, S, W, Zs, Zw, Lambda, Cd, BE, UE = Y.transpose()
        C = Lambda - 1
        return self.a * self.k * (C - Cd)

    def Ttotal(self, Y):
        return self.Ta(Y) + self.F1(Y) + self.F2(Y)

    def Tp(self, Y):
        return self.F1(Y) + self.F2(Y)

    def Ta_ss(self, pCai=None):
        U_ss = self.U_ss(pCai)
        return self.h(self.Lambda_ext) * self.Tref / self.rs * self.kws * self.kuw / self.ksu() / (self.kwu() + self.kws) * U_ss

    def U_ss(self, pCai=None):
        if pCai is None:
            pCai = self.pCai(0)
        CaTRPN_ss, B_ss, S_ss, W_ss, Zs_ss, Zw_ss, Lambda_ss, Cd_ss, BE_ss, UE_ss = self.Get_ss(pCai)
        U_ss = 1.0 - UE_ss - B_ss - BE_ss - W_ss - S_ss
        return U_ss

    def Kub(self, pCai=None):
        CaTRPN_ss, B_ss, S_ss, W_ss, Zs_ss, Zw_ss, Lambda_ss, Cd_ss, BE_ss, UE_ss = self.Get_ss(pCai)
        Kub = self.ku / self.kb() * CaTRPN_ss**(+self.nTm)
        return Kub

    def Tp_ss(self, Lambda=None):
        if Lambda is None:
            Lambda = self.Lambda_ext
        return self.a * (np.exp(self.b * (Lambda - 1)) - 1)

    def Get_ss(self, pCai=None):
        assert self.Dep_k1ork2 is not None
        if pCai is None:
            pCai = self.pCai(0)
        if isinstance(pCai, int):
            pCai = float(pCai)
        if isinstance(pCai, np.ndarray) or isinstance(pCai, list):
            return np.array([self.Get_ss(pCai1) for pCai1 in pCai]).transpose()

        Lambda_ss = self.Lambda_ext
        Cd_ss = Lambda_ss - 1
        Zw_ss = 0
        Zs_ss = 0

        CaTRPN_ss = (self.k_trpn_off / self.k_trpn_on * (10**-pCai /
                     10**-self.pCa50(self.Lambda_ext))**-self.ntrpn + 1)**-1
        KE = self.k1 / self.k2
        Kub = self.ku / self.kb() * CaTRPN_ss**(+self.nTm)

        Q = self.kws * self.kuw / self.ksu() / (self.kwu() + self.kws) + \
            self.kuw / (self.kwu() + self.kws)

        if self.WhichDep in ['force', 'bound', 'totalforce']:
            if self.WhichDep in ['force', 'totalforce']:
                mu = self.koffon * self.h(self.Lambda_ext) * self.Tref / \
                    self.rs * self.kws * self.kuw / self.ksu() / (self.kwu() + self.kws)
            elif self.WhichDep == 'bound':
                mu = self.koffon * self.kuw / \
                    (self.kwu() + self.kws) * (1 + self.kws / self.ksu())

            if self.WhichDep in ['totalforce', 'passiveforce']:
                kfFp = self.koffon * self.Tp_ss()
            elif self.WhichDep in ['force', 'bound']:
                kfFp = 0

            aa = self.ra * mu * (1 + 1 / Kub + Q)
            bb = 1 / KE * (1 + 1 / Kub) - self.ra * mu + (1 + 1 / Kub + Q) * self.ra * (self.rb + kfFp)
            cc = -self.ra * (self.rb + kfFp)

            SmallUCriterion = -4 * aa * cc / bb**2
            if SmallUCriterion > 1e-3:
                U_ss = (-bb + np.sqrt(bb**2 - 4 * aa * cc)) / 2 / aa
            else:
                U_ss = self.ra * (self.rb + kfFp) / bb * (1 - mu * self.ra**2 * (1 + 1 / Kub + Q) * (self.rb + kfFp) / bb**2)

            UE_ss = 1 / KE / (self.rb + mu * U_ss + kfFp) / self.ra * U_ss
            BE_ss = 1 / Kub * UE_ss

        if self.WhichDep == 'Lambda':
            U_ss = ((1 + 1 / Kub) * (1 + 1 / KE / (1 + self.koffon * self.Lambda_ext)) + Q) ** -1
            UE_ss = 1 / KE / (1 + self.koffon * self.Lambda_ext) * U_ss
            BE_ss = 1 / Kub * UE_ss

        if self.WhichDep == 'C':
            U_ss = ((1 + 1 / Kub) * (1 + 1 / KE / (1 + self.koffon * (self.Lambda_ext - 1))) + Q) ** -1
            UE_ss = 1 / KE / (1 + self.koffon * (self.Lambda_ext - 1)) * U_ss
            BE_ss = 1 / Kub * UE_ss

        if self.WhichDep == 'passiveforce':
            Tp = self.a * (np.exp(self.b * (self.Lambda_ext - 1)) - 1)
            U_ss = ((1 + 1 / Kub) * (1 + 1 / KE / (1 + self.koffon * self.Tp_ss(self.Lambda_ext))) + Q) ** -1
            UE_ss = 1 / KE / (1 + self.koffon * Tp) * U_ss
            BE_ss = 1 / Kub * UE_ss

        B_ss = 1 / Kub * U_ss
        W_ss = self.kuw / (self.kwu() + self.kws) * U_ss
        S_ss = self.kws / self.ksu() * self.kuw / (self.kwu() + self.kws) * U_ss

        return np.array([CaTRPN_ss, B_ss, S_ss, W_ss, Zs_ss, Zw_ss, Lambda_ss, Cd_ss, BE_ss, UE_ss], dtype=float)

    # ODE system

    def gwu(self, Y):
        CaTRPN, B, S, W, Zs, Zw, Lambda, Cd, BE, UE = Y
        return self.gw * abs(Zw)

    def gsu(self, Y):
        CaTRPN, B, S, W, Zs, Zw, Lambda, Cd, BE, UE = Y
        if Zs < -self.es:
            return -self.gs * (Zs + self.es)
        elif Zs > 0:
            return self.gs * Zs
        else:
            return 0

    def cw(self, Y=None):
        return self.phi * self.kuw * ((1 - self.rs) * (1 - self.rw)) / ((1 - self.rs) * self.rw)

    def cs(self, Y=None):
        return self.phi * self.kws * ((1 - self.rs) * self.rw) / self.rs

    def U(self, Y):
        CaTRPN, B, S, W, Zs, Zw, Lambda, Cd, BE, UE = Y.T
        return 1. - B - S - W - BE - UE

    def k1_fb(self, Y):
        if self.Dep_k1ork2 == 'k1':
            CaTRPN, B, S, W, Zs, Zw, Lambda, Cd, BE, UE = Y
            if self.WhichDep == 'force':
                return self.k1 * (self.rb + self.koffon * max((self.Ta(Y), 0.)))
            if self.WhichDep == 'totalforce':
                return self.k1 * (self.rb + self.koffon * max((self.Ttotal(Y), 0.)))
            if self.WhichDep == 'passiveforce':
                return self.k1 * (self.rb + self.koffon * max((self.Tp(Y), 0.)))
            elif self.WhichDep == 'Lambda':
                return self.k1 * (self.rb + self.koffon * Lambda)
            elif self.WhichDep == 'bound':
                return self.k1 * (self.rb + self.koffon * (W + S))
            elif self.WhichDep == 'C':
                return self.k1 * (self.rb + self.koffon * (Lambda - 1))
        elif self.Dep_k1ork2 == 'k2':
            return self.k1 * self.ra

    def k2_fb(self, Y):
        if self.Dep_k1ork2 == 'k2':
            CaTRPN, B, S, W, Zs, Zw, Lambda, Cd, BE, UE = Y
            if self.WhichDep == 'force':
                return self.k2 / (self.rb + self.koffon * max((self.Ta(Y), 0.)))
            if self.WhichDep == 'totalforce':
                return self.k2 / (self.rb + self.koffon * max((self.Ttotal(Y), 0.)))
            if self.WhichDep == 'passiveforce':
                return self.k2 / (self.rb + self.koffon * max((self.Tp(Y), 0.)))
            elif self.WhichDep == 'Lambda':
                return self.k2 / (self.rb + self.koffon * Lambda)
            elif self.WhichDep == 'bound':
                return self.k2 / (self.rb + self.koffon * (W + S))
            elif self.WhichDep == 'C':
                return self.k2 / (self.rb + self.koffon * (Lambda - 1))
        elif self.Dep_k1ork2 == 'k1':
            return self.k2 / self.ra

    def dYdt(self, Y, t):
        CaTRPN, B, S, W, Zs, Zw, Lambda, Cd, BE, UE = Y

        dZwdt = self.Aw() * self.dLambdadt_fun(t) - self.cw(Y) * Zw
        dZsdt = self.As() * self.dLambdadt_fun(t) - self.cs(Y) * Zs
        dCaTRPNdt = self.k_trpn_on * \
            (10**-self.pCai(t) / 10**-self.pCa50(Lambda))**self.ntrpn * \
            (1 - CaTRPN) - self.k_trpn_off * CaTRPN
        dBdt = self.kb() * CaTRPN**(-self.nTm / 2) * self.U(Y) \
            - self.ku * CaTRPN**(self.nTm / 2) * B \
            - self.k2_fb(Y) * B \
            + self.k1_fb(Y) * BE
        dWdt = self.kuw * self.U(Y) - self.kwu() * W - \
            self.kws * W - self.gwu(Y) * W
        dSdt = self.kws * W - self.ksu() * S - self.gsu(Y) * S

        dBEdt = self.kb() * CaTRPN**(-self.nTm / 2) * UE \
            - self.ku * CaTRPN**(self.nTm / 2) * BE \
            + self.k2_fb(Y) * B \
            - self.k1_fb(Y) * BE
        dUEdt = -self.kb() * CaTRPN**(-self.nTm / 2) * UE \
            + self.ku * CaTRPN**(self.nTm / 2) * BE \
            + self.k2_fb(Y) * self.U(Y) \
            - self.k1_fb(Y) * UE

        dLambdadt = self.dLambdadt_fun(t)
        if Lambda - 1 - Cd > 0:
            dCddt = self.k / self.eta_l * (Lambda - 1 - Cd)
        else:
            dCddt = self.k / self.eta_s * (Lambda - 1 - Cd)

        return (dCaTRPNdt, dBdt, dSdt, dWdt, dZsdt, dZwdt, dLambdadt, dCddt, dBEdt, dUEdt)

    def dYdt_pas(self, Y, t):
        CaTRPN, B, S, W, Zs, Zw, Lambda, Cd, BE, UE = Y
        dLambdadt = self.dLambdadt_fun(t)
        if Lambda - 1 - Cd > 0:
            dCddt = self.k / self.eta_l * (Lambda - 1 - Cd)
        else:
            dCddt = self.k / self.eta_s * (Lambda - 1 - Cd)
        return (0, 0, 0, 0, 0, 0, dLambdadt, dCddt, 0, 0)

    def DoFpCa(self, DLambda=0.0, pCai_limits=[6.5, 5.0], ifPlot=False):
        pCai_original = self.pCai
        self.Lambda_ext += DLambda
        pCai_array = np.linspace(pCai_limits[1], pCai_limits[0], 50)
        F_array = self.Ta_ss(pCai_array)
        self.pCai = pCai_original
        self.ExpResults[f'FpCa_{self.Lambda_ext:.2f}'] = {
            'pCai': pCai_array, 'F': F_array, 'params': {'DLambda': DLambda}}
        self.Lambda_ext -= DLambda

    def DoDynamic(self, dLambdadt_imposed, t, DLambda_init=0., ifPlot=False):
        self.dLambdadt_fun = dLambdadt_imposed
        CaTRPN_0, B_0, S_0, W_0, Zs_0, Zw_0, Lambda_0, Cd_0, BE_0, UE_0 = self.Get_ss()
        Ta_ss_init = self.Ta_ss()
        Zs_0 += self.As() * DLambda_init
        Zw_0 += self.Aw() * DLambda_init
        self.Lambda_ext = Lambda_0 + DLambda_init
        Y0 = [CaTRPN_0, B_0, S_0, W_0, Zs_0, Zw_0, Lambda_0 + DLambda_init, Cd_0, BE_0, UE_0]
        Ysol = odeint(self.dYdt, Y0, t)
        Tasol = self.Ta(Ysol)
        self.ExpResults['Dynamic'] = {'t': t, 'Tasol': Tasol, 'Ysol': Ysol, 'Ta_ss_init': Ta_ss_init}

    def SinResponse(self, freq, numcycles=10, pointspercycle=30, dLambda_amplitude=0.0001, ifPlot=False):
        from scipy.optimize import curve_fit
        t = np.linspace(0, numcycles / freq, numcycles * pointspercycle)
        self.DoDynamic(lambda t: dLambda_amplitude * np.cos(2 * np.pi * freq * t) * 2 * np.pi * freq, t=t)
        Ysol = self.ExpResults['Dynamic']['Ysol']
        Tasol = self.ExpResults['Dynamic']['Tasol']

        def Sin_fun(t, *a):
            return a[0] * np.sin(2 * np.pi * freq * (t + a[1])) + a[2]

        SinFit, cov = curve_fit(Sin_fun,
                                t[-pointspercycle:], Tasol[-pointspercycle:],
                                p0=((max(Tasol[-pointspercycle:]) - min(Tasol[-pointspercycle:])) / 2,
                                    1 / freq / 4 - np.argmax(Tasol[-pointspercycle:]) / pointspercycle / freq,
                                    np.mean(Tasol[-pointspercycle:])))

        Stiffness = SinFit[0] / dLambda_amplitude
        DphaseTa = SinFit[1] * 2 * np.pi * freq
        if DphaseTa < 0:
            DphaseTa += 2 * np.pi
        elif DphaseTa > 2 * np.pi:
            DphaseTa -= 2 * np.pi
        return Tasol, Ysol, t, Stiffness, DphaseTa

    def GetFeat_FpCa(self, Lambda=1.1, ifPrint=False):
        from scipy.optimize import minimize
        Lambda_initial = self.Lambda_ext
        self.Lambda_ext = Lambda
        Fmax_active = self.Ta_ss(pCai=3)
        pCa50fit = minimize(lambda x: (self.Ta_ss(x) - Fmax_active / 2)**2, [5.], method='Nelder-Mead').x[0]
        dx = 0.05
        def dFdx(x): return (self.Ta_ss(x + dx) - self.Ta_ss(x)) / Fmax_active / dx * -np.log10(np.e)
        nH = 4 * dFdx(pCa50fit)
        self.Lambda_ext = Lambda_initial
        return Fmax_active, pCa50fit, nH
