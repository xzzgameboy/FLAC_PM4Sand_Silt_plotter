import numpy as np
from scipy.stats import norm

class CRR_SPT_IB_2014:
    def __init__(self,
                n_160:float, 
                fc: float, 
                m_w: float, 
                depth: float,
                sigma_v_eff: float,
                liq_prop : float = 0.16,
                pa : float = 101.325
                ) -> None:
        r'''
        n_160 (float): calibrated blow count
        fc (float): fines content -[%]
        m_w (float): magnitude
        depth (float): depth -[ft]
        sigma_v_eff (float): vertical effective stress -[psf or kPa]
        liq_prop (float): probability of liquefaction - [decimal]
        pa (float): atmospheric pressure, same as sigma_v_eff
        '''
        self.n_160 = n_160
        self.fc = fc
        self.m_w = m_w
        self.depth = depth
        self.sigma_v_eff = sigma_v_eff
        self.liq_prop = liq_prop
        self.pa = pa
        self._convert_depth_m()
        self.__post_init()

    def __post_init(self) -> None:
        self.get_rd()
        self.get_msf()
        self.get_clean_sand_correction()
        self.get_n1_60_cs()
        self.get_k_sigma()
        self.get_crr_base()
        self.get_crr()
    
    def _convert_depth_m(self) -> None:
        self.depth_m = self.depth * 0.3048
    
    def get_rd(self) -> None:
        alpha_z = -1.012 - 1.126 * np.sin(self.depth_m/11.73 + 5.133)
        beta_z = 0.106 + 0.118 * np.sin(self.depth_m/11.28 + 5.142)
        self.rd = np.exp(alpha_z + beta_z * self.m_w)
    
    def get_msf(self) -> None:
        msf = 6.9 * np.exp(-self.m_w/4) - 0.058
        self.msf = np.min([1.8, msf])

    def get_clean_sand_correction(self) -> None:
        self.d_n_160 = np.exp(1.63 + 9.7/(self.fc + 0.01) \
                            - (15.7/(self.fc + 0.01))**2)
        
    def get_n1_60_cs(self) -> None:
        self.n1_60_cs = self.n_160 + self.d_n_160

    def get_k_sigma(self) -> None:
        c_sigma = 1/(18.9 - 2.55 * np.sqrt(self.n1_60_cs))
        c_sigma = np.min([0.3, c_sigma])
        k_sigma = 1 - c_sigma * np.log(self.sigma_v_eff/self.pa)
        self.k_sigma = np.min([k_sigma, 1.1])

    def get_crr_base(self) -> None:
        self.crr_m_atm = np.exp(self.n1_60_cs/14.1 + \
                                (self.n1_60_cs/126)**2 -\
                                (self.n1_60_cs/23.6)**3 +\
                                (self.n1_60_cs/25.4)**4 -\
                                2.67 + 0.13*norm.ppf(self.liq_prop))
    
    def get_crr(self) -> None:
        self.crr = self.crr_m_atm * self.msf * self.k_sigma

def CRR_gravel_Vs_Rollins(Mw: float, Vs1: float, PL=0.15):
    PL_term = (1 - PL) / PL
    CRR_term = 3.88e-7 * Vs1**3 - 1.6*Mw - np.log(PL_term)
    CRR_term /= 4.95 
    CRR_7d5 = np.exp(CRR_term)
    MSF = 10.667*np.exp(-0.316*Mw)
    CRR = CRR_7d5*MSF
    return CRR


class CRR_CPT_IB_2014:
    def __init__(
        self,
        qc1n: float,
        fc: float,
        m_w: float,
        depth: float,
        sigma_v_eff: float,
        liq_prop: float = 0.16,
        pa: float = 101.325,
    ) -> None:
        r"""
        Calculate CPT-based cyclic resistance ratio following
        Idriss and Boulanger (2014).

        Parameters
        ----------
        qc1n : float
            Normalized CPT cone resistance, q_c1N, dimensionless.
            This is not the measured cone tip resistance q_c.

        fc : float
            Fines content, in percent.

        m_w : float
            Moment magnitude.

        depth : float
            Depth, in ft.

        sigma_v_eff : float
            Initial vertical effective stress, in psf or kPa.

        liq_prop : float, default 0.16
            Probability of liquefaction, expressed as a decimal.

        pa : float, default 101.325
            Atmospheric pressure in the same units as sigma_v_eff.
            Use approximately 101.325 kPa or 2,116.2 psf.
        """
        self.qc1n = qc1n
        self.fc = fc
        self.m_w = m_w
        self.depth = depth
        self.sigma_v_eff = sigma_v_eff
        self.liq_prop = liq_prop
        self.pa = pa

        self._validate_inputs()
        self._convert_depth_m()
        self.__post_init()

    def __post_init(self) -> None:
        self.get_rd()
        self.get_msf()
        self.get_clean_sand_correction()
        self.get_qc1n_cs()
        self.get_k_sigma()
        self.get_crr_base()
        self.get_crr()

    def _validate_inputs(self) -> None:
        if self.qc1n < 0:
            raise ValueError("qc1n must be nonnegative.")

        if self.fc < 0:
            raise ValueError("fc must be nonnegative.")

        if self.depth < 0:
            raise ValueError("depth must be nonnegative.")

        if self.sigma_v_eff <= 0:
            raise ValueError("sigma_v_eff must be greater than zero.")

        if self.pa <= 0:
            raise ValueError("pa must be greater than zero.")

        if not 0 < self.liq_prop < 1:
            raise ValueError("liq_prop must be between zero and one.")

    def _convert_depth_m(self) -> None:
        """Convert depth from ft to m for the stress-reduction equation."""
        self.depth_m = self.depth * 0.3048

    def get_rd(self) -> None:
        """
        Calculate the depth-dependent stress-reduction coefficient.

        The class calculates rd for compatibility with the SPT class.
        The CRR calculation itself does not use rd.
        """
        alpha_z = (
            -1.012
            - 1.126
            * np.sin(self.depth_m / 11.73 + 5.133)
        )

        beta_z = (
            0.106
            + 0.118
            * np.sin(self.depth_m / 11.28 + 5.142)
        )

        self.rd = np.exp(alpha_z + beta_z * self.m_w)

    def get_msf(self) -> None:
        """Calculate the magnitude scaling factor."""
        msf = 6.9 * np.exp(-self.m_w / 4.0) - 0.058
        self.msf = min(1.8, msf)

    def get_clean_sand_correction(self) -> None:
        """
        Calculate the fines correction applied to q_c1N.

        The resulting correction is added to q_c1N to obtain q_c1Ncs.
        """
        fc_term = self.fc + 2.0

        self.d_qc1n = (
            11.9 + self.qc1n / 14.6
        ) * np.exp(
            1.63
            - 9.7 / fc_term
            - (15.7 / fc_term) ** 2
        )

    def get_qc1n_cs(self) -> None:
        """Calculate clean-sand-equivalent normalized cone resistance."""
        self.qc1n_cs = self.qc1n + self.d_qc1n

    def get_k_sigma(self) -> None:
        """Calculate the overburden correction factor."""
        denominator = (
            37.3
            - 8.27 * self.qc1n_cs**0.264
        )

        if denominator <= 0:
            raise ValueError(
                "The calculated C_sigma denominator is nonpositive. "
                "Check qc1n and the resulting qc1n_cs."
            )

        c_sigma = 1.0 / denominator
        self.c_sigma = min(0.3, c_sigma)

        k_sigma = (
            1.0
            - self.c_sigma
            * np.log(self.sigma_v_eff / self.pa)
        )

        self.k_sigma = min(1.1, k_sigma)

    def get_crr_base(self) -> None:
        """
        Calculate CRR at M = 7.5 and sigma'_v = one atmosphere.

        The probability term follows the probabilistic Idriss and
        Boulanger formulation. A liquefaction probability of 0.16
        gives norm.ppf(0.16) of approximately -1.
        """
        q = self.qc1n_cs

        self.crr_m_atm = np.exp(
            q / 113.0
            + (q / 1000.0) ** 2
            - (q / 140.0) ** 3
            + (q / 137.0) ** 4
            - 2.80
            + 0.13 * norm.ppf(self.liq_prop)
        )

    def get_crr(self) -> None:
        """Calculate CRR for the specified magnitude and effective stress."""
        self.crr = (
            self.crr_m_atm
            * self.msf
            * self.k_sigma
        )