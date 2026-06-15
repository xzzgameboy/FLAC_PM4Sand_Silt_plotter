import numpy as np
import pandas as pd
from scipy.signal import find_peaks
import math

def get_full_data_flac9(filepath: str) -> pd.DataFrame:
    df = pd.read_csv(filepath)
    first_non_zero_index = df['Sigxy'].ne(0).idxmax()
    df = df.iloc[first_non_zero_index-1:]
    peaks, _ = find_peaks(df.Sigxy.to_numpy())
    num_cycles = len(peaks)
    print("Number of cycles:", num_cycles)
    cycles = np.linspace(0, num_cycles, num=len(df))
    df['Cycles'] = cycles
    df['CSR'] = df['Sigxy']/df['SigmaV_eff'].to_numpy()[0]
    df['SigmaV_eff_ratio'] = df['SigmaV_eff']/df['SigmaV_eff'].to_numpy()[0]
    return df

def get_full_data_flac9_pm4sand(filepath: str) -> pd.DataFrame:
    df = pd.read_csv(filepath)
    df['Cycles'] = df['Cycle num']
    df['Shear strain'] = df[' Shear strain (%)']
    df['SigmaV_eff_ratio'] = df[' Vert eff stress ratio']
    return df

def get_full_data_flac8(filepath: str) -> pd.DataFrame:
    column_names = pd.read_csv(filepath, nrows=1, header=None, sep=',').iloc[0]
    # Then, read the rest of the data with space delimiter
    df = pd.read_csv(filepath, skiprows=1, header=None, sep='\\s+')
    # Assign column names to the dataframe
    df.columns = column_names
    return df

class CDSSprocessor:
    #pass model to the class
    def __init__(self,
                csr_data: np.ndarray, 
                strain_data: np.ndarray,
                cycles_data: np.ndarray,
                ru_data: np.ndarray,
                thres_strain: float = 3.0,
                thres_pp: float = 0.98,
                criteria: str = 'SA') -> None:
        '''
        Args:
            csr_data (np.ndarray): cyclic csr data.
            strain_data (np.ndarray): cyclic strain data - [%]
            cycles_data (np.ndarray): number of cycles.
            ru_data (np.ndarray): pore pressure ratio data.
            thres_strain (float): threshold strain, default to 3.0% - [%].
            thres_pp (float): threshold pore pressure, default to 0.98
            - [decimal].
            criteria (str): criteria for count cycles, "SA" or "DA", 
            default to SA.
        '''
        self.csr_data = csr_data
        self.strain_data = strain_data
        self.cycles_data = cycles_data
        self.ru_data = ru_data
        self.thres_strain = thres_strain
        self.thres_pp = thres_pp
        self.criteria = criteria
        self.__post_init()
    
    def __post_init(self):
        self._validate_arrays_length()
        self._validate_input_criteria()
        if self.criteria == 'SA':
            self.get_single_amp_cycle()
        elif self.criteria == 'DA':
            self.get_double_amp_cycle()
    
    def _validate_input_criteria(self) -> None:
        if self.criteria not in ['DA', 'SA']:
            raise ValueError(f'input criteria {self.criteria} is not DA or SA')
    
    def _validate_arrays_length(self) -> None:
        '''
        This method validates the input arrays' lengths
        '''
        try:
            assert len(self.csr_data) == len(self.strain_data), "Stress and strain data lengths do not match"
            assert len(self.strain_data) == len(self.cycles_data), "Strain and cycles data lengths do not match"
            assert len(self.ru_data) == len(self.strain_data), "Pore Pressure Ratio and stress data lengths do not match"
        except AssertionError as e:
            print('lengths of input arrays must be same:')
            print(f'csr_data length: {len(self.csr_data)};')
            print(f'strain_data length: {len(self.strain_data)};')
            print(f'cycles_data length: {len(self.cycles_data)}.')
            print(f'ru_data length: {len(self.ru_data)}.')
    
    @staticmethod
    def find_nearest_first(data: np.ndarray, threshold: float) -> int:
        data = np.abs(data)
        threshold = abs(threshold)
        for index, value in enumerate(data):
            if value > threshold:
                final_index = index
                break
            else:
                final_index = len(data)-1
        if index == len(data)-1:
            print('Value not reached, using the last index')
        return final_index


    def get_ru_cycle(self) -> None:
        ru_index = self.find_nearest_first(self.ru_data, self.thres_pp)
        self.cycle_ru = self.cycles_data[ru_index]
    
    def get_single_amp_cycle(self) -> None:
        sa_index = self.find_nearest_first(self.strain_data, self.thres_strain)
        self.cycle_sa = self.cycles_data[sa_index]
    
    def get_double_amp_cycle(self) -> None:
        strain_abs = np.abs(self.strain_data)
        peaks, _ = find_peaks(strain_abs)

        for i in range(len(peaks)-1):
            sum_of_peaks = strain_abs[peaks[i]] + strain_abs[peaks[i+1]]
            if sum_of_peaks > self.thres_strain:
                index = peaks[i+1]
                break
            else:
                index = len(self.cycles_data)-1
        if index == len(self.cycles_data)-1:
            print('DA not reached, using the last peak')
        self.cycle_da = self.cycles_data[index]


def get_triggering_curve_IB08(N160cs: float, sigma_v: float, PA: float = 2016) -> list:
    """
    Idriss and Boulanger (2008) triggering curve

    Parameters
    ----------
    N160cs : float
        Corrected SPT blow count
    sigma_v : float
        Vertical effective stress term

    Returns
    -------
    list[tuple[float, float]]
        Paired values of `(N_cycles, CSR)`
    """
    sigma_v = sigma_v / PA
    Csigma = min(1.0 / (18.9 - 2.55 * math.sqrt(N160cs)), 3.0)

    Ksigma = min(
        1.1,
        1.0 - Csigma * math.log(sigma_v)
    )

    CSRmid = Ksigma * math.exp(
        N160cs / 14.1
        + (N160cs / 126.0) ** 2
        - (N160cs / 23.6) ** 3
        + (N160cs / 25.4) ** 4
        - 2.8
    )

    IB08_cycle = []
    IB08_CRR = []

    for i in range(1, 42):
        Mw = 5.5 + (i - 1) / 10.0

        MSF = min(
            6.9 * math.exp(-Mw / 4.0) - 0.058,
            1.8
        )

        N_cycles = 15.0 / MSF ** (1.0 / 0.34)
        CSR = MSF * CSRmid

        IB08_cycle.append(N_cycles)
        IB08_CRR.append(CSR)

    return IB08_cycle, IB08_CRR