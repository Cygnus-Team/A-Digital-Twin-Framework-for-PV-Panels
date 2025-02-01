import pandas as pd
from fmpy import *
import copy


class FMUSimulator:
    def __init__(self, model_path: str,
                 start_status: dict):
        self.model_path = model_path
        self.start_status = start_status

    def simulator_desc(self):
        print(dump(self.model_path))

    def first_parameters(self):
        status_dict: dict = copy.deepcopy(self.start_status)
        return status_dict

    def _update_start_values(self, input_dict):
        params = self.start_status
        params.update(input_dict)
        return params

    def run_simulator_data(self, input_dict: dict,
                           output: list,
                           start_time: int = 10800,
                           stop_time: int = 259200,
                           interval: int = 60):
        _input = self._update_start_values(input_dict)
        res = simulate_fmu(self.model_path, start_values=_input, start_time=start_time, stop_time=stop_time,
                           output_interval=interval,
                           output=output)
        return pd.DataFrame(res)

    def run_simulator(self, input_dict: dict,
                      output: list,
                      start_time: int = 10800,
                      stop_time: int = 259200,
                      interval: int = 60):
        _input = self._update_start_values(input_dict)
        res = simulate_fmu(self.model_path, start_values=_input, start_time=start_time, stop_time=stop_time,
                           output_interval=interval,
                           output=output)
        return pd.DataFrame(res)[output].mean().to_dict()


def test():
    import pandas as pd

    def get_values():
        return {
            # variable                                                                                       start   unit             description
            'PhotoVoltaicSystem.constInverterEfficiency': (1.0, '1'),
            # Constant overall efficiency of the DC-AC converter
            'PhotoVoltaicSystem.location.elevation': (273.0, 'm'),  # Height above sea level (elevation) in metres
            'PhotoVoltaicSystem.location.latitude': (49.2553, 'deg'),  # Latitude in decimal degrees
            'PhotoVoltaicSystem.location.longitude': (7.0405, 'deg'),  # Longitude in decimal degrees
            'PhotoVoltaicSystem.plantModel.add.k1': -1.0,  # Gain of input signal 1
            'PhotoVoltaicSystem.plantModel.add.k2': 1.0,  # Gain of input signal 2
            'PhotoVoltaicSystem.plantModel.const.k': 0.0,  # Constant output value
            'PhotoVoltaicSystem.plantModel.constTemperature': (293.15, 'K'),
            # Fixed environment temperature if useTemperatureInput = false
            'PhotoVoltaicSystem.plantModel.constWindSpeed': (1.0, 'm/s'),
            # Fixed wind speed value if useWindSpeedInput = false
            'PhotoVoltaicSystem.plantModel.globalIrradiance.k1': 1.0,  # Gain of input signal 1
            'PhotoVoltaicSystem.plantModel.globalIrradiance.k2': 1.0,  # Gain of input signal 2
            'PhotoVoltaicSystem.plantModel.globalIrradiance.k3': 1.0,  # Gain of input signal 3
            'PhotoVoltaicSystem.plantModel.inclinationAndShadowing.albedo': 0.2,  # Ground reflectance/Albedo
            'PhotoVoltaicSystem.plantModel.inclinationAndShadowing.globalHorizontalIrradiance.k1': 1.0,
            # Gain of input signal 1
            'PhotoVoltaicSystem.plantModel.inclinationAndShadowing.globalHorizontalIrradiance.k2': 1.0,
            # Gain of input signal 2
            'PhotoVoltaicSystem.plantModel.integrator.k': (3.6e-06, '1'),  # Integrator gain
            'PhotoVoltaicSystem.plantModel.integrator.y_start': 0.0,  # Initial or guess value of output (= state)
            'PhotoVoltaicSystem.plantModel.plantIrradianceNormal.T': (293.15, 'K'),
            # Fixed device temperature if useHeatPort = false
            'PhotoVoltaicSystem.plantModel.plantIrradianceNormal.U_0': (25.0, 'W/(m2.K)'),
            # Heat loss coefficient of the PV module
            'PhotoVoltaicSystem.plantModel.plantIrradianceNormal.U_1': (6.84, 'W.s/(m3.K)'),
            # Wind dependent heat loss coefficient of the PV module
            'PhotoVoltaicSystem.plantModel.plantIrradianceNormal.a': (-1.09e-05, 'm2/W'),
            # Irradiance losses according to Heydenreich et al. (2008, equation 3); parameter a
            'PhotoVoltaicSystem.plantModel.plantIrradianceNormal.b': -0.047,
            # Irradiance losses according to Heydenreich et al. (2008, equation 3); parameter b
            'PhotoVoltaicSystem.plantModel.plantIrradianceNormal.beta': (0.0043, '1/K'),
            # Power temperature coefficient
            'PhotoVoltaicSystem.plantModel.plantIrradianceNormal.c': -1.4,
            # Irradiance losses according to Heydenreich et al. (2008, equation 3); parameter c
            'PhotoVoltaicSystem.plantModel.plantIrradianceNormal.physicalIAMmodel.K': (4.0, '1/m'),
            # Glazing extinction coefficient
            'PhotoVoltaicSystem.plantModel.plantIrradianceNormal.physicalIAMmodel.L': (0.002, 'm'),  # Glazing thickness
            'PhotoVoltaicSystem.plantModel.plantIrradianceNormal.physicalIAMmodel.n': (1.526, '1'),
            # Refraction index of the cover glass
            'PhotoVoltaicSystem.plantModel.transformFactor.k': 90.0,  # Constant output value
            'PhotoVoltaicSystem.plantRecord.T_cell_ref': (25.0, 'degC'),
            # PV cell temperature at reference conditions (usually STC)
            'PhotoVoltaicSystem.plantRecord.environmentAlbedo': 0.2,
            # Albedo for isotropic estimation of irradiance by reflection
            'PhotoVoltaicSystem.plantRecord.panelArea': (138.24, 'm2'),  # Overall surface area of all panels (combined)
            'PhotoVoltaicSystem.plantRecord.panelAzimuth': (0.0, 'deg'),
            # Surface azimuth in degree (South equals 0�, positive towards east)
            'PhotoVoltaicSystem.plantRecord.panelTilt': (30.0, 'deg'),
            # Surface tilt in degree (Horizontal equals 0�, vertical equals 90�)
            'PhotoVoltaicSystem.plantRecord.plantEfficiency': (0.2, '1'),  # Overall efficiency

            'inputData.table[1,1]': 1529539200.0,  # Table matrix (grid = first column; e.g., table=[0, 0; 1, 1; 2, 4])
            'inputData.table[1,2]': 30.7548828125,  # Table matrix (grid = first column; e.g., table=[0, 0; 1, 1; 2, 4])
            'inputData.table[1,3]': 74.8021477908,  # Table matrix (grid = first column; e.g., table=[0, 0; 1, 1; 2, 4])
            'inputData.table[1,4]': 287.8843994141,
            # Table matrix (grid = first column; e.g., table=[0, 0; 1, 1; 2, 4])
            'inputData.table[1,5]': 1.1453763949,  # Table matrix (grid = first column; e.g., table=[0, 0; 1, 1; 2, 4])
            'timeAsEpoch.startTime': (0.0, 's'),  # Output y = offset for time < startTime
            'PhotoVoltaicSystem.plantRecord.npModule': 1,
            # Number of parallel connected modules (PhotoVoltaicsLib only)
            'PhotoVoltaicSystem.plantRecord.nsModule': 1,  # Number of series connected modules (PhotoVoltaicsLib only)
            'inputData.columns[1]': 2,  # Columns of table to be interpolated
            'inputData.columns[2]': 3,  # Columns of table to be interpolated
            'inputData.columns[3]': 4,  # Columns of table to be interpolated
            'inputData.columns[4]': 5,  # Columns of table to be interpolated
            'startTime.k': 1530403200,  # Constant output value
            'inputData.verboseRead': True,  # = true, if info message that file is loading is to be printed
            'PhotoVoltaicSystem.location.fileName': 'noFile',  # Filepath to external file storing actual data
            'PhotoVoltaicSystem.plantRecord.fileName': 'noFile',  # Filepath to external file storing actual data
            'inputData.fileName': 'NoName',  # File where matrix is stored
            'inputData.tableName': 'data',  # Table name on file or in function usertab (see docu)
        }

    # data = pd.read_csv("../data/20180606T0000--20180706T2359+0000_uds-campus.txt", sep="\t")
    data = pd.read_csv("../assets/flow_test_data.csv")
    data['TERM'] = data['TERM'] + 273.15  # matematiksel model için Kelvin dönüşümü

    data = data.drop(labels=['PRES', 'IGRO', 'RAFF', 'RAIN', 'DIRV',
                             'PV_PROD', 'SHADOW_TERM', 'DATE'], axis=1)

    data["irr_direct"] = data["RADD"] / 2
    data["irr_diffuse"] = data["RADD"] / 2
    data["temp_air"] = data["TERM"]
    data["wind_speed"] = data["VELV"]

    fmu = 'models/SolarPowerSystems.CoreModels.PVModel.fmu'
    fmu_sim = FMUSimulator(model_path=fmu, start_status=get_values())
    for i in range(len(data)):
        test_sample = data.iloc[i].to_dict()
        _in = {'inputData.table[1,2]': test_sample["irr_direct"],
               'inputData.table[1,3]': test_sample["irr_diffuse"],
               'inputData.table[1,4]': test_sample["temp_air"],
               'inputData.table[1,5]': test_sample["wind_speed"]}
        print(_in)
        print(fmu_sim.run_simulator(input_dict=_in,
                                    output=["PhotoVoltaicSystem.energy"]))
