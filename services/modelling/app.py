from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pandas as pd
import numpy as np

from prometheus_fastapi_instrumentator import Instrumentator
from fastapi.middleware.cors import CORSMiddleware
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from fastapi import FastAPI, Request
import onnxruntime

app = FastAPI(title="Photovoltaic-panels Digital Twin APIs",
              version="0.0.1")

# change model_name with name of model
ML_MODEL = "models/_pv_energy.onnx"
MATH_MODEL = "models/fmu_ubuntu/SolarPowerSystems.CoreModels.PVModel.fmu"


def get_math_model_values():
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


from simulation import FMUSimulator

sim = FMUSimulator(model_path=MATH_MODEL, start_status=get_math_model_values())

ort_session = onnxruntime.InferenceSession(ML_MODEL)


def regression_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)

    return {'MAE': mae, 'MSE': mse, 'RMSE': rmse, 'R²': r2}


def dict_to_dataframe(data):
    df = pd.DataFrame(data, index=[0])
    return df


def get_data(input_data):
    input_d = input_data.dict()
    data = dict_to_dataframe(input_d)

    data['TERM'] = data['TERM'] + 273.15  # matematiksel model için Kelvin dönüşümü

    x = data.drop(labels=['PRES', 'IGRO', 'RAFF', 'RAIN', 'DIRV',
                          'PV_PROD', 'SHADOW_TERM', 'DATE'], axis=1)
    y = data['PV_PROD']
    return data, x, y


def predict(x):
    return ort_session.run(None, {'dense_7_input': x})[0]


def get_ml_model_response(input_data):
    data, x, y = get_data(input_data)

    y_pred = predict(x.iloc[0].to_numpy().reshape(1, -1).astype(np.float32))[0][0]
    y_pred = 0 if y_pred < 0 else y_pred

    # metrics_dict = regression_metrics(y, y_pred)

    return {
        'date': str(data['DATE'].iloc[0]),
        'pv_prod': float(data["PV_PROD"].iloc[0]),
        'pv_prod_predicted': float(y_pred)
    }


def get_math_model_results(input_data):
    data, x, y = get_data(input_data)
    data["irr_direct"] = data["RADD"] / 2
    data["irr_diffuse"] = data["RADD"] / 2
    data["temp_air"] = data["TERM"]
    data["wind_speed"] = data["VELV"]

    _in = {'inputData.table[1,2]': data["irr_direct"],
           'inputData.table[1,3]': data["irr_diffuse"],
           'inputData.table[1,4]': data["temp_air"],
           'inputData.table[1,5]': data["wind_speed"]}

    energy = sim.run_simulator(input_dict=_in,
                               output=["PhotoVoltaicSystem.energy"])
    # metrics_dict = regression_metrics(y, energy["PhotoVoltaicSystem.energy"])

    return {'date': str(data['DATE'].iloc[0]),
            'pv_prod': float(data["PV_PROD"].iloc[0]),
            'pv_prod_predicted': float(energy["PhotoVoltaicSystem.energy"])}


# Input for data validation
class Input(BaseModel):
    PRES: float = Field(..., title="Basınç (hPa)")
    IGRO: float = Field(..., title="Hava Nem Oranı (%)")
    RAFF: float = Field(..., title="Rüzgar Hortum Hızı (m/s)")
    RAIN: float = Field(..., title="Yağış Miktarı (mm)")
    TERM: float = Field(..., title="Sıcaklık (°C)")
    DATE: object = Field(..., title="Zaman")
    DIRV: float = Field(..., title="Rüzgar Yönü (°)")
    RADD: float = Field(..., title="Güneş Işınımı (W/m²)")
    VELV: float = Field(..., title="Rüzgar Hızı (m/s)")
    PV_PROD: float = Field(..., title="İnvertörün Enerji Üretimi (Wh)")
    SHADOW_TERM: float = Field(..., title="Gölge")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {'PRES': 951.9,
                 'IGRO': 100.0,
                 'RAFF': 3.2,
                 'RAIN': 0.0,
                 'TERM': 276.15,
                 'DATE': "2016-02-18 00:00:00",
                 'DIRV': 350.0,
                 'RADD': 0.0,
                 'VELV': 1.0,
                 'PV_PROD': 0.0,
                 'SHADOW_TERM': 0.0}
            ]
        }
    }


# Ouput for data validation
class OutputML(BaseModel):
    date: object
    pv_prod: float
    pv_prod_predicted: float


class Output(BaseModel):
    date: object
    pv_prod: float
    ml_model_prediction: float
    math_model_prediction: float


@app.get("/")
async def home(request: Request):
    return templates.TemplateResponse("home.html", {"request": request})


@app.get('/info')
async def model_info():
    return {
        "name": "Güneş Paneli Dijital İkizi",
        "version": "0.0.1"
    }


@app.get('/health')
async def service_health():
    """Return service health"""
    return {
        "ok"
    }


@app.post('/ml_predict', response_model=OutputML)
async def ml_model_predict(input_d: Input):
    """Predict with input"""
    response = get_ml_model_response(input_d)
    return response


@app.post('/math_predict', response_model=OutputML)
async def math_model_predict(input_d: Input):
    """Predict with input"""
    response = get_math_model_results(input_d)
    return response


@app.post('/predict', response_model=Output)
async def models_predict(input_d: Input):
    """Predict with input"""
    response_ml = get_math_model_results(input_d)
    response_math = get_ml_model_response(input_d)

    # metrics_dict = regression_metrics(response_ml["pv_prod_predicted"], response_math["pv_prod_predicted"])
    return {"date": response_math["date"],
            "pv_prod": response_ml["pv_prod"],
            "ml_model_prediction": response_ml["pv_prod_predicted"],
            "math_model_prediction": response_math["pv_prod_predicted"]}


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

Instrumentator().instrument(app).expose(app)
