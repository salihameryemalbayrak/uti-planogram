
from pydantic import Field, validator
from typing import List, Optional, Union, Literal
from sdks.novavision.src.base.model import Package, Image, Inputs, Configs, Outputs, Response, Detection, Request, Output, Input, Config


class InputReferenceDetections(Input):
    name: Literal["inputReferenceDetections"] = "inputReferenceDetections"
    value: List[Detection]
    type: Literal["list"] = "list"

    class Config:
        title = "Reference"

class InputTestDetections(Input):
    name: Literal["inputTestDetections"] = "inputTestDetections"
    value: List[Detection]
    type: Literal["list"] = "list"

    class Config:
        title = "Test"


class OutputData(Output):
    name: Literal["outputData"] = "outputData"
    value: Union[ str, list]
    type: str = "object"

class FeatureWeight(Config):
    """
        ...
    """
    name: Literal["featureWeight"] = "featureWeight"
    value: float
    type: Literal["string"] = "string"
    field: Literal["textInput"] = "textInput"

    class Config:
        title = "featureWeight"

class IouWeight(Config):
    """
        ...
    """
    name: Literal["iouWeight"] = "iouWeight"
    value: float
    type: Literal["string"] = "string"
    field: Literal["textInput"] = "textInput"

    class Config:
        title = "iouWeight"


class PlanogramInputs(Inputs):
    inputReferenceDetections: InputReferenceDetections
    inputTestDetections: InputTestDetections
    inputImage: InputImage


class PlanogramConfigs(Configs):
    iouWeight: IouWeight
    featureWeight: FeatureWeight


class PlanogramOutputs(Outputs):
    outputData: OutputData


class PlanogramRequest(Request):
    inputs: Optional[PlanogramInputs]
    configs: PlanogramConfigs

    class Config:
        json_schema_extra = {
            "target": "configs"
        }


class PlanogramResponse(Response):
    outputs: PlanogramOutputs


class PlanogramExecutor(Config):
    name: Literal["Planogram"] = "Planogram"
    value: Union[PlanogramRequest, PlanogramResponse]
    type: Literal["object"] = "object"
    field: Literal["option"] = "option"

    class Config:
        title = "Planogram"
        json_schema_extra = {
            "target": {
                "value": 0
            }
        }


class ConfigExecutor(Config):
    name: Literal["ConfigExecutor"] = "ConfigExecutor"
    value: Union[PlanogramExecutor]
    type: Literal["executor"] = "executor"
    field: Literal["dependentDropdownlist"] = "dependentDropdownlist"

    class Config:
        title = "Task"
        json_schema_extra = {
            "target": "value"
        }


class PackageConfigs(Configs):
    executor: ConfigExecutor


class PackageModel(Package):
    configs: PackageConfigs
    type: Literal["component"] = "component"
    name: Literal["Planogram"] = "Planogram"
