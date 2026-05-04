
from pydantic import Field, validator
from typing import List, Optional, Union, Literal
from sdks.novavision.src.base.model import Package, Image, Inputs, Configs, Outputs, Response, Detection, Request, Output, Input, Config

class InputImageOne(Input):
    name: Literal["inputImageOne"] = "inputImageOne"
    value: Union[List[Image], Image]
    type: str = "object"

    @validator("type", pre=True, always=True)
    def set_type_based_on_value(cls, value, values):
        value = values.get('value')
        if isinstance(value, Image):
            return "object"
        elif isinstance(value, list):
            return "list"

    class Config:
        title = "Image"


class InputImageTwo(Input):
    name: Literal["inputImageTwo"] = "inputImageTwo"
    value: Union[List[Image], Image]
    type: str = "object"

    @validator("type", pre=True, always=True)
    def set_type_based_on_value(cls, value, values):
        value = values.get('value')
        if isinstance(value, Image):
            return "object"
        elif isinstance(value, list):
            return "list"

    class Config:
        title = "Image"

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

class OutputImage(Output):
    name: Literal["outputImage"] = "outputImage"
    value: Union[List[Image], Image]
    type: str = "object"

    @validator("type", pre=True, always=True)
    def set_type_based_on_value(cls, value, values):
        value = values.get('value')
        if isinstance(value, Image):
            return "object"
        elif isinstance(value, list):
            return "list"

    class Config:
        title = "Image"

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
    inputImageTwo: InputImageTwo
    inputImageOne: InputImageOne


class PlanogramConfigs(Configs):
    iouWeight: IouWeight
    featureWeight: FeatureWeight


class PlanogramOutputs(Outputs):
    outputData: OutputData
    outputImage: OutputImage


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
