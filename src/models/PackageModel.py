
from pydantic import Field, validator
from typing import List, Optional, Union, Literal
from sdks.novavision.src.base.model import Package, Image, Inputs, Configs, Outputs, Response, Detection, Request, Output, Input, Config


class InputReferenceDetections(Input):
    name: Literal["inputReferenceDetections"] = "inputReferenceDetections"
    value: List[Detection]
    type: Literal["list"] = "list"

    class Config:
        title = "Reference Detections"


class InputTestDetections(Input):
    name: Literal["inputTestDetections"] = "inputTestDetections"
    value: List[Detection]
    type: Literal["list"] = "list"

    class Config:
        title = "Test Detections"


class InputReferenceEmbeddings(Input):
    name: Literal["inputReferenceEmbeddings"] = "inputReferenceEmbeddings"
    value: List
    type: Literal["list"] = "list"

    class Config:
        title = "Reference Embeddings"


class InputTestEmbeddings(Input):
    name: Literal["inputTestEmbeddings"] = "inputTestEmbeddings"
    value: List
    type: Literal["list"] = "list"

    class Config:
        title = "Test Embeddings"


class OutputData(Output):
    name: Literal["outputData"] = "outputData"
    value: Union[str, list]
    type: str = "object"


class FeatureWeight(Config):
    """
    Weight multiplier applied to the visual feature similarity score (extracted via CLIP embeddings)
    when calculating the combined compatibility score between reference and test detections.
    """
    name: Literal["featureWeight"] = "featureWeight"
    value: float
    type: Literal["string"] = "string"
    field: Literal["textInput"] = "textInput"

    class Config:
        title = "Visual similarity weight."


class IouWeight(Config):
    """
    Weight multiplier applied to the spatial Intersection over Union (IoU) overlap score
    when calculating the combined compatibility score between reference and test detections.
    """
    name: Literal["iouWeight"] = "iouWeight"
    value: float
    type: Literal["string"] = "string"
    field: Literal["textInput"] = "textInput"

    class Config:
        title = "Spatial overlap weight."


class Treshold(Config):
    """
    The minimum combined score (IoU + Feature similarity) required for a reference
    and test detection pair to be considered a successful match.
    """
    name: Literal["treshold"] = "treshold"
    value: float
    type: Literal["string"] = "string"
    field: Literal["textInput"] = "textInput"

    class Config:
        title = "Treshold"


class PlanogramInputs(Inputs):
    inputReferenceDetections: InputReferenceDetections
    inputTestDetections: InputTestDetections
    inputReferenceEmbeddings: InputReferenceEmbeddings
    inputTestEmbeddings:      InputTestEmbeddings


class PlanogramConfigs(Configs):
    iouWeight: IouWeight
    featureWeight: FeatureWeight
    treshold: Treshold


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
    type: Literal["capsule"] = "capsule"
    name: Literal["Planogram"] = "Planogram"