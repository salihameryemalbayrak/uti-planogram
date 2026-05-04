
from sdks.novavision.src.helper.package import PackageHelper
from components.Planogram.src.models.PackageModel import PackageModel, PackageConfigs, ConfigExecutor, PlanogramOutputs, PlanogramResponse, PlanogramExecutor, OutputData, OutputImage

from models.PackageModel import OutputImage


def build_response(context):
    outputImage = OutputImage(value=context.image)
    outputData = OutputData(value=context.data)
    outputs = PlanogramOutputs(outputData=outputData, outputImage=outputImage)
    planogramResponse = PlanogramResponse(outputs=outputs)
    planogramExecutor = PlanogramExecutor(value=planogramResponse)
    executor = ConfigExecutor(value=planogramExecutor)
    packageConfigs = PackageConfigs(executor=executor)
    package = PackageHelper(packageModel=PackageModel, packageConfigs=packageConfigs)
    packageModel = package.build_model(context)
    return packageModel