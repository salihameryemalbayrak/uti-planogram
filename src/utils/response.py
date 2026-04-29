
from sdks.novavision.src.helper.package import PackageHelper
from components.Package.src.models.PackageModel import PackageModel, PackageConfigs, ConfigExecutor, PlanogramOutputs, PlanogramResponse, PlanogramExecutor, OutputData


def build_response(context):
    outputData = OutputData(value=context.image)
    outputs = PlanogramOutputs(outputImage=outputData)
    planogramResponse = PlanogramResponse(outputs=outputs)
    planogramExecutor = PlanogramExecutor(value=planogramResponse)
    executor = ConfigExecutor(value=planogramExecutor)
    packageConfigs = PackageConfigs(executor=executor)
    package = PackageHelper(packageModel=PackageModel, packageConfigs=packageConfigs)
    packageModel = package.build_model(context)
    return packageModel