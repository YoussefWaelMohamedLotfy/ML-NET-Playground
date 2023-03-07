using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.ML;

namespace AggressionScorerModel;

public static class AggressionScorerServiceExtensions
{
    private static readonly string _modelFile =
        Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "Model", "AggressionRetrainedModel.zip");

    public static void AddAggressionScorerPredictionEnginePool(this IServiceCollection services)
    {
        services.AddPredictionEnginePool<ModelInput, ModelOutput>()
           .FromFile(filePath: _modelFile, watchForChanges: true);
    }
}