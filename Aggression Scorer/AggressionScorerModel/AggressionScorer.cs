using Microsoft.Extensions.ML;

namespace AggressionScorerModel;

public sealed class AggressionScorer
{
    private readonly PredictionEnginePool<ModelInput, ModelOutput> _predictionEnginePool;

    public AggressionScorer(PredictionEnginePool<ModelInput, ModelOutput> predictionEnginePool)
        => _predictionEnginePool = predictionEnginePool;

    public ModelOutput Predict(string input)
        => _predictionEnginePool.Predict(new() { Comment = input });
}
