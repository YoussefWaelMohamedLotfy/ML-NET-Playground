using Microsoft.ML.Data;

namespace AggressionScorerModel;

public sealed class ModelOutput
{
    public float Probability { get; set; }

    [ColumnName("PredictedLabel")]
    public bool Prediction { get; set; }
}
