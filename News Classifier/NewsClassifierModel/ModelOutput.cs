using Microsoft.ML.Data;

namespace NewsClassifierModel;

public sealed class ModelOutput
{
    [ColumnName("PredictedLabel")]
    public string Category { get; set; }

    public float[] Score { get; set; }
}
