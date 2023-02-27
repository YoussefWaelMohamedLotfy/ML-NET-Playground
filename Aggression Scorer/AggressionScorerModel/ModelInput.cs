using Microsoft.ML.Data;

namespace AggressionScorerModel;

public sealed class ModelInput
{
    [LoadColumn(1)]
    public string Comment { get; set; }

    [LoadColumn(0)]
    [ColumnName("Label")]
    public bool IsAggressive { get; set; }
}
