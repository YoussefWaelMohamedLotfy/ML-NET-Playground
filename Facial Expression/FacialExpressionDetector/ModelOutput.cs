namespace FacialExpressionDetector;

public sealed class ModelOutput
{
    public string Filename { get; set; }

    public List<(string emotion, float Probability)> EmotionProbabilities { get; set; }
}
