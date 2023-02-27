using Microsoft.AspNetCore.Mvc;

namespace AggressionScorerAPI.Controllers;

[Route("api/[controller]")]
[ApiController]
public sealed class AggressionScoreController : ControllerBase
{
    private readonly AggressionScorerModel.AggressionScorer _scorer;

    public AggressionScoreController(AggressionScorerModel.AggressionScorer scorer)
    {
        _scorer = scorer;
    }

    [HttpPost]
    public IActionResult Post(CommentWrapper input)
    {
        return Ok(ScoreComment(input.Comment));
    }

    private AggressionPrediction ScoreComment(string comment)
    {
        var classification = _scorer.Predict(comment);

        return new()
        {
            IsAggressive = classification.Prediction,
            Probability = classification.Probability
        };
    }
}

public sealed class AggressionPrediction
{
    public bool IsAggressive { get; set; }

    public float Probability { get; set; }
}

public sealed class CommentWrapper
{
    public string Comment { get; set; }
}