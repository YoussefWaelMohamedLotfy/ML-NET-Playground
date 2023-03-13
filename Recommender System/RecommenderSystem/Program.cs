using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Trainers;
using RecommenderSystem;
using RecommenderSystem.Helpers;

Console.WriteLine("Restaurant Recommender");

MLContext mlContext = new(0);

var trainingDataFile = @"Data\trainingData.tsv";
DataPreparer.PreprocessData(trainingDataFile);

IDataView trainingDataView = mlContext.Data.LoadFromTextFile<ModelInput>(trainingDataFile, hasHeader: true);

var dataProcessingPipeline = mlContext.Transforms.Conversion
    .MapValueToKey("UserIdEncoded", nameof(ModelInput.UserId))
    .Append(mlContext.Transforms.Conversion.MapValueToKey("RestaurantNameEncoded", nameof(ModelInput.RestaurantName)));

var finalOptions = new MatrixFactorizationTrainer.Options
{
    MatrixColumnIndexColumnName = "UserIdEncoded",
    MatrixRowIndexColumnName = "RestaurantNameEncoded",
    LabelColumnName = "TotalRating",
    NumberOfIterations = 10,
    ApproximationRank = 200,
    Quiet = true
};

var trainer = mlContext.Recommendation().Trainers.MatrixFactorization(finalOptions);
var trainingPipeline = dataProcessingPipeline.Append(trainer);

Console.WriteLine("Creating model");
var model = trainingPipeline.Fit(trainingDataView);

// View results
//var testUserId = "U1134";
var predictionEngine = mlContext.Model.CreatePredictionEngine<ModelInput, ModelOutput>(model);

//var alreadyRatedRestaurants = mlContext.Data
//    .CreateEnumerable<ModelInput>(trainingDataView, false)
//    .Where(i => i.UserId == testUserId)
//    .Select(r => r.RestaurantName)
//    .Distinct();

//var allRestaurantNames = trainingDataView.GetColumn<string>("RestaurantName")
//    .Distinct()
//    .Where(r => !alreadyRatedRestaurants.Contains(r));

//var scoredRestaurants = allRestaurantNames.Select(restName =>
//{
//    var prediction = predictionEngine.Predict(new() { RestaurantName = restName, UserId = testUserId });
//    return (RestaurantName: restName, PredictedRating: prediction.Score);
//});

//var top10Restaurants = scoredRestaurants.OrderByDescending(s => s.PredictedRating).Take(10);

//Console.WriteLine($"\nTop 10 Restaurants for {testUserId}");

//foreach (var (RestaurantName, PredictedRating) in top10Restaurants)
//{
//    Console.WriteLine($"Predicted Rating [{PredictedRating:#.0}] for restaurant: \"{RestaurantName}\"");
//}

var crossValMetrics = mlContext.Recommendation()
                               .CrossValidate(trainingDataView, trainingPipeline, labelColumnName: "TotalRating");

var averageRMSE = crossValMetrics.Average(m => m.Metrics.RootMeanSquaredError);
var averageRSquared = crossValMetrics.Average(m => m.Metrics.RSquared);

Console.WriteLine($"\n--- Metrics before tuning hyper parameters ---");
Console.WriteLine($"Cross validated root mean square error: {averageRMSE:#.000}");
Console.WriteLine($"Cross validated RSquared: {averageRSquared:#.000}");
Console.WriteLine();

//HyperParameterExploration(mlContext, dataProcessingPipeline, trainingDataView);

var prediction = predictionEngine.Predict(
    new ModelInput()
    {
        UserId = "CLONED",
        RestaurantName = "Restaurant Wu Zhuo Yi"
    });

Console.WriteLine($"Predicted: {prediction.Score:#.0} for 'Restaurant Wu Zhuo Yi'");



void HyperParameterExploration(MLContext mlContext, IEstimator<ITransformer> dataProcessingPipeline,
     IDataView trainingDataView)
{
    var results = new List<(double rootMeanSquaredError, double rSquared,
                            int iterations, int approximationRank)>();

    for (int iterations = 5; iterations < 100; iterations += 5)
    {
        for (int approximationRank = 50; approximationRank < 250; approximationRank += 50)
        {
            var option = new MatrixFactorizationTrainer.Options
            {
                MatrixColumnIndexColumnName = "UserIdEncoded",
                MatrixRowIndexColumnName = "RestaurantNameEncoded",
                LabelColumnName = "TotalRating",
                NumberOfIterations = iterations,
                ApproximationRank = approximationRank,
                Quiet = true
            };

            var currentTrainer = mlContext.Recommendation().Trainers.MatrixFactorization(option);

            var completePipeline = dataProcessingPipeline.Append(currentTrainer);

            var crossValMetrics = mlContext.Recommendation()
               .CrossValidate(trainingDataView, completePipeline, labelColumnName: "TotalRating");

            results.Add(
                        (crossValMetrics.Average(m => m.Metrics.RootMeanSquaredError),
                         crossValMetrics.Average(m => m.Metrics.RSquared),
                         iterations,
                         approximationRank)
                       );
        }
    }

    Console.WriteLine("\n--- Hyper parameters and metrics ---");

    foreach (var (rootMeanSquaredError, rSquared, iterations, approximationRank) in results.OrderByDescending(r => r.rSquared))
    {
        Console.Write($"NumberOfIterations: {iterations}");
        Console.Write($" ApproximationRank: {approximationRank}");
        Console.Write($" RootMeanSquaredError: {rootMeanSquaredError}");
        Console.WriteLine($" RSquared: {rSquared}");
    }

    Console.WriteLine();
    Console.WriteLine("Done");

}


