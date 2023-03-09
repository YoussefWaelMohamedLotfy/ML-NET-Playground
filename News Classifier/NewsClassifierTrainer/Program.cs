using Microsoft.ML;
using Microsoft.ML.AutoML;
using NewsClassifierModel;
using NewsClassifierTrainer.Common;
using System.Diagnostics;

Console.WriteLine("News Classifier Trainer started...");

//FindTheBestModel();
TrainTheModel();
TrainTheModelWithUnbalancedData();

void TrainTheModelWithUnbalancedData()
{
    Console.WriteLine("Training the model to use with unbalanced Data...");
    var mlContext = new MLContext(0);

    var trainingDataPath = @"Data\uci-news-aggregator.csv";

    var unbalancedDataFile = @"Data\unbalanced.csv";
    CreateUnbalancedData(trainingDataPath, unbalancedDataFile);

    IDataView trainingDataView = mlContext.Data.LoadFromTextFile<ModelInput>(
        path: unbalancedDataFile,
        hasHeader: true,
        separatorChar: ',',
        allowQuoting: true
    );

    var preProcessingPipeline = mlContext.Transforms.Conversion
        .MapValueToKey("Label", "Category")
        .Append(mlContext.Transforms.Text.FeaturizeText("Features", "Title"))
        .Append(mlContext.Transforms.NormalizeMinMax("Features"))
        .AppendCacheCheckpoint(mlContext);

    var trainer = mlContext.MulticlassClassification.Trainers
        .OneVersusAll(mlContext.BinaryClassification.Trainers.AveragedPerceptron());

    var trainingPipeline = preProcessingPipeline.Append(trainer)
        .Append(mlContext.Transforms.Conversion.MapKeyToValue("PredictedLabel"));

    Console.WriteLine("Cross Validating model...");
    var cvResults = mlContext.MulticlassClassification.CrossValidate(trainingDataView, trainingPipeline);

    var microAccuracy = cvResults.Average(m => m.Metrics.MicroAccuracy);
    var macroAccuracy = cvResults.Average(m => m.Metrics.MacroAccuracy);
    var logLossReduction = cvResults.Average(m => m.Metrics.LogLossReduction);

    Console.WriteLine("\nCross Validation Metrics for our model");
    Console.WriteLine($"MicroAccuracy: {microAccuracy:0.###}");
    Console.WriteLine($"MacroAccuracy: {macroAccuracy:0.###}");
    Console.WriteLine($"LogLossReduction: {logLossReduction:0.###}");

    Console.WriteLine("Training model...");
    var startTime = Stopwatch.GetTimestamp();

    var finalModel = trainingPipeline.Fit(trainingDataView);
    Console.WriteLine($"Model training finished in {Stopwatch.GetElapsedTime(startTime)}...");

    // Save Model
    if (!Directory.Exists("Model"))
    {
        Directory.CreateDirectory("Model");
    }

    var modelPath = @"Model\NewsClassificationModel.zip";
    mlContext.Model.Save(finalModel, trainingDataView.Schema, modelPath);
    Console.WriteLine($"Model saved to {modelPath}");
}

void TrainTheModel()
{
    Console.WriteLine("Training the model to use...");
    var mlContext = new MLContext(0);

    var trainingDataPath = @"Data\uci-news-aggregator.csv";

    IDataView trainingDataView = mlContext.Data.LoadFromTextFile<ModelInput>(
        path: trainingDataPath,
        hasHeader: true,
        separatorChar: ',',
        allowQuoting: true
    );

    var preProcessingPipeline = mlContext.Transforms.Conversion
        .MapValueToKey("Label", "Category")
        .Append(mlContext.Transforms.Text.FeaturizeText("Features", "Title"))
        .Append(mlContext.Transforms.NormalizeMinMax("Features"))
        .AppendCacheCheckpoint(mlContext);

    var trainer = mlContext.MulticlassClassification.Trainers
        .OneVersusAll(mlContext.BinaryClassification.Trainers.AveragedPerceptron());

    var trainingPipeline = preProcessingPipeline.Append(trainer)
        .Append(mlContext.Transforms.Conversion.MapKeyToValue("PredictedLabel"));

    Console.WriteLine("Cross Validating model...");
    var cvResults = mlContext.MulticlassClassification.CrossValidate(trainingDataView, trainingPipeline);

    var microAccuracy = cvResults.Average(m => m.Metrics.MicroAccuracy);
    var macroAccuracy = cvResults.Average(m => m.Metrics.MacroAccuracy);
    var logLossReduction = cvResults.Average(m => m.Metrics.LogLossReduction);

    Console.WriteLine("\nCross Validation Metrics for our model");
    Console.WriteLine($"MicroAccuracy: {microAccuracy:0.###}");
    Console.WriteLine($"MacroAccuracy: {macroAccuracy:0.###}");
    Console.WriteLine($"LogLossReduction: {logLossReduction:0.###}");

    Console.WriteLine("Training model...");
    var startTime = Stopwatch.GetTimestamp();

    var finalModel = trainingPipeline.Fit(trainingDataView);
    Console.WriteLine($"Model training finished in {Stopwatch.GetElapsedTime(startTime)}...");

    // Save Model
    if (!Directory.Exists("Model"))
    {
        Directory.CreateDirectory("Model");
    }

    var modelPath = @"Model\NewsClassificationModel.zip";
    mlContext.Model.Save(finalModel, trainingDataView.Schema, modelPath);
    Console.WriteLine($"Model saved to {modelPath}");
}

void CreateUnbalancedData(string inputDataFile, string outPutDataFile)
{
    var inputFileRows = File.ReadAllLines(inputDataFile);
    var outputRows = new List<string>
    {
        // Add Header to Output
        inputFileRows.First()
    };

    int entertainmentSamples = 0;
    int businessSamples = 0;
    int technologySamples = 0;
    int medicineSamples = 0;

    foreach (var row in inputFileRows.Skip(1))
    {
        if (row.Contains(",b,")) // business sample
        {
            if (Random.Shared.NextDouble() <= .1)
            {
                outputRows.Add(row);
                businessSamples++;
            }
        }
        else if (row.Contains(",e,")) // entertainment sample
        {
            if (Random.Shared.NextDouble() <= .1)
            {
                outputRows.Add(row);
                entertainmentSamples++;
            }
        }
        else if (row.Contains(",t,")) // technology sample
        {
            outputRows.Add(row);
            technologySamples++;
        }
        else if (row.Contains(",m,")) // technology sample
        {
            outputRows.Add(row);
            medicineSamples++;
        }
        
        File.WriteAllLines(outPutDataFile, outputRows);

        Console.WriteLine("\nSampled in the training data");
        Console.WriteLine($"Business: {businessSamples}");
        Console.WriteLine($"Entertainment: {entertainmentSamples}");
        Console.WriteLine($"Technology: {technologySamples}");
        Console.WriteLine($"Medicine: {medicineSamples}\n");

    }
}

void FindTheBestModel()
{
    Console.WriteLine("Finding the best model to use with AutoML...");
    var mlContext = new MLContext(0);

    var trainingDataPath = @"Data\uci-news-aggregator.csv";

    IDataView trainingDataView = mlContext.Data.LoadFromTextFile<ModelInput>(
        path: trainingDataPath,
        hasHeader: true,
        separatorChar: ',',
        allowQuoting: true
    );

    var preProcessingPipeline = mlContext.Transforms.Conversion
        .MapValueToKey("Category", "Category");

    var mappedInputData = preProcessingPipeline.Fit(trainingDataView)
        .Transform(trainingDataView);

    var experimentSettings = new MulticlassExperimentSettings
    {
        MaxExperimentTimeInSeconds = 300,
        CacheBeforeTrainer = CacheBeforeTrainer.On,
        OptimizingMetric = MulticlassClassificationMetric.MicroAccuracy,
        CacheDirectoryName = null
    };

    var experiment = mlContext.Auto().CreateMulticlassClassificationExperiment(experimentSettings);
    Console.WriteLine("Starting Experiments");

    var experimentResults = experiment.Execute(
            trainData: mappedInputData,
            labelColumnName: "Category",
            progressHandler: new MulticlassExperimentProgressHandler()
        );
    Console.WriteLine("Metrics from best run:");

    var metrics = experimentResults.BestRun.ValidationMetrics;
    Console.WriteLine($"MicroAccuracy: {metrics.MicroAccuracy:0.###}");
    Console.WriteLine($"MacroAccuracy: {metrics.MacroAccuracy:0.###}");

}