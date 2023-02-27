using AggressionScorer;
using AggressionScorerModel;
using Microsoft.ML;
using System.Diagnostics;

Console.WriteLine("Aggression Scorer Model builder started");

var mlContext = new MLContext(0);

// Build Pipeline
var inputDataPreparer = mlContext.Transforms.Text
    .FeaturizeText("Features", "Comment")
    .AppendCacheCheckpoint(mlContext);

// Create a training algorithm
var trainer = mlContext.BinaryClassification.Trainers.LbfgsLogisticRegression();

var trainingPipeline = inputDataPreparer.Append(trainer);

// Load Data
var createdInputFile = @"Data\preparedInput.tsv";
DataPreparer.CreatePreparedDataFile(createdInputFile, true);

IDataView trainingDataView = mlContext.Data.LoadFromTextFile<ModelInput>(
        path: createdInputFile,
        hasHeader: true,
        separatorChar: '\t',
        allowQuoting: true
    );

// Fit the model
Console.WriteLine("Start Training Model...");
var startTime = Stopwatch.GetTimestamp();
ITransformer model = trainingPipeline.Fit(trainingDataView);
Console.WriteLine($"Model training finished in {Stopwatch.GetElapsedTime(startTime)} seconds.");

// Save the model
Console.WriteLine("Saving the model...");

if (!Directory.Exists("Model"))
{
    Directory.CreateDirectory("Model");
}

var modelFile = @"Model\AggressionScoreModel.zip";
mlContext.Model.Save(model, trainingDataView.Schema, modelFile);

Console.WriteLine($"The model is saved to {modelFile}");