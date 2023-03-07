using AggressionScorer;
using AggressionScorerModel;
using Microsoft.ML;
using Microsoft.ML.Calibrators;
using Microsoft.ML.Trainers;
using System.Diagnostics;

CrossValidate();

void CrossValidate()
{
    var mlContext = new MLContext(0);

    // Create a small Dataset for faster cross validation
    string createdFileInput = @"Data\preparedInput.tsv";
    DataPreparer.CreatePreparedDataFile(createdFileInput, onlySaveSmallSubset: true);

    IDataView inputDataView = mlContext.Data.LoadFromTextFile<ModelInput>(
        path: createdFileInput,
        hasHeader: true,
        separatorChar: '\t',
        allowQuoting: true
    );

    var dataPipeline = mlContext.Transforms.Text
        .FeaturizeText("Features", "Comment")
        .Append(mlContext.Transforms.NormalizeMeanVariance("Features"))
        .AppendCacheCheckpoint(mlContext);

    // Create the training Algorithms
    var trainers = new IEstimator<ITransformer>[]
    {
        mlContext.BinaryClassification.Trainers.SgdCalibrated(),
        mlContext.BinaryClassification.Trainers.LbfgsLogisticRegression()
    };

    foreach (var trainer in trainers)
    {
        var modelPipeline = dataPipeline.Append(trainer);

        Console.WriteLine($"Cross validating with {trainer.GetType().Name}");

        var crossValidationResults = mlContext.BinaryClassification.CrossValidate(inputDataView, modelPipeline, 5);
        
        var averageAccuracy = crossValidationResults.Average(m => m.Metrics.Accuracy);
        Console.WriteLine($"Cross validated average accuracy: {averageAccuracy:0.###}");
        
        var averageF1Score = crossValidationResults.Average(m => m.Metrics.F1Score);
        Console.WriteLine($"Cross validated average F1Score: {averageF1Score:0.###}\n");
    }
}

return;

Console.WriteLine("Aggression Scorer Model builder started");

var mlContext = new MLContext(0);

// Load Data
var createdInputFile = @"Data\preparedInput.tsv";
DataPreparer.CreatePreparedDataFile(createdInputFile, true);

IDataView trainingDataView = mlContext.Data.LoadFromTextFile<ModelInput>(
        path: createdInputFile,
        hasHeader: true,
        separatorChar: '\t',
        allowQuoting: true
    );

var inputSplitData = mlContext.Data.TrainTestSplit(trainingDataView, .2, seed: 0);

// Build Pipeline
var inputDataPreparer = mlContext.Transforms.Text
    .FeaturizeText("Features", "Comment")
    .Append(mlContext.Transforms.NormalizeMeanVariance("Features"))
    .AppendCacheCheckpoint(mlContext)
    .Fit(inputSplitData.TrainSet);

// Create a training algorithm
var trainer = mlContext.BinaryClassification.Trainers.LbfgsLogisticRegression();

// Fit the model
Console.WriteLine("Start Training Model...");

var startTime = Stopwatch.GetTimestamp();
var transformedData = inputDataPreparer.Transform(inputSplitData.TrainSet);
ITransformer model = trainer.Fit(transformedData);

Console.WriteLine($"Model training finished in {Stopwatch.GetElapsedTime(startTime)} seconds.");

// Test the model
EvaluateModel(mlContext, model, inputDataPreparer.Transform(inputSplitData.TestSet));

// Save the model
Console.WriteLine("Saving the model...");

if (!Directory.Exists("Model"))
{
    Directory.CreateDirectory("Model");
}

var modelFile = @"Model\AggressionScoreModel.zip";
mlContext.Model.Save(model, trainingDataView.Schema, modelFile);
Console.WriteLine($"The model is saved to {modelFile}");

var dataPreparePipelineFile = @"dataPreparePipeline.zip";
mlContext.Model.Save(inputDataPreparer, trainingDataView.Schema, dataPreparePipelineFile);
Console.WriteLine($"The model is saved to {dataPreparePipelineFile}");

var retrainedModel = RetrainModel(modelFile, dataPreparePipelineFile);
var completedRetrainedPipeline = inputDataPreparer.Append(retrainedModel);

string retrainedModelFile = @"Model\AggressionRetrainedModel.zip";
mlContext.Model.Save(completedRetrainedPipeline, trainingDataView.Schema, retrainedModelFile);
Console.WriteLine($"The model is saved to {dataPreparePipelineFile}");

EvaluateModel(mlContext, completedRetrainedPipeline, inputSplitData.TestSet);

ITransformer RetrainModel(string modelFile, string dataPreparePipelineFile)
{
    MLContext context = new(0);

    // Load Pre-Trained model
    ITransformer pretrainedModel = context.Model.Load(modelFile, out _);

    // Extract Pre-Trained model parameters
    var pretrainedModelParameters =
        ((ISingleFeaturePredictionTransformer
        <CalibratedModelParametersBase<LinearBinaryModelParameters, PlattCalibrator>>)pretrainedModel)
        .Model.SubModel;

    string dataFile = @"Data\preparedInput.tsv";
    DataPreparer.CreatePreparedDataFile(dataFile, false);

    IDataView trainingDataView = context.Data.LoadFromTextFile<ModelInput>(
            path: dataFile,
            hasHeader: true,
            separatorChar: '\t',
            allowQuoting: true
        );

    // Load data preparation pipeline
    ITransformer dataPreparationPipeline = context.Model.Load(dataPreparePipelineFile, out _);

    // Prepare input data to a form consumable by ML Model
    var newData = dataPreparationPipeline.Transform(trainingDataView);

    // Retrain Model
    Console.WriteLine("Start retraining model");
    var startTime = Stopwatch.GetTimestamp();

    var retrainedModel = context.BinaryClassification.Trainers
    .LbfgsLogisticRegression()
    .Fit(newData, pretrainedModelParameters);

    Console.WriteLine($"Model retraining finished in {Stopwatch.GetElapsedTime(startTime)}");

    return retrainedModel;
}

void EvaluateModel(MLContext mlContext, ITransformer model, IDataView trainingDataView)
{
    Console.WriteLine("-- Evaluating binary classification model performance --");
    var predictedData = model.Transform(trainingDataView);
    var metrics = mlContext.BinaryClassification.Evaluate(predictedData);
    Console.WriteLine($"Accuracy: {metrics.Accuracy:0.###}");

    Console.WriteLine("Confussion Matrix\n");
    Console.WriteLine(metrics.ConfusionMatrix.GetFormattedConfusionTable());
}