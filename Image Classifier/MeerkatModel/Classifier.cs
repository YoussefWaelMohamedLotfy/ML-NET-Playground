using Microsoft.ML;

namespace MeerkatModel;

public sealed class Classifier
{
    private readonly MLContext _mlContext = new();
    private readonly PredictionEngine<ModelInput, ModelOutput> _predictionEngine;

    public Classifier()
    {
        var loadedModel = _mlContext.Model
           .Load(@"Model\trainedModel.zip", out _);

        _predictionEngine = _mlContext.Model
           .CreatePredictionEngine<ModelInput, ModelOutput>(loadedModel);
    }

    public string Classify(string imagePath)
    {
        // Set up pre processing pipeline just as in training
        var preProcessingPipeline = _mlContext.Transforms
            .LoadRawImageBytes("ImageBytes", Path.GetDirectoryName(imagePath), "ImagePath");

        // Create an array with the image path as a single object
        var imagePathAsArray = new[]
        {
            new
            {
                ImagePath = imagePath,
                Label = string.Empty
            }
        };

        //Load the image into a data view and process it into bytes
        var imagePathDataView = _mlContext.Data.LoadFromEnumerable(imagePathAsArray);

        var imageBytesDataView = preProcessingPipeline.Fit(imagePathDataView).Transform(imagePathDataView);

        // Create a model input to use in the prediction engine
        var modelInput = _mlContext.Data.CreateEnumerable<ModelInput>(imageBytesDataView, true)
                                .First();

        // Make the classification
        ModelOutput prediction = _predictionEngine.Predict(modelInput);

        return prediction.PredictedLabel;
    }
}
