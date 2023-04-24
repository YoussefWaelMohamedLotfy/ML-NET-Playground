using FacialExpressionDetector;
using System.Diagnostics;

Console.WriteLine("Image facial expression detector - using FER+ ONNX model\n");

// Load all image paths
var imageFolder = GetImageFolderFromArgs(args);
var imagePaths = Directory.GetFiles(imageFolder, "*.jpg").Select(Path.GetFullPath).ToArray();


// Run the images through the ONNX model
var facialExpressionDetector = new FacialExpressionDetector.FacialExpressionDetector();

Console.WriteLine("Scoring all images...");
var timeStamp = Stopwatch.GetTimestamp();
var emotionProbabilities = facialExpressionDetector.DetectEmotionsInImageFiles(imagePaths);
Console.WriteLine($"Images scored in {Stopwatch.GetElapsedTime(timeStamp)}\n");

// Print the results
foreach (var fileWithScore in emotionProbabilities)
{
    Console.Write("* ");
    Console.ForegroundColor = ConsoleColor.Cyan;
    Console.WriteLine($"{fileWithScore.Filename}");
    Console.ForegroundColor = ConsoleColor.Gray;

    foreach (var emotion in fileWithScore.EmotionProbabilities)
    {
        if (emotion.Probability > 0.1)
        {
            Console.ForegroundColor = ConsoleColor.Green;
        }

        Console.Write($"[{emotion.Probability:P0}] {emotion.emotion}");
        Console.ForegroundColor = ConsoleColor.Gray;
    }

    Console.WriteLine();
}



string GetImageFolderFromArgs(string[] args)
{
    //Default to "images" folder if no arguments are given
    if (args.Length != 1)
        return "images";

    if (Directory.Exists(args[0]))
        return args[0];

    Console.WriteLine("Given image directory does not exist");
    Environment.Exit(1);

    return null;

}