using NewsClassifierModel;

Console.WriteLine("--- News classifier ---\n");

string[] headlines = new[]
{
    "Stocks move lower on discouraging news from space",
    "Respawn: Patch to increase game resolution likely",
    "'Some movie' sequel is in the works",
    "Measles Outbreak In Some County"
};

foreach (var headline in headlines)
{
    var classification = ConsumeModel.Predict(headline);

    Console.WriteLine($"{classification}: {headline}");
}