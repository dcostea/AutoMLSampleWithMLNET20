using AutoMLSample.Services;
using DataModels.Models;
using Newtonsoft.Json;
using Serilog.Sinks.SystemConsole.Themes;

internal class Program
{
    private static async Task Main(string[] args)
    {
        Log.Logger = new LoggerConfiguration()
            .MinimumLevel.Debug()
            .MinimumLevel.Override("Microsoft", Serilog.Events.LogEventLevel.Warning)
            .MinimumLevel.Override("System", Serilog.Events.LogEventLevel.Warning)
            .WriteTo.Console(outputTemplate: "[{Timestamp:HH:mm:ss} {Level:u3}] {Message:lj}{NewLine}", theme: SystemConsoleTheme.Colored)
            .WriteTo.File(@"logs\log.txt", Serilog.Events.LogEventLevel.Debug, "[{Timestamp:HH:mm:ss} {Level:u3}] {Message:lj}{NewLine}")
            .CreateBootstrapLogger();

        // read settings
        var jsonData = File.ReadAllText(@"c:\Temp\automl.json");
        var trainerSettings = JsonConvert.DeserializeObject<TrainerSettings>(jsonData);
        MachineLearningServices.TrainerSettings = trainerSettings;

        // load dataset and infer columns
        (var trainTestData, var columnInference) = MachineLearningServices.LoadDataAndColumns();

        // auto ml
        var experimentResult = await MachineLearningServices.AutoTrainAsync(trainTestData.TrainSet, columnInference);
        var model = experimentResult.Model as TransformerChain<ITransformer>;

        // evaluate model
        var transformedTestingData = model.Transform(trainTestData.TestSet);
        var transformedData = model.Transform(trainTestData.TestSet);
        //MachineLearningServices.Evaluate(transformedTestingData, columnInference.ColumnInformation.LabelColumnName, showsConfusionMatrix: false);
        
        // permutation feature importance
        MachineLearningServices.PFI(0.01F, model.LastTransformer, transformedData, columnInference.ColumnInformation.LabelColumnName);
        
        // correlation matrix
        MachineLearningServices.PearsonCorrelationMatrix(0.9F, trainTestData.TrainSet);
    }
}