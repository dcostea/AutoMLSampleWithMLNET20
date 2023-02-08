using System.Data;
using MathNet.Numerics.Statistics;
using Microsoft.ML.Trainers.FastTree;
using static AutoMLSample.Helpers.ConsoleHelpers;
using static Microsoft.ML.DataOperationsCatalog;
using Microsoft.ML.Runtime;
using DataModels.Models;

namespace AutoMLSample.Services;

internal static class MachineLearningServices
{
    private static MLContext Context { get; set; } = new MLContext(seed: 1);
    public static TrainerSettings TrainerSettings { get; set; }

    internal static (TrainTestData, ColumnInferenceResults) LoadDataAndColumns()
    {
        var columnInferenceResults = Context.Auto().InferColumns(path: TrainerSettings.CsvDataPath, labelColumnName: TrainerSettings.LabelColumn, groupColumns: false);
        var loader = Context.Data.CreateTextLoader(columnInferenceResults.TextLoaderOptions);
        var data = loader.Load(TrainerSettings.CsvDataPath);
        TrainTestData trainValidationData = Context.Data.TrainTestSplit(data, testFraction: 0.2);

        return (trainValidationData, columnInferenceResults);
    }

    internal static async Task<TrialResult> AutoTrainAsync(IDataView data, ColumnInferenceResults columnInference)
    {
        Context.Log += (_, e) =>
        {
            if (e.Source.Equals("AutoMLExperiment") && e.Kind.Equals(ChannelMessageKind.Info))
            {
                WriteLineColor(e.Message, ConsoleColor.White);
            }
        };

        WriteLineColor($" INCREASE ML MODEL ACCURACY IN THREE STEPS");
        WriteLineColor($" Learning type: {TrainerSettings.Scenario}");
        WriteLineColor($" Training time: {TrainerSettings.TrainingTime} seconds");
        WriteLineColor("----------------------------------------------------------------------------------");

        Context = new MLContext(seed: 1);

        // create training pipeline
        SweepablePipeline pipeline = null;
        switch (TrainerSettings.Scenario)
        {
            case nameof(Scenario.BinaryClassification):
                //Featurizer(IDataView data, string outputColumnName = "Features", string[] catelogicalColumns = null, string[] numericColumns = null, string[] textColumns = null, string[] imagePathColumns = null, string[] excludeColumns = null)
                pipeline = Context.Auto().Featurizer(data, columnInformation: columnInference.ColumnInformation)
                    .Append(Context.Auto().BinaryClassification(labelColumnName: columnInference.ColumnInformation.LabelColumnName));
                break;
            case nameof(Scenario.MulticlassClassification):
                pipeline = Context.Auto().Featurizer(data, columnInformation: columnInference.ColumnInformation)
                    .Append(Context.Transforms.Conversion.MapValueToKey(columnInference.ColumnInformation.LabelColumnName, columnInference.ColumnInformation.LabelColumnName))
                    .Append(Context.Auto().MultiClassification(labelColumnName: columnInference.ColumnInformation.LabelColumnName));
                break;
            case nameof(Scenario.Regression):
                pipeline = Context.Auto().Featurizer(data, columnInformation: columnInference.ColumnInformation)
                    .Append(Context.Auto().Regression(labelColumnName: columnInference.ColumnInformation.LabelColumnName));
                break;
        }

        AutoMLExperiment experiment = Context.Auto().CreateExperiment()
            .SetPipeline(pipeline)
            .SetTrainingTimeInSeconds(TrainerSettings.TrainingTime)
            .SetDataset(data);

        // set sorting metric
        switch (TrainerSettings.Scenario)
        {
            case nameof(Scenario.BinaryClassification):
                experiment.SetBinaryClassificationMetric(BinaryClassificationMetric.Accuracy, labelColumn: columnInference.ColumnInformation.LabelColumnName);
                break;
            case nameof(Scenario.MulticlassClassification):
                experiment.SetMulticlassClassificationMetric(MulticlassClassificationMetric.MicroAccuracy, labelColumn: columnInference.ColumnInformation.LabelColumnName);
                break;
            case nameof(Scenario.Regression):
                experiment.SetRegressionMetric(RegressionMetric.RSquared, labelColumn: columnInference.ColumnInformation.LabelColumnName);
                break;
        }

        // Log experiment trials
        var monitor = AutoMLMonitor.Create(pipeline);
        experiment.SetMonitor(monitor);

        var cts = new CancellationTokenSource();
        var experimentResult = await experiment.RunAsync(cts.Token);

        WriteLineColor("----------------------------------------------------------------------------------");
        WriteLineColor($" STEP 1: AutoML experiment result");
        WriteLineColor("----------------------------------------------------------------------------------");
        WriteLineColor($" Top best trainers:");
        var bestTrials = monitor.GetBestTrials();
        foreach (var trial in bestTrials.OrderByDescending(s => s.Metric))
        {
            var trainer = AutoMLMonitor.ExtractTrainerName(trial.TrialSettings);
            WriteLineColor($" Accuracy: {trial.Metric,-6:F3}   Loss: {trial.Loss,-6:F3}   Training time: {trial.DurationInMilliseconds,5} ms   Trainer: {trainer.EstimatorType}");
        }
        WriteLineColor("----------------------------------------------------------------------------------");
        WriteLineColor($" Top completed trainers:");
        var completedTrials = monitor.GetCompletedTrials();
        foreach (var trial in completedTrials.OrderByDescending(s => s.Metric))
        {
            var trainer = AutoMLMonitor.ExtractTrainerName(trial.TrialSettings);
            WriteLineColor($" Accuracy: {trial.Metric,-6:F3}   Loss: {trial.Loss,-6:F3}   Training time: {trial.DurationInMilliseconds,5} ms   Trainer: {trainer.EstimatorType}");
        }
        WriteLineColor("----------------------------------------------------------------------------------");
        WriteLineColor($" Best trainer: { monitor.GetBestTrial(experimentResult)}");
        WriteLineColor($" Accuracy: {experimentResult.Metric,-6:F3}   Training time: {experimentResult.DurationInMilliseconds,5} ms   CPU: {monitor.PeakCpu,5:P2}   Memory: {monitor.PeakMemoryInMegaByte,5:F2}MB");

        return experimentResult;
    }

    internal static void Evaluate(IDataView transformedData, string label, bool showsConfusionMatrix)
    {
        switch (TrainerSettings.Scenario)
        {
            case nameof(Scenario.BinaryClassification):
                var metricsBinaryClassification = Context.BinaryClassification.Evaluate(transformedData, label);
                //TODO implement next method
                //PrintBinaryClassificationMetrics(metricsBinaryClassification, showsConfusionMatrix);
                break;
            case nameof(Scenario.MulticlassClassification):
                var metricsMulticlassClassification = Context.MulticlassClassification.Evaluate(transformedData, label);
                PrintMulticlassClassificationMetrics(metricsMulticlassClassification, showsConfusionMatrix);
                break;
            case nameof(Scenario.Regression):
                var metricsRegression = Context.Regression.Evaluate(transformedData, label);
                //TODO implement next method
                //PrintRegressionMetrics(metricsRegression, showsConfusionMatrix);
                break;
        }
    }

    internal static void PFI(float threshold, ITransformer transformer, IDataView transformedData, string label)
    {
        WriteLineColor("----------------------------------------------------------------------------------");
        WriteLineColor(" STEP 2: PFI (permutation feature importance)");
        WriteLineColor("----------------------------------------------------------------------------------");
        WriteLineColor($" PFI, threshold: {threshold}");
        WriteLineColor("----------------------------------------------------------------------------------");
        WriteLineColor($"  {"No",4} {"Feature",-15} {"MicroAccuracy",15} {"95% Mean",15}");

        Context.Log += (_, e) =>
        {
            if (e.Source.StartsWith("Permutation") && e.Kind.Equals(ChannelMessageKind.Info))
            {
                if (int.TryParse(e.RawMessage.Split(" ").Last(), out var noOfSlots))
                {
                    if (noOfSlots > 20)
                    {
                        Log.Error("Number of slots is too high {@noOfSlots}! PFI will take very long time to finish.", noOfSlots);
                        return;
                    }
                }
                else 
                {
                    Log.Debug(e.Message);
                }
            }
        };

        uint noCnt = 1;

        switch (TrainerSettings.Scenario)
        {
            case nameof(Scenario.MulticlassClassification):
                var pfiMulticlassClassification = Context.MulticlassClassification.PermutationFeatureImportance(transformer, transformedData, label, permutationCount: 5);
                ////var metricsMulticlassClassification = pfi.Select(p => (p.Key, p.Value.MicroAccuracy)).OrderBy(m => m.MicroAccuracy.Mean);

                // patching dot issue when collecting the multiclass classification metrics
                var patchedPfiMulticlassClassification = pfiMulticlassClassification.Select(p => new KeyValuePair<string, MulticlassClassificationMetricsStatistics>(p.Key.Split(".").First(), p.Value));
                var groupedPfiMulticlassClassification = patchedPfiMulticlassClassification.GroupBy(p => p.Key).ToDictionary(g => g.Key, g => g.Select(x => x.Value));
                var metricsMulticlassClassification = groupedPfiMulticlassClassification.Select(p => new MicroAccuracyModel
                {
                    Key = p.Key,
                    MicroAccuracy = new MetricStatisticsModel
                    {
                        Mean = p.Value.Sum(m => m.MicroAccuracy.Mean),
                        StandardError = p.Value.Sum(m => m.MicroAccuracy.StandardError)
                    }
                })
                .OrderBy(m => m.MicroAccuracy.Mean);
                foreach (var metric in metricsMulticlassClassification)
                {
                    if (Math.Abs(metric.MicroAccuracy.Mean) < threshold)
                    {
                        WriteLineColor($"  {noCnt++,3}. {metric.Key,-15} {metric.MicroAccuracy.Mean,15:F5} {1.95 * metric.MicroAccuracy.StandardError,15:F5} (candidate for deletion!)", ConsoleColor.Red);
                    }
                    else
                    {
                        WriteLineColor($"  {noCnt++,3}. {metric.Key,-15} {metric.MicroAccuracy.Mean,15:F4} {1.95 * metric.MicroAccuracy.StandardError,15:F4}");
                    }
                }
                break;

            case nameof(Scenario.Regression):
                var pfiRegression = Context.Regression.PermutationFeatureImportance(transformer, transformedData, label, permutationCount: 5);
                ////var metricsRegression = pfi.Select(p => (p.Key, p.Value.RSquared)).OrderBy(m => m.RSquared.Mean);

                // patching dot issue when collecting the regression metrics
                var patchedPfiRegression = pfiRegression.Select(p => new KeyValuePair<string, RegressionMetricsStatistics>(p.Key.Split(".").First(), p.Value));
                var groupedPfiRegression = patchedPfiRegression.GroupBy(p => p.Key).ToDictionary(g => g.Key, g => g.Select(x => x.Value));
                var metricsRegression = groupedPfiRegression.Select(p => new RSquaredModel
                {
                    Key = p.Key,
                    RSquared = new MetricStatisticsModel
                    {
                        Mean = p.Value.Sum(m => m.RSquared.Mean),
                        StandardError = p.Value.Sum(m => m.RSquared.StandardError)
                    }
                })
                .OrderBy(m => m.RSquared.Mean);
                foreach (var metric in metricsRegression)
                {
                    if (Math.Abs(metric.RSquared.Mean) < threshold)
                    {
                        WriteLineColor($"  {noCnt++,3}. {metric.Key,-15} {metric.RSquared.Mean,15:F5} {1.95 * metric.RSquared.StandardError,15:F5} (candidate for deletion!)", ConsoleColor.Red);
                    }
                    else
                    {
                        WriteLineColor($"  {noCnt++,3}. {metric.Key,-15} {metric.RSquared.Mean,15:F4} {1.95 * metric.RSquared.StandardError,15:F4}");
                    }
                }
                WriteLineColor("----------------------------------------------------------------------------------");
                break;
        }
    }

    internal static void PearsonCorrelationMatrix(float threshold, IDataView trainingDataView)
    {
        var trainingDataCollection = Context.Data.CreateEnumerable<GeneratedDataModels.InputDataModel>(trainingDataView, reuseRowObject: true);
        (var header, var dataArray) = trainingDataCollection.ExtractDataAndHeader();
        var matrix = Correlation.PearsonMatrix(dataArray.ToArray());
        ////var header = columnInference.ColumnInformation.NumericColumnNames
        ////    .Union(columnInference.ColumnInformation.CategoricalColumnNames)
        ////    .ToArray();
        matrix.ToConsole(header, threshold);

        WriteLineColor("  We can remove one of the next high correlated features!");
        WriteLineColor("    - closer to  0 => low correlated features");
        WriteLineColor("    - closer to  1 => direct high correlated features");
        WriteLineColor("    - closer to -1 => inverted high correlated features");
        WriteLineColor("----------------------------------------------------------------------------------");
        WriteLineColor($"  {"No",4} {"Feature",-15} vs. {"Feature",-15} {"Rate",15}");
        uint noCnt = 1;
        for (int i = 0; i < matrix.ColumnCount; i++)
        {
            for (int j = i; j < matrix.ColumnCount; j++)
            {
                if (i != j && Math.Abs(matrix[i, j]) > threshold)
                {
                    WriteLineColor($"  {noCnt++,3}. {header[i],-15} vs. {header[j],-15} {matrix[i, j],15:F4}", ConsoleColor.Red);
                }
            }
        }
        WriteLineColor("----------------------------------------------------------------------------------");
    }

    private static (string[] header, double[][] dataArray) ExtractDataAndHeader(this IEnumerable<GeneratedDataModels.InputDataModel> trainingDataCollection)
    {
        var record = new GeneratedDataModels.InputDataModel();
        var props = record.GetType().GetProperties();

        var data = new List<List<double>>();
        uint k = 0;
        foreach (var prop in props)
        {
            if (props[k].PropertyType.Name.Equals(nameof(Single)))
            {
                var arr = trainingDataCollection.Select(r => (double)(props[k].GetValue(r) as float?).Value).ToList();
                data.Add(arr);
            }
            k++;
        }
        var header = props.Where(s => s.PropertyType.Name.Equals(nameof(Single))).Select(p => p.Name).ToArray();
        var dataArray = new double[data.Count][];
        for (int i = 0; i < data.Count; i++)
        {
            dataArray[i] = data[i].ToArray();
        }

        return (header, dataArray);
    }

    private class MicroAccuracyModel
    {
        public string Key { get; set; }
        public MetricStatisticsModel MicroAccuracy { get; set; }
    }

    private class RSquaredModel
    {
        public string Key { get; set; }
        public MetricStatisticsModel RSquared { get; set; }
    }

    private class MetricStatisticsModel
    {
        public double Mean { get; set; }
        public double StandardError { get; set; }
        //public double StandardDeviation { get; set; }
    }
}
