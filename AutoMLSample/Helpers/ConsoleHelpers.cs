using MathNet.Numerics.LinearAlgebra;
using static System.Console;

namespace AutoMLSample.Helpers;

public static class ConsoleHelpers
{
    private const int Width = 114;
    private const ConsoleColor YellowColor = ConsoleColor.Yellow;

    internal static void PrintMulticlassClassificationMetrics(MulticlassClassificationMetrics metrics, bool showsConfusionMatrix)
    {
        WriteLineColor($"----------------------------------------------------------------------------------");
        WriteLineColor($"       {"MicroAccuracy",18} {"MacroAccuracy",18} {"LogLoss",18} {"LogLossReduction",18}");
        WriteLineColor($"       {metrics.MicroAccuracy,18:F3} {metrics.MacroAccuracy,18:F3} {metrics.LogLoss,18:F3} {metrics.LogLossReduction,18:F3}");

        if (showsConfusionMatrix)
        {
            WriteLineColor($"----------------------------------------------------------------------------------");
            WriteLineColor(metrics.ConfusionMatrix.GetFormattedConfusionTable());
        }
        WriteLineColor("----------------------------------------------------------------------------------");
    }

    internal static void PrintMulticlassClassificationFoldsAverageMetrics(IEnumerable<TrainCatalogBase.CrossValidationResult<MulticlassClassificationMetrics>> crossValResults)
    {
        var metricsInMultipleFolds = crossValResults.Select(r => r.Metrics);

        var microAccuracyValues = metricsInMultipleFolds.Select(m => m.MicroAccuracy);
        var microAccuracyAverage = microAccuracyValues.Average();
        var microAccuraciesStdDeviation = ModelHelpers.CalculateStandardDeviation(microAccuracyValues);
        var microAccuraciesConfidenceInterval95 = ModelHelpers.CalculateConfidenceInterval95(microAccuracyValues);

        var macroAccuracyValues = metricsInMultipleFolds.Select(m => m.MacroAccuracy);
        var macroAccuracyAverage = macroAccuracyValues.Average();
        var macroAccuraciesStdDeviation = ModelHelpers.CalculateStandardDeviation(macroAccuracyValues);
        var macroAccuraciesConfidenceInterval95 = ModelHelpers.CalculateConfidenceInterval95(macroAccuracyValues);

        var logLossValues = metricsInMultipleFolds.Select(m => m.LogLoss);
        var logLossAverage = logLossValues.Average();
        var logLossStdDeviation = ModelHelpers.CalculateStandardDeviation(logLossValues);
        var logLossConfidenceInterval95 = ModelHelpers.CalculateConfidenceInterval95(logLossValues);

        var logLossReductionValues = metricsInMultipleFolds.Select(m => m.LogLossReduction);
        var logLossReductionAverage = logLossReductionValues.Average();
        var logLossReductionStdDeviation = ModelHelpers.CalculateStandardDeviation(logLossReductionValues);
        var logLossReductionConfidenceInterval95 = ModelHelpers.CalculateConfidenceInterval95(logLossReductionValues);

        WriteLineColor($"----------------------------------------------------------------------------------");
        WriteLineColor($" Cross validation metrics multi-class classification model (against all dataset)");
        WriteLineColor($"       {"",18} {"Avg",18} {"StdDev",18} {"ConfInterval(95%)",18}");
        WriteLineColor($"       {"MicroAccuracy",18} {microAccuracyAverage,18:F3} {microAccuraciesStdDeviation,18:F3} {microAccuraciesConfidenceInterval95,18:F3}");
        WriteLineColor($"       {"MacroAccuracy",18} {macroAccuracyAverage,18:F3} {macroAccuraciesStdDeviation,18:F3} {macroAccuraciesConfidenceInterval95,18:F3}");
        WriteLineColor($"       {"LogLoss",18} {logLossAverage,18:F3} {logLossStdDeviation,18:F3} {logLossConfidenceInterval95,18:F3}");
        WriteLineColor($"       {"LogLossReduction",18} {logLossReductionAverage,18:F3} {logLossReductionStdDeviation,18:F3} {logLossReductionConfidenceInterval95,18:F3}");
        WriteLineColor($"----------------------------------------------------------------------------------");
    }

    internal static void PrintMulticlassClassificationMetricsHeader()
    {
        CreateRow($"  {"No",4} {"Trainer",-35} {"MicroAccuracy",14} {"MacroAccuracy",14} {"Duration",9}", Width);
    }

    internal static void PrintIterationMetrics(int iteration, string trainerName, BinaryClassificationMetrics metrics, double? runtimeInSeconds)
    {
        CreateRow($"  {iteration,3}. {trainerName,-35} {metrics?.Accuracy ?? double.NaN,9:F3} {metrics?.AreaUnderRocCurve ?? double.NaN,8:F3} {metrics?.AreaUnderPrecisionRecallCurve ?? double.NaN,8:F3} {metrics?.F1Score ?? double.NaN,9:F3} {runtimeInSeconds.Value,9:F1}", Width);
    }

    internal static void PrintIterationMetrics(int iteration, string trainerName, MulticlassClassificationMetrics metrics, double? runtimeInSeconds)
    {
        CreateRow($"  {iteration,3}. {trainerName,-35} {metrics?.MicroAccuracy ?? double.NaN,14:F3} {metrics?.MacroAccuracy ?? double.NaN,14:F3} {runtimeInSeconds.Value,9:F1}", Width);
    }

    internal static void PrintIterationMetrics(int iteration, string trainerName, RegressionMetrics metrics, double? runtimeInSeconds)
    {
        CreateRow($"  {iteration,3}. {trainerName,-35} {metrics?.RSquared ?? double.NaN,8:F3} {metrics?.MeanAbsoluteError ?? double.NaN,13:F2} {metrics?.MeanSquaredError ?? double.NaN,12:F2} {metrics?.RootMeanSquaredError ?? double.NaN,8:F2} {runtimeInSeconds.Value,9:F1}", Width);
    }

    internal static void PrintIterationException(Exception ex, string trainerName)
    {
        Log.Debug($"Exception during AutoML iteration (trainer: {trainerName}): OperationCanceledException");
        //Log.Debug($"Exception during AutoML iteration: {ex}");
    }

    private static void CreateRow(string message, int width)
    {
        WriteLineColor(message.PadRight(width - 2));
    }

    internal static void WriteLineColor(string textLine, ConsoleColor color = YellowColor)
    {
        ForegroundColor = color;
        WriteLine(textLine);
        ResetColor();
    }

    internal static void ToConsole(this Matrix<double> matrix, string[] header, float threshold, bool showLegend = false)
    {
        WriteLineColor(" STEP 3: CORRELATION MATRIX");
        WriteLineColor($"----------------------------------------------------------------------------------");
        WriteLineColor($" Correlation Matrix, threshold: {threshold}");
        WriteLineColor($"----------------------------------------------------------------------------------");
        Write(new string(' ', 12));

        for (int i = 0; i < header.Length; i++)
        {
            Write($"{header[i][..Math.Min(header[i].Length, 9)],9}");
        }

        for (int i = 0; i < matrix.ColumnCount; i++)
        {
            WriteLine();
            Write($"{header[i][..Math.Min(header[i].Length, 12)],12}");

            for (int j = 0; j < matrix.ColumnCount; j++)
            {
                switch (matrix[i, j])
                {
                    case double n when n >= -1 && n < -0.8:
                        ForegroundColor = ConsoleColor.Red;
                        break;

                    case double n when n >= -0.8 && n < -0.6:
                        ForegroundColor = ConsoleColor.Yellow;
                        break;

                    case double n when n >= -0.6 && n < -0.4:
                        ForegroundColor = ConsoleColor.Green;
                        break;

                    case double n when n >= -0.4 && n < -0.2:
                        ForegroundColor = ConsoleColor.Blue;
                        break;

                    case double n when n >= -0.2 && n < 0:
                        ForegroundColor = ConsoleColor.Gray;
                        break;

                    case double n when n >= 0 && n < 0.2:
                        ForegroundColor = ConsoleColor.Gray;
                        BackgroundColor = ConsoleColor.Black;
                        break;

                    case double n when n >= 0.2 && n < 0.4:
                        ForegroundColor = ConsoleColor.Blue;
                        BackgroundColor = ConsoleColor.DarkBlue;
                        break;

                    case double n when n >= 0.4 && n < 0.6:
                        ForegroundColor = ConsoleColor.Green;
                        BackgroundColor = ConsoleColor.DarkGreen;
                        break;

                    case double n when n >= 0.6 && n < 0.8:
                        ForegroundColor = ConsoleColor.Yellow;
                        BackgroundColor = ConsoleColor.DarkYellow;
                        break;

                    case double n when n >= 0.8 && n < 1:
                        ForegroundColor = ConsoleColor.Red;
                        BackgroundColor = ConsoleColor.DarkRed;
                        break;
                }

                if (i == j)
                {
                    ForegroundColor = ConsoleColor.Gray;
                    BackgroundColor = ConsoleColor.DarkGray;
                }

                Write($"{matrix[i, j],9:F4}");
                ResetColor();
            }
        }
        WriteLine();
        WriteLineColor($"----------------------------------------------------------------------------------");

        if (showLegend)
        {
            WriteLineColor("  Legend:");

            ForegroundColor = ConsoleColor.Gray;
            BackgroundColor = ConsoleColor.Black;
            Write(" ███████");
            ResetColor();
            WriteLine("-0.0 : 0.2");

            ForegroundColor = ConsoleColor.DarkBlue;
            BackgroundColor = ConsoleColor.Black;
            Write(" ███████");
            ResetColor();
            WriteLine(" 0.2 : 0.4");

            ForegroundColor = ConsoleColor.DarkGreen;
            BackgroundColor = ConsoleColor.Black;
            Write(" ███████");
            ResetColor();
            WriteLine(" 0.4 : 0.6");

            ForegroundColor = ConsoleColor.DarkYellow;
            BackgroundColor = ConsoleColor.Black;
            Write(" ███████");
            ResetColor();
            WriteLine(" 0.6 : 0.8");

            ForegroundColor = ConsoleColor.DarkRed;
            BackgroundColor = ConsoleColor.Black;
            Write(" ███████");
            ResetColor();
            WriteLine(" 0.8 : 1.0");
        }
    }
}
