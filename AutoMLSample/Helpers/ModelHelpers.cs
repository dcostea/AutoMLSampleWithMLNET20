namespace AutoMLSample.Helpers;

internal static class ModelHelpers
{
    internal static double CalculateStandardDeviation(IEnumerable<double> values)
    {
        double average = values.Average();
        double sumOfSquaresOfDifferences = values.Select(val => (val - average) * (val - average)).Sum();
        double standardDeviation = Math.Sqrt(sumOfSquaresOfDifferences / (values.Count() - 1));

        return standardDeviation;
    }

    internal static double CalculateConfidenceInterval95(IEnumerable<double> values)
    {
        double confidenceInterval95 = 1.95 * CalculateStandardDeviation(values) / Math.Sqrt(values.Count() - 1);

        return confidenceInterval95;
    }
}
