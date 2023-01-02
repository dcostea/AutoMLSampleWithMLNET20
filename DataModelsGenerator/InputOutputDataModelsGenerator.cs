using Microsoft.CodeAnalysis.Text;
using Microsoft.CodeAnalysis;
using System.Text;
using Microsoft.ML;
using Microsoft.ML.AutoML;
using System.IO;
using Newtonsoft.Json;
using Microsoft.ML.Data;
using DataModels.Models;
using System.Linq;
using System;

namespace DataModelsGenerator;

[Generator]
public class InputOutputDataModelsGenerator : IIncrementalGenerator
{
    const string InputDataModel = nameof(InputDataModel);
    const string OutputDataModel = nameof(OutputDataModel);
    const string GeneratedDataModels = nameof(GeneratedDataModels);

    public void Initialize(IncrementalGeneratorInitializationContext initContext)
    {
        var mlContext = new MLContext(seed: 1);

        var additionalText = initContext.AdditionalTextsProvider.Where(f => Path.GetFileName(f.Path).Equals("automl.json", StringComparison.OrdinalIgnoreCase));
        var content = additionalText.Collect();

        //https://github.com/dotnet/roslyn/blob/main/docs/features/source-generators.cookbook.md#access-analyzer-config-properties
        //https://github.com/JoanComasFdz/dotnet-how-to-debug-source-generator-vs2022

        ////var automlJson = context.AdditionalFiles.FirstOrDefault(f => Path.GetFileName(f.Path).Equals("automl.json", StringComparison.OrdinalIgnoreCase));
        ////if (automlJson is null)
        ////    throw new Exception("The automl.json file is missing from additionalFiles in csproj!");

        // TODO this is buggy
        ////var options = context.AnalyzerConfigOptions.GetOptions(automlJson);
        ////var hasLabel = options.TryGetValue("build_metadata.additionalFiles.Label", out var label);
        ////if (!hasLabel)
        ////    throw new Exception($"The Label property '{label}' is not in this list: {string.Join(", ", options.Keys)} Label property is missing from additionalFiles in csproj!");

        initContext.RegisterSourceOutput(content, (context, content) =>
        {
            // build input data model

            ////var options = initContext.AnalyzerConfigOptionsProvider.Select((f, cancellationToken) => f.GetOptions(content.Single()));
            ////var optionsContent = options.Select((o, ct) => string.Join(",", o.Keys));
            ////string label = string.Empty;
            ////initContext.RegisterSourceOutput(
            ////            options,
            ////            (context, settings) =>
            ////            {
            ////                settings.TryGetValue("Label", out var label0);
            ////                label = label0;
            ////            });

            var trainerSettings = JsonConvert.DeserializeObject<TrainerSettings>(content.Single().GetText().ToString());
            var columnInferenceResults = mlContext.Auto().InferColumns(trainerSettings.CsvDataPath, labelColumnName: trainerSettings.LabelColumn, groupColumns: false);

            var inputModelSource = new StringBuilder();
            inputModelSource.Append("""
// auto-generated readonly code
using System;
using Microsoft.ML;
using Microsoft.ML.Data;

namespace GeneratedDataModels;

public class InputDataModel
{

"""
            );

            foreach (var column in columnInferenceResults.TextLoaderOptions.Columns)
            {
                // if column is in columnMapping, use the type and name in that
                DataKind dataKind;

                if (trainerSettings.ColumnMapping != null && trainerSettings.ColumnMapping.ContainsKey(column.Name))
                {
                    dataKind = trainerSettings.ColumnMapping[column.Name];
                }
                else
                {
                    dataKind = column.DataKind;
                }

                inputModelSource.AppendLine($"    [LoadColumn({column.Source[0].Min})]");
                inputModelSource.AppendLine($"    public {dataKind} {column.Name} {{ get; set; }}");
                inputModelSource.AppendLine();

                //TODO Accomodate VectorType (array) columns
            }

            inputModelSource.Append("}");

            var inputModelSourceText = SourceText.From(inputModelSource.ToString(), Encoding.UTF8);
            context.AddSource($"{InputDataModel}.cs", inputModelSourceText);




            // build output data model

            StringBuilder outputModelSource = new();

            switch (trainerSettings.Scenario)
            {
                case nameof(Scenario.BinaryClassification):
                    outputModelSource.Append("""
        // auto-generated readonly code
        using System;
        using Microsoft.ML;
        using Microsoft.ML.Data;

        namespace GeneratedDataModels;

        public class OutputDataModel
        {
            [ColumnName("PredictedLabel")]
            public string PredictedLabel { get; set; }

            //TODO float Score for multiclass
            [ColumnName("Score")]
            public float Score { get; set; }

            [ColumnName("Score")]
            public float Probability { get; set; }

            [ColumnName("Features")]
            public float[] Features { get; set; }
        }
        """
                    );
                    break;

                case nameof(Scenario.MulticlassClassification):
                    outputModelSource.Append("""
        // auto-generated readonly code
        using System;
        using Microsoft.ML;
        using Microsoft.ML.Data;

        namespace GeneratedDataModels;

        public class OutputDataModel
        {
            [ColumnName("PredictedLabel")]
            public string PredictedLabel { get; set; }

            //TODO float Score for multiclass
            [ColumnName("Score")]
            public float[] Score { get; set; }

            [ColumnName("Score")]
            public float Probability { get; set; }
        }
        """
                    );
                    break;

                case nameof(Scenario.Regression):
                    outputModelSource.Append("""
        // auto-generated readonly code
        using System;
        using Microsoft.ML;
        using Microsoft.ML.Data;

        namespace GeneratedDataModels;

        public class OutputDataModel
        {
            //TODO float Score for multiclass
            [ColumnName("Score")]
            public float[] Score { get; set; }

            [ColumnName("Score")]
            public float Probability { get; set; }

            [ColumnName("Features")]
            public float[] Features { get; set; }
        }
        """
                    );
                    break;
            }

            var outputModelSourceText = SourceText.From(outputModelSource.ToString(), Encoding.UTF8);
            context.AddSource($"{OutputDataModel}.cs", outputModelSourceText);

        });
    }
}