using Microsoft.ML.Data;
using System.Collections.Generic;

namespace DataModels.Models;

public class TrainerSettings
{
    public string Scenario { get; set; }
    public uint TrainingTime { get; set; }
    public string LabelColumn { get; set; }
    public string CsvDataPath { get; set; }
    public IDictionary<string, DataKind> ColumnMapping { get; set; }
}
