// Libraries
import Foundation
import CreateML

// Getting Data and Create DataTable
let csvFile = URL(contentsOf: "CSVPath")
let dataTable = try MLDataTable(contentsOf: csvFile)

// Separate Relevant Columns
let columns = ["Height" , "Weight"]
let SpecialTable = dataTable[columns]

// Divide MLDataTable as Training and Testing Tables
let (testData, trainData) = SpecialTable.randomSplit(by: 0.20)

// Training the Model - Target Value is Height
let regressor = try MLLinearRegressor(trainingData: trainData,
                                      targetColumn: "Height")

// Start test process with testData
let testReg = regressor.evaluation(on: testData)

// Your Accuracy Values
let TestErorr = testReg.maximumError
let TrainErorr = regressor.trainingMetrics.maximumError

// Save Model
let regData = MLModelMetadata(author: "Oguz Kayra",
                              shortDescription: "Height vs Weight",
                              version: "1.0")

try regressor.write(to: desktopPath.appendingPathComponent("name.mlmodel"),
                    metadata: regressorMetadata)
