param(
    [Parameter(Mandatory = $true)]
    [string]$RequestCsv,

    [Parameter(Mandatory = $true)]
    [string]$OutputCsv,

    [string]$ModelDirectory = "external\i2em_reference",

    [switch]$Force
)

$ErrorActionPreference = "Stop"

function Convert-ToMatlabPath {
    param([Parameter(Mandatory = $true)][string]$Path)
    return $Path.Replace("\", "/").Replace("'", "''")
}

$projectRoot = Split-Path -Parent $PSScriptRoot
$requestPath = [System.IO.Path]::GetFullPath((Join-Path $projectRoot $RequestCsv))
$modelPath = [System.IO.Path]::GetFullPath((Join-Path $projectRoot $ModelDirectory))
$outputPath = [System.IO.Path]::GetFullPath((Join-Path $projectRoot $OutputCsv))

if (-not (Test-Path -LiteralPath $requestPath -PathType Leaf)) {
    throw "Request CSV does not exist: $requestPath"
}
if (-not (Test-Path -LiteralPath $modelPath -PathType Container)) {
    throw "I2EM model directory does not exist: $modelPath"
}
$modelFile = Join-Path $modelPath "I2EM_Backscatter_model.m"
if (-not (Test-Path -LiteralPath $modelFile -PathType Leaf)) {
    throw "Official I2EM_Backscatter_model.m is missing: $modelFile"
}
if ((Test-Path -LiteralPath $outputPath) -and -not $Force) {
    throw "Output already exists. Choose another path or pass -Force: $outputPath"
}
$outputDirectory = Split-Path -Parent $outputPath
New-Item -ItemType Directory -Path $outputDirectory -Force | Out-Null

$adapterPath = Convert-ToMatlabPath $PSScriptRoot
$requestForMatlab = Convert-ToMatlabPath $requestPath
$modelForMatlab = Convert-ToMatlabPath $modelPath
$outputForMatlab = Convert-ToMatlabPath $outputPath
$command = (
    "addpath('$adapterPath'); " +
    "run_i2em_requests_matlab('$requestForMatlab','$modelForMatlab','$outputForMatlab');"
)

$matlab = $null
try {
    $matlab = New-Object -ComObject Matlab.Application.Single
    $response = $matlab.Execute($command)
    if ($response) {
        Write-Output $response
    }
}
finally {
    if ($null -ne $matlab) {
        $matlab.Quit()
        [void][System.Runtime.InteropServices.Marshal]::FinalReleaseComObject($matlab)
    }
}

if (-not (Test-Path -LiteralPath $outputPath -PathType Leaf)) {
    throw "MATLAB did not create the expected result CSV: $outputPath"
}

Write-Output "I2EM results saved: $outputPath"
