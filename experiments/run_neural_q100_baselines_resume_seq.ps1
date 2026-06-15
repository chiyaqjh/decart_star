param(
    [string[]]$Schemes = @('scheme1_ccs23', 'scheme2_server', 'scheme3_offline'),
    [int[]]$Sizes = @(5000, 1000),
    [switch]$Force
)

$ErrorActionPreference = 'Stop'
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new($false)
$OutputEncoding = [System.Text.UTF8Encoding]::new($false)
$PSDefaultParameterValues['Out-File:Encoding'] = 'utf8'
$env:PYTHONIOENCODING = 'utf-8'
$env:PYTHONUTF8 = '1'
$env:PYTHONUNBUFFERED = '1'

$projectRoot = 'E:\decart'
$pythonExe = Join-Path $projectRoot 'venv\Scripts\python.exe'
$logRoot = Join-Path $projectRoot 'experiments\results\data_new\run_logs'

New-Item -ItemType Directory -Force -Path $logRoot | Out-Null

$schemeMap = @{
    'scheme1_ccs23' = @{
        Runner = 'experiments\scheme1_ccs23\runner.py'
        ResultsDir = 'experiments\results\data_new\scheme1_ccs23\neural_network_N10000_n32\q=100'
    }
    'scheme2_server' = @{
        Runner = 'experiments\scheme2_server\runner.py'
        ResultsDir = 'experiments\results\data_new\scheme2_server\neural_network_N10000_n32\q=100'
    }
    'scheme3_offline' = @{
        Runner = 'experiments\scheme3_offline\runner.py'
        ResultsDir = 'experiments\results\data_new\scheme3_offline\neural_network_N10000_n32\q=100'
    }
}

function Test-RunCompleted {
    param(
        [string]$LogPath,
        [string]$DoneMarker
    )

    if (-not (Test-Path $LogPath)) {
        return $false
    }

    return (Get-Content $LogPath | Select-String -SimpleMatch $DoneMarker -Quiet)
}

function Write-LogLine {
    param(
        [string]$LogPath,
        [string]$Text
    )

    $parent = Split-Path -Parent $LogPath
    if (-not (Test-Path $parent)) {
        New-Item -ItemType Directory -Force -Path $parent | Out-Null
    }

    [System.IO.File]::AppendAllText(
        $LogPath,
        $Text + [Environment]::NewLine,
        [System.Text.UTF8Encoding]::new($false)
    )
}

Set-Location $projectRoot

foreach ($schemeName in $Schemes) {
    if (-not $schemeMap.ContainsKey($schemeName)) {
        throw "Unknown scheme: $schemeName"
    }

    $scheme = $schemeMap[$schemeName]

    foreach ($size in $Sizes) {
        $resultsDir = $scheme.ResultsDir
        $runner = $scheme.Runner
        $resultsPath = Join-Path $projectRoot $resultsDir
        New-Item -ItemType Directory -Force -Path $resultsPath | Out-Null

        $logName = ('{0}_neural_q100_{1}x{1}.log' -f ($runner -replace '[\\/\.]+' , '_'), $size)
        $logPath = Join-Path $logRoot $logName
        $doneMarker = 'DONE {0} neural_network q100 {1}x{1}' -f $schemeName, $size

        if ((-not $Force) -and (Test-RunCompleted -LogPath $logPath -DoneMarker $doneMarker)) {
            Write-Host ('[SKIP] {0} neural_network q100 {1}x{1} already completed.' -f $schemeName, $size)
            continue
        }

        $argList = @(
            $runner,
            '--N', '10000',
            '--n', '32',
            '--num-records', [string]$size,
            '--record-dim', [string]$size,
            '--dataset', 'synthetic',
            '--mnist-data-dir', 'data',
            '--model-source', 'synthetic',
            '--trained-models-dir', 'experiments/models/trained',
            '--policy-size', '32',
            '--num-queriers', '100',
            '--num-runs', '1',
            '--results-dir', $resultsDir,
            '--model-types', 'neural_network'
        )

        $header = '[{0}] START {1} neural_network q100 {2}x{2}' -f [DateTime]::Now.ToString('yyyy-MM-dd HH:mm:ss'), $schemeName, $size
        Write-LogLine -LogPath $logPath -Text $header
        Write-Host $header

        & $pythonExe -u @argList *>&1 | ForEach-Object {
            $line = $_.ToString()
            Write-LogLine -LogPath $logPath -Text $line
            Write-Host $line
        }

        if ($LASTEXITCODE -ne 0) {
            $message = '[{0}] FAIL {1} neural_network q100 {2}x{2} exit={3}' -f [DateTime]::Now.ToString('yyyy-MM-dd HH:mm:ss'), $schemeName, $size, $LASTEXITCODE
            Write-LogLine -LogPath $logPath -Text $message
            throw $message
        }

        $footer = '[{0}] DONE {1} neural_network q100 {2}x{2}' -f [DateTime]::Now.ToString('yyyy-MM-dd HH:mm:ss'), $schemeName, $size
        Write-LogLine -LogPath $logPath -Text $footer
        Write-Host $footer
    }
}

Write-Host 'Queued baseline neural_network q=100 reruns finished.'