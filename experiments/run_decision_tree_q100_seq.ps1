param(
    [switch]$DryRun
)

$ErrorActionPreference = 'Stop'
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new($false)
$OutputEncoding = [System.Text.UTF8Encoding]::new($false)
$PSDefaultParameterValues['Out-File:Encoding'] = 'utf8'
$env:PYTHONIOENCODING = 'utf-8'
$env:PYTHONUTF8 = '1'

$projectRoot = 'E:\decart'
$pythonExe = Join-Path $projectRoot 'venv\Scripts\python.exe'
$logRoot = Join-Path $projectRoot 'experiments\results\data_new\run_logs'

New-Item -ItemType Directory -Force -Path $logRoot | Out-Null

$sizes = @(10, 100, 500, 1000, 5000, 10000)
$schemes = @(
    @{
        Name = 'our_decart'
        Runner = 'experiments\our_decart\runner.py'
        ResultsDir = 'experiments\results\data_new\our_decart\decision_tree\q=100'
    },
    @{
        Name = 'our_decart_star'
        Runner = 'experiments\our_decart_star\runner.py'
        ResultsDir = 'experiments\results\data_new\our_decart_star\decision_tree\q=100'
    },
    @{
        Name = 'naive_ccs23'
        Runner = 'experiments\naive_ccs23\runner.py'
        ResultsDir = 'experiments\results\data_new\naive_ccs23\decision_tree\q=100'
    },
    @{
        Name = 'scheme2_server'
        Runner = 'experiments\scheme2_server\runner.py'
        ResultsDir = 'experiments\results\data_new\scheme2_server\decision_tree\q=100'
    },
    @{
        Name = 'scheme3_offline'
        Runner = 'experiments\scheme3_offline\runner.py'
        ResultsDir = 'experiments\results\data_new\scheme3_offline\decision_tree\q=100'
    },
    @{
        Name = 'secpq'
        Runner = 'experiments\secpq\runner.py'
        ResultsDir = 'experiments\results\data_new\secpq\decision_tree\q=100'
    }
)

$items = @()
foreach ($size in $sizes) {
    foreach ($scheme in $schemes) {
        $items += @{
            Name = ('{0} q100 {1}x{1}' -f $scheme.Name, $size)
            Runner = $scheme.Runner
            ResultsDir = $scheme.ResultsDir
            NumRecords = $size
            RecordDim = $size
            NumQueriers = 100
        }
    }
}

Set-Location $projectRoot

foreach ($item in $items) {
    $resultsDir = Join-Path $projectRoot $item.ResultsDir
    New-Item -ItemType Directory -Force -Path $resultsDir | Out-Null

    $logName = ('{0}_q100_{1}x{2}.log' -f ($item.Runner -replace '[\\/\.]+', '_'), $item.NumRecords, $item.RecordDim)
    $logPath = Join-Path $logRoot $logName

    $argList = @(
        $item.Runner,
        '--N', '10000',
        '--n', '32',
        '--num-records', [string]$item.NumRecords,
        '--record-dim', [string]$item.RecordDim,
        '--dataset', 'synthetic',
        '--mnist-data-dir', 'data',
        '--model-source', 'synthetic',
        '--trained-models-dir', 'experiments/models/trained',
        '--policy-size', '32',
        '--num-queriers', [string]$item.NumQueriers,
        '--num-runs', '1',
        '--results-dir', $item.ResultsDir,
        '--model-types', 'decision_tree'
    )

    $header = "[$([DateTime]::Now.ToString('yyyy-MM-dd HH:mm:ss'))] START $($item.Name)"
    $header | Out-File -FilePath $logPath -Append
    Write-Host $header

    if ($DryRun) {
        (("DRY RUN: {0} {1}" -f $pythonExe, ($argList -join ' '))) | Out-File -FilePath $logPath -Append
        Write-Host ("DRY RUN: {0} {1}" -f $pythonExe, ($argList -join ' '))
        continue
    }

    & $pythonExe @argList *>&1 | ForEach-Object {
        $line = $_.ToString()
        $line | Out-File -FilePath $logPath -Append
        Write-Host $line
    }

    if ($LASTEXITCODE -ne 0) {
        $message = "[$([DateTime]::Now.ToString('yyyy-MM-dd HH:mm:ss'))] FAIL $($item.Name) exit=$LASTEXITCODE"
        $message | Out-File -FilePath $logPath -Append
        Write-Error $message
    }

    $footer = "[$([DateTime]::Now.ToString('yyyy-MM-dd HH:mm:ss'))] DONE $($item.Name)"
    $footer | Out-File -FilePath $logPath -Append
    Write-Host $footer
}

Write-Host 'All queued decision_tree q=100 runs finished.'