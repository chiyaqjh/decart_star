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

$items = @(
    @{
        Name = 'our_decart q1 10x10'
        Runner = 'experiments\our_decart\runner.py'
        ResultsDir = 'experiments\results\data_new\our_decart\decision_tree\q=1'
        NumRecords = 10
        RecordDim = 10
        NumQueriers = 1
    },
    @{
        Name = 'our_decart_star q1 10x10'
        Runner = 'experiments\our_decart_star\runner.py'
        ResultsDir = 'experiments\results\data_new\our_decart_star\decision_tree\q=1'
        NumRecords = 10
        RecordDim = 10
        NumQueriers = 1
    },
    @{
        Name = 'our_decart q1 100x100'
        Runner = 'experiments\our_decart\runner.py'
        ResultsDir = 'experiments\results\data_new\our_decart\decision_tree\q=1'
        NumRecords = 100
        RecordDim = 100
        NumQueriers = 1
    },
    @{
        Name = 'our_decart_star q1 100x100'
        Runner = 'experiments\our_decart_star\runner.py'
        ResultsDir = 'experiments\results\data_new\our_decart_star\decision_tree\q=1'
        NumRecords = 100
        RecordDim = 100
        NumQueriers = 1
    },
    @{
        Name = 'our_decart q1 500x500'
        Runner = 'experiments\our_decart\runner.py'
        ResultsDir = 'experiments\results\data_new\our_decart\decision_tree\q=1'
        NumRecords = 500
        RecordDim = 500
        NumQueriers = 1
    },
    @{
        Name = 'our_decart_star q1 500x500'
        Runner = 'experiments\our_decart_star\runner.py'
        ResultsDir = 'experiments\results\data_new\our_decart_star\decision_tree\q=1'
        NumRecords = 500
        RecordDim = 500
        NumQueriers = 1
    },
    @{
        Name = 'our_decart q1 1000x1000'
        Runner = 'experiments\our_decart\runner.py'
        ResultsDir = 'experiments\results\data_new\our_decart\decision_tree\q=1'
        NumRecords = 1000
        RecordDim = 1000
        NumQueriers = 1
    },
    @{
        Name = 'our_decart_star q1 1000x1000'
        Runner = 'experiments\our_decart_star\runner.py'
        ResultsDir = 'experiments\results\data_new\our_decart_star\decision_tree\q=1'
        NumRecords = 1000
        RecordDim = 1000
        NumQueriers = 1
    },
    @{
        Name = 'our_decart q1 5000x5000'
        Runner = 'experiments\our_decart\runner.py'
        ResultsDir = 'experiments\results\data_new\our_decart\decision_tree\q=1'
        NumRecords = 5000
        RecordDim = 5000
        NumQueriers = 1
    },
    @{
        Name = 'our_decart_star q1 5000x5000'
        Runner = 'experiments\our_decart_star\runner.py'
        ResultsDir = 'experiments\results\data_new\our_decart_star\decision_tree\q=1'
        NumRecords = 5000
        RecordDim = 5000
        NumQueriers = 1
    },
    @{
        Name = 'our_decart q1 10000x10000'
        Runner = 'experiments\our_decart\runner.py'
        ResultsDir = 'experiments\results\data_new\our_decart\decision_tree\q=1'
        NumRecords = 10000
        RecordDim = 10000
        NumQueriers = 1
    },
    @{
        Name = 'our_decart_star q1 10000x10000'
        Runner = 'experiments\our_decart_star\runner.py'
        ResultsDir = 'experiments\results\data_new\our_decart_star\decision_tree\q=1'
        NumRecords = 10000
        RecordDim = 10000
        NumQueriers = 1
    }
)

Set-Location $projectRoot

foreach ($item in $items) {
    $resultsDir = Join-Path $projectRoot $item.ResultsDir
    New-Item -ItemType Directory -Force -Path $resultsDir | Out-Null

    $logName = ('{0}_{1}x{2}.log' -f ($item.Runner -replace '[\\/\.]+', '_'), $item.NumRecords, $item.RecordDim)
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

Write-Host 'All queued decision_tree q=1 runs finished.'