param(
    [string[]]$Schemes = @('our_decart', 'our_decart_star'),
    [int[]]$Sizes = @(10, 100, 500, 1000, 5000, 10000)
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

$schemeMap = @{
    'our_decart' = @{
        Runner = 'experiments\our_decart\runner.py'
        ResultsDir = 'experiments\results\data_new\our_decart\decision_tree\q=100'
    }
    'our_decart_star' = @{
        Runner = 'experiments\our_decart_star\runner.py'
        ResultsDir = 'experiments\results\data_new\our_decart_star\decision_tree\q=100'
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

        $logName = ('{0}_q100_{1}x{1}.log' -f ($runner -replace '[\\/\.]+' , '_'), $size)
        $logPath = Join-Path $logRoot $logName
        $doneMarker = 'DONE {0} q100 {1}x{1}' -f $schemeName, $size

        if (Test-RunCompleted -LogPath $logPath -DoneMarker $doneMarker) {
            Write-Host ('[SKIP] {0} q100 {1}x{1} already completed.' -f $schemeName, $size)
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
            '--model-types', 'decision_tree'
        )

        $header = '[{0}] START {1} q100 {2}x{2}' -f [DateTime]::Now.ToString('yyyy-MM-dd HH:mm:ss'), $schemeName, $size
        $header | Out-File -FilePath $logPath -Append
        Write-Host $header

        & $pythonExe @argList *>&1 | ForEach-Object {
            $line = $_.ToString()
            $line | Out-File -FilePath $logPath -Append
            Write-Host $line
        }

        if ($LASTEXITCODE -ne 0) {
            $message = '[{0}] FAIL {1} q100 {2}x{2} exit={3}' -f [DateTime]::Now.ToString('yyyy-MM-dd HH:mm:ss'), $schemeName, $size, $LASTEXITCODE
            $message | Out-File -FilePath $logPath -Append
            throw $message
        }

        $footer = '[{0}] DONE {1} q100 {2}x{2}' -f [DateTime]::Now.ToString('yyyy-MM-dd HH:mm:ss'), $schemeName, $size
        $footer | Out-File -FilePath $logPath -Append
        Write-Host $footer
    }
}

Write-Host 'Queued q=100 reruns finished.'