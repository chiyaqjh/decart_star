param()

$ErrorActionPreference = 'Continue'
Set-Location "E:\decart"

. .\venv\Scripts\Activate.ps1
$env:PYTHONIOENCODING = 'utf-8'

$timestamp = Get-Date -Format yyyyMMdd_HHmmss
$rootOut = "experiments\results\data_new"
$logPath = Join-Path $rootOut "run_data_new_decision_tree_$timestamp.log"
New-Item -ItemType Directory -Force -Path $rootOut | Out-Null
Start-Transcript -Path $logPath -Force | Out-Null
Write-Host "Log: $logPath"

$part1Sizes = @(10, 100, 500, 1000, 5000, 10000)
$part2Sizes = @(10, 100, 1000, 5000)
$part2Queries = @(10, 50, 100)

$methods = @(
    @{ name='secpq'; folder='experiments\results\data_new\secpq'; runner='experiments\secpq\runner.py' },
    @{ name='naive_ccs23'; folder='experiments\results\data_new\naive_ccs23'; runner='experiments\naive_ccs23\runner.py' },
    @{ name='our_decart'; folder='experiments\results\data_new\our_decart'; runner='experiments\our_decart\runner.py' },
    @{ name='our_decart_star'; folder='experiments\results\data_new\our_decart_star'; runner='experiments\our_decart_star\runner.py' },
    @{ name='scheme1_ccs23'; folder='experiments\results\data_new\scheme1_ccs23'; runner='experiments\scheme1_ccs23\runner.py' },
    @{ name='scheme2_server'; folder='experiments\results\data_new\scheme2_server'; runner='experiments\scheme2_server\runner.py' },
    @{ name='scheme3_offline'; folder='experiments\results\data_new\scheme3_offline'; runner='experiments\scheme3_offline\runner.py' }
)

foreach ($method in $methods) {
    New-Item -ItemType Directory -Force -Path $method.folder | Out-Null
    $skipMethod = $false

    Write-Host "========== METHOD $($method.name) =========="

    foreach ($size in $part1Sizes) {
        if ($skipMethod) { break }
        Write-Host "RUN $($method.name) size=${size}x${size} queriers=1"
        python $method.runner --N 10000 --n 32 --num-records $size --record-dim $size --policy-size 32 --num-queriers 1 --num-runs 1 --model-types decision_tree --results-dir $method.folder
        if ($LASTEXITCODE -ne 0) {
            Write-Host "FAILED $($method.name) size=${size} q=1 exit=$LASTEXITCODE"
            $skipMethod = $true
        } else {
            Write-Host "DONE $($method.name) size=${size} q=1"
        }
    }

    foreach ($size in $part2Sizes) {
        if ($skipMethod) { break }
        foreach ($q in $part2Queries) {
            if ($skipMethod) { break }
            Write-Host "RUN $($method.name) size=${size}x${size} queriers=$q"
            python $method.runner --N 10000 --n 32 --num-records $size --record-dim $size --policy-size 32 --num-queriers $q --num-runs 1 --model-types decision_tree --results-dir $method.folder
            if ($LASTEXITCODE -ne 0) {
                Write-Host "FAILED $($method.name) size=${size} q=$q exit=$LASTEXITCODE"
                $skipMethod = $true
            } else {
                Write-Host "DONE $($method.name) size=${size} q=$q"
            }
        }
    }

    if ($skipMethod) {
        Write-Host "SKIP REST OF METHOD $($method.name) DUE TO FAILURE"
    }
}

Stop-Transcript | Out-Null
Write-Host "ALL METHODS FINISHED. Log: $logPath"
