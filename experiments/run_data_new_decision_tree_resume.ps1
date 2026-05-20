param()

$ErrorActionPreference = 'Continue'
Set-Location "E:\decart"

. .\venv\Scripts\Activate.ps1
$env:PYTHONIOENCODING = 'utf-8'

$timestamp = Get-Date -Format yyyyMMdd_HHmmss
$rootOut = "experiments\results\data_new"
$logPath = Join-Path $rootOut "run_data_new_decision_tree_resume_$timestamp.log"
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

function Invoke-Experiment {
    param(
        [hashtable]$Method,
        [int]$Size,
        [int]$Queriers,
        [ref]$SkipMethod
    )

    if ($SkipMethod.Value) {
        return
    }

    Write-Host "RUN $($Method.name) size=${Size}x${Size} queriers=$Queriers"
    python $Method.runner --N 10000 --n 32 --num-records $Size --record-dim $Size --policy-size 32 --num-queriers $Queriers --num-runs 1 --model-types decision_tree --results-dir $Method.folder
    if ($LASTEXITCODE -ne 0) {
        Write-Host "FAILED $($Method.name) size=${Size} q=$Queriers exit=$LASTEXITCODE"
        $SkipMethod.Value = $true
    } else {
        Write-Host "DONE $($Method.name) size=${Size} q=$Queriers"
    }
}

foreach ($method in $methods) {
    New-Item -ItemType Directory -Force -Path $method.folder | Out-Null
    $skipMethod = $false

    Write-Host "========== METHOD $($method.name) =========="

    if ($method.name -eq 'secpq') {
        Invoke-Experiment -Method $method -Size 5000 -Queriers 100 -SkipMethod ([ref]$skipMethod)
    } else {
        foreach ($size in $part1Sizes) {
            Invoke-Experiment -Method $method -Size $size -Queriers 1 -SkipMethod ([ref]$skipMethod)
            if ($skipMethod) { break }
        }

        foreach ($size in $part2Sizes) {
            if ($skipMethod) { break }
            foreach ($q in $part2Queries) {
                Invoke-Experiment -Method $method -Size $size -Queriers $q -SkipMethod ([ref]$skipMethod)
                if ($skipMethod) { break }
            }
        }
    }

    if ($skipMethod) {
        Write-Host "SKIP REST OF METHOD $($method.name) DUE TO FAILURE"
    }
}

Stop-Transcript | Out-Null
Write-Host "RESUME FINISHED. Log: $logPath"