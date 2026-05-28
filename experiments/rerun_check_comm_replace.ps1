$ErrorActionPreference = 'Stop'

$resultsRoot = 'experiments\results\data_new'
$jobs = @(
    @{ size = 10; q = 1 },
    @{ size = 100; q = 1 },
    @{ size = 500; q = 1 },
    @{ size = 1000; q = 1 },
    @{ size = 5000; q = 1 },
    @{ size = 10000; q = 1 },
    @{ size = 10; q = 10 },
    @{ size = 100; q = 10 },
    @{ size = 1000; q = 10 },
    @{ size = 5000; q = 10 },
    @{ size = 10; q = 50 },
    @{ size = 100; q = 50 },
    @{ size = 1000; q = 50 },
    @{ size = 5000; q = 50 },
    @{ size = 10; q = 100 },
    @{ size = 100; q = 100 },
    @{ size = 1000; q = 100 },
    @{ size = 5000; q = 100 }
)

$methods = @(
    @{ name = 'secpq'; runner = 'experiments\secpq\runner.py'; folder = 'secpq' },
    @{ name = 'naive_ccs23'; runner = 'experiments\naive_ccs23\runner.py'; folder = 'naive_ccs23' },
    @{ name = 'our_decart'; runner = 'experiments\our_decart\runner.py'; folder = 'our_decart' },
    @{ name = 'our_decart_star'; runner = 'experiments\our_decart_star\runner.py'; folder = 'our_decart_star' }
)

function Get-MatchingFiles([string]$resultsDir, [int]$size, [int]$q) {
    if (-not (Test-Path $resultsDir)) {
        return @()
    }

    $matchedFiles = @()
    foreach ($file in Get-ChildItem -Path $resultsDir -Filter *.json -File -ErrorAction SilentlyContinue) {
        try {
            $raw = Get-Content $file.FullName -Raw
            $matchesConfig = (
                $raw -match '"N"\s*:\s*10000' -and
                $raw -match '"n"\s*:\s*32' -and
                $raw -match ('"num_records"\s*:\s*' + $size) -and
                $raw -match ('"record_dim"\s*:\s*' + $size) -and
                $raw -match '"policy_size"\s*:\s*32' -and
                (($raw -notmatch '"num_queriers"\s*:') -or ($raw -match ('"num_queriers"\s*:\s*' + $q)))
            )
            if ($matchesConfig) {
                $hasCheckMetric = $raw -match '"comm_check_sizes"\s*:\s*\[\s*[^\]]+'
                $matchedFiles += [PSCustomObject]@{
                    File = $file.FullName
                    Name = $file.Name
                    LastWriteTime = $file.LastWriteTimeUtc
                    HasCheckMetric = $hasCheckMetric
                }
            }
        }
        catch {
        }
    }
    return $matchedFiles
}

foreach ($method in $methods) {
    $resultsDir = Join-Path $resultsRoot $method.folder
    Write-Host "=== METHOD $($method.name) ==="

    foreach ($job in $jobs) {
        $size = $job.size
        $q = $job.q
        Write-Host "RUN $($method.name) size=${size}x${size} q=$q"

        $before = @(Get-MatchingFiles -resultsDir $resultsDir -size $size -q $q)

        python $method.runner --N 10000 --n 32 --num-records $size --record-dim $size --policy-size 32 --num-queriers $q --num-runs 1 --model-types decision_tree --results-dir $resultsDir
        if ($LASTEXITCODE -ne 0) {
            Write-Host "FAILED $($method.name) size=${size} q=$q exit=$LASTEXITCODE"
            break
        }

        $after = @(Get-MatchingFiles -resultsDir $resultsDir -size $size -q $q)
        $valid = @($after | Where-Object { $_.HasCheckMetric })
        if ($valid.Count -eq 0) {
            Write-Host "NO_VALID_RESULT $($method.name) size=${size} q=$q"
            break
        }

        $latest = $valid | Sort-Object LastWriteTime -Descending | Select-Object -First 1
        foreach ($item in $before) {
            if ($item.File -ne $latest.File -and (Test-Path $item.File)) {
                Remove-Item -LiteralPath $item.File -Force -ErrorAction SilentlyContinue
                Write-Host "DELETED_OLD $($item.Name)"
            }
        }

        Write-Host "DONE $($method.name) size=${size} q=$q -> $($latest.Name)"
    }
}