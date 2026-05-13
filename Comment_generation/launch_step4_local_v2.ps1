param(
    [ValidateSet('douyin', 'youtube', 'both')]
    [string]$Platform = 'both',
    [int]$MaxExtractWorkers = 3,
    [double]$SampleSeconds = 3.0,
    [switch]$Reverse,
    [switch]$RetryFailed,
    [int]$Limit = 0,
    [string[]]$Ids,
    [string]$ScratchRoot = '',
    [string]$PythonExe = 'C:\Users\bobzhou\anaconda3\envs\gcg_douyin311\python.exe',
    [string]$RepoRoot = 'E:\pn\new_GCG-main'
)

$ErrorActionPreference = 'Stop'

$runnerPath = Join-Path $PSScriptRoot 'step4_local_v2.py'
$logDir = Join-Path $RepoRoot 'data_pre\logs\step4_local_v2'
New-Item -ItemType Directory -Force -Path $logDir | Out-Null

$timestamp = Get-Date -Format 'yyyyMMdd_HHmmss'
$stdoutLog = Join-Path $logDir "step4_local_v2_$timestamp.out.log"
$stderrLog = Join-Path $logDir "step4_local_v2_$timestamp.err.log"
$workerCmd = Join-Path $logDir "step4_local_v2_worker_$timestamp.cmd"

$argsList = @(
    '-u',
    ('"' + $runnerPath + '"'),
    '--platform', $Platform,
    '--max-extract-workers', [string]$MaxExtractWorkers,
    '--sample-seconds', [string]$SampleSeconds
)
if ($Reverse.IsPresent) { $argsList += '--reverse' }
if ($RetryFailed.IsPresent) { $argsList += '--retry-failed' }
if ($Limit -gt 0) { $argsList += @('--limit', [string]$Limit) }
if ($Ids -and $Ids.Count -gt 0) { $argsList += '--ids'; $argsList += $Ids }
if ($ScratchRoot) { $argsList += @('--scratch-root', ('"' + $ScratchRoot + '"')) }

$pythonCommand = ('"' + $PythonExe + '" ' + ($argsList -join ' '))
$workerText = "@echo off`r`n" +
    "setlocal`r`n" +
    "set ""PATH=%Path%""`r`n" +
    "set ""HF_HUB_DISABLE_SYMLINKS_WARNING=1""`r`n" +
    "set ""HF_HUB_DISABLE_TELEMETRY=1""`r`n" +
    "cd /d ""$RepoRoot""`r`n" +
    "$pythonCommand 1>>""$stdoutLog"" 2>>""$stderrLog""`r`n"

Set-Content -Path $workerCmd -Value $workerText -Encoding ASCII -NoNewline

$psi = New-Object System.Diagnostics.ProcessStartInfo
$psi.FileName = 'C:\Windows\System32\cmd.exe'
$psi.Arguments = '/d /c ""' + $workerCmd + '""'
$psi.WorkingDirectory = $RepoRoot
$psi.UseShellExecute = $false
$psi.CreateNoWindow = $true

$process = [System.Diagnostics.Process]::Start($psi)

[pscustomobject]@{
    ProcessId = $process.Id
    Platform = $Platform
    Reverse = $Reverse.IsPresent
    RetryFailed = $RetryFailed.IsPresent
    Limit = $Limit
    ScratchRoot = $ScratchRoot
    StdoutLog = $stdoutLog
    StderrLog = $stderrLog
    WorkerCmd = $workerCmd
} | ConvertTo-Json -Depth 3
