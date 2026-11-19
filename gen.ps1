param(
    [Parameter(Mandatory)][string]$Items,
    [Parameter(Mandatory)][string]$Out,
    [ValidateSet('cpu','npu')][string]$Vae = 'npu',
    [ValidateSet('auto','off')][string]$Affinity = 'auto',
    [ValidateSet('3','1','off')][string]$DitRestart = '3',
    [ValidateRange(1,256)][int]$TextThreads = 4,
    [ValidateRange(1,256)][int]$ReadWorkers = 8,
    [ValidateRange(1,256)][int]$DitThreads = 2,
    [ValidateRange(1,256)][int]$VaeThreads = 2,
    [string]$Config = (Join-Path $PSScriptRoot 'configs\local.json')
)
$ErrorActionPreference = 'Stop'
try {
    $Config = (Resolve-Path -LiteralPath $Config).Path
    $local = Get-Content -LiteralPath $Config -Raw | ConvertFrom-Json
    $python = [IO.Path]::GetFullPath($local.text_python, (Split-Path $Config))
    if (-not (Test-Path -LiteralPath $python -PathType Leaf)) { throw "Text Python is missing: $python" }
} catch {
    Write-Host "Input error: $_. Run setup.ps1 first."
    exit 2
}
$prefix = Split-Path $python
$env:Path = "$prefix;$prefix\Library\bin;$prefix\Scripts;$env:Path"
$env:PYTHONUTF8 = '1'
$options = @('--items', $Items, '--out', $Out, '--config', $Config, '--vae', $Vae,
    '--affinity', $Affinity, '--dit-restart', $DitRestart, '--text-threads', $TextThreads,
    '--read-workers', $ReadWorkers, '--dit-threads', $DitThreads, '--vae-threads', $VaeThreads)
& $python -B -u (Join-Path $PSScriptRoot 'src\generate.py') @options
exit $LASTEXITCODE

