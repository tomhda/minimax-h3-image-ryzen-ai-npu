param(
    [string]$Out,
    [switch]$Verify,
    [string]$Config = (Join-Path $PSScriptRoot 'configs\local.json')
)
$ErrorActionPreference = 'Stop'
$Config = (Resolve-Path -LiteralPath $Config).Path
$local = Get-Content -LiteralPath $Config -Raw | ConvertFrom-Json
$python = [IO.Path]::GetFullPath($local.text_python, (Split-Path $Config))
$prefix = Split-Path $python
$original = $env:Path
$options = @('--config', $Config)
if ($Out) { $options += @('--out', $ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($Out)) }
if ($Verify) { $options += '--verify' }
try {
    $env:Path = "$prefix;$prefix\Library\bin;$prefix\Scripts;$original"
    & $python -B -u (Join-Path $PSScriptRoot 'tools\convert\convert_all.py') @options
    $code = $LASTEXITCODE
} finally {
    $env:Path = $original
}
exit $code
