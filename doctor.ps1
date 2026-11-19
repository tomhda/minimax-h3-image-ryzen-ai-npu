param(
    [string]$Config = (Join-Path $PSScriptRoot 'configs\local.json'),
    [string]$Out = (Join-Path $PSScriptRoot 'outputs\doctor')
)
$ErrorActionPreference = 'Stop'
$PSNativeCommandUseErrorActionPreference = $true
$Config = (Resolve-Path -LiteralPath $Config).Path
$local = Get-Content -LiteralPath $Config -Raw | ConvertFrom-Json
Get-CimInstance Win32_Processor | Select-Object Name,NumberOfCores,NumberOfLogicalProcessors | Format-List
Get-CimInstance Win32_OperatingSystem | Select-Object Caption,BuildNumber,TotalVisibleMemorySize,FreePhysicalMemory | Format-List
Get-CimInstance Win32_PnPSignedDriver | Where-Object DeviceName -Match 'NPU|IPU' |
    Select-Object DeviceName,DriverVersion | Format-List

$original = $env:Path
$env:PYTHONUTF8 = '1'
$env:PYTHONDONTWRITEBYTECODE = '1'
$env:OMP_WAIT_POLICY = 'PASSIVE'
$env:KMP_BLOCKTIME = '0'
$env:OMP_NUM_THREADS = '2'
$env:MKL_NUM_THREADS = '2'
$Out = $ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($Out)
[IO.Directory]::CreateDirectory($Out) | Out-Null
$env:TEMP = $Out
$env:TMP = $Out
foreach ($key in 'npu_python','text_python','ryzen_ai_path','genai_sd_path','model_path','text_encoder_path','parts_path') {
    $value = [IO.Path]::GetFullPath($local.$key, (Split-Path $Config))
    Write-Host "$key = $value"
    if (-not (Test-Path -LiteralPath $value)) { throw "Configured location is missing: $value" }
}
$env:RYZEN_AI_INSTALLATION_PATH = [IO.Path]::GetFullPath($local.ryzen_ai_path, (Split-Path $Config))
$npu = [IO.Path]::GetFullPath($local.npu_python, (Split-Path $Config))
$text = [IO.Path]::GetFullPath($local.text_python, (Split-Path $Config))
function Use-Python([string]$exe) {
    $prefix = Split-Path $exe
    $env:Path = "$prefix;$prefix\Library\bin;$prefix\Scripts;$original"
}
$packages = Join-Path $PSScriptRoot 'validation\list_packages.py'
Use-Python $text
& $text -B $packages torch transformers diffusers accelerate numpy safetensors pillow psutil huggingface_hub
& $text -B (Join-Path $PSScriptRoot 'tools\download_models.py') --config $Config --check-only
Write-Host '== Parts manifest check =='
& $text -B (Join-Path $PSScriptRoot 'validation\check_parts.py') --config $Config
Use-Python $npu
& $npu -B $packages torch onnx onnxruntime-vitisai onnxruntime-providers-ryzenai ryzenai-dynamic-dispatch ryzenai-onnx-utils numpy safetensors pillow psutil
Push-Location $Out
try {
    $report = Join-Path $Out 'doctor.json'
    & $npu -B -u (Join-Path $PSScriptRoot 'src\doctor_probe.py') --config $Config --output $report
    Get-Content -LiteralPath $report
    Write-Host 'probe exit 0'
} finally {
    Pop-Location
}

