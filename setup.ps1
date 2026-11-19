param(
    [string]$NpuEnv = 'h3npu-npu',
    [string]$TextEnv = 'h3npu-text',
    [string]$RyzenPath = (Join-Path $env:ProgramFiles 'RyzenAI\1.8.0'),
    [string]$MiniMaxPath,
    [string]$TextEncoderPath,
    [string]$GenAiSdPath,
    [string]$DiffusersPath,
    [string]$HfRevision,
    [switch]$SkipDownloads,
    [switch]$SkipParts,
    [switch]$WithDiTWeights,
    [switch]$VerifyModels
)
$ErrorActionPreference = 'Stop'
$PSNativeCommandUseErrorActionPreference = $true
$root = $PSScriptRoot
$example = Get-Content -LiteralPath (Join-Path $root 'configs\example.json') -Raw | ConvertFrom-Json
if (-not $HfRevision) { $HfRevision = $example.parts_revision }
if (-not ($SkipDownloads -or $SkipParts) -and $HfRevision -cnotmatch '\A[0-9a-f]{40}\z') {
    [Console]::Error.WriteLine('HF revision is not pinned in configs/example.json. Set parts_revision to a full commit hash or pass -HfRevision.')
    exit 2
}
if ($NpuEnv -eq $TextEnv) { throw 'Use distinct text and NPU environments.' }
if (-not $MiniMaxPath) { $MiniMaxPath = Join-Path $root 'models\MiniMax-H3-Image' }
if (-not $TextEncoderPath) {
    $TextEncoderPath = Join-Path $MiniMaxPath 'text_encoder'
    if (-not (Test-Path -LiteralPath (Join-Path $TextEncoderPath 'model.safetensors.index.json'))) {
        $TextEncoderPath = Join-Path $root 'models\MiniMax-H3\text_encoder'
    }
}
if (-not $GenAiSdPath) { $GenAiSdPath = Join-Path $RyzenPath 'GenAI-SD' }
$RyzenPath = (Resolve-Path -LiteralPath $RyzenPath).Path
$GenAiSdPath = (Resolve-Path -LiteralPath $GenAiSdPath).Path
$MiniMaxPath = $ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($MiniMaxPath)
$TextEncoderPath = $ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($TextEncoderPath)
if ($DiffusersPath) { $DiffusersPath = (Resolve-Path -LiteralPath $DiffusersPath).Path }

$conda = (Get-Command conda.exe -ErrorAction SilentlyContinue).Source
if (-not $conda) {
    $conda = @(
        (Join-Path $env:USERPROFILE 'miniforge3\Scripts\conda.exe'),
        (Join-Path $env:ProgramFiles 'Miniforge3\Scripts\conda.exe')
    ) | Where-Object { Test-Path -LiteralPath $_ } | Select-Object -First 1
}
if (-not $conda) { throw 'conda.exe was not found. Open a conda PowerShell prompt.' }

$condaRoot = Split-Path (Split-Path $conda)
$envRegistry = Join-Path $env:USERPROFILE '.conda\environments.txt'
$envFolders = @(
    (Join-Path $condaRoot 'envs'),
    (Join-Path $env:USERPROFILE '.conda\envs')
)
if ($env:CONDA_ENVS_PATH) { $envFolders += $env:CONDA_ENVS_PATH -split [IO.Path]::PathSeparator }

function Find-EnvPython([string]$Name) {
    if ($Name -notmatch '^[A-Za-z0-9][A-Za-z0-9_.-]*$') { throw "Invalid conda environment name: $Name" }
    $paths = @($envFolders | ForEach-Object { Join-Path $_ $Name })
    if (Test-Path -LiteralPath $envRegistry) {
        $paths += @(Get-Content -LiteralPath $envRegistry |
            Where-Object { $_ -and (Split-Path $_ -Leaf) -eq $Name })
    }
    foreach ($path in $paths) {
        $python = Join-Path $path 'python.exe'
        if (Test-Path -LiteralPath $python -PathType Leaf) { return $python }
    }
}

function Invoke-Python([string]$Python, [string[]]$Arguments) {
    $prefix = Split-Path $Python
    $original = $env:Path
    try {
        $env:Path = "$prefix;$prefix\Library\bin;$prefix\Scripts;$original"
        $oldUtf8 = $env:PYTHONUTF8
        $env:PYTHONUTF8 = '1'
        & $Python -B @Arguments
    } finally {
        $env:PYTHONUTF8 = $oldUtf8
        $env:Path = $original
    }
}

function Install-Diffusers([string]$Python, [string]$Revision) {
    # Keep the wheel build path short enough for Windows filesystem tools.
    $prefix = Split-Path $Python
    $source = Join-Path $prefix "df-$([guid]::NewGuid().ToString('N').Substring(0, 8))"
    [IO.Directory]::CreateDirectory($source) | Out-Null
    try {
        & git -C $source -c core.longpaths=true init --quiet
        & git -C $source -c core.longpaths=true fetch --quiet --depth=1 https://github.com/huggingface/diffusers.git $Revision
        & git -C $source -c core.longpaths=true checkout --quiet --detach $Revision
        Write-Host "Building Diffusers at revision $Revision"
        Invoke-Python $Python @('-m', 'pip', 'install', $source)
    } finally {
        $resolved = Get-Item -LiteralPath $source
        if ($resolved.Parent.FullName -eq $prefix -and !$resolved.LinkType) {
            Remove-Item -LiteralPath $resolved.FullName -Recurse -Force
        }
    }
}

$npuPython = Find-EnvPython $NpuEnv
if (-not $npuPython) {
    $started = Get-Date
    $envYaml = Join-Path ([IO.Path]::GetTempPath()) "h3npu-$([guid]::NewGuid().ToString('N')).yaml"
    try {
        Copy-Item -LiteralPath (Join-Path $RyzenPath 'env.yaml') -Destination $envYaml
        & $conda env create --name $NpuEnv --file $envYaml --yes
    } finally {
        if (Test-Path -LiteralPath $envYaml) { Remove-Item -LiteralPath $envYaml }
    }
    $npuPython = Find-EnvPython $NpuEnv
    $wheels = @(Get-ChildItem -LiteralPath $RyzenPath -Filter '*.whl' -File | ForEach-Object FullName)
    Invoke-Python $npuPython (@('-m', 'pip', 'install', '-r', (Join-Path $root 'environments\npu-extra.txt')) + $wheels)
    Invoke-Python $npuPython @('-m', 'pip', 'check')
    Write-Host "Created $NpuEnv in $([math]::Round(((Get-Date) - $started).TotalSeconds, 1)) s"
} else {
    Write-Host "Using existing NPU environment $NpuEnv"
}
$textPython = Find-EnvPython $TextEnv
if (-not $textPython) {
    $started = Get-Date
    & $conda create --name $TextEnv --yes python=3.12.16 pip
    $textPython = Find-EnvPython $TextEnv
    Invoke-Python $textPython @('-m', 'pip', 'install', 'torch==2.15.0', '--index-url',
        'https://download.pytorch.org/whl/cpu')
    Invoke-Python $textPython @('-m', 'pip', 'install', '-r', (Join-Path $root 'environments\text.txt'))
    Install-Diffusers $textPython $example.diffusers_revision
    Invoke-Python $textPython @('-m', 'pip', 'check')
    Write-Host "Created $TextEnv in $([math]::Round(((Get-Date) - $started).TotalSeconds, 1)) s"
} else {
    Write-Host "Using existing text environment $TextEnv"
}

$local = [ordered]@{
    npu_python = $npuPython
    text_python = $textPython
    ryzen_ai_path = $RyzenPath
    genai_sd_path = $GenAiSdPath
    model_path = $MiniMaxPath
    text_encoder_path = $TextEncoderPath
    parts_path = Join-Path $root 'parts'
    parts_repo = $example.parts_repo
    parts_revision = $HfRevision
    image_repo = $example.image_repo
    image_revision = $example.image_revision
    text_repo = $example.text_repo
    text_revision = $example.text_revision
    diffusers_revision = $example.diffusers_revision
    min_available_gb = $example.min_available_gb
}
if ($DiffusersPath) { $local.diffusers_source = $DiffusersPath }
$config = Join-Path $root 'configs\local.json'
$pending = Join-Path $root 'configs\local.pending.json'
$local | ConvertTo-Json | Set-Content -LiteralPath $pending -Encoding utf8
try {
    $options = @((Join-Path $root 'tools\download_models.py'), '--config', $pending, '--install-parts')
    if ($SkipDownloads) { $options += '--check-only' }
    if ($SkipParts) { $options += '--skip-parts' }
    if ($WithDiTWeights) { $options += '--with-dit-weights' }
    if ($VerifyModels) { $options += '--verify-existing' }
    Invoke-Python $textPython $options
    [IO.File]::Move($pending, $config, $true)
} finally {
    if (Test-Path -LiteralPath $pending) { Remove-Item -LiteralPath $pending }
}
Write-Host 'Wrote configs/local.json.'

