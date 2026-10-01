param([switch]$TensorRT)
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$tag = 'v8.4.168'
if (-not (Test-Path -LiteralPath 'ultralytics-src/.git')) {
    git clone --branch $tag https://github.com/ultralytics/ultralytics.git ultralytics-src
    if ($LASTEXITCODE -ne 0) { throw 'Ultralytics clone failed.' }
}
if (-not (Test-Path -LiteralPath 'vendor/ultralytics/.git')) {
    git -C ultralytics-src worktree add -b ym-gray16 ../vendor/ultralytics $tag
    if ($LASTEXITCODE -ne 0) { throw 'Ultralytics worktree creation failed.' }
}
uv python install 3.11.13
if ($LASTEXITCODE -ne 0) { throw 'Python installation failed.' }
if ($TensorRT) { uv sync --locked --extra tensorrt } else { uv sync --locked }
if ($LASTEXITCODE -ne 0) { throw 'uv sync failed.' }
uv run --no-sync python -m tools.check_environment
if ($LASTEXITCODE -ne 0) { throw 'GPU verification failed.' }
