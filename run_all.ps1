if (Test-Path ".\out") {
    Write-Host "🧹 Cleaning previous outputs..."
    Remove-Item -Recurse -Force ".\out"
}

py -m src.batch_run
py -m src.auto_analysis
py -m src.make_showcase
