param(
    [string]$Config = "src/configs/officehome_art2real.yaml"
)

Write-Host "[Local] Task1 - Spatial CycleGAN"
python src/train_task1_style.py --config $Config --mode spatial

Write-Host "[Local] Task1 - Spectral CycleGAN"
python src/train_task1_style.py --config $Config --mode spectral

$strategies = @("source_only", "cyclegan", "spectral_cyclegan", "cycada", "fda")
foreach ($s in $strategies) {
    Write-Host "[Local] Task2 - $s"
    python src/train_task2_uda.py --config $Config --strategy $s
}
