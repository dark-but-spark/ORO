param(
    [string]$ManualReviewDir = "runsTemp\manual_review_package_20260724",
    [string]$OutputDir = "runsTemp\inflammation_gt_review_20260727",
    [int]$MaxSamples = 30
)

$ErrorActionPreference = "Stop"
Add-Type -AssemblyName System.Drawing

$sheetPath = Join-Path $ManualReviewDir "manual_review_sheet.csv"
if (-not (Test-Path $sheetPath)) {
    throw "manual_review_sheet.csv not found: $sheetPath"
}

$outRoot = New-Item -ItemType Directory -Force -Path $OutputDir
$imageDir = New-Item -ItemType Directory -Force -Path (Join-Path $outRoot.FullName "images")
$fullPanelDir = New-Item -ItemType Directory -Force -Path (Join-Path $outRoot.FullName "full_panels")

$rows = Import-Csv $sheetPath |
    Where-Object { $_.review_group -eq "03_class2_hard" } |
    Sort-Object {[int]$_.review_order}

$seen = @{}
$selected = @()
foreach ($row in $rows) {
    if ($seen.ContainsKey($row.image_file)) {
        continue
    }
    $seen[$row.image_file] = $true
    $selected += $row
    if ($selected.Count -ge $MaxSamples) {
        break
    }
}

function Convert-ToNumber($value) {
    if ([string]::IsNullOrWhiteSpace($value)) {
        return 0.0
    }
    return [double]$value
}

function Copy-CropPanel($sourcePath, $destPath) {
    $resolved = Resolve-Path $sourcePath
    $src = [System.Drawing.Bitmap]::FromFile($resolved)
    try {
        $panelWidth = [int]($src.Width / 7)
        $panelHeight = $src.Height
        $columns = @(0, 1, 5, 6)
        $out = New-Object System.Drawing.Bitmap ($panelWidth * $columns.Count), $panelHeight
        try {
            $graphics = [System.Drawing.Graphics]::FromImage($out)
            try {
                $graphics.Clear([System.Drawing.Color]::Black)
                for ($i = 0; $i -lt $columns.Count; $i++) {
                    $sourceRect = New-Object System.Drawing.Rectangle ($columns[$i] * $panelWidth), 0, $panelWidth, $panelHeight
                    $destRect = New-Object System.Drawing.Rectangle ($i * $panelWidth), 0, $panelWidth, $panelHeight
                    $graphics.DrawImage($src, $destRect, $sourceRect, [System.Drawing.GraphicsUnit]::Pixel)
                }
            }
            finally {
                $graphics.Dispose()
            }
            $out.Save($destPath, [System.Drawing.Imaging.ImageFormat]::Jpeg)
        }
        finally {
            $out.Dispose()
        }
    }
    finally {
        $src.Dispose()
    }
}

$reviewRows = @()
$index = 1
foreach ($row in $selected) {
    $panelPath = $row.tta_panel_package_path
    $panelMode = "tta"
    if ([string]::IsNullOrWhiteSpace($panelPath) -or -not (Test-Path $panelPath)) {
        $panelPath = $row.no_tta_panel_package_path
        $panelMode = "no_tta"
    }
    if ([string]::IsNullOrWhiteSpace($panelPath) -or -not (Test-Path $panelPath)) {
        continue
    }

    $stem = [System.IO.Path]::GetFileNameWithoutExtension($row.image_file)
    $shortStem = $stem
    if ($shortStem.Length -gt 80) {
        $shortStem = $shortStem.Substring(0, 80)
    }
    $compactName = "{0:000}_{1}_{2}.jpg" -f $index, $panelMode, $shortStem
    $compactPath = Join-Path $imageDir.FullName $compactName
    Copy-CropPanel $panelPath $compactPath

    $fullName = "{0:000}_{1}_{2}.jpg" -f $index, $panelMode, $shortStem
    $fullPath = Join-Path $fullPanelDir.FullName $fullName
    Copy-Item -Force $panelPath $fullPath

    $class2True = Convert-ToNumber $row.class_2_true_pixels
    $class2Pred = Convert-ToNumber $row.class_2_pred_pixels
    $class2Fp = Convert-ToNumber $row.class_2_fp_pixels
    $class2Fn = Convert-ToNumber $row.class_2_fn_pixels
    $class2Dice = Convert-ToNumber $row.class_2_dice
    $class2Prob = Convert-ToNumber $row.class_2_prob_mean

    $autoHint = "Check whether the blue GT class2 region is regional inflammatory-cell infiltration."
    if ($class2True -eq 0 -and $class2Pred -gt 0) {
        $autoHint = "Prediction has class2 but GT has none; check single-cell/scattered-cell false positive."
    }
    elseif ($class2True -gt 0 -and $class2Dice -lt 0.1) {
        $autoHint = "GT has class2 but prediction barely overlaps; check if GT is too broad or only scattered cells."
    }

    $reviewRows += [PSCustomObject]@{
        review_order = $index
        image_file = $row.image_file
        compact_gt_review_image = $compactPath
        full_panel_image = $fullPath
        panel_mode = $panelMode
        class2_dice = $class2Dice
        class2_true_pixels = $class2True
        class2_pred_pixels = $class2Pred
        class2_fp_pixels = $class2Fp
        class2_fn_pixels = $class2Fn
        class2_prob_mean = $class2Prob
        auto_hint = $autoHint
        gt_inflammation_valid = ""
        inflammation_pattern = ""
        model_single_cell_false_positive = ""
        suggested_action = ""
        reviewer = ""
        notes = ""
    }
    $index += 1
}

$csvPath = Join-Path $outRoot.FullName "inflammation_gt_review_sheet.csv"
$reviewRows | Export-Csv -NoTypeInformation -Encoding UTF8 $csvPath

$readmePath = Join-Path $outRoot.FullName "README_inflammation_gt_review.md"
$readme = @"
# Inflammation GT Review

This package focuses on class2 hard cases.

Image layout in images/:

```text
image / ground_truth / gt_overlay / pred_overlay
```

Class color map:

```text
class0 red
class1 green
class2 blue
class3 yellow
```

Please review class2 blue regions and fill inflammation_gt_review_sheet.csv.
"@
$readme | Set-Content -Encoding UTF8 $readmePath

$summaryPath = Join-Path $outRoot.FullName "summary.json"
$summary = [PSCustomObject]@{
    source_manual_review_dir = (Resolve-Path $ManualReviewDir).Path
    output_dir = (Resolve-Path $OutputDir).Path
    selected_samples = $reviewRows.Count
    source_group = "03_class2_hard"
    generated_at = (Get-Date).ToString("yyyy-MM-dd HH:mm:ss")
}
$summary | ConvertTo-Json -Depth 4 | Set-Content -Encoding UTF8 $summaryPath

Write-Output "Inflammation GT review package created: $($outRoot.FullName)"
Write-Output "Selected samples: $($reviewRows.Count)"
Write-Output "Review sheet: $csvPath"
Write-Output "README: $readmePath"
