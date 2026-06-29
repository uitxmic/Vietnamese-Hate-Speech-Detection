$ErrorActionPreference = "Stop"

Add-Type -AssemblyName System.Drawing

$heatmapDir = Join-Path $PSScriptRoot "Heatmap"

function From-Utf8Base64 {
    param([string]$Text)

    return [System.Text.Encoding]::UTF8.GetString([System.Convert]::FromBase64String($Text))
}

$captionMap = @{
    "TP_Toxic_to_Toxic" = From-Utf8Base64 "4bqibmggaMaw4bufbmcgY+G7p2EgdOG7qyDEkeG6v24gZOG7sSDEkW/DoW4ga+G6v3QgcXXhuqMgVG94aWMgKGThu7EgxJFvw6FuIMSRw7puZyk="
    "TN_Clean_to_Clean" = From-Utf8Base64 "4bqibmggaMaw4bufbmcgY+G7p2EgdOG7qyDEkeG6v24gZOG7sSDEkW/DoW4ga+G6v3QgcXXhuqMgQ2xlYW4gKGThu7EgxJFvw6FuIMSRw7puZyk="
    "FP_Clean_to_Toxic" = From-Utf8Base64 "4bqibmggaMaw4bufbmcgY+G7p2EgdOG7qyDEkeG6v24gZOG7sSDEkW/DoW4ga+G6v3QgcXXhuqMgVG94aWMgKHNhaTogbmjDo24gdGjhuq10IENsZWFuKQ=="
    "FN_Toxic_to_Clean" = From-Utf8Base64 "4bqibmggaMaw4bufbmcgY+G7p2EgdOG7qyDEkeG6v24gZOG7sSDEkW/DoW4ga+G6v3QgcXXhuqMgQ2xlYW4gKHNhaTogbmjDo24gdGjhuq10IFRveGljKQ=="
}

$sampleLabel = From-Utf8Base64 "IC0gTeG6q3Ug"

function Get-CaptionForFile {
    param([string]$FileName)

    foreach ($key in $captionMap.Keys) {
        if ($FileName.StartsWith($key)) {
            $sample = [System.IO.Path]::GetFileNameWithoutExtension($FileName).Substring($key.Length + 1)
            return "$($captionMap[$key])$sampleLabel$sample"
        }
    }

    return $null
}

Get-ChildItem -Path $heatmapDir -Filter "*.png" | ForEach-Object {
    $caption = Get-CaptionForFile -FileName $_.Name
    if (-not $caption) {
        Write-Warning "Skipping unknown heatmap file: $($_.Name)"
        return
    }

    $image = [System.Drawing.Image]::FromFile($_.FullName)
    try {
        $bitmap = New-Object System.Drawing.Bitmap $image.Width, $image.Height
        $graphics = [System.Drawing.Graphics]::FromImage($bitmap)
        try {
            $graphics.SmoothingMode = [System.Drawing.Drawing2D.SmoothingMode]::AntiAlias
            $graphics.TextRenderingHint = [System.Drawing.Text.TextRenderingHint]::AntiAliasGridFit
            $graphics.Clear([System.Drawing.Color]::White)
            $graphics.DrawImage($image, 0, 0, $image.Width, $image.Height)

            $titleHeight = [Math]::Max(95, [Math]::Round($image.Height * 0.13))
            $graphics.FillRectangle([System.Drawing.Brushes]::White, 0, 0, $image.Width, $titleHeight)

            $fontSize = [Math]::Max(24, [Math]::Min(46, [Math]::Round($image.Width / 95)))
            $font = New-Object System.Drawing.Font("Arial", $fontSize, [System.Drawing.FontStyle]::Regular, [System.Drawing.GraphicsUnit]::Pixel)
            try {
                $format = New-Object System.Drawing.StringFormat
                $format.Alignment = [System.Drawing.StringAlignment]::Center
                $format.LineAlignment = [System.Drawing.StringAlignment]::Center

                $rect = New-Object System.Drawing.RectangleF(0, 0, $image.Width, $titleHeight)
                $graphics.DrawString($caption, $font, [System.Drawing.Brushes]::Black, $rect, $format)
            }
            finally {
                if ($format) { $format.Dispose() }
                $font.Dispose()
            }
        }
        finally {
            $graphics.Dispose()
        }
    }
    finally {
        $image.Dispose()
    }

    $tempPath = "$($_.FullName).tmp"
    try {
        $bitmap.Save($tempPath, [System.Drawing.Imaging.ImageFormat]::Png)
        Move-Item -LiteralPath $tempPath -Destination $_.FullName -Force
    }
    finally {
        if ($bitmap) { $bitmap.Dispose() }
        if (Test-Path -LiteralPath $tempPath) { Remove-Item -LiteralPath $tempPath -Force }
    }

    Write-Host "Updated: $($_.Name)"
}
