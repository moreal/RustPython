# Puts the Cargo target directory of a Windows job on a Dev Drive with
# antivirus filtering disabled, which is where most of the I/O of a Rust
# build happens. Based on
# https://github.com/DioxusLabs/dioxus/blob/main/.github/workflows/setup-dev-drive.ps1
#
# The volume is mounted at the workspace's target directory instead of the
# job moving to a drive letter, so the paths the workflow and its caches use
# stay the same. Run it after checkout and before restoring the cache.

$ErrorActionPreference = "Stop"

$Vhd = "C:\dev_drive.vhdx"
$Target = Join-Path $env:GITHUB_WORKSPACE "target"

$Partition = New-VHD -Path $Vhd -SizeBytes 50GB -Dynamic |
    Mount-VHD -Passthru |
    Initialize-Disk -Passthru |
    New-Partition -AssignDriveLetter -UseMaximumSize
$Volume = $Partition | Format-Volume -DevDrive -Confirm:$false -Force
$Drive = "$($Volume.DriveLetter):"

fsutil devdrv trust $Drive
fsutil devdrv enable /disallowAv

# Remount so the changes take effect.
Dismount-VHD -Path $Vhd
Mount-VHD -Path $Vhd
fsutil devdrv query $Drive

$Partition = Get-Partition -DriveLetter $Volume.DriveLetter
New-Item $Target -ItemType Directory -Force | Out-Null
$Partition | Add-PartitionAccessPath -AccessPath $Target

$Tmp = "$Drive\tmp"
New-Item $Tmp -ItemType Directory -Force | Out-Null

Write-Output "Dev Drive $Drive is mounted at $Target"
Write-Output `
    "DEV_DRIVE=$Drive" `
    "TMP=$Tmp" `
    "TEMP=$Tmp" `
    >> $env:GITHUB_ENV
