# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Preview by default. Pass -apply to save a backup and update the stale paths.
param(
    [string]$grasp_root = '\\wsl.localhost\Ubuntu\home\nmt\Projects\GraspXL',
    [switch]$apply
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$registry_path = 'Software\RaiSim Tech\RaiSimUnity'
$old_prefix = 'C:\Linux\GraspXL'
$count_name = 'NoRscDir_h464876921'
$encoding = New-Object System.Text.UTF8Encoding($false, $true)
$read_key = [Microsoft.Win32.Registry]::CurrentUser.OpenSubKey($registry_path, $false)
if ($null -eq $read_key) {
    throw "The viewer settings do not exist: HKCU\$registry_path"
}

try {
    if ($read_key.GetValueKind($count_name) -ne [Microsoft.Win32.RegistryValueKind]::DWord) {
        throw 'The resource-directory count has an unexpected registry type.'
    }
    $resource_count = $read_key.GetValue($count_name)
    $resources = @(
        foreach ($name in $read_key.GetValueNames()) {
            if ($name -notmatch '^RscDir[0-9]+_h[0-9]+$') {
                continue
            }
            $kind = $read_key.GetValueKind($name)
            if ($kind -ne [Microsoft.Win32.RegistryValueKind]::Binary) {
                throw "Unexpected registry type for $name; no changes were made."
            }
            [byte[]]$bytes = $read_key.GetValue($name)
            $old_path = $encoding.GetString($bytes).TrimEnd([char]0)
            $new_path = $old_path
            if ($old_path.Equals($old_prefix, [StringComparison]::Ordinal) -or
                $old_path.StartsWith($old_prefix + '\', [StringComparison]::Ordinal)) {
                $new_path = $grasp_root.TrimEnd([char]92) + $old_path.Substring($old_prefix.Length)
            }
            [PSCustomObject]@{
                name = $name
                kind = $kind.ToString()
                bytes_base64 = [Convert]::ToBase64String($bytes)
                old_path = $old_path
                new_path = $new_path
            }
        }
    )
}
finally {
    $read_key.Dispose()
}

$changes = @($resources | Where-Object { $_.old_path -cne $_.new_path })
if ($changes.Count -eq 0) {
    Write-Output 'No stale GraspXL viewer resource paths remain; no changes were made.'
    exit 0
}
if ($null -ne (Get-Process -Name 'RaiSimUnity' -ErrorAction SilentlyContinue)) {
    throw 'Close RaiSimUnity before migrating settings so it cannot overwrite them on exit.'
}
foreach ($entry in $changes) {
    if (-not (Test-Path -LiteralPath $entry.new_path -PathType Container)) {
        throw "Replacement resource directory does not exist: $($entry.new_path)"
    }
    Get-ChildItem -LiteralPath $entry.new_path -Force -ErrorAction Stop |
        Select-Object -First 1 | Out-Null
}
Write-Output "Verified $($changes.Count) replacement resource directories through the Windows WSL share."
if (-not $apply) {
    Write-Output 'Preview only. Pass -apply to back up and update these paths.'
    exit 0
}

$backup_directory = Join-Path $grasp_root 'build'
[IO.Directory]::CreateDirectory($backup_directory) | Out-Null
$timestamp = [DateTime]::UtcNow.ToString('yyyyMMddTHHmmssfffZ')
$backup_path = Join-Path $backup_directory "viewer-resource-settings-before-migration-$timestamp.json"
$backup = [ordered]@{
    saved_utc = [DateTime]::UtcNow.ToString('o')
    registry_path = "HKEY_CURRENT_USER\$registry_path"
    count = [ordered]@{ name = $count_name; kind = 'DWord'; value = $resource_count }
    resources = $resources
}
$backup_text = $backup | ConvertTo-Json -Depth 6
$backup_bytes = $encoding.GetBytes($backup_text)
$backup_file = [IO.File]::Open($backup_path, [IO.FileMode]::CreateNew, [IO.FileAccess]::Write)
try {
    $backup_file.Write($backup_bytes, 0, $backup_bytes.Length)
    $backup_file.Flush($true)
}
finally {
    $backup_file.Dispose()
}
Write-Output "Saved original resource values, types, and count to $backup_path"

$write_key = [Microsoft.Win32.Registry]::CurrentUser.OpenSubKey($registry_path, $true)
if ($null -eq $write_key) {
    throw 'Unable to open the existing viewer registry key for writing.'
}
$written = New-Object 'System.Collections.Generic.List[object]'
try {
    if ($write_key.GetValueKind($count_name) -ne [Microsoft.Win32.RegistryValueKind]::DWord -or
        $write_key.GetValue($count_name) -ne $resource_count) {
        throw 'The resource count changed during migration; no resource paths were updated.'
    }
    foreach ($entry in $resources) {
        if ($write_key.GetValueKind($entry.name) -ne [Microsoft.Win32.RegistryValueKind]::Binary -or
            [Convert]::ToBase64String([byte[]]$write_key.GetValue($entry.name)) -cne $entry.bytes_base64) {
            throw "Viewer settings changed during migration: $($entry.name)"
        }
    }
    foreach ($entry in $changes) {
        [byte[]]$new_bytes = $encoding.GetBytes($entry.new_path + [char]0)
        $write_key.SetValue($entry.name, $new_bytes, [Microsoft.Win32.RegistryValueKind]::Binary)
        $written.Add($entry)
        if ([Convert]::ToBase64String([byte[]]$write_key.GetValue($entry.name)) -cne
            [Convert]::ToBase64String($new_bytes)) {
            throw "Registry verification failed: $($entry.name)"
        }
    }
    $write_key.Flush()
}
catch {
    foreach ($entry in $written) {
        $write_key.SetValue(
            $entry.name,
            [Convert]::FromBase64String($entry.bytes_base64),
            [Microsoft.Win32.RegistryValueKind]::Binary
        )
    }
    $write_key.Flush()
    throw
}
finally {
    $write_key.Dispose()
}
Write-Output "Updated $($changes.Count) resource paths. The resource count and other viewer settings were preserved."
