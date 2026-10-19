# One-time publish of saudi-housing-finance-macro to GitHub (run from PowerShell).
#   cd D:\download\claude\Coding\saudi-housing-finance-macro
#   powershell -ExecutionPolicy Bypass -File _setup\publish_to_github.ps1            # private repo (default)
#   powershell -ExecutionPolicy Bypass -File _setup\publish_to_github.ps1 -Public    # public repo
param([switch]$Public, [string]$Owner = "waheedsaif78", [string]$Repo = "saudi-housing-finance-macro")
$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)

# 1) Put the CI workflow in place (remote tools may not write into .github)
New-Item -ItemType Directory -Force .github\workflows | Out-Null
Copy-Item _setup\ci.yml .github\workflows\ci.yml -Force

# 2) Remove files left over from the earlier version of the project
foreach ($f in @("pytest.ini", "data\raw\gastat__REPI_Q3_2024.xlsx", "data\raw\kapsarc__interest-rates-and-sama-average-bills.csv")) {
    if (Test-Path $f) { Remove-Item $f; Write-Host "removed $f" }
}

# 3) Local repository + first commit
if (-not (Test-Path .git)) { git init -b main | Out-Null }
git config user.name  "Waheed Saif"
git config user.email "$Owner@users.noreply.github.com"   # keeps your work e-mail out of public history
git config core.autocrlf true
git add -A

git commit -m "Saudi housing finance & macro analysis pipeline" -m "Links GASTAT's Real Estate Price Index to SAMA residential new-mortgage flows and interest rates: lead/lag tests, finance mix vs prices, rate sensitivity, 4-quarter forecasts and SAIBOR scenarios." -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"

# 4) Create the GitHub repo and push
$visibility = if ($Public) { "--public" } else { "--private" }
if (Get-Command gh -ErrorAction SilentlyContinue) {
    gh repo create "$Owner/$Repo" $visibility --source . --remote origin --push --description "Saudi real-estate prices vs mortgage finance and interest rates (GASTAT, SAMA)"
} else {
    Write-Host "GitHub CLI not found. Create an EMPTY repo named '$Repo' at https://github.com/new (no README), then press Enter."
    Read-Host | Out-Null
    git remote add origin "https://github.com/$Owner/$Repo.git"
    git push -u origin main
}
Write-Host "Done: https://github.com/$Owner/$Repo"
