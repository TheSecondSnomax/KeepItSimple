function Test-Command {
    if ($LASTEXITCODE -ne 0) {
        Write-Output "`nERROR! Check above."
        exit $LASTEXITCODE
    }
}


isort .
Test-Command

black .
Test-Command

docformatter src
Test-Command

mypy src
Test-Command

pylint src
Test-Command

coverage run -m unittest discover ./tests/unit
Test-Command

coverage combine --quiet
coverage report
Test-Command

Write-Output "`nAll is well."
